using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32, :img_channels => 1,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :λpatch => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0), :dropout => 0.2,
)
args[:imzprod] = prod(args[:img_size])

## =====

device!(0)

dev = gpu

##=====

all_chars = load("../Recur_generative/data/exp_pro/omniglot_train.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
num_train = trunc(Int, 0.8 * length(xs))

new_chars = load("../Recur_generative/data/exp_pro/omniglot_eval.jld2")
xs_new = shuffle(vcat((new_chars[key] for key in keys(new_chars))...))

function fast_cat(xs)
    x_array = zeros(Float32, size(xs[1])..., length(xs))
    Threads.@threads for i in 1:length(xs)
        x_array[:, :, i] = xs[i]
    end
    x_array
end

xs_cat = fast_cat(xs)
train_chars = xs_cat[:, :, 1:num_train]
val_chars = xs_cat[:, :, num_train+1:end]

new_xs_cat = fast_cat(xs_new)
## ====
train_loader = DataLoader(train_chars |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
val_loader = DataLoader(val_chars |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(new_xs_cat |> dev, batchsize=args[:bsz], shuffle=true, partial=false)

## =====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev
## =====

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], sin), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=sin)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)

    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function init_H!(H; f=0.4f0)
    for p in Flux.params(H)
        p = f * p
    end
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    Flux.trainmode!((Hx, Ha, Encoder))
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                log_value(lg, "KL loss", klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            full_loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        foreach(x -> clamp!(x, -1.0f0, 1.0f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

## ====

# todo don't bind RNN size to args[:π]

args[:π] = 100

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_enc_e_z = (784 + 1) * args[:π]

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x]


l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

Hx = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    Dropout(args[:dropout]),
    LayerNorm(64, elu),
    Dropout(args[:dropout]),
    Dense(64, 64),
    Dropout(args[:dropout]),
    LayerNorm(64, elu),
    Dense(64, 64),
    Dropout(args[:dropout]),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    Dropout(args[:dropout]),
    LayerNorm(64, elu),
    Dense(64, 64),
    Dropout(args[:dropout]),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
) |> gpu


Encoder = let
    enc1 = Chain(
        x -> reshape(x, 28, 28, 1, :),
        Conv((5, 5), 1 => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        flatten,
    )
    outsz = Flux.outputsize(enc1, (28, 28, 1, args[:bsz]))
    Chain(
        enc1,
        Dense(outsz[1], 64,),
        LayerNorm(64, elu),
        Dense(64, 64,),
        Dropout(args[:dropout]),
        LayerNorm(64, elu),
        Dense(64, 64,),
        Dropout(args[:dropout]),
        LayerNorm(64, elu),
        Split(
            Dense(64, args[:π]),
            Dense(64, args[:π])
        )
    )
end |> gpu

ps = Flux.params(Hx, Ha, Encoder)

## ======

## ======

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "gen_2lvl"
alias = "double_H_omni_generative_v02_dropout"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 4
args[:scale_offset] = 2.2f0

# args[:λpatch] = Float32(1 / 2 * args[:seqlen])
args[:λpatch] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.2f0


args[:η] = 4e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
## ====
begin
    Ls = []
    for epoch in 1:200
        if epoch % 100 == 0
            opt.eta = max(0.67 * opt.eta, 5e-7)
            log_value(lg, "learning_rate", opt.eta)
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        Flux.testmode!((Hx, Ha, Encoder))
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        p2 = plot_recs(sample_loader(val_loader), inds)
        p3 = plot(p, p2, layout=(1, 2))
        display(p3)
        log_image(lg, "recs_$(epoch)", p3)


        Ltest = test_model(test_loader)
        Lval = test_model(val_loader)
        log_value(lg, "test_loss", Ltest)
        log_value(lg, "val_loss", Lval)
        @info "Test loss: $Ltest - Val loss: $Lval"
        push!(Ls, ls)
        if epoch % 250 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

