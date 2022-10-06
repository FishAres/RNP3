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
    :esz => 32, :asz => 6, :add_offset => true, :fa_out => identity, :f_z => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :λpatch => Float32(1 / 4),
    :scale_offset => 2.8f0, :D => Normal(0.0f0, 1.0f0),
)
args[:imzprod] = prod(args[:img_size])

## =====

device!(1)

dev = gpu

##=====
train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)


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
function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z], fa=args[:fa_out])
    inds = Zygote.ignore() do
        [0; cumsum([Hx_bounds...; args[:π]; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=fz)

    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[3], relu6), flatten)

    z0 = fz.(Θ[4])
    a0 = fa.(Θ[5])
    return (Enc_za_z, f_state, Dec_z_x̂,), z0, a0
end

function get_models(θsz; args=args, Hx_bounds=Hx_bounds, Ha_bounds=Ha_bounds)
    (Enc_za_z, f_state, Dec_z_x̂), z0, a0 = get_fstate_models(θsz, Hx_bounds; args=args)
    models = f_state, Enc_za_z, Dec_z_x̂, fpolicy
    return models, z0, a0
end


function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, Enc_za_z, Dec_z_x̂, fpolicy = models
    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    z1 = f_state(ez)
    a1 = fpolicy(za)

    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset) |> flatten

    return z1, a1, x̂, patch_t
end

function full_sequence(models::Tuple, z0, a0, x; args=args, scale_offset=args[:scale_offset])
    f_state, Enc_za_z, Dec_z_x̂, fpolicy = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end

function full_sequence(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = Hx(z)
    models, z0, a0 = get_models(θsz; args=args)
    return full_sequence(models, z0, a0, x; args=args, scale_offset=scale_offset)
end

function model_loss(x, r; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    models, z0, a0 = get_models(θsz; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

function get_loop(x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    r = rand(args[:D], args[:π], args[:bsz]) |> gpu
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    models, z0, a0 = get_models(θsz; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, out_small, a1, patch_t), outputs)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out, out_small, a1, patch_t), outputs)
    end
    return outputs
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
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
            full_loss + args[:λ] * (norm(Flux.params(Hx)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

## ====

args[:π] = 32

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_enc_e_z = (784 + 1) * args[:π]

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x]

Hx = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π] + args[:asz], bias=false),
) |> gpu

l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]


Enc_za_a = Dense((args[:π] + args[:asz]), args[:π])
f_policy = GRU(args[:π], args[:π])
Dec_z_a = Dense(args[:π], args[:asz], sin, bias=false)

fpolicy = Chain(
    Enc_za_a,
    LayerNorm(args[:π], elu),
    f_policy,
    LayerNorm(args[:π], elu),
    Dec_z_a,
) |> gpu

Encoder = let
    enc1 = Chain(
        x -> reshape(x, 28, 28, 1, :),
        Conv((5, 5), 1 => 32, relu, pad=(1, 1)),
        Conv((5, 5), 32 => 32, relu, pad=(1, 1)),
        Conv((5, 5), 32 => 32, relu, pad=(1, 1)),
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
        LayerNorm(64, elu),
        Dense(64, 64,),
        LayerNorm(64, elu),
        Split(
            Dense(64, args[:π]),
            Dense(64, args[:π])
        )
    )
end |> gpu

ps = Flux.params(Hx, fpolicy, Encoder)

## =====

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "gen_2lvl"
alias = "separate_fpolicy_no_z_dependence"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:seqlen] = 4
args[:scale_offset] = 2.4f0

# args[:λpatch] = Float32(1 / 2 * args[:seqlen])
args[:λpatch] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.5f0

args[:η] = 4e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
## ====
begin
    Ls = []
    for epoch in 1:20
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
    end
end

