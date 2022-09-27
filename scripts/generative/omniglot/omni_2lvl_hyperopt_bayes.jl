using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Hyperopt
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
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0)
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
## ===== functions

function get_param_sizes(model)
    nw = []
    for m in Flux.modules(model)
        if hasproperty(m, :weight)
            wprod = prod(size(m.weight)[1:end-1])
            if hasproperty(m, :bias)
                wprod += size(m.bias)[1]
            end
            push!(nw, wprod)
        end
    end
    return nw
end


function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)

    # conv decoder
    Dec_z_x̂ = Chain(
        HyDense(args[:π], 64, Θ[3], elu),
        flatten,
        HyDense(64, 64, Θ[4], elu),
        flatten,
        HyDense(64, 64, Θ[5], elu),
        flatten,
        HyDense(64, 64, Θ[6], elu),
        flatten,
        HyDense(64, args[:imzprod], Θ[7], relu)
    )
    z0 = fz.(Θ[8])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)

    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function model_loss(model, x, r; args=args)
    Encoder, Hx, Ha = model
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

function train_model(model, opt, ps, train_data, args; epoch=1, logger=nothing, D=args[:D])
    Encoder, Hx, Ha = model
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(model, x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                log_value(lg, "KL loss", klqp)
            end
            full_loss = args[:α] * rec_loss + args[:β] * klqp
            full_loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(model, test_data; D=args[:D])
    Encoder, Hx, Ha = model
    rs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(model, x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp
    end
    return L / length(test_data)
end


## ====== model

function get_model(args)

    l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
    l_fx = get_rnn_θ_sizes(args[:π], args[:π]) # μ, logvar

    mdec = Chain(
        HyDense(args[:π], 64, args[:bsz], elu),
        flatten,
        HyDense(64, 64, args[:bsz], elu),
        flatten,
        HyDense(64, 64, args[:bsz], elu),
        flatten,
        HyDense(64, 64, args[:bsz], elu),
        flatten,
        HyDense(64, args[:imzprod], args[:bsz], relu)
    )

    l_dec_x = get_param_sizes(mdec)

    Hx_bounds = [l_enc_za_z; l_fx; l_dec_x...]

    l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
    l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
    l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

    Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

    Hx = Chain(
        LayerNorm(args[:π],),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 256),
        LayerNorm(256, elu),
        x -> reshape(x, 8, 8, 4, :),
        ConvTranspose((4, 4), 4 => 16, stride=(2, 2), pad=(0, 0)),
        GroupNorm(16, 4, elu),
        ConvTranspose((4, 4), 16 => 16, stride=(2, 2), pad=(2, 2)),
        GroupNorm(16, 4, elu),
        ConvTranspose((4, 4), 16 => 16, stride=(2, 2), pad=(2, 2)),
        GroupNorm(16, 4, elu),
        ConvTranspose((4, 4), 16 => 8, stride=(2, 2), pad=(2, 2), bias=false),
        flatten,
    ) |> gpu

    for p in Flux.params(Hx)
        p ./= 10.0f0
    end

    Ha = Chain(
        LayerNorm(args[:π],),
        Dense(args[:π], 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
    ) |> gpu


    Encoder = let
        enc1 = Chain(
            x -> reshape(x, args[:img_size]..., args[:img_channels], :),
            Conv((5, 5), args[:img_channels] => 32),
            BatchNorm(32, relu),
            Conv((5, 5), 32 => 32),
            BatchNorm(32, relu),
            Conv((5, 5), 32 => 32),
            BatchNorm(32, relu),
            Conv((5, 5), 32 => 32),
            BatchNorm(32, relu),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            BasicBlock(32 => 32, +),
            flatten,
        )
        outsz = Flux.outputsize(enc1, (args[:img_size]..., args[:img_channels], args[:bsz]))
        Chain(
            enc1,
            Dense(outsz[1], 64,),
            LayerNorm(64, elu),
            Dense(64, 64,),
            LayerNorm(64, elu),
            Dense(64, 64,),
            LayerNorm(64, elu),
            Dense(64, 64,),
            LayerNorm(64, elu),
            Split(
                Dense(64, args[:π]),
                Dense(64, args[:π])
            )
        ) |> gpu
    end
    return Encoder, Hx, Ha, Hx_bounds, Ha_bounds
end

## =====

save_folder = "gen_2lvl"
alias = "omni_2lvl_convH_hyperopt_v0"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:α] = 1.0f0
args[:β] = 0.05f0
args[:seqlen] = 4

# function run_training_thingy(πsz, scale_offset, λ, λpatch, η)
# function run_training_thingy(model, opt, ps, args)
function run_training_thingy(opt, args)
    Encoder, Hx, Ha, Hx_bounds, Ha_bounds = get_model(args)
    model = Encoder, Hx, Ha
    ps = Flux.params(Hx, Ha, Encoder)
    for epoch in 1:20
        ls = train_model(model, opt, ps, train_loader, args; epoch=epoch, logger=nothing)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        p2 = plot_recs(sample_loader(val_loader), inds)
        p3 = plot(p, p2, layout=(1, 2))
        display(p3)
    end
    ltest = test_model(model, val_loader)
    ltest

end

## =====

# beware of confusing macro syntax - check commas

ho = @hyperopt for i = 10,
    sampler = Hyperband(R=50, η=3, inner=BOHB()),
    πsz = LinRange(64, 256, 20),
    scale_offset = LinRange(1.8, 2.2, 4),
    λpatch = LinRange(0.0, 0.00, 5),
    λ = LinRange(0.0, 0.00, 5),
    η = LinRange(1e-5, 1e-4, 5)

    args[:scale_offset] = Float32(scale_offset)
    args[:λ] = Float32(λ)
    args[:λpatch] = Float32(λpatch)
    args[:η] = η
    args[:π] = trunc(Int, πsz)

    opt = ADAM(args[:η])

    # println("π = $(trunc(Int, πsz)), sc_offset = $(scale_offset), λpatch, λ, η")
    cost = run_training_thingy(opt, args)

end
## =====


lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
## ====
begin
    Ls = []
    for epoch in 1:400
        if epoch % 100 == 0
            opt.eta = max(0.6 * opt.eta, 1e-7)
            log_value(lg, "learning_rate", opt.eta)
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        p2 = plot_recs(sample_loader(val_loader), inds)
        p3 = plot(p, p2, layout=(1, 2))
        display(p3)
        log_image(lg, "recs_$(epoch)", p3)

        Lval = test_model(val_loader)
        log_value(lg, "val_loss", Lval)
        @info "Val loss: $Lval"
        if epoch % 10 == 0
            @time Ltest = test_model(test_loader)
            log_value(lg, "test_loss", Ltest)
            @info "Test loss: $Ltest"
        end

        push!(Ls, ls)
        if epoch % 250 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

