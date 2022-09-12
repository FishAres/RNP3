using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using Images
using StatsBase: sample
using Random: shuffle
using ParameterSchedulers

# todo - refactor so scripts share same code
include(srcdir("double_H_vae_utils.jl"))
include(srcdir("cifar_2lvl_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (32, 32), :img_channels => 3, :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => elu,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
)
args[:imszprod] = prod(args[:img_size])
## =====

device!(0)

dev = gpu

##=====

train_digits, train_labels = CIFAR10(split=:train)[:]
test_digits, test_labels = CIFAR10(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

x = first(train_loader)

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

function full_sequence(models::Tuple, z0, a0, x; args=args, scale_offset=args[:scale_offset])
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    err_emb = err_rnn(Δz) # update error embedding
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        err_emb = err_rnn(Δz) # update error embedding
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, err_emb
end


function model_loss(z, x, rs; args=args)
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    Lfull = Flux.mse(flatten(out_full), flatten(x); agg=sum)
    for t = 2:args[:glimpse_len]-1
        μ, logvar = RN2(err_emb_1)
        z = sample_z(μ, logvar, rs[t-1])
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])

        # Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end
    μ, logvar = RN2(err_emb_1)
    z = sample_z(μ, logvar, rs[end])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
    Lfull += Flux.mse(flatten(out_full), flatten(x); agg=sum)
    klqp = kl_loss(μ, logvar)
    return Lfull, klqp
end

function get_loop(z, x, rs; args=args)
    outputs = full_recs, patches, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x)
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
    sense_patch = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])

    push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]-1
        μ, logvar = RN2(err_emb_1)
        z = sample_z(μ, logvar, rs[t-1])
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
        sense_patch = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])
        push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)


    end
    μ, logvar = RN2(err_emb_1)
    z = sample_z(μ, logvar, rs[end])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)
    return outputs
end

## ====

# todo don't bind RNN size to args[:π]
args[:π] = 128
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_err_rnn = get_rnn_θ_sizes(args[:π], args[:π]) # also same size lol
l_dec_x = args[:imszprod] * args[:π] # decoder z -> x̂, no bias
l_enc_e_z = ((args[:imszprod] * args[:img_channels]) + 1) * args[:π]

Hx_bounds = [l_enc_za_z; l_fx; l_err_rnn; l_dec_x; l_dec_x; l_dec_x; l_enc_e_z]


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
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    # Dense(64, 64),
    # LayerNorm(64, elu),
    # Dense(64, 64),
    # LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
) |> gpu


RN2 = Chain(
    Dense(args[:π], 64,),
    LayerNorm(64, elu),
    LSTM(64, 64,),
    Split(
        Dense(64, args[:π],),
        Dense(64, args[:π],),
    )
) |> gpu

z0 = rand(args[:D], args[:π], args[:bsz]) |> gpu

ps = Flux.params(Hx, Ha, RN2, z0)


## ======

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "enc_rnn_2lvl"
alias = "2lvl_double_H_cifar_vae_z0emb_1rec_step"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 2
args[:glimpse_len] = 4
args[:scale_offset] = 0.0f0
args[:scale_offset_sense] = 2.8f0

# args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0

args[:α] = 1.0f0
args[:β] = 0.1f0

## =====
args[:η] = 1e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

# using ParameterSchedulers
# using ParameterSchedulers: Stateful
# 
# s = Stateful(SinExp(args[:η], 4e-7, 20, 0.99))

## ====

begin
    Ls = []
    for epoch in 1:20
        if epoch % 50 == 0
            opt.eta = 0.6 * opt.eta
            log_value(lg, "learning_rate", opt.eta)
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        println("z0 mean: ", mean(z0), ", ", "std: ", std(z0))
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)

        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 50 == 0
            save_model((Hx, Ha, RN2, z0), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

## ====

# save_model((Hx, Ha, RN2, z0), joinpath(save_folder, alias, savename(args) * "_67eps"))




# rs = [rand(args[:D], args[:π], args[:bsz]) for _ in 1:args[:seqlen]] |> gpu


# full_recs, patches, errs, zs, as, patches_t = get_loop(z0, x, rs)



# aa = z0 |> cpu
# heatmap(aa)

# histogram(aa', legend=false, alpha=0.2, bins=-5:0.5:5)