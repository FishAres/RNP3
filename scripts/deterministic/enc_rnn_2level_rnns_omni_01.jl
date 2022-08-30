using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using Flux, Zygote, CUDA

using MLDatasets
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Plots
using Distributions
using StatsBase: sample
using Random: shuffle
using ProgressMeter
using ProgressMeter: Progress

include(srcdir("interp_utils.jl"))
include(srcdir("hypernet_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))
include(srcdir("utils.jl"))

include(srcdir("z_rnn_model_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :λf => 0.167f0
)

## =====

device!(1)

dev = gpu

##=====

all_chars = load("../Recur_generative/data/exp_pro/omniglot.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
x_batches = batch.(partition(xs, args[:bsz]))

ntrain, ntest = 286, 15
xs_train = flatten.(x_batches[1:ntrain] |> gpu)
xs_test = flatten.(x_batches[ntrain+1:ntrain+ntest] |> gpu)
x = xs_test[1]

## ====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]] |> dev
const diag_mat = cat(diag_vec..., dims=3) |> dev

const diag_offs = 1.0f-4 .* diag_vec

const diag_off = cat(1.0f-6 .* diag_vec..., dims=3)
## =====

function get_models(θs, model_bounds; args=args, init_zs=true)
    inds = Zygote.ignore() do
        init_zs ? [0; cumsum([model_bounds...; args[:π]; args[:asz]])] : [0; cumsum(model_bounds)]
    end

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    f_state = ps_to_RN(get_rn_θs(Θ[1], args[:π], args[:π]); f_out=elu)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    err_rnn = ps_to_RN(get_rn_θs(Θ[3], args[:π], args[:π]); f_out=elu)

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[4], elu), flatten)
    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[5], elu), flatten)

    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[6], elu), flatten)

    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[7], relu6), flatten)

    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[8], sin), flatten)

    z0 = elu.(Θ[9])
    a0 = sin.(Θ[10])

    return (f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a), z0, a0
end

"todo: calculate error outside gradient"
function forward_pass(z1, a1, models, x)
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid) |> flatten
    # ϵ = patch_t .- x̂
    ϵ = Zygote.ignore() do
        patch_t .- x̂
    end
    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
end

function full_sequence(z::AbstractArray, x; args=args, model_bounds=model_bounds)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    err_emb = err_rnn(Δz) # update error embedding
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        err_emb = err_rnn(Δz) # update error embedding
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, err_emb
end

function full_sequence(models::Tuple, z0, a0, x; args=args, model_bounds=model_bounds)
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    err_emb = err_rnn(Δz) # update error embedding
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        err_emb = err_rnn(Δz) # update error embedding
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, err_emb
end

function model_loss(z, x; args=args, model_bounds=model_bounds)
    Flux.reset!(RN2)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_1, err_emb_1 = full_sequence(z1, patch_t)
    out_full, err_emb_full = full_sequence(models, z0, a0, x)

    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t))
    Lfull = Flux.mse(flatten(out_full), flatten(x))
    for t = 2:args[:glimpse_len]
        z = RN2(err_emb_1)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_1, err_emb_1 = full_sequence(z1, patch_t)

        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end

    local_loss = args[:δL] * Lpatch
    return local_loss + args[:λf] * Lfull
end

function get_loop(z, x; args=args, model_bounds=model_bounds)
    outputs = patches, recs, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, err_rnn, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_full, err_emb_full = full_sequence(models, z0, a0, x)

    out_1, err_emb_1 = full_sequence(z1, patch_t)
    out = sample_patch(out_full .+ 0.1f0, a1, sampling_grid)

    push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]
        z = RN2(err_emb_1)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, err_rnn, Dec_z_x̂, Dec_z_a = models
        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_1, err_emb_1 = full_sequence(z1, patch_t)
        out += sample_patch(out_1 .+ 0.1f0, a1, sampling_grid)
        push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)
    end
    return outputs
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    zs = [rand(Uniform(-1.0f0, 1.0f0), args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            loss = model_loss(zs[i], x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "loss", loss)
            end
            loss + args[:λ] * norm(Flux.params(H))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data)
    zs = [rand(Uniform(-1.0f0, 1.0f0), args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        L += model_loss(zs[i], x)
    end
    return L / length(test_data)
end

function plot_recs(x, inds; plot_seq=true)
    z = rand(Uniform(-1.0f0, 1.0f0), args[:π], args[:bsz]) |> gpu

    patches, preds, errs, xys, zs = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        # [plot_rec(preds[end], x, preds, ind) for ind in inds]
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end


## =====

args[:π] = 128
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_err_rnn = get_rnn_θ_sizes(args[:π], args[:π]) # also same size lol

l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

l_enc_e_z = (784 + 1) * args[:π]

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1

model_bounds = [l_fx; l_fa; l_err_rnn; l_enc_za_z; l_enc_za_a; l_enc_e_z; l_dec_x; l_dec_a]

## =====

lθ = sum(model_bounds)
println(lθ, " parameters for primary")
# ! initializes z_0, a_0 
H = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

println("# hypernet params: $(sum(map(prod, size.(Flux.params(H)))))")

RN2 = Chain(
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    LSTM(64, 64),
    Dense(64, args[:π], elu),
) |> gpu

ps = Flux.params(H, RN2)
## =====

modelpath = "saved_models/enc_rnn_2l2v2l/fx_fa_fe_rnns_omni_v01/add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_scale_offset=2.8_seqlen=5_δL=0.2_λ=0.006_λf=1.0_π=128_50eps.bson"

H, RN2 = load(modelpath)[:model] |> gpu


## =====
z = randn(args[:π], args[:bsz]) |> gpu
model_loss(z, x)

thetas = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> gpu

A_rot, A_s, A_shear, b = get_affine_mats(thetas; scale_offset=args[:scale_offset])
A = batched_mul(batched_mul(A_rot, A_shear), A_s)

Ac = collect(eachslice(cpu(A), dims=3))

sum(diag(a) .== 0 for a in Ac)

diag_vec[1]


## =====

save_folder = "enc_rnn_2l2v2l"
alias = "fx_fa_fe_rnns_omni_v01"
save_dir = get_save_dir(save_folder, alias)


## =====

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(rand(xs_test), inds)

## =====

args[:seqlen] = 5
args[:glimpse_len] = 4
args[:scale_offset] = 2.8f0
args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
# args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.006f0
opt = ADAM(1e-4)
lg = new_logger(save_dir, args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:100
        if epoch % 20 == 0
            opt.eta = 0.8 * opt.eta
        end
        ls = train_model(opt, ps, xs_train; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(rand(xs_test), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(xs_test)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 25 == 0
            save_model((H, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

## =====

z = randn(Float32, args[:π], args[:bsz]) |> gpu
full_recs, patches, errs, zs, a1s, patches_t = get_loop(z, x)

ind = 0
begin
    ind = mod(ind + 1, args[:bsz]) + 1
    # p = [plot_digit(reshape(patch[:, :, 1, ind], 28, 28)) for patch in patches_t]
    p = [heatmap(reshape(patch[:, ind], 28, 28)) for patch in patches_t]
    plot(p...)
end

# es = reshape(full_recs[2], 28, 28, 64)
# heatmap(es[:, :, 1])

# zz = cat(zs..., dims=3)

# plot(zz[:, 4, :]', legend=false)

# xx = -10:0.1:10
# plot(xx, sin.(xx))
# plot!(xx, tanh.(xx))
# plot!(xx, 2 .* (σ.(xx) .- 0.5))