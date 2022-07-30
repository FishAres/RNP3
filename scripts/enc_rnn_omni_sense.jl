using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using Flux, Zygote, CUDA

using MLDatasets
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Random: shuffle
using Plots
using StatsBase: sample
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
    :asz => 6, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 0.0f0,
)

## =====

device!(0)

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
const diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
## ====

manzrelu(x) = min(relu(x), 5.0f0)

function get_models(θs, model_bounds; args=args, init_zs=true)
    inds = init_zs ? [0; cumsum([model_bounds...; args[:π]; args[:asz]])] : [0; cumsum(model_bounds)]

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Vx = reshape(Θ[1], args[:π], args[:π], args[:bsz])
    Va = reshape(Θ[2], args[:asz], args[:asz], args[:bsz])

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[3], tanh), flatten)
    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:asz], Θ[4], tanh), flatten)

    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[5], tanh), flatten)
    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[6], relu6), flatten)

    Dec_z_a = Chain(HyDense(args[:asz], args[:asz], Θ[7],), flatten)

    z0 = Θ[8]
    a0 = Θ[9]

    return (Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a), z0, a0
end

Δa(a, Va, Dec_z_a) = sin.(Dec_z_a(elu.(bmul(a, Va))))

function forward_pass(z1, a1, models, x)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = elu.(bmul(ez, Vx)) # non-linear transition
    a1 = Δa(ea, Va, Dec_z_a)
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid) |> flatten
    ϵ = patch_t .- x̂
    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
end


function full_sequence(z, x; args=args)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)

    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end

function model_loss(z, x; args=args, model_bounds=model_bounds)
    Flux.reset!(RN2)
    θs = H(z)

    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    # z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    # z = update_z(z, Δz)
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_full = full_sequence(z, x)
    # out = sample_patch(x̂, a1, sampling_grid)
    Lfull = Flux.mse(flatten(out_full), flatten(x))
    for t = 2:args[:seqlen]
        z = RN2(Δz)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_full = full_sequence(z, x)
        # out += sample_patch(x̂, a1, sampling_grid)
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end
    # rec_loss = Flux.mse(flatten(out), flatten(x))
    # return rec_loss + 0.167f0 * Lfull
    return Lfull
end

function get_loop(z, x; args=args, model_bounds=model_bounds)
    Flux.reset!(RN2)
    outputs = patches, recs, full_out, errs, xys, z1s, zs, patches_t, Vxs = [], [], [], [], [], [], [], [], []

    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    # z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    # z = RN2(Δz)
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_full = full_sequence(z, x)
    out = sample_patch(x̂, a1, sampling_grid)

    push_to_arrays!((x̂, out, out_full, ϵ, a1, z1, z, patch_t, Vx), outputs)
    for t = 2:args[:seqlen]
        z = RN2(Δz)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_full = full_sequence(z, x)
        out += sample_patch(x̂, a1, sampling_grid)

        push_to_arrays!((x̂, out, out_full, ϵ, a1, z1, z, patch_t, Vx), outputs)
    end
    return outputs
end

function plot_recs(x, inds; plot_seq=true)
    z = randn(Float32, args[:π], args[:bsz]) |> gpu

    patches, preds, full_out, errs, xys, zs = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), full_out)
        # [plot_rec(preds[end], x, preds, ind) for ind in inds]
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

## ====
function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
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
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        L += model_loss(zs[i], x)
    end
    return L / length(test_data)
end



## ======

Vx_sz = (args[:π], args[:π],)
Va_sz = (args[:asz], args[:asz],)

l_enc = 784 * args[:π] + args[:π] # encoder ϵ -> z, with bias
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_dec_a = args[:asz] * args[:asz] + args[:asz] # decoder z -> a, with bias

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_enc_za_a = (args[:π] + args[:asz]) * args[:asz] # encoder (z_t, a_t) -> a_t+1

model_bounds = [map(sum, (prod(Vx_sz), prod(Va_sz),))...; l_enc_za_z; l_enc_za_a; l_enc; l_dec_x; l_dec_a]

lθ = sum(model_bounds)
println(lθ, " parameters for primary")
# ! initializes z_0, a_0 
H = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, gelu),
    Dense(64, 64),
    LayerNorm(64, gelu),
    Dense(64, 64),
    LayerNorm(64, gelu),
    Dense(64, 64),
    LayerNorm(64, gelu),
    Dense(64, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

println("# hypernet params: $(sum(map(prod, size.(Flux.params(H)))))")


RN2 = RNN(args[:π], args[:π],) |> gpu

ps = Flux.params(H, RN2)

## ======
save_folder = "full_seq_tanh_v0"
alias = "omni_sense"
save_dir = get_save_dir(save_folder, alias)

## =====

# inds = sample(1:args[:bsz], 6, replace=false)
# p = plot_recs(rand(xs_test), inds)

## =====

args[:seqlen] = 7
args[:scale_offset] = 2.4f0
args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
# args[:δL] = 0.0f0
args[:λ] = 0.006f0
opt = ADAM(1e-3)
lg = new_logger(save_dir, args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:1000
        if epoch % 15 == 0
            opt.eta = 0.8 * opt.eta
        end
        ls = train_model(opt, ps, xs_train; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(rand(xs_test), inds)
        display(p)
        L = test_model(xs_test)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 50 == 0
            save_model((H, RN2), "full_seq_omni_sense_v01_$(epoch)eps")
        end
    end
end
# modelpath = joinpath(save_dir, "full_seq_v01_10eps")

## ====
# L = vcat(Ls...)
# plot(L)
# plot(log.(1:length(L)), log.(L))
# ## ===