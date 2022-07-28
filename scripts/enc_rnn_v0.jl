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

train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev, train_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev, test_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

x, y = first(test_loader)
## ====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]] |> dev

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
    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[6], manzrelu), flatten)

    Dec_z_a = Chain(HyDense(args[:asz], args[:asz], Θ[7],), flatten)

    z0 = Θ[8]
    a0 = Θ[9]

    return (Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a), z0, a0
end

Δa(a, Va, Dec_z_a) = sin.(Dec_z_a(bmul(a, Va)))

function forward_pass(z1, a1, models, x)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = bmul(ez, Vx) # linear transition
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

function model_loss(z, x; args=args)
    Flux.reset!(RN2)
    θs = H(z)

    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    z = update_z(z, Δz)
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
    out_full = full_sequence(z1, x)
    out = sample_patch(x̂, a1, sampling_grid)
    Lx = mean(ϵ .^ 2)
    Lfull = Flux.mse(flatten(out_full), flatten(x))
    for t = 2:args[:seqlen]
        z = update_z(z, Δz)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_full = full_sequence(z1, x)
        out += sample_patch(x̂, a1, sampling_grid)
        Lx += mean(ϵ .^ 2)
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end
    rec_loss = Flux.mse(flatten(out), flatten(x))
    local_loss = args[:δL] * Lx
    return rec_loss + local_loss + 0.167f0 * Lfull
end

function get_loop(z, x; args=args)
    Flux.reset!(RN2)
    outputs = patches, recs, full_out, errs, xys, z1s, zs, patches_t, Vxs = [], [], [], [], [], [], [], [], []

    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    z = update_z(z, Δz)
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
    out_full = full_sequence(z, x)
    out = sample_patch(x̂, a1, sampling_grid)

    push_to_arrays!((x̂, out, out_full, ϵ, a1, z1, z, patch_t, Vx), outputs)
    for t = 2:args[:seqlen]
        z = update_z(z, Δz)
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
    Dense(64, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

println("# hypernet params: $(sum(map(prod, size.(Flux.params(H)))))")


RN2 = RNN(args[:π], args[:π],) |> gpu

ps = Flux.params(H, RN2)

## ======

save_folder = "full_seq_tanh_v0"
save_dir = get_save_dir(save_folder, savename(args))

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

args[:seqlen] = 6
args[:scale_offset] = 2.4f0
args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
# args[:δL] = 0.0f0
args[:λ] = 0.006f0
opt = ADAM(1e-3)
lg = new_logger(save_dir, args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:100
        if epoch % 10 == 0
            opt.eta = 0.8f0 * opt.eta
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 10 == 0
            save_model((H, RN2), save_dir * "$(epoch)eps")
        end
    end
end

# ## ====
# L = vcat(Ls...)
# plot(L)
# plot(log.(1:length(L)), log.(L))
# ## ===
# p = plot_recs(sample_loader(test_loader), inds)

# z = randn(args[:π], args[:bsz]) |> dev

# @time patches, preds, full_out, errs, xys, z1s, zs, patches_t, Vxs = get_loop(z, x)
# zz = cat(zs..., dims=3)

# # recs = [sample_patch(x, xy, sampling_grid) for (x, xy) in zip(gpu(patches), gpu(xys))] |> cpu
# # ind = 0

# # sem(x; dims=1) = std(x, dims=dims) / sqrt(size(x, dims))

# # begin
# #     ind = mod(ind + 1, 64) + 1
# #     plot_tings(x, y) =
# #         let
# #             p1 = plot_digit(x, c=:grays)
# #             plot_digit!(y, alpha=0.3, c=:grays)
# #             p1
# #         end
# #     p1 = [plot_tings(rec[:, :, 1, ind], cpu(x)[:, :, ind]) for rec in recs]
# #     plot(p1...)
# # end

# # # patches_t[1]
# using LaTeXStrings
# # ind = 0
# p1 = begin
#     ind = mod(ind + 1, 64) + 1
#     pp = plot([plot_digit(reshape(x[:, :, 1, ind], 28, 28), c=:jet, boundc=false, colorbar=true) for x in recs]...)
#     # pp = plot([plot_digit(reshape(x[:, ind], 28, 28), c=:jet, clim=(0, 1), colorbar=true) for x in patches]..., title="predictions", titlefontsize=10)
#     pe = plot([plot_digit(reshape(x[:, ind], 28, 28), c=:jet, clim=(0, 1), colorbar=true) for x in patches_t]..., title="patches", titlefontsize=10)
#     # 
#     p1 = plot(pp, pe, layout=(2, 1))
#     # 
#     px = plot_digit(reshape(cpu(x)[:, :, ind], 28, 28))
#     pout = plot_digit(preds[end][:, :, 1, ind])
#     pz = plot(zz[:, ind, :]', legend=false, title=L"z^2")
#     cc = heatmap(cor(zz[:, ind, :]))
#     plot(plot(px, pout,), p1, plot(pz, cc), layout=(3, 1), size=(600, 900))
# end
