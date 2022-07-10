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
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))
include(srcdir("utils.jl"))
CUDA.allowscalar(false)

## =====

args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :seqlen => 4, :λ => 1.0f-3
)

## ======

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
sampling_grid = get_sampling_grid(args[:img_size]...) |> gpu
sampling_grid_2d = sampling_grid[1:2, :, :]

scale_offset = let
    tmp = gpu(zeros(Float32, 6, args[:bsz]))
    tmp[[1, 4], :] .= 3.2f0
    tmp
end

scale_offset_3d = let
    tmp = gpu(zeros(Float32, 4, args[:bsz]))
    tmp[1:2, :] .= 3.2f0
    tmp
end


## ====

function get_models(θs, model_bounds; init_zs=true)
    inds = init_zs ? [0; cumsum([model_bounds...; args[:π]; args[:asz]])] : [0; cumsum(model_bounds)]

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Vx = reshape(Θ[1], args[:π], args[:π], args[:bsz])
    Va = reshape(Θ[2], 6, 6, args[:bsz])
    # Vϵ = reshape(Θ[3], args[:π], args[:π], args[:bsz])

    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[3], σ), flatten)
    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[4], relu), flatten)

    z0 = init_zs ? Θ[5] : nothing
    a0 = init_zs ? Θ[6] : nothing

    return Vx, Va, Enc_ϵ_z, Dec_z_x̂, z0, a0
end

Δa(a, Va) = sin.(bmul(a, Va))


function full_loop(z, x)
    θs = H(z)
    Vx, Va, Enc_ϵ_z, Dec_z_x̂, z0, a0 = get_models(θs, model_bounds)

    z1 = bmul(z0, Vx)
    a1 = a0 + Δa(a0, Va)
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in(x, a1, sampling_grid) |> flatten
    ϵ = patch_t .- x̂
    Δz = Enc_ϵ_z(ϵ)
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        # z = z + Δz
        z = Δz
        θs = H(z)
        Vx, Va, Enc_ϵ_z, Dec_z_x̂, z0, a0 = get_models(θs, model_bounds)

        z1 = bmul(z1, Vx)
        # a1 = bmul(a1, Va)
        a1 = a1 + Δa(a1, Va)
        x̂ = Dec_z_x̂(z1)
        patch_t = zoom_in(x, a1, sampling_grid) |> flatten
        ϵ = patch_t .- x̂
        Δz = Enc_ϵ_z(ϵ)

        out += sample_patch(x̂, a1, sampling_grid)
    end
    out
end

function get_loop(z, x)
    patches, recs, errs = [], [], []
    θs = H(z)
    Vx, Va, Enc_ϵ_z, Dec_z_x̂, z0, a0 = get_models(θs, model_bounds)

    z1 = bmul(z0, Vx)
    # a1 = bmul(a0, Va)
    a1 = a0 + Δa(a0, Va)
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in(x, a1, sampling_grid) |> flatten
    ϵ = patch_t .- x̂
    Δz = Enc_ϵ_z(ϵ)
    out = sample_patch(x̂, a1, sampling_grid)

    push!(patches, cpu(x̂))
    push!(recs, cpu(out))
    push!(errs, cpu(ϵ))

    for t = 2:args[:seqlen]
        # z = z + Δz
        z = Δz
        θs = H(z)
        Vx, Va, Enc_ϵ_z, Dec_z_x̂, _, _ = get_models(θs, model_bounds)

        z1 = bmul(z1, Vx)
        # a1 = bmul(a1, Va)
        a1 = a1 + Δa(a1, Va)
        x̂ = Dec_z_x̂(z1)
        patch_t = zoom_in(x, a1, sampling_grid) |> flatten
        ϵ = patch_t .- x̂
        Δz = Enc_ϵ_z(ϵ)

        out += sample_patch(x̂, a1, sampling_grid)

        push!(patches, cpu(x̂))
        push!(recs, cpu(out))
        push!(errs, cpu(ϵ))
    end
    patches, recs, errs
end

function model_loss(z, x)
    L = Flux.mse(full_loop(z, x) |> flatten, x |> flatten)
    L + args[:λ] * norm(Flux.params(H))
end
## ====

function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, (x, y)) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    x_
end

function train_model(opt, ps, train_data; epoch=1)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, (x, y)) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            model_loss(zs[i], x)
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
    for (i, (x, y)) in enumerate(test_data)
        L += model_loss(zs[i], x)
    end
    return L / length(test_data)
end

## ====
function plot_rec(out, x, ind)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    return plot(p1, p2)
end

function plot_rec(out, x, xs, ind)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    p3 = plot([plot_digit(x[:, :, 1, ind]) for x in xs]...)
    return plot(p1, p2, p3, layout=(1, 3))
end


function plot_recs(x, inds; plot_seq=true)
    z = randn(Float32, args[:π], args[:bsz]) |> gpu

    patches, preds, errs = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        # [plot_rec(preds[end], x, preds, ind) for ind in inds]
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

## =====
Vx_sz = (args[:π], args[:π],)
Va_sz = (args[:asz], args[:asz],)

# Vϵ_sz = (args[:π], args[:π],)
l_enc = 784 * args[:π] + args[:π] # encoder ϵ -> z, with bias
l_dec = 784 * args[:π] # decoder z -> x̂, no bias

# model_bounds = [map(sum, (prod(Vx_sz), prod(Va_sz), Vϵ_sz))...; l_enc; l_decs]
model_bounds = [map(sum, (prod(Vx_sz), prod(Va_sz),))...; l_enc; l_dec]

lθ = sum(model_bounds)
# ! initializes z_0, a_0 
H = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 128),
    LayerNorm(128, elu),
    Dense(128, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

ps = Flux.params(H)
## ====
args[:seqlen] = 6
opt = ADAM(1e-4)

for epoch in 1:20
    ls = train_model(opt, ps, train_loader; epoch=epoch)
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
    L = test_model(test_loader)
    @info "Test loss: $L"
end
## ====
thetas = randn(4, 64) |> gpu
sc = thetas[1:2, :]
bs = sc .* thetas[3:4, :]
tr_grid = batched_mul(unsqueeze(bs, 1), sampling_grid_2d)
xy = randn(6, 64) |> gpu

affine_grid_generator(sampling_grid_2d, thetas)
affine_grid_generator(sampling_grid, xy)

sampling_grid_2d

## =====
inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

z = randn(Float32, args[:π], args[:bsz]) |> gpu
patches, preds, errs = get_loop(z, x)

ind = 0
begin
    ind = mod(ind + 1, 64) + 1
    p = [plot_digit(reshape(x[:, ind], 28, 28),) for x in errs]
    plot(p...)
end

p1 = begin
    ind = mod(ind + 1, 64) + 1
    pp = plot([plot_digit(reshape(x[:, ind], 28, 28),) for x in patches]...)
    pe = plot([plot_digit(reshape(x[:, ind], 28, 28), boundc=false) for x in errs]...)
    p1 = plot(pp, pe, layout=(2, 1))
    px = plot_digit(reshape(cpu(x)[:, :, ind], 28, 28),)
    plot(p1, px)
end




