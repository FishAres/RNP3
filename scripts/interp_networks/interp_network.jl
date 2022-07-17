using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using MLDatasets
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using Plots
using StatsBase: sample
using ProgressMeter
using ProgressMeter: Progress

using DiffEqFlux, DifferentialEquations

include(srcdir("interp_utils.jl"))
include(srcdir("hypernet_utils.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("logging_utils.jl"))
include(srcdir("utils.jl"))

CUDA.allowscalar(false)

## =====

args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 4, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 0.0f0,
)

## ======

device!(1)

dev = gpu

##=====

train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev, train_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev, test_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

x, y = first(test_loader)

## =====

sampling_grid = get_sampling_grid(args[:img_size]...) |> gpu

scale_offset = 0.0f0

## ====
model_loss(x, xy, y) = Flux.mse(m((x, xy)), flatten(y))

function train_model(opt, ps, train_data; epoch=1)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    for (i, (x, _)) in enumerate(train_data)
        xy = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> gpu
        y = sample_patch(x, xy, sampling_grid)
        loss, grad = withgradient(ps) do
            model_loss(x, xy, y)
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data)
    L = 0.0f0
    for (i, (x, _)) in enumerate(test_data)
        xy = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> gpu
        y = sample_patch(x, xy, sampling_grid)
        L += model_loss(x, xy, y)
    end
    return L / length(test_data)
end

function plot_rec(out, x, xi, ind)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    xi = reshape(cpu(xi), 28, 28, size(xi)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    px = plot_digit(xi[:, :, ind])
    return plot(px, p1, p2, layout=(1, 3))
end

function plot_recs(x, inds)
    xy = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> gpu
    out = m((x, xy))
    y = sample_patch(x, xy, sampling_grid)
    p = [plot_rec(out, y, x, i) for i in inds]
    plot(p..., layout=(length(inds), 1), size=(600, 800))
end

## +====

# m = Chain(
#     Parallel(
#         vcat,
#         Chain(
#             flatten,
#             Dense(784, 32),
#             BatchNorm(32, relu),
#             Dense(32, 32),
#             BatchNorm(32, relu),
#         ),
#         Chain(
#             Dense(6, 32),
#             BatchNorm(32, elu),
#             Dense(32, 32),
#             BatchNorm(32, elu),
#         )
#     ),
#     Dense(64, 32),
#     BatchNorm(32, elu),
#     Dense(32, 784, relu, bias=false),
# ) |> gpu

# ps = Flux.params(m)

modelpath = "saved_models/mnist_interp_mini_v0_epoch1000.bson"
m = load(modelpath)[:model] |> gpu


## ====

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(x, inds)

opt = ADAM(1e-5)

begin
    Ls = []
    for epoch in 1:2000
        ls = train_model(opt, ps, train_loader; epoch=epoch)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(x, inds)
        display(p)

        L = test_model(test_loader)
        @info "Test loss: $L"
        push!(Ls, ls)

        if epoch % 100 == 0
            save_model(m, "mnist_interp_mini_v0_epoch$(epoch)")
        end
    end
end

# xx = 1:length(vcat(Ls...))
# plot(log.(xx), log.(vcat(Ls...)))

## =====
