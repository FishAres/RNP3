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

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 0.0f0,
)

## =====

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
## ====
dev = has_cuda() ? gpu : cpu

const sampling_grid = get_sampling_grid(args[:img_size]...) |> dev
const sampling_grid_2d = sampling_grid[1:2, :, :]

const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1 0; 0 1] for _ in 1:args[:bsz]] |> dev


## =====

thetas = randn(6, args[:bsz]) |> dev

function affine_grid_generator(sampling_grid, thetas)
    sc = thetas[[1, 4], :] .+ args[:scale_offset]
    bs = sc .* thetas[5:6, :]
    thetas = vcat(thetas[1:4, :], bs)
    thetas = reshape(thetas, 2, 3, :)
    return batched_mul(thetas, sampling_grid)
end

function sample_patch(x, thetas, sampling_grid; sz=args[:img_size], grid_gen=grid_generator_3d)
    grid = grid_gen(sampling_grid, thetas)
    tg = reshape(grid, 2, sz..., size(grid)[end])
    x = reshape(x, sz..., 1, size(x)[end])
    grid_sample(x, tg; padding_mode=:zeros)
end

gr = affine_grid_generator(sampling_grid, thetas)
out = sample_patch(x, gr) |> cpu
out1 = sample_patch(x, thetas, sampling_grid_2d)


