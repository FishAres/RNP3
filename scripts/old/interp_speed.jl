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

include(srcdir("interp_utils.jl"))
include(srcdir("plotting_utils.jl"))
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
# dev = has_cuda() ? gpu : cpu
## =====

train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev, train_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev, test_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

x, y = first(test_loader)
## ====

const sampling_grid = (get_sampling_grid(args[:img_size]...) |> dev)
const sampling_grid_2d = sampling_grid[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]] |> dev

const diag_mat = cat(diag_vec..., dims=3)

## =====

thetas = randn(6, args[:bsz]) |> gpu


function affine_grid_generator(thetas, sampling_grid)
    sc = thetas[[1, 4], :]
    thetas = vcat(thetas[1:4, :], sc .* thetas[5:6, :])
    thetas = reshape(thetas, 2, 3, :)
    return batched_mul(thetas, sampling_grid)
end

@time g = affine_grid_generator(thetas, sampling_grid)

@time gr = grid_generator_3d(sampling_grid_2d, thetas)
@time gri = get_inv_grid(sampling_grid_2d, thetas)
@time gri2 = get_inv_grid2(sampling_grid_2d, thetas)

out = sample_patch(x, gr)
outi = sample_patch(out, gri)

begin
    ind = mod(ind + 1, args[:bsz]) + 1
    p1 = plot_digit(cpu(out)[:, :, 1, ind])
    p2 = plot_digit(cpu(outi)[:, :, 1, ind])
    plot(p1, p2)
end

A_rot, A_s, A_shear, b = get_affine_mats(thetas; scale_offset=args[:scale_offset])
