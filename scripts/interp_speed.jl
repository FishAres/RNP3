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

thetas = let
    thetas = randn(6, args[:bsz])
    thetas[3, :] .= 0.0f0
    thetas |> dev
end


function affine_grid_generator(thetas, sampling_grid)
    sc = thetas[[1, 4], :]
    thetas = vcat(thetas[1:4, :], sc .* thetas[5:6, :])
    thetas = reshape(thetas, 2, 3, :)
    return batched_mul(thetas, sampling_grid)
end

@time g = affine_grid_generator(thetas, sampling_grid) |> cpu
@time grid_generator_3d(sampling_grid_2d, thetas)




@inline function get_affine_mats_fast(thetas; scale_offset=0.0f0)
    sc = add_sc(thetas; scale_offset=scale_offset)
    b = sc .* (@view thetas[5:6, :])
    A_rot = get_rot_mat(@view thetas[2, :])
    A_sc = unsqueeze(sc, 2) .* diag_mat
    A_shear = get_shear_mat(thetas[3, :])
    return A_rot, A_sc, A_shear, b
end


function get_inv_grid2(sampling_grid_2d, thetas)
    sc = thetas[[1, 4], :]
    rot = thetas[2, :]
    b = sc .* thetas[5:6, :]

    cos_rot = reshape(cos.(rot), 1, 1, :)
    sin_rot = reshape(sin.(rot), 1, 1, :)

    A_rot = hcat(vcat(cos_rot, -sin_rot), vcat(sin_rot, cos_rot))
    A_s = cat(map(.*, eachcol(sc), diag_vec)..., dims=3)

    rot_inv = cat(map(inv, eachslice(cpu(A_rot), dims=3))..., dims=3)
    sc_inv = cat(map(inv, eachslice(cpu(A_s), dims=3))..., dims=3)

    Ainv = batched_mul(sc_inv, rot_inv) |> gpu

    inv_grid = batched_mul(Ainv, sampling_grid_2d .- unsqueeze(b, 2))
end

inv_grid = get_inv_grid2(sampling_grid_2d, thetas)

out = sample_patch(x, g |> gpu)
outi = sample_patch(out, inv_grid)
ind = 0

begin
    ind = mod(ind + 1, 64) + 1
    p1 = plot_digit(cpu(out)[:, :, 1, ind])
    p2 = plot_digit(cpu(outi)[:, :, 1, ind])
    plot_digit!(cpu(x)[:, :, ind], alpha=0.4)
    plot(p1, p2)
end