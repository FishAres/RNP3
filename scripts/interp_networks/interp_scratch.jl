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

function sample_loader(loader, ind)
    for (i, (x, y)) in enumerate(loader)
        i == ind && return (x, y)
    end
end

## =====

sampling_grid = get_sampling_grid(args[:img_size]...) |> gpu
sampling_grid_2d = sampling_grid[1:2, :, :]

scale_offset = 0.0f0

scale_offset_mat = let
    tmp = zeros(6, args[:bsz])
    tmp[[1, 4], :] .= scale_offset
    tmp |> gpu
end

## =====

thetas = rand(Uniform(-2.0f0, 2.0f0), 6, args[:bsz]) |> gpu

thetas_2d = let
    tmp = rand(Uniform(-2.0f0, 2.0f0), 6, args[:bsz])
    tmp[2:3, :] .= 0.0f0
    tmp
end |> gpu



function grid_generator_2d(sampling_grid_2d, thetas)
    thetas = thetas + scale_offset_mat
    sc = thetas[[1, 4], :]
    bs = sc .* thetas[5:6, :]
    grid2 = unsqueeze(sc, 2) .* sampling_grid_2d .+ unsqueeze(bs, 2)
    grid2
end


tg = grid_generator_3d(sampling_grid, thetas_2d) |> cpu
tg2 = grid_generator_2d(sampling_grid_2d, thetas_2d) |> cpu

ind = 0
begin
    ind = mod(ind + 1, 64) + 1
    scatter(tg[1, :, ind], tg[2, :, ind])
    # scatter!(tg1[1, :, ind], tg1[2, :, ind])
    scatter!(tg2[1, :, ind], tg2[2, :, ind])
end

## ====

function get_inv_grid2d(sampling_grid_2d, thetas)
    sc = thetas[[1, 4], :] .+ scale_offset
    bs = sc .* thetas[5:6, :]
    A = map(scalemat, eachcol(cpu(sc)))
    Ai = map(x -> diag(inv(x)), A)
    Ainv = hcat(Ai...) |> gpu
    return unsqueeze(Ainv, 2) .* (sampling_grid_2d .- unsqueeze(bs, 2))
end


function get_inv_grid(sampling_grid_2d, thetas)
    sc = thetas[[1, 4], :] .+ scale_offset
    bs = sc .* thetas[5:6, :]
    # bs = thetas[5:6, :]
    Asc = map(x -> scalemat(x, 0.0f0), eachcol(cpu(sc)))
    Ashear = map(shearmat, cpu(thetas)[3, :])
    Arot = map(rotmat, cpu(thetas)[2, :])
    # Ab = map(bmat, eachcol(cpu(bs)))
    A = map(*, Ashear, Asc, Arot)
    Ai = map(x -> diag(inv(x)), A)

    # A = map(*, Fxto, Arot, Ashear, Asc, Ab, Fcto)
    Ai = map(x -> diag(inv(x)), A)

    Ainv = hcat(Ai...) |> gpu
    Ainv, bs
    # unsqueeze(Ainv, 2) .* (sampling_grid_2d .- unsqueeze(bs, 2))
end

function sample_patch(grid, x)
    tg = reshape(grid, 2, 28, 28, size(grid)[end])
    x = reshape(x, 28, 28, 1, size(x)[end])
    grid_sample(x, tg; padding_mode=:zeros)
end


thetas[2:3, :] .= 0.0f0
thetas = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> gpu

function grid_generator_3d(sampling_grid, thetas)
    thetas = thetas + scale_offset_mat
    sc = thetas[[1, 4], :]
    thetas = vcat(thetas[1:4, :], sc .* thetas[5:6, :])
    thetas = reshape(thetas, 2, 3, size(thetas)[end])
    tr_grid = batched_mul(thetas, sampling_grid)
    return tr_grid
end

begin
    tg = grid_generator_3d(sampling_grid, thetas)
    tginv = get_inv_grid2d(sampling_grid_2d, thetas)
    tginv2 = get_inv_grid(sampling_grid_2d, thetas)

    out = sample_patch(tg |> gpu, x)
    outi = sample_patch(tginv |> gpu, out)
    outi2 = sample_patch(tginv2 |> gpu, out)
end

begin
    ind = mod(ind + 1, 64) + 1
    # ind = 32
    px = plot_digit(cpu(x)[:, :, ind])
    p1 = plot_digit(cpu(out)[:, :, 1, ind])
    p2 = let
        plot_digit(cpu(outi)[:, :, 1, ind])
        plot_digit!(cpu(x)[:, :, ind], alpha=0.3)
    end
    p3 = let
        plot_digit(cpu(outi2)[:, :, 1, ind])
        plot_digit!(cpu(x)[:, :, ind], alpha=0.3)
    end
    plot(px, p1, p2, p3,)
    # plot(p1, p3)
end

fcto = [1 0 -1/2; 0 1 -1/2; 0 0 1]
fxto = [1 0 1/2; 0 1 1/2; 0 0 1]

Fcto = [fcto for _ in 1:64]
Fxto = [fxto for _ in 1:64]



rotmat(θ) = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 1; 0 0 1]
scalemat(sc, offs::Float32=0.0f0) = [sc[1]+offs 0 0; 0 sc[2]+offs 1; 0 0 1]
shearmat(s) = [1 s 0; 0 1 1; 0 0 1]
bmat(bs) = [1 0 bs[1]; 0 1 bs[2]; 0 0 1]

get_inv_grid(sampling_grid, thetas)

sc = thetas[[1, 4], :] .+ scale_offset
# bs = sc .* thetas[5:6, :]
bs = thetas[5:6, :]
Asc = map(x -> scalemat(x, 0.0f0), eachcol(cpu(sc)))
Ashear = map(shearmat, cpu(thetas)[3, :])
Arot = map(rotmat, cpu(thetas)[2, :])
Ab = map(bmat, eachcol(cpu(bs)))

A = map(*, Fxto, Arot, Ashear, Asc, Ab, Fcto)

Ai = map(x -> diag(inv(x)), A)
Ainv = hcat(Ai...) |> gpu
sampling_grid

batched_mul(unsqueeze(Ainv, 1), sampling_grid)

unsqueeze(Ainv, 2) .* (sampling_grid_2d .- unsqueeze(bs, 2))
