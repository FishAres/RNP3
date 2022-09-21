using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32, :img_channels => 1,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :λpatch => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0)
)
args[:imzprod] = prod(args[:img_size])

## =====

device!(0)

dev = gpu

##=====

all_chars = load("../Recur_generative/data/exp_pro/omniglot_train.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
num_train = trunc(Int, 0.8 * length(xs))

new_chars = load("../Recur_generative/data/exp_pro/omniglot_eval.jld2")
xs_new = shuffle(vcat((new_chars[key] for key in keys(new_chars))...))

function fast_cat(xs)
    x_array = zeros(Float32, size(xs[1])..., length(xs))
    Threads.@threads for i in 1:length(xs)
        x_array[:, :, i] = xs[i]
    end
    x_array
end

xs_cat = fast_cat(xs)
train_chars = xs_cat[:, :, 1:num_train]
val_chars = xs_cat[:, :, num_train+1:end]

new_xs_cat = fast_cat(xs_new)
## ====
train_loader = DataLoader(train_chars |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
val_loader = DataLoader(val_chars |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(new_xs_cat |> dev, batchsize=args[:bsz], shuffle=true, partial=false)

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

x = unsqueeze(first(train_loader), 3)

W = randn(Float32, 5, 5, 1, 32, 64) |> gpu
b = randn(Float32, 32, 64) |> gpu

w = [flatten(W); b]

m = HyConv((5, 5), 1, 32, w)

A = zeros(Float32, 2, 2)

A[diagind(A)[1]] = 1.0f-6
A
inv(A)

A[diagind(A)[1]] ≈ 0.0f0

