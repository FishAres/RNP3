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
    :asz => 4, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 0.0f0,
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
sampling_grid = sampling_grid[1:2, :, :]

## ====

Vx_sz = (args[:π], args[:π],)
Va_sz = (args[:asz], args[:asz],)

# Vϵ_sz = (args[:π], args[:π],)
l_enc = 784 * args[:π] + args[:π] # encoder ϵ -> z, with bias
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_dec_a = args[:asz] * args[:asz] + args[:asz] # decoder z -> a, with bias

# model_bounds = [map(sum, (prod(Vx_sz), prod(Va_sz), Vϵ_sz))...; l_enc; l_decs]
model_bounds = [map(sum, (prod(Vx_sz), prod(Va_sz),))...; l_enc; l_dec_x; l_dec_a]

lθ = sum(model_bounds)
# ! initializes z_0, a_0 
H = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

ps = Flux.params(H)

sum(map(prod, size.(ps)))
## =====
z = randn(args[:π], args[:bsz]) |> gpu
θs = H(z)
Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0 = get_models(θs, model_bounds)
models = Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a

Δa(a, Va, Dec_z_a) = sin.(Dec_z_a(bmul(a, Va)))

