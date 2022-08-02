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

include(srcdir("rnp_v1_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :seqlen => 4, :λ => 1.0f-3,
    :scale_offset => 1.8f0, :λ2 => 0.25f0
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

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]] |> dev
const diag_mat = cat(diag_vec..., dims=3) |> dev
## ====


Encoder = let
    enc1 = Chain(
        x -> unsqueeze(x, 3),
        Conv((5, 5), 1 => 32),
        BatchNorm(32, relu),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        Conv((5, 5), 32 => 8, stride=2),
        flatten,
    )

    n_in = Flux.outputsize(enc1, size(x))[1]

    Chain(
        enc1,
        Dense(n_in, 64),
        LayerNorm(64, elu),
        Dense(64, 64),
        LayerNorm(64, elu),
        Dense(64, args[:π], elu),
    )
end |> gpu



l_enc = (args[:π] + args[:asz] + 1) * args[:esz] # 2 of thems
l_fx = get_rnn_θ_sizes(args[:esz], args[:π])
l_fa = get_rnn_θ_sizes(args[:esz], args[:π])
l_decz = args[:π] * 784 # no bias
l_deca = (args[:π]) * args[:asz] # no bias

# model_bounds = [l_enc, l_enc, l_fx, l_fa, l_decz, l_deca]
model_bounds_x = [l_enc, l_fx, l_decz]
model_bounds_a = [l_enc, l_fa, l_deca]

# try different head for each output
Hx = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Split(
        [Dense(64, θ, bias=false) for θ in model_bounds_x]...,
        Dense(64, args[:π], elu), # init z0
    )
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Split(
        [Dense(64, θ, bias=false) for θ in model_bounds_a]...,
        Dense(64, args[:asz], cos), # init a0
    )
) |> gpu

ps = Flux.params(Encoder, Hx, Ha)
## =====

args[:λ] = 0.00f0
args[:λ2] = 0.25f0
args[:scale_offset] = 2.0f0
lg = nothing

opt = ADAM(1e-4)
begin
    Ls = []
    for epoch in 1:20

        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)

        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)

        L = test_model(test_loader)

        @info "Test loss: $L"
        push!(Ls, ls)
    end
end
## =====

loss, grad = withgradient(ps) do
    Lout, L2 = model_loss(x)
    Lout + args[:λ2] * L2
end


## ====

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

patches, preds, a1s, z1s, patches_t = get_loop(x)

preds[1]

heatmap(cpu(preds[1])[:, :, 1, 1])
heatmap(cpu(preds[2])[:, :, 1, 1])
heatmap(cpu(preds[end])[:, :, 1, 1])

