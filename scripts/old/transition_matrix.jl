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

include(srcdir("model_utils.jl"))

CUDA.allowscalar(false)
## =====

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

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1 0; 0 1] for _ in 1:args[:bsz]] |> dev

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
println(lθ, " parameters for primary")
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

## ===='
args[:seqlen] = 6
args[:scale_offset] = 2.0f0
args[:δL] = round(Float32(1 / 6), digits=3)
args[:λ] = 0.004f0
opt = ADAM(1e-3)

begin
    Ls = []
    for epoch in 1:20
        ls = train_model(opt, ps, train_loader; epoch=epoch)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        L = test_model(test_loader)
        @info "Test loss: $L"
        push!(Ls, ls)
    end
end

## ====
z = randn(args[:π], args[:bsz]) |> dev
inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

@time patches, preds, errs, xys, z1s, zs, patches_t, Vxs = get_loop(z, x)
zz = cat(zs..., dims=3)

recs = [sample_patch(x, xy, sampling_grid) for (x, xy) in zip(gpu(patches), gpu(xys))] |> cpu

begin
    ind = mod(ind + 1, 64) + 1
    p = [plot_digit(rec[:, :, 1, ind]) for rec in recs]
    plot(p...)
end



ind = 0
p1 = begin
    ind = mod(ind + 1, 64) + 1
    pp = plot([plot_digit(reshape(x[:, ind], 28, 28), c=:jet, boundc=false, colorbar=true) for x in patches]...)
    pe = plot([plot_digit(reshape(x[:, ind], 28, 28), c=:jet, boundc=false, colorbar=true) for x in errs]...)
    p1 = plot(pp, pe, layout=(2, 1))

    px = plot_digit(reshape(cpu(x)[:, :, ind], 28, 28))
    pout = plot_digit(preds[end][:, :, 1, ind])
    pz = plot(zz[:, ind, :]', legend=false)
    cc = heatmap(cor(zz[:, ind, :]))
    plot(plot(px, pout,), p1, plot(pz, cc), layout=(3, 1), size=(600, 900))
end

θs = H(z)
Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0 = get_models(θs, model_bounds)
models = Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a

# is z0, a0 necessary?
z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)

a1[3:4, :]