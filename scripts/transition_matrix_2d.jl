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

include(srcdir("model_utils_2d.jl"))


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
using Distributions
thetas = rand(Uniform(-1.0f0, 1.0f0), 4, args[:bsz]) |> gpu
out = sample_patch(x, thetas, sampling_grid)
outi = zoom_in2d(x, thetas, sampling_grid)
outix = zoom_in2d(x, thetas, sampling_grid)


begin
    ind = mod(ind + 1, args[:bsz]) + 1
    px = plot_digit(cpu(x)[:, :, ind])
    po = plot_digit(cpu(out)[:, :, 1, ind])
    poi = let
        plot_digit(cpu(outi)[:, :, 1, ind])
        plot_digit!(cpu(x)[:, :, ind], alpha=0.4)
    end

    plot(px, po, poi,)
end



## =====

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

## ====
args[:seqlen] = 5
args[:scale_offset] = 3.2f0
# args[:δL] = Float32(1 / args[:seqlen])
args[:δL] = round(Float32(1 / 6), digits=3)
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


## =====
inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

z = randn(Float32, args[:π], args[:bsz]) |> gpu


ind = 0
begin
    ind = mod(ind + 1, 64) + 1
    p = [plot_digit(reshape(x[:, ind], 28, 28),) for x in errs]
    plot(p...)
end

patches, preds, errs, xys, zs = get_loop(z, x)
p1 = begin
    ind = mod(ind + 1, 64) + 1
    pp = plot([plot_digit(reshape(x[:, ind], 28, 28),) for x in patches]...)
    pe = plot([plot_digit(reshape(x[:, ind], 28, 28), boundc=false, colorbar=true) for x in errs]...)
    p1 = plot(pp, pe, layout=(2, 1))

    px = plot_digit(reshape(cpu(x)[:, :, ind], 28, 28))
    plot(px, p1, layout=(2, 1), size=(600, 800))
end

xys[1]


begin
    ind = mod(ind + 1, 64) + 1
    as = [sample_patch(patch, xy, sampling_grid |> cpu) for (patch, xy) in zip(patches, xys)]
    p = [plot_digit(a[:, :, 1, ind]) for a in as]
    pe = plot_digit(preds[end][:, :, 1, ind])
    plot(p..., pe)
end

a = hcat(xys...)
histogram(a[1, :])
histogram!(a[2, :])

histogram(a[3, :] .* a[1, :])
histogram!(a[4, :] .* a[2, :])