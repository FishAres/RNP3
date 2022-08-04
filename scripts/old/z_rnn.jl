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

include(srcdir("z_rnn_model_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 0.0f0,
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
    Dense(args[:π], 32),
    LayerNorm(32, gelu),
    Dense(32, 64),
    LayerNorm(64, gelu),
    Dense(64, 64),
    LayerNorm(64, gelu),
    Dense(64, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

println("# hypernet params: $(sum(map(prod, size.(Flux.params(H)))))")

RN2 = RNN(args[:π], args[:π], elu) |> gpu

ps = Flux.params(H, RN2)

## ======

args[:seqlen] = 6
args[:scale_offset] = 2.4f0
args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
# args[:δL] = 0.0f0
args[:λ] = 0.005f0
# opt = ADAM(2e-3)

opt = ADAM(2e-4)
begin
    # Ls = []
    for epoch in 37:150
        if epoch % 5 == 0
            opt.eta = 0.8 * opt.eta
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        L = test_model(test_loader)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 50 == 0
            save_model((RN2, H), "z_rnn_model_gelu_7step_$(epoch)eps")
        end
    end
end
## ====

# ## ====
z = randn(args[:π], args[:bsz]) |> dev
# inds = sample(1:args[:bsz], 6, replace=false)
# p = plot_recs(sample_loader(test_loader), inds)

@time patches, preds, errs, xys, z1s, zs, patches_t, Vxs = get_loop(z, x)
zz = cat(zs..., dims=3)
# 
recs = [sample_patch(x, xy, sampling_grid) for (x, xy) in zip(gpu(patches), gpu(xys))] |> cpu
ind = 0

sem(x; dims=1) = std(x, dims=dims) / sqrt(size(x, dims))

begin
    ind = mod(ind + 1, 64) + 1
    plot_tings(x, y) =
        let
            p1 = plot_digit(x, c=:grays)
            plot_digit!(y, alpha=0.3, c=:grays)
            p1
        end
    p1 = [plot_tings(rec[:, :, 1, ind], cpu(x)[:, :, ind]) for rec in recs]
    plot(p1...)
end

patches_t[1]

ind = 0
p1 = begin
    ind = mod(ind + 1, 64) + 1
    # pp = plot([plot_digit(reshape(x[:, :, 1, ind], 28, 28), c=:jet, boundc=false, colorbar=true) for x in recs]...)
    pp = plot([plot_digit(reshape(x[:, ind], 28, 28), c=:jet, boundc=false, colorbar=true) for x in patches]..., title="predictions", titlefontsize=10)
    pe = plot([plot_digit(reshape(x[:, ind], 28, 28), c=:jet, boundc=false, colorbar=true) for x in errs]..., title="patches", titlefontsize=10)

    p1 = plot(pp, pe, layout=(2, 1))

    px = plot_digit(reshape(cpu(x)[:, :, ind], 28, 28))
    pout = plot_digit(preds[end][:, :, 1, ind])
    pz = plot(zz[:, ind, :]', legend=false, title=L"z^2")
    cc = heatmap(cor(zz[:, ind, :]))
    plot(plot(px, pout,), p1, plot(pz, cc), layout=(3, 1), size=(600, 900))
end

begin
    ind = mod(ind + 1, 64) + 1
    px = plot_digit(reshape(cpu(x)[:, :, ind], 28, 28))
    pout = plot_digit(preds[end][:, :, 1, ind])
    pz = plot(zz[:, ind, :]', legend=false, title=L"z^2")
    cc = heatmap(cor(zz[:, ind, :]), title="corr. coeff")

    patches_ = [reshape(x[:, ind], 28, 28) for x in patches]
    patches_t_ = [reshape(x[:, ind], 28, 28) for x in patches_t]

    p = let
        l = length(patches_)
        p1 = plot([plot_digit(patch) for patch in patches_]..., layout=(l, 1), title="prediction")
        p2 = plot([plot_digit(patch_t) for patch_t in patches_t_]..., layout=(l, 1), title="patch")
        plot(p1, p2,)
    end

    p1 = plot(px, pout, pz, cc)
    plot(p1, p, size=(1000, 700))
end

## =====

thetas = randn(6, args[:bsz]) |> dev

out = sample_patch(x, thetas, sampling_grid)
ginv = get_inv_grid(sampling_grid, thetas)
outi = sample_patch(out, ginv) |> cpu
outi = zoom_in2d(x, thetas, sampling_grid) |> cpu
begin
    ind = mod(ind + 1, 64) + 1
    p1 = plot_digit(cpu(out)[:, :, 1, ind])
    p2 = plot_digit(outi[:, :, 1, ind])
    plot(p1, p2)
end

begin
    ind = mod(ind + 1, 64) + 1
    p = [heatmap(V[:, :, ind], clim=(-2, 2)) for V in Vxs]
    plot(p...)
end
Vv = cat(Vxs..., dims=4)

a = Vv[:, :, 13, :]
heatmap(cor(flatten(a)))

## ====

zz
