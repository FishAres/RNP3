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

include(srcdir("z_rnn_draw_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 4, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
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

function get_xgrids(w, h, bsz)
    x1 = LinRange(-1, 1, w)
    y1 = LinRange(-1, 1, h)
    xx, yy = meshgrid(x1, y1)
    xx = repeat(xx, 1, 1, bsz)
    yy = repeat(yy, 1, 1, bsz)
    return xx, yy
end

const xgrids = get_xgrids(28, 28, args[:bsz]) |> dev
## =====
## ====

Va_sz = (args[:asz], args[:asz],)

l_dec_a = args[:asz] * args[:asz] #  decoder z -> a, no bias
l_enc_z = 784 * args[:π] + args[:π]

model_bounds = [prod(Va_sz); l_dec_a; l_enc_z]

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
    Dense(64, lθ + args[:asz], bias=false),
) |> gpu

println("# hypernet params: $(sum(map(prod, size.(Flux.params(H)))))")

RN2 = RNN(args[:π], args[:π], gelu) |> gpu

ps = Flux.params(H, RN2)

## =====

xx = rand(1:32, 32)

gradient(xx -> sum(mod.(xx, 12)), xx)

## =====



args[:seqlen] = 8
args[:δL] = 0.002f0
args[:λ] = 0.02f0
opt = ADAM(1e-5)
begin
    Ls = []
    for epoch in 1:40
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

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====
z = randn(args[:π], args[:bsz]) |> gpu

loss, grad = withgradient(ps) do
    model_loss(z, x)
end

replace_nan!(x) = isnan(x) ? 0.0f0 : x


foreach(x -> replace_nan!.(x), grad)



Flux.update!(opt, ps, grad)

[sum(isnan.(p)) for p in ps]


@time patches, preds, errs, a1s, zs, Vas = get_loop(z, x)
zz = cat(zs..., dims=3)

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
    p1 = [plot_tings(rec[:, :, ind], cpu(x)[:, :, ind]) for rec in patches]
    plot(p1...)
end

