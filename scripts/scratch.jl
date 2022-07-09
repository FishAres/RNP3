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
    :esz => 32, :add_offset => true,
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
sampling_grid_2d = sampling_grid[1:2, :, :]

scale_offset = let
    tmp = zeros(Float32, 6, args[:bsz])
    tmp[[1, 4], :] .= 2.0f0
    tmp
end |> gpu

## todo: fₓ = Wh(z)*h + Wϵ(z, ϵ_(t-1))*ϵ_t + bₓ
## todo  where We(z)(h_ϵ, ϵ_1:t) is an RNN parameterized
## todo  by z through some hypernetworkd


## ====

# * RNNs

fx_param_lenghts = rec_rnn_θ_sizes(args[:esz], args[:π])
# ! cuidado, los decoderos son los que tienen el mismo tamaño que los encoderos
fa_param_lenghts = rec_rnn_θ_sizes(args[:π], 6)
fϵ_param_lenghts = rec_rnn_θ_sizes(args[:π], args[:π])
l_encs = 784 * args[:esz] + args[:esz] # same length for both right now
l_decs = 784 * args[:π] # same but backwards + no bias

model_bounds = [map(sum, (fx_param_lenghts, fa_param_lenghts, fϵ_param_lenghts))...; l_encs; l_encs; l_decs]

lθ = sum(model_bounds)

H = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 128),
    LayerNorm(128, elu),
    Dense(128, lθ, bias=false),
) |> gpu

fz2 = RNN(args[:π], args[:π]) |> gpu

ps = Flux.params(H, fz2)
## =====
function get_models(θs, model_bounds)
    inds = [0; cumsum(model_bounds)]

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    W̄x = get_rn_θs(Θ[1], args[:esz], args[:π])
    W̄a = get_rn_θs(Θ[2], args[:π], 6)
    W̄ϵ = get_rn_θs(Θ[3], args[:π], args[:π])

    fx = ps_to_RN(W̄x; rn_fun=RN)
    fa = ps_to_RN(W̄a; rn_fun=RN, f_out=identity)
    fϵ = ps_to_RN(W̄ϵ; rn_fun=RN)

    Enc_ϵ_fx = Chain(HyDense(784, args[:esz], Θ[4], elu), flatten)
    Enc_ϵ_z = Chain(HyDense(784, args[:esz], Θ[5], elu), flatten)
    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[6], elu), flatten)

    fx, fa, fϵ, Enc_ϵ_fx, Enc_ϵ_z, Dec_z_x̂
end

function model_forward(z, x, model_bounds)
    θs = H(z)
    fx, fa, fϵ, Enc_ϵ_fx, Enc_ϵ_z, Dec_z_x̂ = get_models(θs, model_bounds)

    err0 = randn(Float32, 784, args[:bsz]) |> gpu
    ê_fx = Enc_ϵ_fx(err0)
    a1 = fa(fx.state)
    z1 = fx(ê_fx)

    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in(x, xy, sampling_grid)
    out = sample_patch(x̂, xy, sampling_grid)

    err = Zygote.ignore() do
        flatten(patch_t) .- x̂
    end
    hϵ = fϵ(Enc_ϵ_z(err))
    for t = 2:2
        err0 = randn(Float32, 784, args[:bsz]) |> gpu
        ê_fx = Enc_ϵ_fx(err0)
        a1 = fa(fx.state)
        z1 = fx(ê_fx)

        x̂ = Dec_z_x̂(z1)
        patch_t = zoom_in(x, xy, sampling_grid)
        err = Zygote.ignore() do
            flatten(patch_t) .- x̂
        end

        hϵ = fϵ(Enc_ϵ_z(err))
        out += sample_patch(x̂, xy, sampling_grid)
    end
    out, hϵ
end

function model_loop(z, x)
    out, hϵ = model_forward(z, x, model_bounds)
    for i in 2:2
        z = fz2(hϵ)
        out2, hϵ = model_forward(z, x, model_bounds)
        out += out2
    end
    out
end

function model_loss(x, z)
    out = model_loop(z, x)
    Flux.mse(flatten(out), flatten(x))
end

function plot_rec(out, x, ind)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    return plot(p1, p2)
end

function plot_recs(xs, inds)
    z = randn(Float32, args[:π], args[:bsz]) |> gpu
    out = model_loop(z, xs)
    p = [plot_rec(out, xs, ind) for ind in inds]
    return plot(p...; layout=(length(inds), 1), size=(400, 800))
end


## ===== 
function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, (x, y)) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    x_
end



## =====
err0 = randn(Float32, 784, args[:bsz]) |> gpu

opt = ADAM(1e-4)

for epoch in 1:20
    ls = train_model(opt, ps, train_loader; epoch=epoch)
    p = plot_recs(sample_loader(test_loader), 1:6)
    display(p)
    L = test_model(test_loader)
    @info "Test loss: $L"
end


## =====

function train_model(opt, ps, train_data; epoch=1)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, (x, y)) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            model_loss(x, zs[i])
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data)
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, (x, y)) in enumerate(test_data)
        L += model_loss(x, zs[i])
    end
    return L / length(test_data)
end

