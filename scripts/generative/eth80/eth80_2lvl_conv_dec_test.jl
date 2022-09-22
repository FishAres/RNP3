using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using Images
using StatsBase: sample
using Random: shuffle
using ParameterSchedulers

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)

## ======

args = Dict(
    :bsz => 64, :img_size => (50, 50), :img_channels => 3, :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => elu,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
)
args[:imszprod] = prod(args[:img_size])
## =====

device!(0)

dev = gpu
## =====

data = load(datadir("exp_pro", "eth80_segmented_train_test.jld2"))

train_data = data["train_data"]
test_data = data["test_data"]

train_loader = DataLoader(train_data |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(test_data |> dev, batchsize=args[:bsz], shuffle=true, partial=false)

## =====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev
## ===== functions

function get_param_sizes(model)
    nw = []
    for m in Flux.modules(model)
        if hasproperty(m, :weight)
            wprod = prod(size(m.weight)[1:end-1])
            if hasproperty(m, :bias)
                wprod += size(m.bias)[1]
            end
            push!(nw, wprod)
        end
    end
    return nw
end

function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)

    # conv decoder
    Dec_z_x̂ = Chain(
        HyDense(args[:π], 256, Θ[3], elu),
        x -> reshape(x, 8, 8, 4, :),
        HyConvTranspose((5, 5), 4 => 32, Θ[4], relu; stride=1),
        HyConvTranspose((6, 6), 32 => 32, Θ[5], relu; stride=2, pad=2),
        HyConvTranspose((6, 6), 32 => 3, Θ[6], relu; stride=2, pad=1)
    )
    z0 = fz.(Θ[7])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], sin), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=sin)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)

    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function imview_cifar(x)
    colorview(RGB, permutedims(batched_adjoint(x), [3, 1, 2]))
end

function plot_rec_cifar(x, out, xs::Vector, ind)
    out_ = reshape(cpu(out), args[:img_size]..., 3, :)
    x_ = reshape(cpu(x), args[:img_size]..., 3, size(x)[end])
    p1 = plot(imview_cifar(out_[:, :, :, ind]), axis=nothing,)
    p2 = plot(imview_cifar(x_[:, :, :, ind]), axis=nothing, size=(20, 20))
    p3 = plot([plot(imview_cifar(x[:, :, :, ind]), axis=nothing) for x in xs]...)
    return plot(p1, p2, p3, layout=(1, 3))
end


function plot_recs(x, inds; plot_seq=true, args=args)
    full_recs, patches, xys, patches_t = get_loop(x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, args[:img_size]..., args[:img_channels], size(x)[end]), patches)
        [plot_rec_cifar(x, full_recs[end], patches_, ind) for ind in inds]
    end : [plot_rec_cifar(full_recs[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end


"one iteration"
function forward_pass(z1, a1, models, x; args=args, scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    za = vcat(z1, a1) # todo parallel layer?
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    # μ, logvar = z1_pre[1:args[:π], :], z1_pre[args[:π]+1:end, :]
    # z1 = sample_z(μ, logvar, rand(args[:D], args[:π], args[:bsz]) |> gpu)
    a1 = Dec_z_a(f_policy(ea))
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset) |> flatten

    return z1, a1, x̂, patch_t
end


## ====== model

# todo don't bind RNN size to args[:π]
args[:π] = 200
args[:depth_Hx] = 6
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π]) # μ, logvar
# l_dec_x = args[:imszprod] * args[:π] # decoder z -> x̂, no bias
mdec = Chain(
    HyDense(args[:π], 256, args[:bsz]),
    x -> reshape(x, 8, 8, 4, :),
    HyConvTranspose((5, 5), 4 => 32, args[:bsz]; stride=1),
    HyConvTranspose((6, 6), 32 => 32, args[:bsz], relu; stride=2, pad=2),
    HyConvTranspose((6, 6), 32 => 3, args[:bsz], relu; stride=2, pad=1)
)
l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x...]

l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

Hx = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    [Chain(Dense(64, 64), LayerNorm(64, elu)) for _ in 1:args[:depth_Hx]]...,
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
) |> gpu


Encoder = let
    enc1 = Chain(
        x -> reshape(x, args[:img_size]..., args[:img_channels], :),
        Conv((5, 5), args[:img_channels] => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        flatten,
    )
    outsz = Flux.outputsize(enc1, (args[:img_size]..., args[:img_channels], args[:bsz]))
    Chain(
        enc1,
        Dense(outsz[1], 64,),
        LayerNorm(64, elu),
        Dense(64, 64,),
        LayerNorm(64, elu),
        Dense(64, 64,),
        LayerNorm(64, elu),
        Dense(64, 64,),
        LayerNorm(64, elu),
        Split(
            Dense(64, args[:π]),
            Dense(64, args[:π])
        )
    )
end |> gpu
ps = Flux.params(Hx, Ha, Encoder)

## ======

c = HyConvTranspose((5, 5), 4 => 32, W) |> gpu
conv_transpose_dims(c, x)
let
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end
## =====

save_folder = "gen_2lvl"
alias = "2lvl_double_H_eth80_50x50_vae_v01_conv_dec"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 4
args[:scale_offset] = 1.8f0

args[:λpatch] = 0.0f0
args[:λ] = 0.001f0

args[:α] = 1.0f0
args[:β] = 0.2f0

args[:η] = 4e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

## ====
begin
    log_value(lg, "learning_rate", opt.eta)
    Ls = []
    for epoch in 1:2000
        if epoch > 100
            opt.eta = max(0.67 * opt.eta, 1e-7)
            log_value(lg, "learning_rate", opt.eta)
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        if epoch % 250 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

## ====
