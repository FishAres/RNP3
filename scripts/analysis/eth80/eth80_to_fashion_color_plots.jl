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
train_data, train_labels = FashionMNIST(split=:train)[:]
test_data, test_labels = FashionMNIST(split=:test)[:]

train_data = imresize(train_data, args[:img_size])
test_data = imresize(test_data, args[:img_size])

train_data = unsqueeze(train_data, 3)
test_data = unsqueeze(test_data, 3)
begin
    color_train_data = zeros(Float32, size(train_data)[1:2]..., 3, size(train_data)[end])
    Threads.@threads for i in 1:size(train_data)[end]
        x_ = train_data[:, :, :, i]
        a = rand(Float32, 3)
        a = 1.4f0 * a / sum(a)
        r, g, b = a
        color_train_data[:, :, :, i] = cat(r * x_, g * x_, b * x_, dims=3)
    end
end

begin
    color_test_data = zeros(Float32, size(test_data)[1:2]..., 3, size(test_data)[end])
    Threads.@threads for i in 1:size(test_data)[end]
        x_ = test_data[:, :, :, i]
        a = rand(Float32, 3)
        a = 1.4f0 * a / sum(a)
        r, g, b = a
        color_test_data[:, :, :, i] = cat(r * x_, g * x_, b * x_, dims=3)
    end
end

train_loader = DataLoader(color_train_data |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(color_test_data |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
x = first(test_loader)
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
        HyDense(args[:π], 400, Θ[3], elu),
        x -> reshape(x, 10, 10, 4, :),
        HyConvTranspose((5, 5), 4 => 64, Θ[4], relu; stride=1),
        HyConvTranspose((4, 4), 64 => 64, Θ[5], relu; stride=2, pad=2),
        HyConvTranspose((4, 4), 64 => 3, Θ[6], relu; stride=2, pad=2)
    )
    z0 = fz.(Θ[7])

    return (Enc_za_z, f_state, Dec_z_x̂), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
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

function model_loss(x, r; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

## ======

args[:seqlen] = 4
args[:scale_offset] = 2.0f0

# args[:λpatch] = 0.001f0
args[:λpatch] = 0.0f0
args[:λ] = 1.0f-5

args[:α] = 1.0f0
args[:β] = 0.3f0

args[:π] = 256
args[:depth_Hx] = 6
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π]) # μ, logvar
# l_dec_x = args[:imszprod] * args[:π] # decoder z -> x̂, no bias
mdec = Chain(
    HyDense(args[:π], 400, args[:bsz]),
    x -> reshape(x, 10, 10, 4, :),
    HyConvTranspose((5, 5), 4 => 64, args[:bsz]; stride=1),
    HyConvTranspose((4, 4), 64 => 64, args[:bsz], relu; stride=2, pad=2),
    HyConvTranspose((4, 4), 64 => 3, args[:bsz], relu; stride=2, pad=2)
)
l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x...]

l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

## ======

# modelpath = "saved_models/gen_2lvl/2lvl_double_H_eth80_50x50_vae_v01_conv_dec_denseH/add_offset=true_asz=6_bsz=64_depth_Hx=6_esz=32_glimpse_len=4_img_channels=3_imszprod=2500_scale_offset=2.0_scale_offset_sense=3.2_seqlen=4_α=1.0_β=0.2_δL=0.25_η=4e-5_λ=1e-5_λf=0.167_λpatch=0.0_π=256_500eps.bson"

modelpath = "saved_models/gen_2lvl/eth80_50x50_to_fashion_v0/add_offset=true_asz=6_bsz=64_depth_Hx=6_esz=32_glimpse_len=4_img_channels=3_imszprod=2500_scale_offset=2.0_scale_offset_sense=3.2_seqlen=4_α=1.0_β=0.3_δL=0.25_η=4e-5_λ=1e-5_λf=0.167_λpatch=0.0_π=256_60eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

## =====

let
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end

## ======

save_folder = "gen_2lvl"
alias = "eth80_50x50_to_fashion_v0"
save_dir = get_save_dir(save_folder, alias)

## =====

out_ = reshape(cpu(out), args[:img_size]..., 3, :)
x_ = reshape(cpu(x), args[:img_size]..., 3, size(x)[end])
p1 = plot(imview_cifar(out_[:, :, :, ind]), axis=nothing,)
p2 = plot(imview_cifar(x_[:, :, :, ind]), axis=nothing, size=(20, 20))
p3 = plot([plot(imview_cifar(x[:, :, :, ind]), axis=nothing) for x in xs]...)







p = plot_seq ? let
    patches_ = map(x -> reshape(x, args[:img_size]..., args[:img_channels], size(x)[end]), patches)
    [plot_rec_cifar(x, full_recs[end], patches_, ind) for ind in inds]
end : [plot_rec_cifar(full_recs[end], x, ind) for ind in inds]



full_recs[end]
ind = 0

ind = ind - 1

x = sample_loader(test_loader)
full_recs, patches, xys, patches_t = get_loop(x)
begin
    ind = mod(ind + 1, args[:bsz]) + 1

    p = imresize(imview_cifar(full_recs[end][:, :, :, ind]), (100, 100))
end

save("plots/reconstructions/paper_reconstructions/eth80_to_fashion/bag2.png", map(clamp01nan, p))