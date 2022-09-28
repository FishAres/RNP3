using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

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
## ===== functions

function get_param_sizes(model)
    nw = []
    for m in Flux.modules(model)
        if hasproperty(m, :weight)
            wprod = prod(size(m.weight)[1:end-1])
            if hasproperty(m, :bias)
                if m.bias == true
                    wprod += size(m.bias)[1]
                end
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
        HyDense(args[:π], 64, Θ[3], elu),
        flatten,
        HyDense(64, 64, Θ[4], elu),
        flatten,
        HyDense(64, 64, Θ[5], elu),
        flatten,
        HyDense(64, 64, Θ[6], elu),
        flatten,
        HyDense(64, args[:imzprod], Θ[7], relu)
    )
    z0 = fz.(Θ[8])

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



## ====== model
using DiffEqFlux, DifferentialEquations

# todo don't bind RNN size to args[:π]
args[:π] = 64
args[:D] = Normal(0.0f0, 1.0f0)
args[:norm_groups] = 8
args[:seqlen] = 4
args[:scale_offset] = 2.0f0

# args[:λpatch] = Float32(1 / 3args[:seqlen])
# args[:λpatch] = 1.0f-4
args[:λpatch] = 0.0f0
args[:λ] = 1.0f-6
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.05f0


l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π]) # μ, logvar

mdec = Dec_z_x̂ = Chain(
    HyDense(args[:π], 64, args[:bsz], elu),
    flatten,
    HyDense(64, 64, args[:bsz], elu),
    flatten,
    HyDense(64, 64, args[:bsz], elu),
    flatten,
    HyDense(64, 64, args[:bsz], elu),
    flatten,
    HyDense(64, args[:imzprod], args[:bsz], relu)
)

l_dec_x = get_param_sizes(mdec)

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x...]

l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

modelpath = "saved_models/gen_2lvl/omni_2lvl_conv_node_H_v0/add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_img_channels=1_imzprod=784_norm_groups=8_scale_offset=2.0_scale_offset_sense=3.2_seqlen=4_α=1.0_β=0.05_η=0.0001_λ=1e-6_λf=0.167_λpatch=0.0_π=64_100eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu
## ======

let
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end

## ====== 
include(srcdir("fancy_plotting_utils.jl"))

x = first(test_loader)

patches, sub_patches, xys, xys1 = get_loop_patches(x) |> gpu
pasted_patches =
    [sample_patch(patches[i], xys[i], sampling_grid) for i = 1:length(patches)] |> cpu

digits = orange_on_rgb_array(pasted_patches)
# digit_patches = [orange_on_rgb_array(map(x -> reshape(x, 28, 28, 1, 64), z)) for z in sub_patches]
xys[1]
digit_patches = [
    orange_on_rgb_array([
        sample_patch(sample_patch(sub_patch, xy, sampling_grid), xy2, sampling_grid) for
        (sub_patch, xy, xy2) in zip(subpatch_vec, xy_vec, xys)
    ]) for (subpatch_vec, xy_vec) in zip(sub_patches, xys1)
] |> cpu

digit_patches[1]

patches_ = [
    [
        sample_patch(sub_patch, xy, sampling_grid) for
        (sub_patch, xy) in zip(subpatch_vec, xy_vec)
    ] for (subpatch_vec, xy_vec) in zip(sub_patches, xys1)
]

patches_[1][1]
patches_[1]
length(patches_)


xys[1]
a = orange_on_rgb_array([sample_patch(patches_[k][k], xys[1], sampling_grid) for k in 1:4] |> cpu)
colorview(RGB, a[2])






## ======
digits, digit_patches, digit_im, patch_ims, xys, xys1 = get_digit_parts(x, 1)

colorview(RGB, digits[1])

digit_patches[1]

ind = 0
x = first(test_loader)
begin
    # ind = mod(ind + 1, 64) + 1
    println(ind)
    digits, digit_patches, digit_im, patch_ims, xys, xys1 = get_digit_parts(x, ind;
        savestring="plots/reconstructions/paper_reconstructions/parts_subparts/omniglot/omni_digit_01")
    digit_im
end


begin
    ind += 1
    digits, digit_patches, digit_im, patch_ims, xys, xys1 = get_digit_parts(first(test_loader), ind)
    digit_im
end
ind