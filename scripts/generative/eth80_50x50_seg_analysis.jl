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

device!(1)

dev = gpu
## =====

data = load(datadir("exp_pro", "eth80_segmented_train_test.jld2"))

train_data = data["train_data"]
test_data = data["test_data"]

# save(datadir("exp_pro", "eth80_segmented_train_test.jld2"), Dict("train_data" => train_data, "test_data" => test_data))

## =====
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
function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)

    # rgb
    Dec_z_x̂r = Chain(HyDense(args[:π], args[:imszprod], Θ[3], relu6), flatten)
    Dec_z_x̂g = Chain(HyDense(args[:π], args[:imszprod], Θ[4], relu6), flatten)
    Dec_z_x̂b = Chain(HyDense(args[:π], args[:imszprod], Θ[5], relu6), flatten)

    Dec_z_x̂ = Chain(
        Split(
            Dec_z_x̂r,
            Dec_z_x̂g,
            Dec_z_x̂b,
        ),
        x -> hcat(unsqueeze.(x, 2)...)
    )

    z0 = fz.(Θ[6])

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

## ====== model

# todo don't bind RNN size to args[:π]
args[:π] = 200
args[:depth_Hx] = 6
args[:D] = Normal(0.0f0, 1.0f0)

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_err_rnn = get_rnn_θ_sizes(args[:π], args[:π]) # also same size lol
l_dec_x = args[:imszprod] * args[:π] # decoder z -> x̂, no bias
l_enc_e_z = ((args[:imszprod] * args[:img_channels]) + 1) * args[:π]

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x; l_dec_x; l_dec_x]


l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]
## ======

modelpath = "saved_models/gen_2lvl/2lvl_double_H_eth80_50x50_segmented_vae_v01/add_offset=true_asz=6_bsz=64_depth_Hx=6_esz=32_glimpse_len=4_img_channels=3_imszprod=2500_scale_offset=2.0_scale_offset_sense=3.2_seqlen=4_α=1.0_β=0.2_δL=0.25_η=4e-5_λ=0.001_λf=0.167_λpatch=0.0_π=200_750eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

## ======
let
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end
## =====

save_folder = "gen_2lvl"
alias = "2lvl_double_H_eth80_50x50_segmented_vae_v01"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 4
args[:scale_offset] = 2.0f0

args[:λpatch] = 0.0f0
args[:λ] = 0.001f0

args[:α] = 1.0f0
args[:β] = 0.2f0
## =====

function get_loop(z, x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, out_small, a1, patch_t), outputs)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out, out_small, a1, patch_t), outputs)
    end
    return outputs
end


function stack_ims(x; nrow=5)
    a = collect(partition(eachslice(x[:, :, :, 1:nrow^2], dims=4), nrow))
    b = map(x -> vcat(x...), a)
    c = hcat(b...)
    return colorview(RGB, permutedims(c, [3, 1, 2]))
end

## ======
# x = sample_loader(test_loader)
z = randn(Float32, args[:π], args[:bsz]) |> gpu

full_recs, patches, xys, patches_t = get_loop(z, x)
stacked_rec = stack_ims(full_recs[end]; nrow=5)
plot(stacked_rec, size=(500, 500), axis=nothing)

## =======

function stack_zs(xs)
    z2s, z1s = [], []
    for x in xs
        μ, logvar = Encoder(x)
        push!(z2s, μ |> cpu)
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)
        z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
        push!(z1s, z1 |> cpu)
        for t = 2:args[:seqlen]
            z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
            push!(z1s, z1 |> cpu)
        end
    end
    return z2s, z1s
end

z2s, z1s = stack_zs(test_loader)

z2vec = hcat(z2s...)
z1vec = hcat(z1s...)

using TSne

Y = tsne([z2vec'; z1vec'])

scatter(Y[:, 1], Y[:, 2])