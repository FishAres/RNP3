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
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
)
args[:imzprod] = prod(args[:img_size])

## =====

device!(1)

dev = gpu

##=====
train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)


## =====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev
## =====


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

## ====

args[:π] = 32

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_enc_e_z = (784 + 1) * args[:π]

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x]


l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

modelpath = "saved_models/gen_2lvl/double_H_mnist_generative_v03/add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_img_channels=1_imzprod=784_scale_offset=2.0_scale_offset_sense=3.2_seqlen=4_α=1.0_β=0.5_η=4e-5_λ=0.001_λf=0.167_λpatch=0.0_π=32_100eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

## ======

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

args[:seqlen] = 4
args[:scale_offset] = 2.0f0
args[:D] = Normal(0.0f0, 1.0f0)

## ====

# todo - get centroids of z2 and train a new embedding for Ha
# todo - does it look like the original z2 means after?

function get_zs(x; args=args)
    z2s, z1s = [], []
    μ, logvar = Encoder(x)
    # z = sample_z(μ, logvar, r)
    z = μ
    push!(z2s, cpu(μ))
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    push!(z1s, cpu(z1))
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        push!(z1s, cpu(z1))
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
    end
    return vcat(z2s...), hcat(z1s...)
end

x = first(test_loader)

z2, z1 = let
    z2, z1 = [], []
    for x in test_loader
        z2_, z1_ = get_zs(x)
        push!(z2, z2_)
        push!(z1, z1_)
    end
    z2, z1
end

z1s = hcat(z1[1:10]...)
z2s = hcat(z2[1:10]...)

using TSne

zz = hcat(shuffle(collect(eachcol(hcat(z1s, z2s))))...)

Y = tsne(zz')

scatter(Y[:, 1], Y[:, 2], legend=false)
scatter(Y[:, 1], Y[:, 2], c=inds, legend=false)


using Clustering

R = kmeans(zz, 40; maxiter=200, display=:iter)

zout = [R.centers zeros(Float32, args[:π], args[:bsz] - size(R.centers, 2))] |> gpu

function get_loop(z, x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    θsz = Hx(z)
    θsa = Ha(za)
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

get_loop(zout, x)

function plot_recs(z, x, inds; plot_seq=true, args=args)
    full_recs, patches, xys, patches_t = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        [plot_rec(full_recs[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(full_recs[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(zout, sample_loader(test_loader), inds)

## =====

function get_centroid_z(x; n_clusters=10)
    # z2, z1 = get_zs(x)
    z2, _ = get_zs(x)
    R = kmeans(z2, n_clusters; maxiter=200)
    zout = [R.centers zeros(Float32, args[:π], args[:bsz] - size(R.centers, 2))] |> gpu
    zout
end

function model_loss_centroids(x; args=args)
    z = Zygote.ignore() do
        get_centroid_z(x)
    end
    θsz = Hx(z)
    θsa = Ha(za)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    end
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch
end

function train_model_centroids(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss = model_loss_centroids(x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
            end
            rec_loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model_centroids(test_data; D=args[:D])
    rs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss = model_loss_centroids(x)
        L += rec_loss
    end
    return L / length(test_data)
end


zout = get_centroid_z(x)

r = randn(Float32, args[:π], args[:bsz]) |> gpu

za = randn(Float32, args[:π], args[:bsz]) |> gpu

ps = Flux.params(za)

test_model_centroids(test_loader)

opt = ADAM(1e-4)
ls = train_model_centroids(opt, ps, train_loader)


inds = sample(1:args[:bsz], 6, replace=false)
x = sample_loader(test_loader)
zout = get_centroid_z(x)
p = plot_recs(zout, x, inds)
