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

device!(0)

dev = gpu

## =====

function get_rand_thetas(args; blims=0.1f0)
    thetas = zeros(Float32, 6, args[:bsz])
    # thetas[[1, 4], :] .= -2.0f0
    thetas[[1, 4], :] .= -3.0f0
    thetas[5:6, :] .= rand(Uniform(-blims, blims), 2, args[:bsz])
    thetas[2, :] .= 2.0f0 * π32 .* rand(Uniform(-1.0f0, 1.0f0), args[:bsz])
    thetas[3, :] .= rand(Uniform(-0.5f0, 0.5f0), args[:bsz])
    return thetas
end


function get_quad_mnist(x::Tuple{Array{Float32,4},Array{Float32,4}})
    zero_canv = zeros(Float32, args[:img_size]..., 1, args[:bsz])
    canv = [zero_canv for _ in 1:4]
    canvas = cat(canv..., dims=3)
    for batch in 1:args[:bsz]
        quads = sample(1:4, 2, replace=false)
        for (i, j) in enumerate(quads)
            canvas[:, :, j, batch] = x[i][:, :, 1, batch]
        end
    end
    canvas
end

function squish_quads(x)
    a = collect(eachslice(x, dims=3))
    b = vcat(a[1], a[2])
    c = vcat(a[3], a[4])
    d = hcat(b, c)
    return unsqueeze(imresize(d, args[:img_size]), 3)
end

function gen_quad_data(xs; n_samples=100)
    xout = []
    for n in 1:n_samples
        x1, x2 = rand(xs, 2)
        a1, a2 = [get_rand_thetas(args) for _ in 1:2]
        x1 = sample_patch(gpu(x1), gpu(a1), sampling_grid) |> cpu
        x2 = sample_patch(gpu(x2), gpu(a2), sampling_grid) |> cpu
        push!(xout, get_quad_mnist((x1, x2)) |> squish_quads)
    end
    xout
end


function fast_concat(xs; args=args)
    xout = zeros(Float32, args[:img_size]..., 1, args[:bsz] * length(xs))
    Threads.@threads for i in 1:length(xs)
        xout[:, :, 1, (i-1)*args[:bsz]+1:i*args[:bsz]] = xs[i]
    end
    return xout
end

function prep_mnist_quads(args; ntrain=2000, ntest=200)

    train_digits, train_labels = MNIST(split=:train)[:]
    test_digits, test_labels = MNIST(split=:test)[:]

    train_vec = collect(eachslice(train_digits, dims=3))
    test_vec = collect(eachslice(test_digits, dims=3))
    train_batches = unsqueeze.(batch.(shuffle(collect(partition(train_vec, args[:bsz])))), 3)
    test_batches = unsqueeze.(batch.(shuffle(collect(partition(test_vec, args[:bsz])))), 3)

    train_data = fast_concat(gen_quad_data(train_batches; n_samples=ntrain))
    test_data = fast_concat(gen_quad_data(test_batches; n_samples=ntest))

    return train_data, test_data
end

## ====

train_data, test_data = prep_mnist_quads(args; ntrain=2000, ntest=200)

train_loader = DataLoader((train_data |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_data |> dev), batchsize=args[:bsz], shuffle=true, partial=false)


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

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)

    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function full_sequence2(models::Tuple, z0, a0, x; args=args, scale_offset=args[:scale_offset])
    f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a = models
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    out_small = full_sequence(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        out_small = full_sequence(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
    end
    return out
end

function full_sequence2(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence2(models, z0, a0, x; args=args, scale_offset=scale_offset)
end

function model_loss(x, r; args=args)
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence2(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    Lpatch = Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence2(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        Lpatch += Flux.mse(flatten(out_small), flatten(patch_t); agg=sum)
    end
    klqp = kl_loss(μ, logvar)
    rec_loss = Flux.mse(flatten(out), flatten(x); agg=sum)
    return rec_loss + args[:λpatch] * Lpatch, klqp
end

"output sequence: full recs, local recs, xys (a1), patches_t"
function get_loop(x; args=args)
    outputs = patches, recs, as, patches_t = [], [], [], [], []
    r = rand(args[:D], args[:π], args[:bsz]) |> gpu
    μ, logvar = Encoder(x)
    z = sample_z(μ, logvar, r)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
    out_small = full_sequence2(z1, patch_t)
    out = sample_patch(out_small, a1, sampling_grid)
    push_to_arrays!((out, out_small, a1, patch_t), outputs)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
        out_small = full_sequence2(z1, patch_t)
        out += sample_patch(out_small, a1, sampling_grid)
        push_to_arrays!((out, out_small, a1, patch_t), outputs)
    end
    return outputs
end


## ====

# todo don't bind RNN size to args[:π]

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

Hx = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Hx_bounds) + args[:π], bias=false),
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
) |> gpu


Encoder = let
    enc1 = Chain(
        x -> reshape(x, 28, 28, 1, :),
        Conv((5, 5), 1 => 32),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32),
        BatchNorm(32, relu),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        flatten,
    )
    outsz = Flux.outputsize(enc1, (28, 28, 1, args[:bsz]))
    Chain(
        enc1,
        Dense(outsz[1], 64,),
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

## =====

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "gen_3lvl"
alias = "double_H_mnist_noisy_2quad_generative_v0"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 2
args[:scale_offset] = 1.5f0

# args[:λpatch] = Float32(1 / 2 * args[:seqlen])
args[:λpatch] = 0.0f0
args[:λ] = 0.0001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 2.0f0
args[:β] = 0.01f0


args[:η] = 4e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
## ====
begin
    Ls = []
    for epoch in 1:200
        if epoch % 50 == 0
            opt.eta = 0.67 * opt.eta
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
        push!(Ls, ls)
        if epoch % 50 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end


## ======
