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

##=====
train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

## =====

function mnist_quadrants(xs; args=args)
    tmp = collect(partition(eachslice(xs, dims=3), 4))
    a = map(x -> vcat(x[1:2]...), tmp)
    b = map(x -> vcat(x[3:4]...), tmp)
    c = map(x -> hcat(x...), zip(a, b))
    resized_vec = map(x -> imresize(x, args[:img_size]), c)
    # return fast_img_concat(resized_vec)
end

function fast_img_concat(xs; args=args)
    cat_ims = zeros(Float32, args[:img_size]..., length(xs))
    Threads.@threads for i in 1:length(xs)
        cat_ims[:, :, i] = xs[i]
    end
    return cat_ims
end

function gen_mnist_quad_data_mixed(args)
    train_digits, train_labels = MNIST(split=:train)[:]
    test_digits, test_labels = MNIST(split=:test)[:]
    train_ims = mnist_quadrants(train_digits)
    test_ims = mnist_quadrants(test_digits)
    train_digit_vec = collect(eachslice(train_digits, dims=3))
    test_digit_vec = collect(eachslice(test_digits, dims=3))

    train_data = fast_img_concat(shuffle([train_ims; train_digit_vec]))
    test_data = fast_img_concat(shuffle([test_ims; test_digit_vec]))

    return train_data, test_data

end

function shuffle_concat_mnist(digits)
    train_digit_vec = shuffle(collect(eachslice(digits, dims=3)))
    fast_img_concat(train_digit_vec)
end

function gen_mnist_quad_data(args; n_repeats=2)
    train_digits, train_labels = MNIST(split=:train)[:]
    test_digits, test_labels = MNIST(split=:test)[:]
    train_ims = [mnist_quadrants(shuffle_concat_mnist(train_digits)) for _ in 1:n_repeats]
    test_ims = [mnist_quadrants(shuffle_concat_mnist(test_digits)) for _ in 1:n_repeats]

    train_data = fast_img_concat(vcat(train_ims...))
    test_data = fast_img_concat(vcat(test_ims...))

    return train_data, test_data
end

train_data, test_data = gen_mnist_quad_data(args)

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

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], sin), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=sin)
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
        x -> reshape(x, 28, 28, 1, :),
        Conv((5, 5), 1 => 32,),
        BatchNorm(32, relu),
        BasicBlock(32 => 32, +),
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
alias = "double_H_mnist_quad_generative_v0"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 4
args[:scale_offset] = 2.0f0

# args[:λpatch] = Float32(1 / 2 * args[:seqlen])
args[:λpatch] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 4.0f0
args[:β] = 0.02f0


args[:η] = 4e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
log_value(lg, "learning_rate", opt.eta)
## ====
begin
    Ls = []
    for epoch in 1:400
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
        if epoch % 25 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end


## ======
