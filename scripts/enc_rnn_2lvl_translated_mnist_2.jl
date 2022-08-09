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

include(srcdir("double_H_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :λf => 0.167f0, :f_z => elu, :D => Normal(0.0f0, 1.0f0),
    :translation_bounds => 0.5f0,
)

## =====

device!(0)

dev = gpu

##=====
# 1 - transform first then move to gpu in train loop
# 2 - faster gpu -> gpu?

train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

# train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
# test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader(train_digits |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(test_digits |> dev, batchsize=args[:bsz], shuffle=true, partial=false)

x = first(test_loader)
## ====
dev = has_cuda() ? gpu : cpu

const sampling_grid = get_sampling_grid(args[:img_size]...)[1:2, :, :] |> dev
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev
## =====

function trans_thetas(args; bounds=args[:translation_bounds])
    thetas = zeros(Float32, 6, args[:bsz])
    thetas[5:6, :] .= rand(Uniform(-bounds, bounds), 2, args[:bsz])
    return thetas |> dev
end

function stack_batch(x)
    x = cpu(x)
    x = length(size(x)) > 3 ? dropdims(x, dims=3) : x
    b = collect(partition(eachslice(x, dims=3), 8))
    c = map(x -> vcat(x...), b)
    return hcat(c...)
end

function translate_batch(x, args)
    thetas = trans_thetas(args)
    out = sample_patch(x, thetas, sampling_grid; scale_offset=0.0f0)
    return out
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from distribution D
    zs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        x = translate_batch(x, args)
        loss, grad = withgradient(ps) do
            loss = model_loss(zs[i], x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "loss", loss)
            end
            loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    zs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        x = translate_batch(x, args)
        L += model_loss(zs[i], x)
    end
    return L / length(test_data)
end

## ====


# todo don't bind RNN size to args[:π]

args[:π] = 32

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_err_rnn = get_rnn_θ_sizes(args[:π], args[:π]) # also same size lol
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_enc_e_z = (784 + 1) * args[:π]

Hx_bounds = [l_enc_za_z; l_fx; l_err_rnn; l_dec_x; l_enc_e_z]


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


RN2 = Chain(
    Dense(args[:π], 64,),
    LayerNorm(64, elu),
    GRU(64, 64,),
    Dense(64, args[:π], elu),
) |> gpu

ps = Flux.params(Hx, Ha, RN2)

## ======

function get_loop(z, x; args=args)
    outputs = patches, recs, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_full, err_emb_full = full_sequence(models, z0, a0, x)

    out_1, err_emb_1 = full_sequence(z1, patch_t)
    out = sample_patch(out_full .+ 0.1f0, a1, sampling_grid)

    push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]
        z = RN2(err_emb_1)
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_1, err_emb_1 = full_sequence(z1, patch_t)
        out = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid)
        push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)
    end
    return outputs
end

function plot_rec(x, out::Vector, xs::Vector, ind)
    out_ = [reshape(cpu(k), 28, 28, 1, size(k)[end]) for k in out]
    x_ = reshape(cpu(x), 28, 28, size(x)[end])

    p1 = plot([
        begin
            plot_digit(x_[:, :, ind], boundc=false, alpha=0.5)
            plot_digit!(x[:, :, 1, ind], boundc=false, alpha=0.7)

        end
        for x in out_]...)

    p2 = plot_digit(x_[:, :, ind])
    p3 = plot([plot_digit(x[:, :, 1, ind], boundc=false) for x in xs]...)
    return plot(p1, p2, p3, layout=(1, 3))
end

function plot_recs(x, inds; args=args)
    z = rand(args[:D], args[:π], args[:bsz]) |> gpu
    full_recs, patches, errs, xys, zs = get_loop(z, x)
    full_recs = map(x -> reshape(x, 28, 28, 1, size(x)[end]), full_recs)

    p = [plot_rec(x, patches, full_recs, ind) for ind in inds]
    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

## =====
inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(translate_batch(sample_loader(test_loader), args), inds)

## =====

save_folder = "enc_rnn_2lvl"
alias = "double_H_GRU_translated__uniform_z_init_elu"
save_dir = get_save_dir(save_folder, alias)

## =====

args[:seqlen] = 4
args[:glimpse_len] = 5
args[:scale_offset] = 3.2f0
args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0
args[:D] = Uniform(-1.0f0, 1.0f0)
opt = ADAM(1e-4)
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:20

        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(translate_batch(sample_loader(test_loader), args), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
    end
end

## ====
L = vcat(Ls...)
plot(L)

