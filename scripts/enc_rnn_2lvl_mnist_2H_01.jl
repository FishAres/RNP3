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
using ParameterSchedulers
include(srcdir("double_H_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => elu,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
)

## =====

device!(0)

dev = gpu

##=====

train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

x = first(test_loader)
## ====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev
## =====

## ======
# todo: better variable names (patches, out, etc.)
function get_loop(z, x; args=args)
    outputs = patches, recs, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x)
    out_1, err_emb_1 = full_sequence(z1, patch_t)
    out = sample_patch(out_full .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])

    push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]
        z = RN2(err_emb_1)
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t)
        out = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])
        push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)
    end
    z = RN2(err_emb_1)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)
    return outputs
end


function plot_rec(x, out::Vector, xs::Vector, ind)
    out_ = [reshape(cpu(k), 28, 28, 1, size(k)[end]) for k in out]
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot([
        begin
            # plot_digit(x_[:, :, ind], boundc=false, alpha=0.5)
            # plot_digit!(x[:, :, 1, ind], boundc=false, alpha=0.6)
            xnew = 0.2f0 .* x_[:, :, ind] + x[:, :, 1, ind]
            plot_digit(xnew, boundc=false)
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

# todo don't bind RNN size to args[:π]

args[:π] = 90

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
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
) |> gpu


RN2 = Chain(
    Dense(args[:π], 64,),
    LayerNorm(64, elu),
    LSTM(64, 64,),
    LayerNorm(64, elu),
    Dense(64, args[:π], elu),
) |> gpu

ps = Flux.params(Hx, Ha, RN2)

## ======

# modelpath = "saved_models/enc_rnn_2lvl/double_H_lstm_mnist_var_offset_4recs/add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=3_scale_offset=3.2_scale_offset_sense=4.2_seqlen=4_δL=0.0_η=1e-5_λ=0.001_λf=1.0_π=64_75eps.bson"

# Hx, Ha, RN2 = load(modelpath)[:model] |> gpu

## ====

# inds = sample(1:args[:bsz], 6, replace=false)
# p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "enc_rnn_2lvl"
alias = "double_H_lstm_mnist_var_offset_4recs_v2"
save_dir = get_save_dir(save_folder, alias)

## =====

args[:seqlen] = 4
args[:glimpse_len] = 3
args[:scale_offset] = 3.2f0
args[:scale_offset_sense] = 4.6f0

# args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:η] = 1e-4
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

using ParameterSchedulers
using ParameterSchedulers: Stateful

s = Stateful(SinExp(args[:η], 4e-7, 20, 0.99))

## ====
begin
    Ls = []
    for epoch in 1:200
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        log_value(lg, "learning_rate", opt.eta)
        if epoch > 1
            opt.eta = ParameterSchedulers.next!(s)
        end
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 25 == 0
            save_model((Hx, Ha, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

# ## ====
# L = vcat(Ls...)
# plot(L)
# plot(log.(1:length(L)), log.(L))
# ## ==== Analysis

# z = rand(args[:D], args[:π], args[:bsz]) |> gpu
# full_recs, glimpses, errs, zs, as, patches_t = get_loop(z, x)

# ind = 1
# glimpses[1]
# plot([plot_digit(x[:, :, 1, ind], boundc=false) for x in glimpses]...)


# aa = [(a[[1, 4], :] .+ args[:scale_offset]) .* a[5:6, :] for a in as]
# theme(:ggplot2)
# p = let
#     p1 = plot()
#     for a in aa
#         # scatter!(a[1, :], a[2, :])
#         histogram!(a[1, :], a[2, :], alpha=0.33, bins=-1.2:0.1:1.2)
#     end
#     p1
# end

# plot([histogram(a[1, :], a[2, :], bins=-1.2:0.1:1.2, title="$i") for (i, a) in enumerate(aa)]...,
#     layout=(4, 1), size=(400, 600))



# ## ===== Tune hypernet weight range?

# Hx = Chain(
#     LayerNorm(args[:π],),
#     Dense(args[:π], 64),
#     LayerNorm(64, elu),
#     Dense(64, 64),
#     LayerNorm(64, elu),
#     Dense(64, 64),
#     LayerNorm(64, elu),
#     Dense(64, sum(Hx_bounds) + args[:π], bias=false),
# ) |> gpu

# Ha = Chain(
#     LayerNorm(args[:π],),
#     Dense(args[:π], 64),
#     LayerNorm(64, elu),
#     Dense(64, 64),
#     LayerNorm(64, elu),
#     Dense(64, 64),
#     LayerNorm(64, elu),
#     Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
# ) |> gpu


# RN2 = Chain(
#     Dense(args[:π], 64,),
#     LayerNorm(64, elu),
#     LSTM(64, 64,),
#     Dense(64, args[:π], elu),
# ) |> gpu

# ps = Flux.params(Hx, Ha, RN2)

# θx, re_Hx = Flux.destructure(Hx)
# θa, re_Ha = Flux.destructure(Ha)
# θrn, re_RN2 = Flux.destructure(RN2)

# ## ====

# # function 
# z = rand(D2, args[:π], args[:bsz]) |> gpu
# sigma = 0.1f0
# Ha = re_Ha(rand(Normal(0.0f0, sigma), length(θa)) |> gpu)

# full_recs, glimpses, errs, zs, as, patches_t = get_loop(z, x)
# aa = [(a[[1, 4], :] .+ args[:scale_offset]) .* a[5:6, :] for a in as]
# plot([histogram(a[1, :], a[2, :], bins=-1.2:0.1:1.2, title="$i") for (i, a) in enumerate(aa)]...,
#     layout=(4, 1), size=(400, 600))

# ind = 1
# plot([plot_digit(x[:, :, 1, ind], boundc=false) for x in glimpses]...)


# theme(:ggplot2)
# p = let
#     p1 = plot()
#     for a in aa
#         # scatter!(a[1, :], a[2, :])
#         histogram!(a[1, :], a[2, :], alpha=0.33, bins=-1.2:0.1:1.2)
#     end
#     p1
# end







# ## ==== where to initialize
# ps = Flux.params(Hx, Ha, RN2)

# z = rand(args[:D], args[:π], args[:bsz]) |> gpu
# full_recs, glimpses, errs, zs, as, patches_t = get_loop(z, x)
# histogram(cpu(z)[:])

# histogram(zs[1][:])
# histogram(map(x -> x[:], zs), alpha=0.5, bins=-1:0.1:1,)

# D2 = Normal(0.0f0, 0.002f0)

# z2 = rand(D2, args[:π], args[:bsz]) |> gpu

# g = gradient(() -> model_loss(z2, x), ps)
# g.grads