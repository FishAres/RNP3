using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using Flux, Zygote, CUDA
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle
using ParameterSchedulers

# todo - refactor so scripts share same code
include(srcdir("double_H_vae_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32, :img_channels => 1,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => elu,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0), :z_rnn => "LSTM"
)
args[:imszprod] = prod(args[:img_size])
## =====

device!(0)

dev = gpu

##=====

datapath = "/gscratch/rao/aresf/Code/datasets/celeba/img_align_celeba/"

using Images

imgs = [Float32.(channelview(imresize(load(img), (50, 50)))) for img in readdir(datapath, join=true)]

allims = cat(imgs..., dims=4)


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

"one iteration"
function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1) # todo parallel layer?
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset) |> flatten

    ϵ = patch_t .- x̂

    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
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

function model_loss(z, x, rs; args=args)
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])


    Lfull = Flux.mse(flatten(out_full), flatten(x); agg=sum)
    for t = 2:args[:glimpse_len]-1
        μ, logvar = RN2(err_emb_1)
        z = sample_z(μ, logvar, rs[t-1])
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])

        # Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end
    μ, logvar = RN2(err_emb_1)
    z = sample_z(μ, logvar, rs[end])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
    Lfull += Flux.mse(flatten(out_full), flatten(x); agg=sum)
    klqp = kl_loss(μ, logvar)
    return Lfull, klqp

end

function get_loop(z, x, rs; args=args)
    outputs = full_recs, patches, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x)
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
    sense_patch = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])

    push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]-1
        μ, logvar = RN2(err_emb_1)
        z = sample_z(μ, logvar, rs[t-1])
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
        sense_patch = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])
        push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)


    end
    μ, logvar = RN2(err_emb_1)
    z = sample_z(μ, logvar, rs[end])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)
    return outputs
end


function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = [[rand(D, args[:π], args[:bsz]) for _ in 1:args[:glimpse_len]]
          for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(z0, x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                log_value(lg, "KL loss", klqp)
            end

            args[:α] * rec_loss + args[:β] * klqp + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end

        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    rs = [[rand(D, args[:π], args[:bsz]) for _ in 1:args[:glimpse_len]]
          for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(z0, x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp

    end
    return L / length(test_data)
end

function plot_recs(x, inds; args=args)
    rs = [rand(args[:D], args[:π], args[:bsz]) for _ in 1:args[:seqlen]] |> gpu
    full_recs, patches, errs, xys, zs = get_loop(z0, x, rs)
    full_recs = map(x -> reshape(x, 28, 28, 1, size(x)[end]), full_recs)

    p = [plot_rec(x, patches, full_recs, ind) for ind in inds]
    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

## ====

# todo don't bind RNN size to args[:π]

args[:π] = 96
args[:D] = Normal(0.0f0, 1.0f0)

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

zrnn = args[:z_rnn] == "LSTM" ? LSTM : GRU

RN2 = Chain(
    Dense(args[:π], 64,),
    LayerNorm(64, elu),
    zrnn(64, 64,),
    Split(
        Dense(64, args[:π],),
        Dense(64, args[:π],),
    )
) |> gpu

z0 = rand(args[:D], args[:π], args[:bsz]) |> gpu
ps = Flux.params(Hx, Ha, RN2, z0)
## ====
# modelpath = "saved_models/enc_rnn_2lvl/2lvl_double_H_omni_vae_z0emb/add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=5_img_channels=1_imszprod=784_scale_offset=2.0_scale_offset_sense=2.2_seqlen=4_z_rnn=LSTM_α=1.0_β=0.1_δL=0.0_η=0.0001_λ=0.001_λf=1.0_π=96_123eps.bson"

# Hx, Ha, RN2, z0 = load(modelpath)[:model] |> gpu

## ======

# inds = sample(1:args[:bsz], 6, replace=false)
# p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "enc_rnn_2lvl"
alias = "2lvl_double_H_omni_vae_z0emb_3"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 4
args[:glimpse_len] = 5
args[:scale_offset] = 2.0f0
args[:scale_offset_sense] = 2.6f0

# args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0

args[:α] = 1.0f0
args[:β] = 0.1f0


args[:η] = 1e-4
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

# using ParameterSchedulers
# using ParameterSchedulers: Stateful
# s = Stateful(SinExp(args[:η], 4e-7, 20, 0.99))

## ====

begin
    Ls = []
    for epoch in 1:1000
        if epoch > 80 && epoch % 80 == 0
            opt.eta = 0.33 * opt.eta
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
            save_model((Hx, Ha, RN2, z0), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

## ====

# save_model((Hx, Ha, RN2, z0), joinpath(save_folder, alias, savename(args) * "_123eps"))