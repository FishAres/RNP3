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

device!(1)

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

dudt = Chain(
    GroupNorm(32, args[:norm_groups]),
    Conv((3, 3), 32 => 32, relu, pad=(1, 1)),
    GroupNorm(32, args[:norm_groups]),
) |> gpu

nn_ode = NeuralODE(dudt, (0.0f0, 2.0f0), Tsit5(), save_everystep=false,
    reltol=1e-3, abstol=1e-3, save_start=false) |> gpu

function DiffEqArray_to_Array(x)
    xarr = gpu(x)
    return xarr[:, :, :, :, 1]
end

modelpath = "saved_models/gen_2lvl/omni_2lvl_conv_node_H_v0/add_offset=true_asz=6_bsz=64_esz=32_glimpse_len=4_img_channels=1_imzprod=784_norm_groups=8_scale_offset=2.0_scale_offset_sense=3.2_seqlen=4_α=1.0_β=0.05_η=0.0001_λ=1e-6_λf=0.167_λpatch=0.0_π=64_100eps.bson"

Hx, Ha, Encoder = load(modelpath)[:model] |> gpu

Encoder_old = Encoder
Encoder_new = Encoder = let
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

# Ha = Chain(
#     LayerNorm(args[:π],),
#     Dense(args[:π], 64),
#     LayerNorm(64, elu),
#     Dense(64, 64),
#     LayerNorm(64, elu),
#     Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
# ) |> gpu

ps = Flux.params(Encoder)
## ======



## ======
# z = randn(Float32, args[:π], args[:bsz]) |> gpu
# a = Hx(z)
# b = sum(Hx_bounds) + sum(Ha_bounds)
# size(a, 1) / b

let
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
    display(p)
end
## =====

save_folder = "gen_2lvl_analysis"
alias = "omni_2lvl_conv_node_H_v0_transfer_enc"
save_dir = get_save_dir(save_folder, alias)

## =====

args[:η] = 1e-4
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
        ls = train_model(opt, ps, test_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        p2 = plot_recs(sample_loader(val_loader), inds)
        p3 = plot(p, p2, layout=(1, 2))
        display(p3)
        log_image(lg, "recs_$(epoch)", p3)

        @time Lval = test_model(test_loader)
        log_value(lg, "val_loss", Lval)
        @info "Val loss: $Lval"
        push!(Ls, ls)
        if epoch % 50 == 0
            save_model((Hx, Ha, Encoder), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end
## =====
include(srcdir("fancy_plotting_utils.jl"))
ind = 0

x = sample_loader(test_loader)
img_ind = 6
begin
    # ind = mod(ind + 1, args[:bsz]) + 1
    # ind = 37
    digits, digit_patches, digit_im, patch_ims, xys, xys1 = get_digit_parts(x, ind)
    p1 = plot_digit(cpu(x)[:, :, ind])
    p2 = plot(digit_im, axis=nothing)
    savefig(p1, "plots/transfer/omniglot_eg$(img_ind)_orig.png")
    savefig(p2, "plots/transfer/omniglot_eg$(img_ind)_rec.png")
    plot(p1, p2)
end
x

ind

# ## +==== sampling

# function get_outputs(z)
#     θsz = Hx(z)
#     θsa = Ha(z)
#     models, z0, a0 = get_models(θsz, θsa; args=args)
#     z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
#     out_small = full_sequence(z1, patch_t)
#     out = sample_patch(out_small, a1, sampling_grid)
#     for t = 2:args[:seqlen]
#         z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
#         out_small = full_sequence(z1, patch_t)
#         out += sample_patch(out_small, a1, sampling_grid)
#         push!(z1s, cpu(z1))
#     end
#     return out
# end


# function plot_sample(z; n=nothing)
#     bsz = size(z)[end]
#     n = n === nothing ? trunc(Int, sqrt(bsz)) : n
#     n_samples = n^2
#     out = dropdims(get_outputs(z) |> cpu, dims=3)[:, :, 1:n_samples]
#     a = partition(collect(eachslice(out, dims=3)), n)
#     b = map(x -> vcat(x...), a)
#     c = hcat(b...)
#     plot_digit(c)
# end

# begin
#     z = rand(args[:D], args[:π], args[:bsz]) |> gpu
#     plot_sample(z; n=5)
# end

# ## === clustering
# function all_zs(x)
#     z1s, z2s = [], []
#     μ, logvar = Encoder(x)
#     θsz = Hx(μ)
#     θsa = Ha(μ)
#     models, z0, a0 = get_models(θsz, θsa; args=args)
#     z1, a1, x̂, patch_t = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset])
#     out_small = full_sequence(z1, patch_t)
#     out = sample_patch(out_small, a1, sampling_grid)
#     push!(z1s, cpu(z1))
#     for t = 2:args[:seqlen]
#         z1, a1, x̂, patch_t = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
#         out_small = full_sequence(z1, patch_t)
#         out += sample_patch(out_small, a1, sampling_grid)
#         push!(z1s, cpu(z1))
#     end
#     push!(z2s, cpu(μ))
#     return z2s, z1s
# end

# x = first(test_loader)
# z2s, z1s = all_zs(x)

# hcat(z2s...)

# begin
#     z2, z1 = [], []
#     for x in test_loader
#         z2s, z1s = all_zs(x)
#         push!(z2, hcat(z2s...))
#         push!(z1, hcat(z1s...))
#     end
# end

# n_samples = 3000
# z2ss = hcat(z2...)
# z2ss = z2ss[:, 1:n_samples]

# z1ss = hcat(z1...)
# z1ss = z1ss[:, 1:n_samples]

# zz = hcat(z2ss, z1ss)

# using TSne
# Y = tsne(z2ss')

# scatter(Y[:, 1], Y[:, 2])