using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using LinearAlgebra, Statistics
using Flux, Zygote, CUDA

using MLDatasets
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Plots
using StatsBase: sample
using ProgressMeter
using ProgressMeter: Progress

include(srcdir("interp_utils.jl"))
include(srcdir("hypernet_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))
include(srcdir("utils.jl"))

include(srcdir("z_rnn_model_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :seqlen => 4, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 0.0f0, :λf => 0.167f0
)

## =====

device!(1)

dev = gpu

##=====

train_digits, train_labels = MNIST(split=:train)[:]
test_digits, test_labels = MNIST(split=:test)[:]

train_labels = Float32.(Flux.onehotbatch(train_labels, 0:9))
test_labels = Float32.(Flux.onehotbatch(test_labels, 0:9))

train_loader = DataLoader((train_digits |> dev, train_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader((test_digits |> dev, test_labels |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

x, y = first(test_loader)
## ====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]] |> dev
const diag_mat = cat(diag_vec..., dims=3) |> dev
## =====

function get_models(θs, model_bounds; args=args, init_zs=true)
    inds = Zygote.ignore() do
        init_zs ? [0; cumsum([model_bounds...; args[:π]; args[:asz]])] : [0; cumsum(model_bounds)]
    end

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    f_state = ps_to_RN(get_rn_θs(Θ[1], args[:π], args[:π]); f_out=elu)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    err_rnn = ps_to_RN(get_rn_θs(Θ[3], args[:π], args[:π]); f_out=elu)

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[4], elu), flatten)
    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[5], elu), flatten)

    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[6], elu), flatten)

    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[7], relu6), flatten)

    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[8], sin), flatten)

    z0 = Θ[9]
    a0 = Θ[10]

    return (f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a), z0, a0
end

"todo: calculate error outside gradient"
function forward_pass(z1, a1, models, x)
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1)
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid) |> flatten
    ϵ = patch_t .- x̂
    # ϵ = Zygote.ignore() do
    # patch_t .- x̂
    # end
    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
end

function full_sequence(z::AbstractArray, x; args=args, model_bounds=model_bounds)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    err_emb = err_rnn(Δz) # update error embedding
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        err_emb = err_rnn(Δz) # update error embedding
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, err_emb
end

function full_sequence(models::Tuple, z0, a0, x; args=args, model_bounds=model_bounds)
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    err_emb = err_rnn(Δz) # update error embedding
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        err_emb = err_rnn(Δz) # update error embedding
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, err_emb
end

function model_loss(z, x; args=args, model_bounds=model_bounds)
    Flux.reset!(RN2)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_1, err_emb_1 = full_sequence(z1, patch_t)
    out_full, err_emb_full = full_sequence(models, z0, a0, x)

    Lpatch = Flux.mse(flatten(x̂), flatten(patch_t))
    Lfull = Flux.mse(flatten(out_full), flatten(x))
    for t = 2:args[:seqlen]
        z = RN2(err_emb_1)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_1, err_emb_1 = full_sequence(z1, patch_t)

        Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end

    local_loss = args[:δL] * Lpatch
    return local_loss + args[:λf] * Lfull
end

function get_loop(z, x; args=args, model_bounds=model_bounds)
    outputs = patches, recs, errs, zs, patches_t = [], [], [], [], [], []
    Flux.reset!(RN2)
    θs = H(z)
    models, z0, a0 = get_models(θs, model_bounds)
    Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, err_rnn, Dec_z_x̂, Dec_z_a = models
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)
    out_full, err_emb_full = full_sequence(models, z0, a0, x)

    out_1, err_emb_1 = full_sequence(z1, patch_t)
    out = sample_patch(out_full, a1, sampling_grid)

    push_to_arrays!((out_full, out, ϵ, z, patch_t), outputs)

    for t = 2:args[:seqlen]
        z = RN2(err_emb_1)
        θs = H(z)
        models, z0, a0 = get_models(θs, model_bounds)
        Vx, Va, Enc_za_z, Enc_za_a, Enc_ϵ_z, err_rnn, Dec_z_x̂, Dec_z_a = models
        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out_1, err_emb_1 = full_sequence(z1, patch_t)
        out = sample_patch(out_1, a1, sampling_grid)
        push_to_arrays!((out_full, out, ϵ, z, patch_t), outputs)
    end
    return outputs
end


## =====
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_err_rnn = get_rnn_θ_sizes(args[:π], args[:π]) # also same size lol

l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

l_enc_e_z = (784 + 1) * args[:π]

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1

model_bounds = [l_fx; l_fa; l_err_rnn; l_enc_za_z; l_enc_za_a; l_enc_e_z; l_dec_x; l_dec_a]

## =====

lθ = sum(model_bounds)
println(lθ, " parameters for primary")
# ! initializes z_0, a_0 
H = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, lθ + args[:π] + args[:asz], bias=false),
) |> gpu

println("# hypernet params: $(sum(map(prod, size.(Flux.params(H)))))")

RN2 = Chain(
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    GRU(64, 64),
    Dense(64, args[:π], elu),
) |> gpu

ps = Flux.params(H, RN2)

## =====
modelpath = "saved_models/enc_rnn_2l2v2l/fx_fa_fe_rnns_v01/add_offset=true_asz=6_bsz=64_esz=32_scale_offset=2.0_seqlen=4_δL=0.0_λ=0.006_λf=1.0_π=32_50eps.bson"

H, RN2 = load(modelpath)[:model] |> gpu

# model_loss(z, x)
## =====

save_folder = "enc_rnn_2l2v2l"
alias = "fx_fa_fe_rnns_v01"
save_dir = get_save_dir(save_folder, alias)


## =====

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)
## =====


args[:seqlen] = 4
args[:scale_offset] = 2.0f0
# args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.006f0
opt = ADAM(1e-4)
lg = new_logger(save_dir, args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:200
        if epoch % 20 == 0
            opt.eta = 0.8 * opt.eta
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
        if epoch % 10 == 0
            save_model((H, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

## =====

z = randn(Float32, args[:π], args[:bsz]) |> gpu
full_recs, patches, errs, zs, patches_t = get_loop(z, x)

plot_digit(patches[1][:, :, 1, 1])

patches_t
errs
es = reshape(patches_t[4], 28, 28, 64)
heatmap(es[:, :, 4])

zz = cat(zs..., dims=3)

plot(zz[:, 4, :]', legend=false)