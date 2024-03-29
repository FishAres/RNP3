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
    :bsz => 64, :img_size => (50, 50), :π => 32, :img_channels => 3,
    :esz => 32, :add_offset => true, :fa_out => identity, :f_z => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :λpatch => Float32(1 / 4),
    :scale_offset => 2.8f0, :scale_offset_sense => 3.2f0,
    :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
)
args[:imszprod] = prod(args[:img_size])

## =====

device!(1)

dev = gpu

##=====

data = load(datadir("exp_pro", "eth80_segmented_train_test.jld2"))

train_data = data["train_data"]
test_data = data["test_data"]

train_loader = DataLoader(train_data |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
test_loader = DataLoader(test_data |> dev, batchsize=args[:bsz], shuffle=true, partial=false)
x = first(train_loader)


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

function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    z0 = fz.(Θ[3])

    return (Enc_za_z, f_state), z0
end

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)

    a0 = sin.(Θ[3])

    return (Enc_za_a, f_policy), a0
end


function get_models(θsz, θsa; args=args, Hx_bounds=Hx_bounds, Ha_bounds=Ha_bounds)
    (Enc_za_z, f_state), z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    (Enc_za_a, f_policy), a0 = get_fpolicy_models(θsa, Ha_bounds; args=args)
    models = f_state, f_policy, Enc_za_z, Enc_za_a, Dec_z_x̂, Dec_z_a
    return models, z0, a0
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


## ====


args[:π] = 128

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_dec_x = args[:imszprod] * args[:π] # decoder z -> x̂, no bias

Hx_bounds = [l_enc_za_z; l_fx]


l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa]

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
        x -> reshape(x, args[:img_size]..., args[:img_channels], :),
        Conv((5, 5), args[:img_channels] => 32, pad=(1, 1)),
        BatchNorm(32, relu),
        Conv((5, 5), 32 => 32, pad=(1, 1)),
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

Dec_z_a = Dense(args[:π], args[:asz], sin, bias=false) |> gpu

Dec_z_x̂ = Chain(
    Dense(args[:π], 256),
    LayerNorm(256, elu),
    x -> reshape(x, 8, 8, 4, :),
    ConvTranspose((5, 5), 4 => 32),
    BatchNorm(32, relu),
    ConvTranspose((4, 4), 32 => 32, stride=(2, 2), pad=(2, 2)),
    BatchNorm(32, relu),
    ConvTranspose((4, 4), 32 => 32, stride=(2, 2), pad=(2, 2)),
    BatchNorm(32, relu),
    ConvTranspose((5, 5), 32 => 32,),
    BatchNorm(32, relu),
    ConvTranspose((5, 5), 32 => 3, relu)
) |> gpu

ps = Flux.params(Hx, Ha, Encoder, Dec_z_x̂, Dec_z_a)

## =====
save_folder = "gen_2lvl_thru"
alias = "double_H_eth80_50x50_generative_thru_v01"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:seqlen] = 4
args[:scale_offset] = 1.5f0

args[:λpatch] = Float32(1 / 8)
args[:λpatch] = 0.0f0
args[:λ] = 0.001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.5f0


args[:η] = 4e-5
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)

## ====
let
    inds = sample(1:args[:bsz], 6, replace=false)
    p = plot_recs(sample_loader(test_loader), inds)
end
## =====

begin
    Ls = []
    for epoch in 1:2000
        if epoch % 200 == 0
            opt.eta = max(0.67 * opt.eta, 1e-7)
            log_value(lg, "learning_rate", opt.eta)
        end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        if epoch % 10 == 0
            inds = sample(1:args[:bsz], 6, replace=false)
            p = plot_recs(sample_loader(test_loader), inds)
            display(p)
            log_image(lg, "recs_$(epoch)", p)
        end
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 500 == 0
            save_model((Hx, Ha, Encoder, Dec_z_x̂, Dec_z_a), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end

