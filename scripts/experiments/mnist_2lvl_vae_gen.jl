using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle
using ParameterSchedulers

# todo - refactor so scripts share same code
include(srcdir("gen_2lvl_utils.jl"))

CUDA.allowscalar(false)
## ====
begin
    args = Dict(
        :bsz => 64, :img_size => (28, 28), :π => 32, :img_channels => 1,
        :esz => 32, :add_offset => true, :fa_out => identity, :f_z => identity,
        :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :λpatch => Float32(1 / 4),
        :scale_offset => 2.8f0,
        :α => 1.0f0, :β => 0.5f0,
        :λf => 0.167f0, :D => Normal(0.0f0, 1.0f0),
    )
    args[:imszprod] = prod(args[:img_size])
end
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

args[:π] = 32

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias

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
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false),
) |> gpu

Encoder = let
    enc1 = Chain(
        x -> reshape(x, args[:img_size]..., args[:img_channels], :),
        Conv((5, 5), args[:img_channels] => 32),
        BatchNorm(32, elu),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        BasicBlock(32 => 32, +),
        Conv((5, 5), 32 => 8),
        BatchNorm(8, elu),
        flatten,
    )
    Chain(
        enc1,
        Dense(3200, 128),
        LayerNorm(128, elu),
        Dense(128, 128),
        LayerNorm(128, elu),
        Dense(128, 64),
        LayerNorm(64, elu),
        Split(
            Dense(64, args[:π]),
            Dense(64, args[:π]),
        )
    )
end |> gpu

ps = Flux.params(Encoder, Hx, Ha)
## =====
inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====
save_folder = "generative_2lvl"
alias = "mnist_vae_01"
save_dir = get_save_dir(save_folder, alias)

## =====
# todo - separate sensing network?
args[:scale_offset] = 0.5f0
args[:seqlen] = 4
args[:λf] = 1.0f0
args[:λpatch] = round(Float32(1 / args[:seqlen]), digits=3)
args[:λ] = 0.001f0
args[:D] = Normal(0.0f0, 1.0f0)

args[:α] = 1.0f0
args[:β] = 0.5f0

args[:η] = 1e-4
opt = ADAM(args[:η])
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

# using ParameterSchedulers
# using ParameterSchedulers: Stateful
# 
# s = Stateful(SinExp(args[:η], 4e-7, 20, 0.99))

## ====
begin
    Ls = []
    for epoch in 1:20
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
            save_model((Hx, Ha, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end
## +====

full_recs, patches, zs, as, patches_t = get_loop(x)

patches_t[1]
plot_digit(full_recs[1][:, :, 1, 1])

plot_digit(reshape(patches_t[3][:, 12], 28, 28))

heatmap(vcat(patches_t...))

a = cat(as..., dims=3)

heatmap(a[:, 1, :])

bs = (a[[1, 4], :, :] .+ 1.0f0 .+ args[:scale_offset]) .* a[5:6, :, :]

begin
    p = plot()
    for i in 1:4
        scatter!(bs[1, :, i], bs[2, :, i])
    end
    p
end

