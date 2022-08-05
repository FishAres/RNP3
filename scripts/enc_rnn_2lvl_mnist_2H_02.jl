using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Distributions
using StatsBase: sample
using Random: shuffle

include(srcdir("triple_H_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :λf => 0.167f0
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
args[:f_z] = elu # activation function used for z vectors


## =====

# todo don't bind RNN size to args[:π]

args[:π] = 32

l_enc_za_z = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> z_t+1
l_fx = get_rnn_θ_sizes(args[:π], args[:π])
l_dec_x = 784 * args[:π] # decoder z -> x̂, no bias

Hx_bounds = [l_enc_za_z; l_fx; l_dec_x]

l_enc_za_a = (args[:π] + args[:asz]) * args[:π] # encoder (z_t, a_t) -> a_t+1
l_fa = get_rnn_θ_sizes(args[:π], args[:π]) # same size for now
l_dec_a = args[:asz] * args[:π] + args[:asz] # decoder z -> a, with bias

Ha_bounds = [l_enc_za_a; l_fa; l_dec_a]

l_enc_e_z = (784 + 1) * args[:π]
l_err_rnn = get_rnn_θ_sizes(args[:π], args[:π]) # also same size lol
He_bounds = [l_enc_e_z; l_err_rnn]

Hx = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Split(
        [Dense(64, θ, bias=false) for θ in Hx_bounds]...,
        Dense(64, args[:π], elu),
        Dense(64, args[:π], elu), # for He
    ),
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Split(
        [Dense(64, θ, bias=false) for θ in Ha_bounds]...,
        Dense(64, args[:asz], sin),
        Dense(64, args[:π], elu), # for He
    ),
) |> gpu

He = Chain(
    Parallel(
        vcat,
        Chain(Dense(args[:π], 64), LayerNorm(64, elu)),
        Chain(Dense(args[:π], 64), LayerNorm(64, elu)),
    ),
    Dense(128, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Split(
        [Dense(64, θ, elu) for θ in He_bounds]...,
    ),
) |> gpu

RN2 = Chain(
    Dense(args[:π], 64,),
    LayerNorm(64, elu),
    LSTM(64, 64,),
    Dense(64, args[:π], elu),
) |> gpu

ps = Flux.params(Hx, Ha, He, RN2)

## ======
z = randn(args[:π], args[:bsz]) |> gpu
inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(sample_loader(test_loader), inds)

## =====

save_folder = "enc_rnn_2lvl"
alias = "double_H_v01"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:D] = Normal(0.0f0, 1.0f0)
args[:seqlen] = 4
args[:glimpse_len] = 4
args[:scale_offset] = 2.2f0
args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
# args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.001f0
opt = ADAM(1e-4)
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:20
        # if epoch % 20 == 0
        # opt.eta = 0.8 * opt.eta
        # end
        ls = train_model(opt, ps, train_loader; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(sample_loader(test_loader), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(test_loader)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        # if epoch % 25 == 0
        # save_model((H, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        # end
    end
end

## ====

L = vcat(Ls...)
plot(L)
plot(log.(1:length(L)), log.(L))