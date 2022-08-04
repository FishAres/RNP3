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

include(srcdir("double_H_utils.jl"))

CUDA.allowscalar(false)
## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :π => 32,
    :esz => 32, :add_offset => true, :fa_out => identity,
    :asz => 6, :glimpse_len => 4, :seqlen => 5, :λ => 1.0f-3, :δL => Float32(1 / 4),
    :scale_offset => 2.8f0, :λf => 0.167f0
)

## =====

device!(1)

dev = gpu

##=====

all_chars = load("../Recur_generative/data/exp_pro/omniglot.jld2")
xs = shuffle(vcat((all_chars[key] for key in keys(all_chars))...))
x_batches = batch.(partition(xs, args[:bsz]))

ntrain, ntest = 286, 15
xs_train = flatten.(x_batches[1:ntrain] |> gpu)
xs_test = flatten.(x_batches[ntrain+1:ntrain+ntest] |> gpu)
x = xs_test[1]

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

args[:π] = 128

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
    Split(
        [Dense(64, θ, bias=false) for θ in Hx_bounds]...,
        Dense(64, args[:π], elu),
    ),
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Split(
        [Dense(64, θ, bias=false) for θ in Ha_bounds]...,
        Dense(64, args[:asz], sin),
    ),
) |> gpu


RN2 = Chain(
    Dense(args[:π], 64,),
    LayerNorm(64, elu),
    GRU(64, 64,),
    Dense(64, args[:π], elu),
) |> gpu

ps = Flux.params(Hx, Ha, RN2)

## ======

inds = sample(1:args[:bsz], 6, replace=false)
p = plot_recs(rand(xs_test), inds)

## =====

save_folder = "enc_rnn_2lvl"
alias = "double_H_v01"
save_dir = get_save_dir(save_folder, alias)

## =====

args[:seqlen] = 5
args[:glimpse_len] = 4
args[:scale_offset] = 2.8f0
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
        ls = train_model(opt, ps, xs_train; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(rand(xs_test), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(xs_test)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        # if epoch % 25 == 0
        # save_model((H, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        # end
    end
end