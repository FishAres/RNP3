using DrWatson
@quickactivate "RNP3"
ENV["GKSwstype"] = "nul"

using MLDatasets
using IterTools: partition, iterated
using Flux: batch, unsqueeze, flatten
using Flux.Data: DataLoader
using Images: imresize
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


# train_chars, train_labels = Omniglot(split=:train)[:]
# test_chars, test_labels = Omniglot(split=:test)[:]
# train_chars = 1.0f0 .- imresize(train_chars, args[:img_size])
# test_chars = 1.0f0 .- imresize(test_chars, args[:img_size])

# train_loader = DataLoader((train_chars |> dev), batchsize=args[:bsz], shuffle=true, partial=false)
# test_loader = DataLoader((test_chars |> dev), batchsize=args[:bsz], shuffle=true, partial=false)

# x = first(test_loader)

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

args[:π] = 64

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
    Dense(64, sum(Hx_bounds) + args[:π], bias=false)
) |> gpu

Ha = Chain(
    LayerNorm(args[:π],),
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(Ha_bounds) + args[:asz], bias=false)
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

alias = "double_H_v01_no_w_heads_omni_2σ"
save_dir = get_save_dir(save_folder, alias)

## =====
args[:D] = Normal(0.0f0, 1.0f0)
args[:seqlen] = 5
args[:glimpse_len] = 4
args[:scale_offset] = 2.8f0
args[:δL] = round(Float32(1 / args[:seqlen]), digits=3)
# args[:δL] = 0.0f0
args[:λf] = 1.0f0
args[:λ] = 0.0002f0
opt = ADAM(1e-4)
lg = new_logger(joinpath(save_folder, alias), args)
# todo try sinusoidal lr schedule

begin
    Ls = []
    for epoch in 1:200
        if epoch % 20 == 0
            opt.eta = 0.9 * opt.eta
        end
        ls = train_model(opt, ps, xs_train; epoch=epoch, logger=lg)
        inds = sample(1:args[:bsz], 6, replace=false)
        p = plot_recs(rand(xs_test), inds)
        display(p)
        log_image(lg, "recs_$(epoch)", p)
        L = test_model(xs_test)
        log_value(lg, "test_loss", L)
        @info "Test loss: $L"
        push!(Ls, ls)
        if epoch % 25 == 0
            save_model((Hx, Ha, RN2), joinpath(save_folder, alias, savename(args) * "_$(epoch)eps"))
        end
    end
end




## ====

L = vcat(Ls...)
plot(L)
plot(log.(1:length(L)), log.(L))