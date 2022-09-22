using DrWatson
using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Flux: batch, unsqueeze, flatten
using Distributions
using Plots

include(srcdir("gen_vae_utils.jl"))

CUDA.allowscalar(false)

## ====
args = Dict(
    :bsz => 64, :img_size => (28, 28), :img_channels => 1, :scale_offset => 2.0f0,
    :π => 32, :asz => 6,
)
## =====
device!(2)

## =====
dev = has_cuda() ? gpu : cpu

const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev

## ======

canvas = let
    canvas = zeros(Float32, 28, 28, args[:bsz])
    canvas[8:21, 8:21, :] .= 1.0f0
    canvas
end |> dev

thetas_targ = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> dev
targ = sample_patch(canvas, thetas_targ, sampling_grid)

thetas_init = rand(Uniform(-1.0f0, 1.0f0), 6, args[:bsz]) |> dev
init = sample_patch(canvas, thetas_init, sampling_grid)

Encoder = Chain(
    Parallel(
        vcat,
        Chain(
            Conv((5, 5), 1 => 32, stride=2),
            Conv((5, 5), 32 => 32, stride=2),
            BatchNorm(32, relu),
            flatten,
            Dense(512, 64),
            BatchNorm(64, relu),
        ),
        Chain(
            Conv((5, 5), 1 => 32, stride=2),
            Conv((5, 5), 32 => 32, stride=2),
            BatchNorm(32, relu),
            flatten,
            Dense(512, 64),
            BatchNorm(64, relu),
        )),
    Dense(128, 64),
    BatchNorm(64, relu),
    Dense(64, 64),
    BatchNorm(64, relu),
    Dense(64, args[:π])
) |> gpu

z = Encoder((init, targ))

l_fa = get_rnn_θ_sizes(args[:asz], args[:π])
l_dec_a = args[:asz] * args[:π] + args[:asz]
H_bounds = [l_fa; l_dec_a]

H = Chain(
    Dense(args[:π], 64),
    LayerNorm(64, elu),
    Dense(64, 64),
    LayerNorm(64, elu),
    Dense(64, sum(H_bounds) + args[:asz], bias=false),
) |> gpu

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    f_policy = ps_to_RN(get_rn_θs(Θ[1], args[:asz], args[:π]); f_out=sin)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[2], sin), flatten)

    a0 = sin.(Θ[3])

    return (f_policy, Dec_z_a), a0
end

θs = H(z)
(fx, Dec_z_a), a0 = get_fpolicy_models(θs, H_bounds)

a1 = Dec_z_a(fx(a0))
x̂ = sample_patch(canvas, a, sampling_grid)







