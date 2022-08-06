using DrWatson
@quickactivate "RNP3"

using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Flux: flatten, unsqueeze, batch
using Plots

include(srcdir("interp_utils.jl"))

## ====
args = Dict(
    :img_size => (28, 28),
    :bsz => 64,
)


device!(0)
## ======
dev = gpu
const sampling_grid = (get_sampling_grid(args[:img_size]...)|>dev)[1:2, :, :]
# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
diag_vec = [[1.0f0 0.0f0; 0.0f0 1.0f0] for _ in 1:args[:bsz]]
const diag_mat = cat(diag_vec..., dims=3) |> dev
const diag_off = cat(1.0f-6 .* diag_vec..., dims=3) |> dev

## +====

patch_vert, patch_horz = let
    a, b = zeros(28, 28, args[:bsz]), zeros(28, 28, args[:bsz])
    a[:, 13:15, :] .= 1
    b[13:15, :, :] .= 1
    a, b
end |> dev


sc = unsqueeze(2 * rand(2, args[:bsz]), 2) |> dev
b = randn(2, args[:bsz]) |> dev

a = patch_vert + patch_horz
c = sample_patch(a, sc .* sampling_grid .+ unsqueeze(b, 2))
plot([heatmap(cpu(c)[:, :, 1, ind]) for ind in rand(1:64, 6)]...)

d = unsqueeze(a, 3) + c
plot([heatmap(cpu(d)[:, :, 1, ind]) for ind in rand(1:64, 6)]...)

