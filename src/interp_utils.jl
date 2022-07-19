using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Flux: unsqueeze
using Images

dev = has_cuda() ? gpu : cpu

# constant matrices for "nice" affine transformation
const ones_vec = ones(1, 1, args[:bsz]) |> dev
const zeros_vec = zeros(1, 1, args[:bsz]) |> dev
const diag_vec = [[1 0; 0 1] for _ in 1:args[:bsz]] |> dev


function filter_batch(x, f, p::Number=1.0f0)
    sz = size(x)
    x = length(sz) > 3 ? dropdims(x, dims=3) : x
    x = collect(eachslice(cpu(x), dims=3))
    out = cat([imfilter(k[:, :, 1], f(p)) for k in x]..., dims=3)
    out = length(sz) > 3 ? unsqueeze(out, 3) : out
end

"generate sampling grid 3 x (width x height) x (batch size)"
function get_sampling_grid(width, height; args=args)
    x = LinRange(-1, 1, width)
    y = LinRange(-1, 1, height)
    x_t_flat = reshape(repeat(x, height), 1, height * width)
    y_t_flat = reshape(repeat(transpose(y), width), 1, height * width)
    all_ones = ones(eltype(x_t_flat), 1, size(x_t_flat)[2])
    sampling_grid = vcat(x_t_flat, y_t_flat, all_ones)
    sampling_grid = reshape(
        transpose(repeat(transpose(sampling_grid), args[:bsz])),
        3,
        size(x_t_flat, 2),
        args[:bsz],
    )
    return Float32.(sampling_grid)
end

function get_affine_mats(thetas; scale_offset=0.0f0)
    b = thetas[1:2, :]
    sc = thetas[3:4, :]
    theta_rot = thetas[5, :]
    theta_sh = reshape(thetas[6, :], 1, 1, :)

    cos_rot = reshape(cos.(theta_rot), 1, 1, :)
    sin_rot = reshape(sin.(theta_rot), 1, 1, :)

    A_rot = hcat(vcat(cos_rot, -sin_rot), vcat(sin_rot, cos_rot))
    A_s = cat(map((x, y) -> (x .+ 1.0f0 .+ scale_offset) .* y, eachcol(sc), diag_vec)..., dims=3)
    A_shear = hcat(vcat(ones_vec, theta_sh), vcat(zeros_vec, ones_vec))

    return A_rot, A_s, A_shear, b
end

function grid_generator_3d(sampling_grid_2d, thetas)
    A_rot, A_s, A_shear, b = get_affine_mats(thetas)
    A = batched_mul(batched_mul(A_rot, A_shear), A_s)
    return batched_mul(A, sampling_grid_2d) .+ unsqueeze(b, 2)
end

function get_inv_grid(sampling_grid_2d, thetas)
    A_rot, A_s, A_shear, b = get_affine_mats(thetas)
    sh_inv = cat(map(inv, eachslice(cpu(A_shear), dims=3))..., dims=3)
    sc_inv = cat(map(inv, eachslice(cpu(A_s), dims=3))..., dims=3)
    rot_inv = cat(map(inv, eachslice(cpu(A_rot), dims=3))..., dims=3)
    Ainv = batched_mul(batched_mul(sc_inv, sh_inv), rot_inv) |> gpu
    return batched_mul(Ainv, (sampling_grid_2d .- unsqueeze(b, 2)))
end

function sample_patch(grid, x; sz=args[:img_size])
    tg = reshape(grid, 2, sz..., size(grid)[end])
    x = reshape(x, sz..., 1, size(x)[end])
    grid_sample(x, tg; padding_mode=:zeros)
end
