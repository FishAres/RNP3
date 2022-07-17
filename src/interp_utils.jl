using LinearAlgebra, Statistics
using Flux, Zygote
using Flux: unsqueeze

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

function grid_generator_2d(sampling_grid_2d, thetas; args=args)
    sc = abs(args[:scale_offset]) > 0.0f0 ? thetas[1:2, :] .+ args[:scale_offset] : thetas[1:2, :]
    bs = sc .* thetas[3:4, :]
    return unsqueeze(sc, 2) .* sampling_grid_2d .+ unsqueeze(bs, 2)
end

function grid_generator_3d(sampling_grid_3d, thetas; args=args)

    sc = thetas[1:2, :] .+ args[:scale_offset]
    # thetas = vcat(sc[1:1, :], thetas[2:3, :], sc[2:2, :], sc .* thetas[5:6, :])
    thetas = vcat(sc, thetas[2:3, :], sc .* thetas[5:6, :])
    thetas = reshape(thetas, 2, 3, size(thetas)[end])
    tr_grid = batched_mul(thetas, sampling_grid_3d)
    return tr_grid
end

function affine_grid_generator(sampling_grid, thetas; args=args, sz=args[:img_size])
    bsz = size(thetas)[end]
    tr_grid = if size(sampling_grid, 1) > 2
        grid_generator_3d(sampling_grid, thetas)
    else
        grid_generator_2d(sampling_grid, thetas)
    end
    return reshape(tr_grid, 2, sz..., bsz)
end


function sample_patch(x, xy, sampling_grid; sz=args[:img_size])
    ximg = reshape(x, sz..., 1, size(x)[end])
    tr_grid = affine_grid_generator(sampling_grid, xy; sz=sz)
    grid_sample(ximg, tr_grid; padding_mode=:zeros)
end

## === for inverting the grid


rotmat(θ) = [cos(θ) sin(θ); -sin(θ) cos(θ)]
shearmat(s) = [1 s; 0 1]

function make_invertible!(A)
    for (i, a_) in enumerate(A)
        a_[diagind(a_)[findall(iszero, diag(a_))]] .= 1.0f-3
    end
    # A
end

"from 6-param xy get A, inv(A) of Ax + b"
function get_transform_matrix(xy; scale_b=true)
    xy = cpu(xy)
    S = xy[[1, 4], :]
    θ = xy[2, :]
    s = xy[3, :]
    b = xy[5:6, :]
    bsz = size(xy)[end]
    Arot = map(rotmat, θ)
    Ashear = map(shearmat, s)

    # add scale_offset
    As = if args[:add_offset]
        offs = args[:scale_offset]
        [[(S[1, i]+1+offs) 0; 0 (S[2, i]+1+offs)] for i in 1:bsz]
    else
        [[(S[1, i]+1) 0; 0 (S[2, i]+1)] for i in 1:bsz]
    end
    make_invertible!(As)
    A = map(*, Arot, As, Ashear)

    # Ainv = map((x, y, z) -> inv(x) * inv(y) * inv(z), Arot, As, Ashear)
    Ainv = map(inv, A)
    b_ = collect(eachcol(b))
    b_ = scale_b ? map(*, As, b_) : b_
    A, Ainv, b_
end

Zygote.@nograd function zoom_in(x, xy, sampling_grid; args=args, scale_b=true)
    dev = isa(xy, CuArray) ? gpu : cpu
    sampling_grid_2d = sampling_grid[1:2, :, :]
    A, Ainv, b_ = get_transform_matrix(xy; scale_b=scale_b)
    Ai = cat(Ainv..., dims=3)
    xy_ = cpu(sampling_grid_2d) .- unsqueeze(hcat(b_...), 2)
    gn = batched_mul(Ai, xy_)
    gn = reshape(gn, 2, 28, 28, size(gn)[end]) |> dev
    x = reshape(x, args[:img_size]..., 1, size(x)[end])
    grid_sample(x, gn; padding_mode=:zeros)
end


