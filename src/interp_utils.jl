using LinearAlgebra, Statistics
using Flux, Zygote, NNlib
using Flux: unsqueeze

function get_pre_grid(width, height; args=args)
    x = collect(LinRange(-1, 1, width))
    y = collect(LinRange(-1, 1, height))
    x_t_flat = reshape(repeat(x, height), 1, height * width)
    y_t_flat = reshape(repeat(transpose(y), width), 1, height * width)
    sampling_grid = vcat(x_t_flat, y_t_flat)
    sampling_grid = reshape(
        transpose(repeat(transpose(sampling_grid), args[:bsz])),
        2,
        size(x_t_flat)[2],
        args[:bsz],
    )
    return collect(Float32.(sampling_grid))
end

# caution - scale_offset is provided from outside the function input
function affine_grid_generator(sampling_grid, thetas; args=args, sz=args[:img_size])
    bsz = size(thetas)[end]
    thetas = isa(thetas, CuArray) ? thetas .+ scale_offset : thetas .+ cpu(scale_offset)
    theta = Zygote.Buffer(thetas)
    theta[1:4, :] = thetas[1:4, :]
    theta[[5, 6], :] = thetas[[1, 4], :] .* thetas[[5, 6], :]
    th_ = reshape(copy(theta), 2, 3, bsz)
    tr_grid = batched_mul(th_, sampling_grid)
    return reshape(tr_grid, 2, sz..., bsz)
    return tr_grid
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
    for (i,a_) in enumerate(A)
        a_[diagind(a_)[findall(iszero, diag(a_))]] .= 1f-3
    end
    # A
end

"from 6-param xy get A, inv(A) of Ax + b"
function get_transform_matrix(xy; add_offset=true)
    xy = cpu(xy)
    S = xy[[1, 4], :]
    θ = xy[2, :]
    s = xy[3, :]
    b = xy[5:6, :]

    Arot = map(rotmat, θ)
    Ashear = map(shearmat, s)

    # add scale_offset
    As = if add_offset
        [[S[1, i].+1 0; 0 S[2, i].+1] for i in 1:64]
    else
        offs = cpu(scale_offset)[1,1]
        [[S[1, i].+1 .+ offs  0; 0 S[2, i] .+1 .+ offs] for i in 1:64]
    end
    make_invertible!(As)
    A = map(*, Arot, As, Ashear)
    
    Ainv = map((x, y, z) -> inv(x) * inv(y) * inv(z), Arot, As, Ashear)
    
    b_inv = map(*, map(inv, As), eachcol(b))
    b_ = map(*, As, eachcol(b))
    A, Ainv, b_, b_inv
end

Zygote.@nograd function zoom_in(x, xy, sampling_grid_2d; args=args)
    A, Ainv, b_, b_inv = get_transform_matrix(xy)
    Ai = cat(Ainv..., dims=3)
    gn = batched_mul(gpu(Ai), sampling_grid_2d) .+ gpu(unsqueeze(hcat(b_inv...), 2))
    gn = reshape(gn, 2, 28, 28, size(gn)[end])
    x = reshape(x, args[:img_size]..., 1, size(x)[end])
    grid_sample(x, gn; padding_mode=:zeros)
end