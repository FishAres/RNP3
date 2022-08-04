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


"""
here size(thetas) = (6, bsz)
do we multiply scale (thetas[[1,4],:]) by translation (thetas[5:6,:])?

"""
function grid_generator_3d(sampling_grid, thetas)
    sc = thetas[[1, 4], :]
    thetas = vcat(thetas[1:4, :], sc .* thetas[5:6, :])
    thetas = reshape(thetas, 2, 3, size(thetas)[end])
    tr_grid = batched_mul(thetas, sampling_grid)
    return tr_grid
end

"for 2d sampling grid and no rotation or shear (thetas[[2:3,:] = 0)"
function grid_generator_2d(sampling_grid_2d, thetas)
    sc = thetas[[1, 4], :]
    bs = sc .* thetas[5:6, :]
    grid2 = unsqueeze(sc, 2) .* sampling_grid_2d .+ unsqueeze(bs, 2)
    grid2
end

## ====

rotmat(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)] # rotation matrix
shearmat(s) = [1 s; 0 1] # shear matrix
scalemat(sc, offs::Float32=0.0f0) = [sc[1]+offs 0; 0 sc[2]+offs] # scale matrix

"works with (thetas[[2:3,:] = 0)"
function get_inv_grid2d(sampling_grid_2d, thetas)
    sc = thetas[[1, 4], :] .+ scale_offset
    bs = sc .* thetas[5:6, :]
    A = map(scalemat, eachcol(cpu(sc)))
    Ai = map(x -> diag(inv(x)), A)
    Ainv = hcat(Ai...) |> gpu
    return unsqueeze(Ainv, 2) .* (sampling_grid_2d .- unsqueeze(bs, 2))
end

"i can't get this right!! - works with (thetas[[2:3,:] = 0)"
function get_inv_grid(sampling_grid_2d, thetas)
    sc = thetas[[1, 4], :] # scale
    bs = sc .* thetas[5:6, :] # element-wise multiplication
    # apply "scalemat()" to each column of sc etc.
    Asc = map(scalemat, eachcol(cpu(sc)))
    Ashear = map(shearmat, cpu(thetas)[3, :])
    Arot = map(rotmat, cpu(thetas)[2, :])

    A = map(*, Ashear, Asc, Arot)
    Ai = map(x -> diag(inv(x)), A)

    Ainv = hcat(Ai...) |> gpu

    unsqueeze(Ainv, 2) .* (sampling_grid_2d .- unsqueeze(bs, 2))
end

"sample from image x at grid grid"
function sample_patch(grid, x)
    tg = reshape(grid, 2, 28, 28, size(grid)[end])
    x = reshape(x, 28, 28, 1, size(x)[end])
    grid_sample(x, tg; padding_mode=:zeros) # does bilinear interpolation
end

