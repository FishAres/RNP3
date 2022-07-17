rotmat(θ) = [cos(θ) -sin(θ); sin(θ) cos(θ)]
shearmat(s) = [1 s; 0 1]
scalemat(sc, offs::Float32=0.0f0) = [sc[1]+offs 0; 0 sc[2]+offs]

function make_invertible!(A)
    for (i, a_) in enumerate(A)
        a_[diagind(a_)[findall(iszero, diag(a_))]] .= 1.0f-4
    end
    # A
end

"from 6-param xy get A, inv(A) of Ax + b"
function get_transform_matrix(xy; scale_b=true, scale_offset=args[:scale_offset])
    xy = cpu(xy)
    sc = xy[[1, 4], :] .+ scale_offset
    θ = xy[2, :]
    s = xy[3, :]
    b = xy[5:6, :]
    bsz = size(xy)[end]
    Arot = map(rotmat, θ)
    Ashear = map(shearmat, s)


    # Sc = map(x -> scalemat(x, args[:scale_offset]), eachcol(S))

    Sc = map(x -> I(2) .* x, eachcol(sc))
    make_invertible!(Sc)
    # A = map(*, Arot, Sc, Ashear)
    A = map(*, Sc, Ashear, Arot)

    # Ainv = map((x, y, z) -> inv(x) * inv(y) * inv(z), Arot, As, Ashear)
    Ainv = map(inv, A)
    b_ = collect(eachcol(b))
    b_ = scale_b ? map(*, Sc, b_) : b_
    A, Ainv, b_
end

Zygote.@nograd function zoom_in(x, xy, sampling_grid; args=args, scale_b=true)
    dev = isa(xy, CuArray) ? gpu : cpu
    sampling_grid_2d = sampling_grid[1:2, :, :]
    A, Ainv, b_ = get_transform_matrix(xy; scale_b=scale_b)
    Ai = cat(Ainv..., dims=3)
    # xy_ = cpu(sampling_grid_2d) .- unsqueeze(hcat(b_...), 2)
    xy_ = cpu(sampling_grid_2d)
    gn = batched_mul(Ai, xy_) .- unsqueeze(hcat(b_...), 2)
    gn = reshape(gn, 2, 28, 28, size(gn)[end]) |> dev
    x = reshape(x, args[:img_size]..., 1, size(x)[end])
    grid_sample(x, gn; padding_mode=:zeros)
end


