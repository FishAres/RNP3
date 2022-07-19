
function get_inv(thetas; scale_offset=args[:scale_offset])
    thetas = cpu(thetas)
    sc = thetas[1:2, :] .+ scale_offset
    bs = thetas[3:4, :] .* sc
    As = map(x -> I(2) .* x, eachcol(sc))
    Ainv = hcat(map(x -> diag(inv(x)), As)...)
    return Ainv, bs
end

Zygote.@nograd function zoom_in2d(x, xy, sampling_grid; args=args)
    dev = isa(x, CuArray) ? gpu : cpu

    # y = sample_patch(x, xy, sampling_grid)

    Ai, bs = get_inv(xy) |> dev
    tr_grid = unsqueeze(Ai, 2) .* (sampling_grid .- unsqueeze(bs, 2))
    tr_grid = reshape(tr_grid, 2, args[:img_size]..., args[:bsz])

    ximg = reshape(x, args[:img_size]..., 1, args[:bsz])
    grid_sample(ximg, tr_grid; padding_mode=:zeros)
end

## ====

σ1(x) = σ(x) - typeof(x)(0.5)


function get_models(θs, model_bounds; init_zs=true)
    inds = init_zs ? [0; cumsum([model_bounds...; args[:π]; args[:asz]])] : [0; cumsum(model_bounds)]

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Vx = reshape(Θ[1], args[:π], args[:π], args[:bsz])
    Va = reshape(Θ[2], args[:asz], args[:asz], args[:bsz])

    # dudt ?
    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[3], σ1), flatten)
    # tanh to cancel out previous patches    
    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[4], relu), flatten)

    Dec_z_a = Chain(HyDense(args[:asz], args[:asz], Θ[5],), flatten)

    z0 = init_zs ? Θ[6] : nothing
    a0 = init_zs ? Θ[7] : nothing

    return Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0
end

Δa(a, Va, Dec_z_a) = sin.(Dec_z_a(bmul(a, Va)))

# update_z(z, Δz) = max.(z, Δz)
update_z(z, Δz) = z + Δz

function forward_pass(z1, a1, models, x)
    Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models
    z1 = bmul(z1, Vx) # linear transition
    a1 = Δa(a1, Va, Dec_z_a)
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid) |> flatten
    ϵ = patch_t .- x̂
    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
end


Zygote.@nograd function push_to_arrays!(outputs, arrays)
    for (output, array) in zip(outputs, arrays)
        push!(array, cpu(output))
    end
end


function model_loss(z, x)
    θs = H(z)
    Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0 = get_models(θs, model_bounds)
    models = Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a

    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)

    out = sample_patch(x̂, a1, sampling_grid)
    Lx = mean(ϵ .^ 2)
    for t = 2:args[:seqlen]
        z = update_z(z, Δz)
        θs = H(z)
        Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0 = get_models(θs, model_bounds)
        models = Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out += sample_patch(x̂, a1, sampling_grid)
        Lx += mean(ϵ .^ 2)
    end
    rec_loss = Flux.mse(flatten(out), flatten(x))
    local_loss = args[:δL] * Lx
    return rec_loss + local_loss + args[:λ] * norm(Flux.params(H))
end

function get_loop(z, x)
    outputs = patches, recs, errs, xys, z1s, zs, patches_t, Vxs = [], [], [], [], [], [], [], []

    θs = H(z)
    Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0 = get_models(θs, model_bounds)
    models = Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x)

    out = sample_patch(x̂, a1, sampling_grid)
    Lx = mean(ϵ .^ 2)

    push_to_arrays!((x̂, out, ϵ, a1, z1, z, patch_t, Vx), outputs)
    for t = 2:args[:seqlen]
        z = update_z(z, Δz)
        θs = H(z)
        Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a, z0, a0 = get_models(θs, model_bounds)
        models = Vx, Va, Enc_ϵ_z, Dec_z_x̂, Dec_z_a
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x)
        out += sample_patch(x̂, a1, sampling_grid)
        Lx += mean(ϵ .^ 2)

        push_to_arrays!((x̂, out, ϵ, a1, z1, z, patch_t, Vx), outputs)
    end
    return outputs
end




# function model_loss(z, x)
# L = Flux.mse(full_loop(z, x) |> flatten, x |> flatten)
# L + args[:λ] * norm(Flux.params(H))
# end
## ====

function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, (x, y)) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    x_
end

function train_model(opt, ps, train_data; epoch=1)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, (x, y)) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            model_loss(zs[i], x)
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data)
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, (x, y)) in enumerate(test_data)
        L += model_loss(zs[i], x)
    end
    return L / length(test_data)
end

## ====
function plot_rec(out, x, ind)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    return plot(p1, p2)
end

function plot_rec(out, x, xs, ind)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    p3 = plot([plot_digit(x[:, :, 1, ind], boundc=false) for x in xs]...)
    return plot(p1, p2, p3, layout=(1, 3))
end


function plot_recs(x, inds; plot_seq=true)
    z = randn(Float32, args[:π], args[:bsz]) |> gpu

    patches, preds, errs, xys, zs = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        # [plot_rec(preds[end], x, preds, ind) for ind in inds]
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end
