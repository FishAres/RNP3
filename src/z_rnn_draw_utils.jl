function get_models(θs, model_bounds; init_zs=true)
    inds = init_zs ? [0; cumsum([model_bounds...; args[:asz]])] : [0; cumsum(model_bounds)]

    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Va = reshape(Θ[1], args[:asz], args[:asz], args[:bsz])

    Dec_a_ps = Chain(HyDense(args[:asz], args[:asz], Θ[2],), flatten)
    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[3], gelu), flatten)
    # tanh to cancel out previous patches    

    a0 = init_zs ? Θ[4] : nothing

    return (Va, Dec_a_ps, Enc_ϵ_z), a0
end

get_err(out, x) = (out .* x) .- out

function nd_gaussian(xgrids, μ, σ)
    g_inner(xgrid, μ, σ) = ((flatten(xgrid) .- unsqueeze(μ, 1)) .^ 2) ./ (unsqueeze(σ, 1) .^ 2)
    sz = size(xgrids[1])[1:2]
    inners = [g_inner(xgrids[i], μ[i, :], σ[i, :]) for i in 1:length(xgrids)]
    reshape(exp.(-sum(inners)), sz..., :)
end

function forward_pass(a1, models, x)
    Va, Da, fϵ = models
    a1 = bmul(a1, Va)
    ms = Da(a1)
    μs = 0.75f0 .* tanh.(ms[1:2, :])
    σs = 0.3f0 .* σ.(ms[3:4, :])
    out = nd_gaussian(xgrids, μs, σs)
    err = get_err(out, x)
    Δa = fϵ(flatten(err))
    a1, out, err, Δa
end

function model_loss(z, x)
    Flux.reset!(RN2)
    θs = H(z)
    models, a0 = get_models(θs, model_bounds)
    a1, x̂, err, Δa = forward_pass(a0, models, x)
    z = RN2(Δa)
    out = x̂
    Lx = mean(err .^ 2)
    for t = 2:args[:seqlen]
        θs = H(z)
        models, a0 = get_models(θs, model_bounds)
        a1, x̂, err, Δa = forward_pass(a1, models, x)
        out += x̂
        z = RN2(Δa)
        Lx += mean(err .^ 2)
    end
    rec_loss = Flux.mse(out, x)
    local_loss = args[:δL] * Lx
    rec_loss + local_loss + args[:λ] * norm(Flux.params(H))
end



Zygote.@nograd function push_to_arrays!(outputs, arrays)
    for (output, array) in zip(outputs, arrays)
        push!(array, cpu(output))
    end
end

function get_loop(z, x)
    outputs = patches, recs, errs, a1s, zs, Vas = [], [], [], [], [], []
    Flux.reset!(RN2)
    θs = H(z)
    models, a0 = get_models(θs, model_bounds)
    a1, x̂, err, Δa = forward_pass(a0, models, x)
    z = RN2(Δa)
    out = x̂
    push_to_arrays!((x̂, out, err, a1, z, models[1]), outputs)
    for t = 2:args[:seqlen]
        θs = H(z)
        models, a0 = get_models(θs, model_bounds)
        a1, x̂, err, Δa = forward_pass(a1, models, x)
        out += x̂
        z = RN2(Δa)
        push_to_arrays!((x̂, out, err, a1, z, models[1]), outputs)
    end
    return outputs
end

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


replace_nan!(x; dev=gpu) = (isnan(x) ? 0.0f0 : x) |> dev


function train_model(opt, ps, train_data; epoch=1)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    zs = [randn(Float32, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, (x, y)) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            model_loss(zs[i], x)
        end
        foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        # foreach(x -> replace_nan!.(x), grad)
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
function plot_rec(out, x, ind; kwargs...)
    out_ = reshape(cpu(out), 28, 28, size(out)[end])
    x_ = reshape(cpu(x), 28, 28, size(x)[end])
    p1 = plot_digit(out_[:, :, ind])
    p2 = plot_digit(x_[:, :, ind])
    return plot(p1, p2, kwargs...)
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
    patches, preds, errs, a1s, zs, Vas = get_loop(z, x)
    # patches, preds, errs, xys, zs = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        # [plot_rec(preds[end], x, preds, ind) for ind in inds]
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end
