function get_models(θs, dec_out, f_out)
    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:esz], θs[1], elu), flatten)
    f = ps_to_RN(get_rn_θs(θs[2], args[:esz], args[:π]); f_out=elu)
    Dec_z_ = Chain(HyDense(args[:π], dec_out, θs[3], f_out), flatten)
    Enc_za_z, f, Dec_z_, θs[end]
end


function forward_pass(models, z1, a1)
    Encx_za_z, fx, Dec_z_x̂, Enca_za_z, fa, Dec_z_a = models
    za_cat = vcat(z1, a1)
    z1 = fx(Encx_za_z(za_cat))
    a1 = Dec_z_a(fa(Enca_za_z(za_cat)))

    return z1, a1
end

function full_sequence(z)
    θsx = Hx(z)
    θsa = Ha(z)
    Encx_za_z, fx, Dec_z_x̂, z0 = get_models(θsx, 784, relu)
    Enca_za_z, fa, Dec_z_a, a0 = get_models(θsa, 6, sin)
    models = Encx_za_z, fx, Dec_z_x̂, Enca_za_z, fa, Dec_z_a

    z1, a1 = forward_pass(models, z0, a0)
    x̂ = Dec_z_x̂(z1)
    out = sample_patch(x̂, a1, sampling_grid)
    for t in 2:args[:seqlen]
        z1, a1 = forward_pass(models, z0, a0)
        x̂ = Dec_z_x̂(z1)
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out
end


function model_loss(x)
    z = Encoder(x)
    θsx = Hx(z)
    θsa = Ha(z)
    Encx_za_z, fx, Dec_z_x̂, z0 = get_models(θsx, 784, relu)
    Enca_za_z, fa, Dec_z_a, a0 = get_models(θsa, 6, sin)
    models = Encx_za_z, fx, Dec_z_x̂, Enca_za_z, fa, Dec_z_a

    z1, a1 = forward_pass(models, z0, a0)
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid)
    out1 = full_sequence(z1)
    out2 = sample_patch(out1, a1, sampling_grid)
    L2 = Flux.mse(flatten(x̂), flatten(patch_t))
    for t in 2:args[:seqlen]
        z1, a1 = forward_pass(models, z0, a0)
        x̂ = Dec_z_x̂(z1)
        out1 = full_sequence(z1)
        out2 += sample_patch(out1, a1, sampling_grid)
        L2 += Flux.mse(flatten(x̂), flatten(patch_t))
    end
    return Flux.mse(flatten(out2), flatten(x)), L2
end

## === get variables from loop
Zygote.@nograd function push_to_arrays!(outputs, arrays)
    for (output, array) in zip(outputs, arrays)
        push!(array, cpu(output))
    end
end

function get_loop(x)
    outputs = patches, recs, a1s, z1s, patches_t = [], [], [], [], [], []
    z = Encoder(x)
    θsx = Hx(z)
    θsa = Ha(z)
    Encx_za_z, fx, Dec_z_x̂, z0 = get_models(θsx, 784, relu)
    Enca_za_z, fa, Dec_z_a, a0 = get_models(θsa, 6, sin)
    models = Encx_za_z, fx, Dec_z_x̂, Enca_za_z, fa, Dec_z_a

    z1, a1 = forward_pass(models, z0, a0)
    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid)
    out1 = full_sequence(z1)
    out2 = sample_patch(out1, a1, sampling_grid)

    push_to_arrays!((out1, out2, a1, z1, patch_t), outputs)

    for t in 2:args[:seqlen]
        z1, a1 = forward_pass(models, z0, a0)
        x̂ = Dec_z_x̂(z1)
        out1 = full_sequence(z1)
        out2 += sample_patch(out1, a1, sampling_grid)

        push_to_arrays!((out1, out2, a1, z1, patch_t), outputs)
    end
    return outputs
end

## === Training


function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, (x, y)) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    x_
end

function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing)
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))

    for (i, (x, y)) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            Lout, L2 = model_loss(x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "loss", loss)
            end
            loss = Lout + args[:λ2] * L2
            reg = args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
            loss + reg
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; args=args)
    L = 0.0f0
    for (i, (x, y)) in enumerate(test_data)
        Lout, L2 = model_loss(x)
        loss = Lout + args[:λ2] * L2
        L += loss
    end
    return L / length(test_data)
end


## === plotting

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
    patches, preds, a1s, z1s, patches_t = get_loop(x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end
