using LinearAlgebra, Statistics
using Flux, Zygote, CUDA
using Distributions
using ProgressMeter
using ProgressMeter: Progress
using Plots


include(srcdir("interp_utils.jl"))
include(srcdir("hypernet_utils.jl"))
include(srcdir("nn_utils.jl"))
include(srcdir("plotting_utils.jl"))
include(srcdir("logging_utils.jl"))

include(srcdir("utils.jl"))

function get_fstate_models(θs, Hx_bounds; args=args, fz=args[:f_z])
    inds = Zygote.ignore() do
        [0; cumsum([Hx_bounds...; args[:π]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_z = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)

    f_state = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=fz)
    err_rnn = ps_to_RN(get_rn_θs(Θ[3], args[:π], args[:π]); f_out=elu)

    Dec_z_x̂ = Chain(HyDense(args[:π], 784, Θ[4], relu6), flatten)
    Enc_ϵ_z = Chain(HyDense(784, args[:π], Θ[5], elu), flatten)

    z0 = fz.(Θ[6])

    return (Enc_za_z, f_state, err_rnn, Dec_z_x̂, Enc_ϵ_z,), z0
end

twoσ(x) = 2.0f0 * (σ(x) - 0.5f0)

function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], twoσ), flatten)

    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function get_models(θsz, θsa; args=args, Hx_bounds=Hx_bounds, Ha_bounds=Ha_bounds)
    (Enc_za_z, f_state, err_rnn, Dec_z_x̂, Enc_ϵ_z,), z0 = get_fstate_models(θsz, Hx_bounds; args=args)
    (Enc_za_a, f_policy, Dec_z_a,), a0 = get_fpolicy_models(θsa, Ha_bounds; args=args)
    models = f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a
    return models, z0, a0
end



"one iteration"
function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1) # todo parallel layer?
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1)
    patch_t = zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset) |> flatten

    ϵ = Zygote.ignore() do
        # block gradients for pred. error
        patch_t .- x̂
    end
    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
end

function full_sequence(models::Tuple, z0, a0, x; args=args, scale_offset=args[:scale_offset])
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=scale_offset)
    err_emb = err_rnn(Δz) # update error embedding
    out = sample_patch(x̂, a1, sampling_grid)
    for t = 2:args[:seqlen]
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=scale_offset)
        err_emb = err_rnn(Δz) # update error embedding
        out += sample_patch(x̂, a1, sampling_grid)
    end
    return out, err_emb
end

function full_sequence(z::AbstractArray, x; args=args, scale_offset=args[:scale_offset])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    return full_sequence(models, z0, a0, x; args=args, scale_offset=scale_offset)
end


function model_loss(z, x; args=args)
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset])
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    # Lpatch = Flux.mse(flatten(x̂), flatten(patch_t))
    Lfull = Flux.mse(flatten(out_full), flatten(x))
    for t = 2:args[:glimpse_len]
        z = RN2(err_emb_1)
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset])

        # Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end
    z = RN2(err_emb_1)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    # Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
    Lfull += Flux.mse(flatten(out_full), flatten(x))

    # local_loss = args[:δL] * Lpatch

    # return local_loss + args[:λf] * Lfull
    return Lfull

end

Zygote.@nograd function push_to_arrays!(outputs, arrays)
    for (output, array) in zip(outputs, arrays)
        push!(array, cpu(output))
    end
end

function get_loop(z, x; args=args)
    outputs = patches, recs, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset])
    out = sample_patch(out_full .+ 0.1f0, a1, sampling_grid)

    push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]
        z = RN2(err_emb_1)
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t)
        out += sample_patch(out_1 .+ 0.1f0, a1, sampling_grid)
        push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)
    end
    z = RN2(err_emb_1)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    push_to_arrays!((out_full, out, ϵ, z, a1, patch_t), outputs)
    return outputs
end



function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    zs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            loss = model_loss(zs[i], x)
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "loss", loss)
            end
            loss + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end
        # foreach(x -> clamp!(x, -0.1f0, 0.1f0), grad)
        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    zs = [rand(D, args[:π], args[:bsz]) for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        L += model_loss(zs[i], x)
    end
    return L / length(test_data)
end

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


function plot_recs(x, inds; plot_seq=true, args=args)
    z = rand(args[:D], args[:π], args[:bsz]) |> gpu

    patches, preds, errs, xys, zs = get_loop(z, x)
    p = plot_seq ? let
        patches_ = map(x -> reshape(x, 28, 28, 1, size(x)[end]), patches)
        # [plot_rec(preds[end], x, preds, ind) for ind in inds]
        [plot_rec(preds[end], x, patches_, ind) for ind in inds]
    end : [plot_rec(preds[end], x, ind) for ind in inds]

    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end

function sample_loader(loader)
    rand_int = rand(1:length(loader))
    x_ = for (i, x) in enumerate(loader)
        if i == rand_int
            return x
        end
    end
    x_
end
