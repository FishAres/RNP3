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

    # rgb
    Dec_z_x̂r = Chain(HyDense(args[:π], args[:imszprod], Θ[4], relu6), flatten)
    Dec_z_x̂g = Chain(HyDense(args[:π], args[:imszprod], Θ[5], relu6), flatten)
    Dec_z_x̂b = Chain(HyDense(args[:π], args[:imszprod], Θ[6], relu6), flatten)

    Dec_z_x̂ = Chain(
        Split(
            Dec_z_x̂r,
            Dec_z_x̂g,
            Dec_z_x̂b,
        ),
        x -> hcat(unsqueeze.(x, 2)...)
    )

    Enc_ϵ_z = Chain(HyDense(args[:imszprod] * args[:img_channels], args[:π], Θ[7], elu), flatten)

    z0 = fz.(Θ[8])

    return (Enc_za_z, f_state, err_rnn, Dec_z_x̂, Enc_ϵ_z,), z0
end

"one iteration"
function forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset])
    f_state, f_policy, err_rnn, Enc_za_z, Enc_za_a, Enc_ϵ_z, Dec_z_x̂, Dec_z_a = models

    za = vcat(z1, a1) # todo parallel layer?
    ez = Enc_za_z(za)
    ea = Enc_za_a(za)
    z1 = f_state(ez)
    a1 = Dec_z_a(f_policy(ea))

    x̂ = Dec_z_x̂(z1) |> flatten
    patch_t = zoom_in2d(x, a1, sampling_grid; scale_offset=scale_offset) |> flatten

    ϵ = patch_t .- x̂

    Δz = Enc_ϵ_z(ϵ)
    return z1, a1, x̂, patch_t, ϵ, Δz
end


function get_fpolicy_models(θs, Ha_bounds; args=args)
    inds = Zygote.ignore() do
        [0; cumsum([Ha_bounds...; args[:asz]])]
    end
    Θ = [θs[inds[i]+1:inds[i+1], :] for i in 1:length(inds)-1]

    Enc_za_a = Chain(HyDense(args[:π] + args[:asz], args[:π], Θ[1], elu), flatten)
    f_policy = ps_to_RN(get_rn_θs(Θ[2], args[:π], args[:π]); f_out=elu)
    Dec_z_a = Chain(HyDense(args[:π], args[:asz], Θ[3], sin), flatten)

    a0 = sin.(Θ[4])

    return (Enc_za_a, f_policy, Dec_z_a), a0
end

function model_loss(z, x, rs; args=args)
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    # is z0, a0 necessary?
    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])


    Lfull = Flux.mse(flatten(out_full), flatten(x); agg=sum)
    for t = 2:args[:glimpse_len]-1
        μ, logvar = RN2(err_emb_1)
        z = sample_z(μ, logvar, rs[t-1])
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])

        # Lpatch += Flux.mse(flatten(x̂), flatten(patch_t))
        Lfull += Flux.mse(flatten(out_full), flatten(x))
    end
    μ, logvar = RN2(err_emb_1)
    z = sample_z(μ, logvar, rs[end])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])
    Lfull += Flux.mse(flatten(out_full), flatten(x); agg=sum)
    klqp = kl_loss(μ, logvar)
    return Lfull, klqp
end

function get_loop(z, x, rs; args=args)
    outputs = full_recs, patches, errs, zs, as, patches_t = [], [], [], [], [], [], []
    Flux.reset!(RN2)
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)

    z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z0, a0, models, x; scale_offset=args[:scale_offset_sense])
    out_full, err_emb_full = full_sequence(models, z0, a0, x)
    out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
    sense_patch = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])

    push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)

    for t = 2:args[:glimpse_len]-1
        μ, logvar = RN2(err_emb_1)
        z = sample_z(μ, logvar, rs[t-1])
        θsz = Hx(z)
        θsa = Ha(z)
        models, z0, a0 = get_models(θsz, θsa; args=args)

        out_full, err_emb_full = full_sequence(models, z0, a0, x)
        z1, a1, x̂, patch_t, ϵ, Δz = forward_pass(z1, a1, models, x; scale_offset=args[:scale_offset_sense])
        out_1, err_emb_1 = full_sequence(z1, patch_t; scale_offset=args[:scale_offset_sense])
        sense_patch = sample_patch(out_1 .+ 0.1f0, a1, sampling_grid; scale_offset=args[:scale_offset_sense])
        push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)


    end
    μ, logvar = RN2(err_emb_1)
    z = sample_z(μ, logvar, rs[end])
    θsz = Hx(z)
    θsa = Ha(z)
    models, z0, a0 = get_models(θsz, θsa; args=args)
    out_full, err_emb_full = full_sequence(models, z0, a0, x; scale_offset=args[:scale_offset])

    push_to_arrays!((out_full, sense_patch, ϵ, z, a1, patch_t), outputs)
    return outputs
end


function train_model(opt, ps, train_data; args=args, epoch=1, logger=nothing, D=args[:D])
    progress_tracker = Progress(length(train_data), 1, "Training epoch $epoch :)")
    losses = zeros(length(train_data))
    # initial z's drawn from N(0,1)
    rs = [[rand(D, args[:π], args[:bsz]) for _ in 1:args[:glimpse_len]]
          for _ in 1:length(train_data)] |> gpu
    for (i, x) in enumerate(train_data)
        loss, grad = withgradient(ps) do
            rec_loss, klqp = model_loss(z0, x, rs[i])
            logger !== nothing && Zygote.ignore() do
                log_value(lg, "rec_loss", rec_loss)
                log_value(lg, "KL loss", klqp)
            end

            args[:α] * rec_loss + args[:β] * klqp + args[:λ] * (norm(Flux.params(Hx)) + norm(Flux.params(Ha)))
        end

        Flux.update!(opt, ps, grad)
        losses[i] = loss
        ProgressMeter.next!(progress_tracker; showvalues=[(:loss, loss)])
    end
    return losses
end

function test_model(test_data; D=args[:D])
    rs = [[rand(D, args[:π], args[:bsz]) for _ in 1:args[:glimpse_len]]
          for _ in 1:length(test_data)] |> gpu
    L = 0.0f0
    for (i, x) in enumerate(test_data)
        rec_loss, klqp = model_loss(z0, x, rs[i])
        L += args[:α] * rec_loss + args[:β] * klqp

    end
    return L / length(test_data)
end

## =====
using Images, Plots

function imview_cifar(x)
    colorview(RGB, permutedims(batched_adjoint(x), [3, 1, 2]))
end


function plot_rec(x, out::Vector, xs::Vector, ind)
    l = @layout [a b c]
    out_ = [reshape(cpu(k), args[:img_size]..., 3, size(k)[end]) for k in out]
    x_ = reshape(cpu(x), args[:img_size]..., 3, size(x)[end])
    p1 = plot([ # plot glimpses over image
        begin
            xnew = 0.3f0 .* x_[:, :, :, ind] + x[:, :, :, ind]
            plot(imview_cifar(xnew), axis=nothing,)
        end
        for x in out_]...)
    p2 = plot(imview_cifar(x_[:, :, :, ind]), axis=nothing, size=(20, 20))
    p3 = plot([plot(imview_cifar(x[:, :, :, ind]), axis=nothing) for x in xs]...)
    return plot(p1, p2, p3, layout=l)
end


function plot_recs(x, inds; args=args)
    rs = [rand(args[:D], args[:π], args[:bsz]) for _ in 1:args[:seqlen]] |> gpu
    full_recs, patches, errs, xys, zs = get_loop(z0, x, rs)
    full_recs = map(x -> reshape(x, args[:img_size]..., 3, size(x)[end]), full_recs)

    p = [plot_rec(x, patches, full_recs, ind) for ind in inds]
    return plot(p...; layout=(length(inds), 1), size=(600, 800))
end
