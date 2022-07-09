using Plots

function plot_digit(x; color = :grays, alpha = 1, kwargs...)
    return heatmap(
        x[:, end:-1:1]';
        color = color,
        clim = (0, 1),
        axis = nothing,
        colorbar = false,
        alpha = alpha,
        kwargs...,
    )
end

function plot_digit!(x; color = :grays, alpha = 1, kwargs...)
    return heatmap!(
        x[:, end:-1:1]';
        color = color,
        clim = (0, 1),
        axis = nothing,
        colorbar = false,
        alpha = alpha,
        kwargs...,
    )
end
