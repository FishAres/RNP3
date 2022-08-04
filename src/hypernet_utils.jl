using Flux

## Indexing helpers

function get_module_sizes(m::Flux.Chain; args=args)
    sizes, modules = [], []
    function get_module_sizes_(m, sizes, modules)
        for l in m.layers
            if hasfield(typeof(l), :layers)
                get_module_sizes_(l, sizes, modules)
            elseif hasfield(typeof(l), :weight)
                wsz = size(l.weight)
                b_ = l.bias
                b_sz = b_ == false ? 0 : size(b_)
                push!(sizes, (wsz, b_sz))
                push!(modules, typeof(l))
            elseif isempty(Flux.params(l))
                nothing
            elseif Flux.hasaffine(l) # check for activity normalization
                psz = size.(Flux.params(l))
                push!(sizes, (1,))
                push!(modules, typeof(l))
            end
        end
        return modules, sizes
    end
    modules, sizes = get_module_sizes_(m, sizes, modules)
end

function get_params_length(p)
    ms, szs = get_module_sizes(p)
    sizes_ = map(x -> prod.(x), szs)
    map(sum, sizes_)
end


function split_weights(θs, offsets)
    msz, bsz = size(θs)
    offsets_full = [0; offsets]
    ws = [θs[offsets_full[i]+1:offsets_full[i+1], :][:] for i in 1:length(offsets)]
    ws_flat = vcat(ws...)
    @assert length(ws_flat) == msz * bsz
    return ws_flat
end



## Dense

struct HyDense
    weight
    bias
    σ
    HyDense(weight, b, σ) = new(weight, b, σ)
end


Flux.@functor HyDense

## ==== create


function create_bias(weights::AbstractArray, bias::Bool, dims::Tuple...)
    bias ? fill!(similar(weights, dims...), 0) : false
end


function HyDense(
    in::Integer,
    out::Integer,
    bsz::Integer,
    σ=identity;
    init=Flux.glorot_uniform,
    bias=true
)
    w = init(out, in, bsz)
    b = create_bias(w, bias, (size(w, 1), size(w, 3)))
    HyDense(init(out, in, bsz), b, σ)
end

function split_bias(in::Integer, out::Integer, W::AbstractMatrix)
    ab, bsz = size(W)
    in_out = (in * out)
    bsize = ab - in_out
    W_ = reshape(W[1:in_out, :], out, in, :)
    b_ = bsize > 0 ? W[in_out+1:end, :] : false
    W_, b_
end

"single parameter matrix W, known input-output dims"
function HyDense(in::Integer, out::Integer, W::AbstractMatrix, σ=identity)
    W_, b_ = split_bias(in, out, W)
    HyDense(W_, b_, σ)
end


## ==== 

function (m::HyDense)(x::AbstractArray)
    σ = NNlib.fast_act(m.σ, x)  # replaces tanh => tanh_fast, etc
    x = length(size(x)) > 2 ? x : unsqueeze(x, 2)
    b_ = isa(m.bias, AbstractArray) ? unsqueeze(m.bias, 2) : m.bias
    return σ.(batched_mul(m.weight, x) .+ b_)
end


function Base.show(io::IO, l::HyDense)
    print(
        io,
        "HyDense(",
        size(l.weight, 2),
        " => ",
        size(l.weight, 1),
        ", batch size: ",
        size(l.weight, 3),
    )
    l.σ == identity || print(io, ", ", l.σ)
    l.bias == false && print(io, "; bias=false")
    print(io, ")")
end

## == RNNs
@inline bmul(a, b) = dropdims(batched_mul(unsqueeze(a, 1), b); dims=1)

"RNN parameterized by Wh, Wx, b, with initial state h"
function RN(Wh, Wx, b, h, x; f=elu)
    h = f.(bmul(h, Wh) + bmul(x, Wx) + b)
    return h, h
end

gate(h, n) = (1:h) .+ h * (n - 1)
gate(x::AbstractVector, h, n) = @view x[gate(h, n)]
gate(x::AbstractMatrix, h, n) = @view x[gate(h, n), :]

function gru_output(Wx, Wh, b, x, h)
    o = size(h, 1)
    gx = bmul(x, Wx)
    gh = bmul(h, Wh)
    r = σ.(gate(gx, o, 1) .+ gate(gh, o, 1) .+ gate(b, o, 1))
    z = σ.(gate(gx, o, 2) .+ gate(gh, o, 2) .+ gate(b, o, 2))
    return gx, gh, r, z
end

function manzGRU(Wh, Wx, b, h, x)
    b, o = b, size(h, 1)
    gx, gh, r, z = gru_output(Wx, Wh, b, x, h)
    h̃ = tanh.(gate(gx, o, 3) .+ r .* gate(gh, o, 3) .+ gate(b, o, 3))
    h′ = (1 .- z) .* h̃ .+ z .* h
    sz = size(x)
    return h′, reshape(h′, :, sz[2:end]...)
end

"(θ: b, Wh, Wx, h) -> Recur(RNN(..., h))"
function ps_to_RN(θ; rn_fun=RN, args=args, f_out=elu)
    Wh, Wx, b, h = θ
    ft = (h, x) -> rn_fun(Wh, Wx, b, h, x; f=f_out)
    return Flux.Recur(ft, h)
end

"size for RNN Wh, Wx, b"
get_rnn_θ_sizes(esz, hsz) = esz * hsz + hsz^2 + 2 * hsz

function rec_rnn_θ_sizes(esz, hsz)
    [hsz^2, esz * hsz, hsz, hsz]
end

function get_rn_θs(rnn_θs, esz, hsz)
    fx_sizes = rec_rnn_θ_sizes(esz, hsz)
    @assert sum(fx_sizes) == size(rnn_θs, 1)
    fx_inds = [0; cumsum(fx_sizes)]
    # split rnn_θs vector according to cumulative fx_inds
    Wh_, Wx_, b, h =
        collect(rnn_θs[fx_inds[ind-1]+1:fx_inds[ind], :] for ind = 2:length(fx_inds))
    Wh = reshape(Wh_, hsz, hsz, size(Wh_)[end])
    Wx = reshape(Wx_, esz, hsz, size(Wx_)[end])
    Wh, Wx, b, h
end




## maybe useful soon

function get_offsets(m::Flux.Chain)
    θ, re = Flux.destructure(m)
    m_offsets = re.offsets
    offsets_ = []
    function get_offsets_(offsets, offsets_)
        for mo in offsets
            for key in keys(mo)
                a = mo[key]
                if isa(a, Number)
                    push!(offsets_, a)
                elseif isa(a, NamedTuple) || isa(a, Tuple) && !isempty(a)
                    get_offsets_(a, offsets_)
                end
            end
        end
        return offsets_
    end
    return get_offsets_(m_offsets, offsets_)
end