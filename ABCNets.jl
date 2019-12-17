using Flux
using Zygote
using CuArrays

# fix https://github.com/FluxML/Flux.jl/issues/947
Zygote.@adjoint function Base.convert(::Type{T}, xs::Array{K,N}) where {T<:CuArray, K, N}
  Base.convert(T, xs), Δ -> (nothing, Base.convert(Array, Δ),)
end

function weight_masks(W, us)
    dims = size(W)

    W̅ = W .- mean(W)

    std_W = std(W)

    W̃s = cat((sign.(W̅ .+ (u * std_W)) for u in us)...,
            dims=length(size(W)) + 1)

    W̃s
end

function binarize_weights(W, us)
    W̃s = weight_masks(W, us)

    dims = size(W)

    Wv = reshape(W, :)
    W̃vs = reshape(W̃s, :, length(us))

    αs = W̃vs \ Wv

    W̃v = W̃vs * αs
    
    W̃ = reshape(W̃v, dims...)

    W̃, αs
end

Zygote.@adjoint function binarize_weights(W, us) 
    W̃ = binarize_weights(W, us)
    function adjoint((∇_W̃, ∇_αs)) 
        # the "straight-through estimator"
        (∇_W̃, nothing)
    end
    W̃, adjoint
end

even_us(M) = if M == 1
        [0.0f0]
    else
        Array(range(-1.0f0, stop=1.0f0, length=M))
end
function even_err(M, W)
    us = even_us(M)
    W̃, _ = binarize_weights(W, us)
    mean((W .- W̃).^2)
end

# don't solve for alphas
function binarize_weights(W, us, αs)
    W̃s = weight_masks(W, us)

    dims = size(W)

    Wv = reshape(W, :)
    W̃vs = reshape(W̃s, :, length(us))

    W̃v = W̃vs * αs
    
    W̃ = reshape(W̃v, dims...)

    W̃
end

Zygote.@adjoint function binarize_weights(W, us, αs)
    # not sure why you'd need this, but might as well...
    W̃ = binarize_weights(W, us, αs)
    function adjoint(∇_W̃) 
        # the "straight-through estimator"
        (∇_W̃, nothing, nothing)
    end
    W̃, adjoint
end

function binarize_weights(W :: CuArray, us)
    W̃, αs = binarize_weights(cpu(W), cpu(us))
    gpu(W̃), cpu(αs) # note alphas are still on CPU!!
end
function binarize_weights(W :: CuArray, us, αs)
    W̃ = binarize_weights(cpu(W), cpu(us), cpu(αs))
    gpu(W̃)
end

function z(q)
    if q > 0.5f0
        1.0f0
    else
        -1.0f0
    end
end

function _zrev(q, ∇_q)
    if 0.0f0 < q < 1.0f0
        ∇_q
    else
        0.0f0
    end
end

Zygote.@adjoint z(q) = z(q), (∇_q) -> _zrev(q, ∇_q)

function zb(Q)
    z.(Q)
end

Zygote.@adjoint zb(Q) = zb(Q), (∇_Q) -> (_zrev.(Q, ∇_Q),)

function binarize_activations(A, vs, βs)
    shape = size(A)
    
    Av_x = reshape( (A), :, 1)
    
    vs_x = reshape( (vs), 1, :)
    βs_x = reshape( (βs), 1, :)
    
    Av1 =  (Av_x) .+  (vs_x)
    #Av2 = z.( (Av1))
    Av2 = zb( (Av1))
    Av3 =  (Av2) .*  (βs_x)
    
    Ãv = sum( (Av3), dims=2)
    
    result = reshape( (Ãv), shape)
        
    result
end

mutable struct BinWeights{U <: AbstractVector, A <: AbstractVector}
    us :: U
    αs :: A
    active :: Bool
end

function BinWeights(W, us :: U; active=false) where U
    # note: W is not stored! it's just used to initialize alphas.
    _, αs = binarize_weights(W, us)
    BinWeights(us, αs, active)
end

# note: no args, because the parameters aren't trainable, and shouldn't be moved to GPU.
Flux.@functor BinWeights ()

function (bw :: BinWeights{U, A})(W) where {U, A}
    if bw.active
        if Flux.istraining()
            W̃, αs = binarize_weights(W, bw.us)
            bw.αs = αs
            W̃
        else
            # todo: cache W̃?
            W̃ = binarize_weights(W, bw.us, bw.αs)
            W̃
        end
    else
        W
    end
end

mutable struct BinActs{V, B}
    vs :: V
    βs :: B
    active :: Bool
end

function BinActs(vs :: V, βs :: B; active=false) where {V, B}
    # note: W is not stored! it's just used to initialize alphas.
    BinActs(vs, βs, active)
end

function (ba :: BinActs{V, B})(A) where {V, B}
    if ba.active
        binarize_activations(A, ba.vs, ba.βs)
    else
        A
    end
end

Flux.@functor BinActs (vs, βs)

# from https://github.com/FluxML/Flux.jl/blob/fb4a48f970ba40d0022a7488b48d19cd563867c4/src/layers/conv.jl
# note: does not include activation, that should go before (?)

"""
Standard convolutional layer with ABC-based quantization.
"""
struct ABCCrossCor{W,Z, S,P, U,A, V,B} # that's a lotta parameters!!
    weight::W
    bias::Z
    
    stride::NTuple{S,Int}
    pad::NTuple{P,Int}
    dilation::NTuple{S,Int}
    
    bin_weights :: BinWeights{U, A}
    bin_acts :: BinActs{V, B}
end

function ABCCrossCor(weight::AbstractArray{T,K}, bias::AbstractVector{T},
                 us::AbstractVector{T}, vs::AbstractVector{T}, βs::AbstractVector{T};
              stride = 1, pad = 0, dilation = 1, bin_active=false) where {T, K}
    @assert size(vs) == size(βs)
    
    stride = expand(Val(K-2), stride)
    pad = expand(Val(2*(K-2)), pad)
    dilation = expand(Val(K-2), dilation)
    
    bin_weights = BinWeights(weight, us, active=bin_active) # note: weights is used to initialize αs, not stored
    bin_acts = BinActs(vs, βs, active=bin_active)
    
    ABCCrossCor(weight, bias, stride, pad, dilation, bin_weights, bin_acts)
end

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

function ABCCrossCor(k::NTuple{D,Integer}, ch::Pair{<:Integer,<:Integer}, N::Integer, M::Integer;
    weight_init = Flux.glorot_uniform, bias_init=k -> zeros(Float32, k), 
    us_init=even_us, vs_init=even_us, βs_init=k -> ones(Float32, k), 
    stride = 1, pad = 0, dilation = 1,
    bin_active=false) where D
        
    ABCCrossCor(weight_init(k..., ch...), bias_init(ch[2]),
           us_init(M), vs_init(N), βs_init(N),
           stride = stride, pad = pad, dilation = dilation, bin_active=bin_active)
end

Flux.@functor ABCCrossCor

function (c::ABCCrossCor)(A::AbstractArray)
    b = reshape(c.bias, map(_->1, c.stride)..., :, 1)
    
    W = c.weight
    W̃ = c.bin_weights(W)
    
    Ã = c.bin_acts(A)
    
    cdims = DenseConvDims(Ã, W̃; stride=c.stride, padding=c.pad, dilation=c.dilation)
    #println("W̃: ", typeof(W̃), " ", size(W̃), " Ã: ", typeof(Ã), " ", size(Ã), " b: ", typeof(b), " ", size(b), " dims: ", cdims)
    
    conv(Ã, W̃, cdims) .+ b
end

# Base.show(io :: IO, ::Type{ABCCrossCor}) = print(io, "ABCCrossCor")
function Base.show(io::IO, l::ABCCrossCor)
    print(io, "ABCCrossCor(", size(l.weight)[1:ndims(l.weight)-2])
    print(io, ", ", size(l.weight, ndims(l.weight)-1), "=>", size(l.weight, ndims(l.weight)))
    print(io, ", ", size(l.bin_acts.vs), ", ", size(l.bin_weights.us))
    print(io, ", active=", l.bin_acts.active)

    print(io, ")")
end

function binarize(c :: ABCCrossCor; active=true)
    c.bin_acts.active = active
    c.bin_weights.active = active
    ()
end

