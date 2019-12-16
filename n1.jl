using CuArrays, CUDAnative, BenchmarkTools, Test

using Flux

# which order does ballot go in?
# indexed from LSB to MSB
# inactive threads get 0

function gpu_test_ballot!(o, b)
    index = threadIdx().x
    ballot = vote_ballot(b[index])
    if (index - 1) % 32 == 0
        o[1] = ballot
    end
    return
end

b = CuArrays.fill(true, 32)
CuArrays.allowscalar(true)
b[1] = false
CuArrays.allowscalar(false)
o = CuArrays.fill(UInt32(0), 1)
@cuda threads=32 gpu_test_ballot!(o, b)
println(o)
@cuda threads=4 gpu_test_ballot!(o, b)
println(o)

# which order does ballot go in?
# indexed from LSB to MSB
# inactive threads get 0

function gpu_test_ballot_64!(o, b)
    #@inbounds begin
        index = threadIdx().x    # this example only requires linear indexing, so just use `x`
        ballot_low = vote_ballot(b[index])
        ballot_high = vote_ballot(b[index + 32])
        if (index - 1) % 32 == 0
            o[1] = UInt64(ballot_low) | (UInt64(ballot_high) << 32)
        end
    #end
    return
end

b = CuArrays.fill(true, 64)
CuArrays.allowscalar(true)
b[1] = false
CuArrays.allowscalar(false)
o = CuArrays.fill(UInt64(0), 1)
CuArrays.@sync @cuda threads=32 gpu_test_ballot_64!(o, b)
println(o)
CuArrays.@sync @cuda threads=4 gpu_test_ballot_64!(o, b)
println(o)

# hm, uneven things on the end there.

# todo: how does julia's multi-dimensional indexing computation work? 
# todo: array-walking helpers
# todo: helper functions for computing sizes of output arrays (ceil to multiples of 64)
# todo: 32 and 64 bit versions?
# todo: iterate channels
# todo: maximize occupancy

simplequant(x) = x > 0
bit_to_float(b :: Bool) = if b 1.0f0 else -1.0f0 end

# layout:

# todo: optimize this
# todo: support more than 1024 channels

# thread x: C index

# block x: W
# block y: H
# block z: N

# note: P is packed C

# todo: type for output w/ channel annotation

function quant_pack_WHCN_PWHN_32_kernel!(quantized, input, f)
    tidx = threadIdx()
    bidx = blockIdx()
    
    c = tidx.x
    
    w = bidx.x
    h = bidx.y
    n = bidx.z
    
    p = div(c - 1, 32) + 1
    @inbounds begin
        ballot = vote_ballot(f(input[w,h,c,n]))
        if (c - 1) % 32 == 0
            quantized[p,w,h,n] = ballot
        end
    end
    return
end

(W,H,C,N) = (10, 10, 128, 128)
P = ceil(Int64, C/32)

input_WHCN = CuArray(randn(Float32, W, H, C, N))
quantized_PWHN = CuArray(zeros(UInt32, P, W, H, N))

CuArrays.@sync @cuda threads=C blocks=(W, H, N) quant_pack_WHCN_PWHN_32_kernel!(quantized_PWHN, input_WHCN, simplequant)
()

function quant_pack_WHCN_PWHN_32(input_WHCN; quant=simplequant)
    (W,H,C,N) = size(input_WHCN)
    P = ceil(Int64, C/32)
    quantized_PWHN = CuArray{UInt32}(undef, P, W, H, N) # todo allow this as arg
    @cuda threads=C blocks=(W, H, N) quant_pack_WHCN_PWHN_32_kernel!(quantized_PWHN, input_WHCN, simplequant)
    quantized_PWHN
end

@test quant_pack_WHCN_PWHN_32(input_WHCN) == quantized_PWHN

conv_ = gpu(Conv((3, 3), 128=>128))

function unpack_PWHN_WHCN_32_kernel!(unquantized, quantized)
    tidx = threadIdx()
    bidx = blockIdx()
    
    c = tidx.x
    
    w = bidx.x
    h = bidx.y
    n = bidx.z
    
    p = div(c - 1, 32) + 1
    
    @inbounds begin
        q = quantized[p,w,h,n]
        unquantized[w,h,c,n] = ((q >> rem(c - 1, 32)) & 1) == 1
    end
    return
end
unquantized_WHCN = CuArray(zeros(Bool, W, H, C, N))
CuArrays.@sync @cuda threads=C blocks=(W, H, N) unpack_PWHN_WHCN_32_kernel!(unquantized_WHCN, quantized_PWHN)

@test simplequant.(input_WHCN) == unquantized_WHCN
@test bit_to_float.(simplequant.(input_WHCN)) == bit_to_float.(unquantized_WHCN)

()

function unpack_PWHN_WHCN_32(quantized; channels=size(quantized, 1) * 32)
    (W,H,P,N) = size(input_WHCN)
    C = channels
    @assert C <= P * 32
    unquantized_WHCN = CuArray{Bool}(undef, W, H, C, N)

    @cuda threads=C blocks=(W, H, N) unpack_PWHN_WHCN_32_kernel!(unquantized_WHCN, quantized_PWHN)
    unquantized_WHCN
end

@test unpack_PWHN_WHCN_32(quantized_PWHN) == unquantized_WHCN


# https://github.com/FluxML/Flux.jl/blob/fb4a48f970ba40d0022a7488b48d19cd563867c4/src/layers/conv.jl

# todo activation
# todo padding
# todo shared memory
# todo multipliers?
# todo bias
# todo teach ABCConv about 1-padding??
# -> no; just use whatever value it gives for 0??
# todo fuse padding (into above?)
# todo dispatch functions

function qconv_32_kernel!(output, input, kernel, channels_in_kernel)
    bidx = blockIdx()
    tidx = threadIdx()

    w₂ = bidx.x
    h₂ = bidx.y
    n = bidx.z

    o = tidx.x

    # PWHO
    Hₖ = size(kernel, 3)
    Wₖ = size(kernel, 2)
    
    h_top = h₂ - (div(Hₖ, 2) + 1)
    w_left = w₂ - (div(Wₖ, 2) + 1)
    
    P = size(input, 1)
    
    plus_1s = 0
    
    for p in 1:P
        for x in 1:Hₖ
            for y in 1:Wₖ
                w₁ = w_left + x
                h₁ = h_top + y
                input_value = input[p,w₁,h₁,o]
                kernel_value = kernel[p,x,y,o]
                #plus_1s += popc(~xor(in_value, kernel_value))
            end
        end
    end
    
    # todo precompute factor?
    minus_1s = (channels_in_kernel - plus_1s)
    output_value = Float32(plus_1s - minus_1s)
    
    output[w₂,h₂,o,n] = output_value
    
    return
end
# input layout: PWHN
# kernel layout: PWHO
# output layout: W₂H₂ON

# (P: packed in channels, O: unpacked out channels)

# (default conv kernel layout: WHCO; can reuse other quantize op.)

function qconv_32(input, kernel; in_channels=size(kernel, 1) * 32)
    (P, W₁, H₁, N) = size(input)
    (P_, Wₖ, Hₖ, O) = size(kernel)
    @assert P == P_
    @assert in_channels <= P * 32
    
    W₂ = W₁ - div(Wₖ, 2)
    H₂ = H₁ - div(Hₖ, 2)
    
    output = CuArray{Float32}(undef, W₂, H₂, O, N)
        
    channels_in_kernel = in_channels * size(kernel, 3) * size(kernel, 2)
    
    @cuda threads=O blocks=(W₂, H₂, N) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
    
    output
end

# make kernel consistent w/ results

weightq = bit_to_float.(simplequant.(conv_.weight))
bq = (x -> 0.0f0).(conv_.bias)
conv_ = gpu(Conv(weightq, bq, identity, stride=conv_.stride, pad=conv_.pad, dilation=conv_.dilation))

unquantizedf_WHCN = bit_to_float.(unquantized_WHCN)
correct = conv_(unquantizedf_WHCN)

quantized_kernel_PWHO = quant_pack_WHCN_PWHN_32(conv_.weight)
output = qconv_32(quantized_PWHN, quantized_kernel_PWHO)

()




