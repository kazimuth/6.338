using CuArrays, CUDAnative, Test

# cuda-compatible strides()
# is there a better way to implement this?? probably.

"""
function _strides(dims :: NTuple{1})
    return (1,)
end
function _strides(dims :: NTuple{2})
    (d1, d2) = dims
    return (1, d1)
end
function _strides(dims :: NTuple{3})
    (d1, d2, d3) = dims
    d12 = d1 * d2
    return (1, d1, d12)
end
function _strides(dims :: NTuple{4})
    (d1, d2, d3, d4) = dims
    d12 = d1 * d2
    d123 = d12 * d3
    return (1, d1, d12, d123)
end
function _strides(dims :: NTuple{5})
    (d1, d2, d3, d4, d5) = dims
    d12 = d1 * d2
    d123 = d12 * d3
    d1234 = d123 * d4

    return (1, d1, d12, d123, d1234)
end

A = (10, 2, 3, 5, 2)
@test strides(randn(A...)) == _strides(A)
A = (10, 2, 3, 5)
@test strides(randn(A...)) == _strides(A)
A = (10, 2, 3)
@test strides(randn(A...)) == _strides(A)
A = (10, 2)
@test strides(randn(A...)) == _strides(A)
A = (10,)
@test strides(randn(A...)) == _strides(A)

function _ind2sub0(strides :: NTuple{1}, ind0)
    s1 = div(ind0, strides[1])
    (s1,)
end
function _ind2sub0(strides :: NTuple{2}, ind0)
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2)
end
function _ind2sub0(strides :: NTuple{3}, ind0)
    s3 = div(ind0, strides[3]) 
    ind0 -= s3 * strides[3]
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2, s3)
end
function _ind2sub0(strides :: NTuple{4}, ind0)
    s4 = div(ind0, strides[4])
    ind0 -= s4 * strides[4]
    s3 = div(ind0, strides[3]) 
    ind0 -= s3 * strides[3]
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2, s3, s4)
end
function _ind2sub0(strides :: NTuple{5}, ind0)
    s5 = div(ind0, strides[5])
    ind0 -= s5 * strides[5]
    s4 = div(ind0, strides[4])
    ind0 -= s4 * strides[4]
    s3 = div(ind0, strides[3]) 
    ind0 -= s3 * strides[3]
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2, s3, s4, s5)
end

struct Ind2Sub{O, S}
    offsets :: O
    strides :: S
end

function Ind2Sub(ranges :: NTuple{N, UnitRange}) where N
    Ind2Sub(
        (r -> r.start).(ranges),
        _strides((r -> 1 + r.stop - r.start).(ranges))
    )
end

function Base.getindex(i2s :: Ind2Sub, i)
    _ind2sub0(i2s.strides, i - 1) .+ i2s.offsets
end

i2s = Ind2Sub((1:3, 4:5, 6:7))

#@btime i2s[1]

A = (1:5, 3:7, 10:12, 30:33)
c = CartesianIndices(A) # doesn't work on gpu >:/
i2s = Ind2Sub(A)
for i in 1:length(c)
    @test Tuple(c[i]) == i2s[i]
end

#@btime prod((1,2,3))

# Copy a slice of an array to local shared memory, using an entire block.
# 
# NOTE: for this to work you must be only varying the X thread index!!
# NOTE: target should be contiguous!

function kernel_copy!(target, source, source_slices)
    #@inbounds begin
        #source_view = view(source, source_slices...)
        i2s_source = Ind2Sub(source_slices)
        i2s_target = Ind2Sub((s -> 1:s).(size(target)))

        tidx = CUDAnative.threadIdx_x()
        tmax = CUDAnative.blockDim_x()
        
        l = length(target)
    
        i = tidx
        while i <= l
            i2st = i2s_target[i]
            i2ss = i2s_source[i]
            #if tidx == 1
                @cuprintf("%ld (%ld %ld) (%ld %ld)\n", i, i2st[1], i2st[2], i2ss[1], i2ss[2])
            #end
            target[i2s_target[i]...] = source[i2s_source[i]...]
                @cuprintf("%ld (%ld %ld) (%ld %ld): GOOD\n", i, i2st[1], i2st[2], i2ss[1], i2ss[2])
            i += tmax
        end
    #end

    CUDAnative.sync_threads()
    
    return 
end
function kernel_copy_shmem_test!(input, output)
    input_cache = @cuDynamicSharedMem(Float32, (10, 10, 1))

    kernel_copy!(input_cache, input, (1:10, 1:10))
    
    kernel_copy!(output, input_cache, (1:10, 1:10, 1:1))

    return
end

#input = CuArray(randn(Float32, 10, 10))
#output = CuArray{Float32}(undef, 10, 10)
#CuArrays.@sync @cuda threads=32 shmem=(10*10*8) kernel_copy_shmem_test!(input, output)

#function kernel_copy_shmem_test!(input, output)
#    input_cache = @cuDynamicSharedMem(Float32, (10, 10))
#
#    kernel_copy!(input_cache, input, (1:10, 1:10))
#    
#    kernel_copy!(output, input_cache, (1:10, 1:10))
#
#    return
#end
#input = CuArray(randn(Float32, 10, 10))
#output = CuArray{Float32}(undef, 10, 10)
#
#CuArrays.@sync @cuda threads=32 shmem=(10*10*sizeof(Float32)) kernel_copy_shmem_test!(input, output)
"""

# cuda-compatible strides()
# is there a better way to implement this?? probably.

function _strides(dims :: NTuple{1, T}) :: NTuple{1, T} where T <: Integer
    return (T(1),)
end
function _strides(dims :: NTuple{2, T}) :: NTuple{2, T} where T <: Integer
    (d1, d2) = dims
    return (T(1), d1)
end
function _strides(dims :: NTuple{3, T}) :: NTuple{3, T} where T <: Integer
    (d1, d2, d3) = dims
    d12 = d1 * d2
    return (T(1), d1, d12)
end
function _strides(dims :: NTuple{4, T}) :: NTuple{4, T} where T <: Integer
    (d1, d2, d3, d4) = dims
    d12 = d1 * d2
    d123 = d12 * d3
    return (T(1), d1, d12, d123)
end
function _strides(dims :: NTuple{5, T}) :: NTuple{5, T} where T <: Integer
    (d1, d2, d3, d4, d5) = dims
    d12 = d1 * d2
    d123 = d12 * d3
    d1234 = d123 * d4

    return (T(1), d1, d12, d123, d1234)
end

function _ind2sub0(strides :: NTuple{1, T}, ind0 :: T) :: NTuple{1, T} where T <: Integer
    s1 = div(ind0, strides[1])
    (s1,)
end
function _ind2sub0(strides :: NTuple{2, T}, ind0 :: T) :: NTuple{2, T} where T <: Integer
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2)
end
function _ind2sub0(strides :: NTuple{3, T}, ind0 :: T) :: NTuple{3, T} where T <: Integer
    s3 = div(ind0, strides[3]) 
    ind0 -= s3 * strides[3]
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2, s3)
end
function _ind2sub0(strides :: NTuple{4, T}, ind0 :: T) :: NTuple{4, T} where T <: Integer
    s4 = div(ind0, strides[4])
    ind0 -= s4 * strides[4]
    s3 = div(ind0, strides[3]) 
    ind0 -= s3 * strides[3]
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2, s3, s4)
end
function _ind2sub0(strides :: NTuple{5, T}, ind0 :: T) :: NTuple{5, T} where T <: Integer
    s5 = div(ind0, strides[5])
    ind0 -= s5 * strides[5]
    s4 = div(ind0, strides[4])
    ind0 -= s4 * strides[4]
    s3 = div(ind0, strides[3]) 
    ind0 -= s3 * strides[3]
    s2 = div(ind0, strides[2]) 
    ind0 -= s2 * strides[2]
    s1 = div(ind0, strides[1])
    
    (s1, s2, s3, s4, s5)
end

struct Ind2Sub{N, I <: Integer}
    offsets :: NTuple{N, I}
    strides :: NTuple{N, I}
end

function Ind2Sub(ranges :: NTuple{N, UnitRange{I}}) :: Ind2Sub{N, I} where {N, I <: Integer}
    Ind2Sub(
        (r -> r.start).(ranges),
        _strides((r -> 1 + r.stop - r.start).(ranges))
    )
end

function Base.getindex(i2s :: Ind2Sub{N, I}, i :: I) :: NTuple{N, I} where {N, I}
    _ind2sub0(i2s.strides, i - 1) .+ i2s.offsets
end

"""
Copy a slice of an array to local shared memory, using an entire block.

NOTE: for this to work you must be only varying the X thread index!!
NOTE: target should be contiguous!
"""

function kernel_copy!(
        target :: CuDeviceArray{T},
        source :: CuDeviceArray{T, NS},
        source_slices :: NTuple{NS, UnitRange{I}}) :: Nothing where {T, NS, I <: Integer}
    #@inbounds begin
        #source_view = view(source, source_slices...)
        i2s_source = Ind2Sub(source_slices)
        i2s_target = Ind2Sub((s -> 1:s).(size(target)))

        tidx = CUDAnative.threadIdx_x()
        tmax = CUDAnative.blockDim_x()
        
        l = length(target)
        
        i = tidx
      #  while i <= l
      #      i2st = i2s_target[i]
      #      i2ss = i2s_source[i]
      #      #if tidx == 1
      #          @cuprintf("%ld (%ld %ld) (%ld %ld)\n", i, i2st[1], i2st[2], i2ss[1], i2ss[2])
      #      #end
      #      target[i2s_target[i]...] = source[i2s_source[i]...]
      #          @cuprintf("%ld (%ld %ld) (%ld %ld): GOOD\n", i, i2st[1], i2st[2], i2ss[1], i2ss[2])
      #      i += tmax
      #  end
    
        i = tidx
        while i <= l
            target[i2s_target[i]...] = source[i2s_source[i]...]
            i += tmax
        end
    #end

    CUDAnative.sync_threads()
    
    return 
end

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

#CuArrays.@sync @cuda threads=C blocks=(W, H, N) quant_pack_WHCN_PWHN_32_kernel!(quantized_PWHN, input_WHCN, simplequant)
#()

function quant_pack_WHCN_PWHN_32(input_WHCN; quant=simplequant)
    (W,H,C,N) = size(input_WHCN)
    P = ceil(Int64, C/32)
    quantized_PWHN = CuArray{UInt32}(undef, P, W, H, N) # todo allow this as arg
    @cuda threads=C blocks=(W, H, N) quant_pack_WHCN_PWHN_32_kernel!(quantized_PWHN, input_WHCN, simplequant)
    quantized_PWHN
end
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
    
    h_top = h₂ - (div(Hₖ, 2))
    w_left = w₂ - (div(Wₖ, 2))
    
    P = size(input, 1)
    
    plus_1s = 0
        
    ##@cuStaticSharedMem(T::Type, dims) -> CuDeviceArray{T,AS.Shared}
    
    #input_cache = @cuStaticSharedMem(UInt32, (P, Wₖ, Hₖ))
    #input_cache = @cuStaticSharedMem(UInt32, (10, 10, 10))
    input_cache = @cuDynamicSharedMem(UInt32, (P, Wₖ, Hₖ))
    #input_cache = @cuDynamicSharedMem(UInt32, (3, 3, 3))

    #input_cache[1, 1, 1] = 1
    
    kernel_copy!(input_cache,
                 input, (1:P, w_left+1:w_left+Wₖ, h_top+1:h_top+Hₖ, n:n))
        
    #@cuprintf("%ld %ld %ld %ld\n", w₂, h₂, n, o)

    #@inbounds begin
        for p in 1:P
            for x in 1:Hₖ
                for y in 1:Wₖ
                    ## pre-shared:
                    w₁ = w_left + x
                    h₁ = h_top + y
                    #input_value = input[p,w₁,h₁,n]
                    
                    input_value = input_cache[p, x, y]
                    #kernel_value = kernel[p,x,y,o]

                    plus_1s += CUDAnative.popc(reinterpret(Int32, ~xor(input_value, kernel_value)))
                end
            end
        end
    #end
    
    #@cuprintf("%ld %ld %ld %ld: success!\n", w₂, h₂, n, o)
    
    # todo precompute factor?
    minus_1s = (channels_in_kernel - plus_1s)
    output_value = Float32(plus_1s - minus_1s)
    
    output[w₂,h₂,o,n] = output_value
    
    return
end

# input layout: P W₁ H₁ N
# kernel layout: P Wₖ Hₖ O
# output layout: W₂ H₂ O N

# (P: packed in channels, O: unpacked out channels)

# (default conv kernel layout: WHCO; can reuse other quantize op.)

function qconv_32(input, kernel; in_channels=size(kernel, 1) * 32)
    (P, W₁, H₁, N) = size(input)
    (P_, Wₖ, Hₖ, O) = size(kernel)
    @assert P == P_
    @assert in_channels <= P * 32
    @assert (Wₖ % 2) == 1
    @assert (Hₖ % 2) == 1
    
    # spurious: could adjust logic here

    W₂ = W₁ - 2 * div(Wₖ, 2)
    H₂ = H₁ - 2 * div(Hₖ, 2)
    
    output = CuArray{Float32}(undef, W₂, H₂, O, N)
    
    channels_in_kernel = in_channels * size(kernel, 3) * size(kernel, 2)
    
    #@device_code_warntype @cuda threads=O blocks=(W₂, H₂, N) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
    #println(size(output), " ", size(input), " ", size(kernel), " ", channels_in_kernel)
    @cuda threads=O blocks=(W₂, H₂, N) shmem=(P*Wₖ*Hₖ*sizeof(UInt32) * 2) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
    
    output
end

function test_eq_conv(input, kernel)
    input = CuArray(bit_to_float.(simplequant.(input)))
    kernel = CuArray(bit_to_float.(simplequant.(input)))

    input = CuArray(bit_to_float.(simplequant.(randn(Float32, 10, 10, 128, 128))))
    kernel = CuArray(bit_to_float.(simplequant.(randn(Float32, 3, 3, 128, 128))))
    input_quantized = quant_pack_WHCN_PWHN_32(input)
    kernel_quantized = quant_pack_WHCN_PWHN_32(kernel)

    CuArrays.@sync output = qconv_32(input_quantized, kernel_quantized)
    correct = CuArrays.conv(input, kernel, CuArrays.DenseConvDims(input, kernel, flipkernel=true))
    @assert output == correct
end

test_eq_conv(ones(Float32, 3, 3, 32, 1), ones(Float32, 3, 3, 32, 1))