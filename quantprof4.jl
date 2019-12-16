println("init")
using CuArrays, CUDAnative, CUDAdrv
println("imports complete")

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
    @inbounds begin
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
    end

    CUDAnative.sync_threads()
    
    return 
end

const N_var = 4

function qconv_32_kernel!(output, input, kernel, channels_in_kernel)
    bidx = blockIdx()
    tidx = threadIdx()

    w₂ = bidx.x
    h₂ = bidx.y
    n_o = (bidx.z - 1) * N_var
    n_min = n_o + 1
    n_max = n_o + N_var

    o = tidx.x

    # PWHO
    Hₖ = size(kernel, 3)
    Wₖ = size(kernel, 2)
    
    h_top = h₂ - (div(Hₖ, 2))
    w_left = w₂ - (div(Wₖ, 2))
    
    P = size(input, 1)
    
    input_cache = @cuDynamicSharedMem(UInt32, (P, Wₖ, Hₖ))

    @inbounds begin
        for n_i in 1:N_var
            plus_1s = 0
            n = n_o + n_i
            kernel_copy!(input_cache,
                 input, (1:P, w_left+1:w_left+Wₖ, h_top+1:h_top+Hₖ, n:n))
            for p in 1:P
                for x in 1:Hₖ
                    for y in 1:Wₖ
                        input_value = input_cache[p, x, y]
                        kernel_value = kernel[p, x, y, o]

                        plus_1s += CUDAnative.popc(reinterpret(Int32, ~xor(input_value, kernel_value)))
                    end
                end
            end
            minus_1s = (channels_in_kernel - plus_1s)
            output_value = Float32(plus_1s - minus_1s)
            output[w₂,h₂,o,n_o + n_i] = output_value
        end
    end
    
    #@cuprintf("%ld %ld %ld %ld: success!\n", w₂, h₂, n, o)
    
    return
end

# input layout: P W₁ H₁ N
# kernel layout: P Wₖ Hₖ O
# output layout: W₂ H₂ O N
(P, W₁, H₁, N) = (div(128, 32), 100, 100, 32)
(Wₖ, Hₖ, O) = (3, 3, 128)

W₂ = W₁ - 2 * div(Wₖ, 2)
H₂ = H₁ - 2 * div(Hₖ, 2)

input = CuArray(rand(UInt32, P, W₁, H₁, N))
kernel = CuArray(rand(UInt32, P, Wₖ, Hₖ, O))
output = CuArray{Float32}(undef, W₂, H₂, O, N)

in_channels = size(kernel, 1) * 32
channels_in_kernel = in_channels * size(kernel, 3) * size(kernel, 2)

println("warmup")
CuArrays.@sync @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32)*N_var + 10) qconv_32_kernel!(output, input, kernel, channels_in_kernel)

println("import bench")
#CUDAdrv.@profile @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32) + 100) qconv_32_kernel!(output, input, kernel, channels_in_kernel)

using BenchmarkTools

println("bench")
@btime CuArrays.@sync @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32)*N_var + 10) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
@btime CuArrays.@sync @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32)*N_var + 10) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
@btime CuArrays.@sync @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32)*N_var + 10) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
@btime CuArrays.@sync @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32)*N_var + 10) qconv_32_kernel!(output, input, kernel, channels_in_kernel)
@btime CuArrays.@sync @cuda threads=O blocks=(W₂, H₂, div(N, N_var)) shmem=(P*Wₖ*Hₖ*sizeof(UInt32)*N_var + 10) qconv_32_kernel!(output, input, kernel, channels_in_kernel)