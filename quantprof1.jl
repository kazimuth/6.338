println("~~ julia booted ~~")

using CuArrays, CUDAnative, BenchmarkTools, Test, CUDAdrv

println("~~ imports booted ~~")

simplequant(x) = x > 0

function quant_pack_WHCN_PWHN_32!(quantized, input, f)
    tidx = threadIdx()
    bidx = blockIdx()
    
    C = tidx.x
    
    W = bidx.x
    H = bidx.y
    N = bidx.z
    
    P = div(C - 1, 32) + 1
    @inbounds begin
        ballot = vote_ballot(f(input[W,H,C,N]))
        if (C - 1) % 32 == 0
            quantized[P,W,H,N] = ballot
        end
    end
    return
end

(W,H,C,N) = (10, 10, 128, 128)
P = ceil(Int64, C/32)

input_WHCN = CuArray(randn(Float32, W, H, C, N))
quantized_PWHN = CuArray(zeros(UInt32, P, W, H, N))

CuArrays.@sync @cuda threads=C blocks=(W, H, N) quant_pack_WHCN_PWHN_32!(quantized_PWHN, input_WHCN, simplequant)

println("~~ benching ~~")

CUDAdrv.@profile begin
    @btime CuArrays.@sync @cuda threads=C blocks=(W, H, N) quant_pack_WHCN_PWHN_32!(quantized_PWHN, input_WHCN, simplequant)
end

println("~~ done. ~~")