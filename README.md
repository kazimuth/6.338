# 18.337 / 6.338 
This is my (James Gilles)'s class notes and projects for MIT 6.338 in Fall 2019.

## ABCNets implementation
This repository contains a [Flux.jl](https://github.com/FluxML/Flux.jl/) implementation of the algorithm described in "Towards Accurate Binary Convolutional Networks" (https://arxiv.org/abs/1711.11294). To use it, simply copy `ABCNets.jl` to your project; make sure you have Flux.jl, Zygote.jl, and CuArrays.jl.

`ABCNets.jl` provides a Flux.jl layer: `ABCCrossCor`. This mimics the `CrossCor` layer, and implements a standard cross-correlation operation (what is usually termed "convolution" in the deep learning literature). In addition, once enabled with `binarize`, its weights and input activations are quantized, such that they're represented by linear combinations of +1/-1 valued bases. The full algorithm is described in the paper, and also in my [writeup](https://github.com/kazimuth/6.338/blob/master/writeup/writeup.pdf).

`ABCCrossCor` has two constructors:

```julia
function ABCCrossCor(k::NTuple{D,Integer}, ch::Pair{<:Integer,<:Integer}, N::Integer, M::Integer;
    weight_init = Flux.glorot_uniform, bias_init=k -> zeros(Float32, k), 
    us_init=even_us, vs_init=even_us, βs_init=k -> ones(Float32, k), 
    stride = 1, pad = 0, dilation = 1,
    bin_active=false) where D

function ABCCrossCor(weight::AbstractArray{T,K}, bias::AbstractVector{T},
            us::AbstractVector{T}, vs::AbstractVector{T}, βs::AbstractVector{T})
            stride = 1, pad = 0, dilation = 1, bin_active=false) where {T, K}
```

The first generates a randomly-initialized layer using the provided dimensions and initializers. The second allows construction directly via the given parameters.

After construction, `ABCCrossCor` will behave like a standard `CrossCor` layer. You can train it as usual and move it to the GPU. In addition, you can call the `binarize` function:

```julia
binarize(c :: ABCCrossCor; active=true)
```

This activates the quantization operators. This should probably be done after some pre-training without quantization. For the quantization to work well you should make sure there's a `BatchNorm` followed by a `ReLU` before the `ABCCrossCor`. `CrossCor` has no fused activation; you should apply batchnorm first.

Quantization can be disabled by calling `binarize(c, active=false)`

Additional development is present in the notebooks `FinalProject.ipynb` and `Eval.ipynb`, including in-progress CUDA kernels that compute convolution using bitwise operations instead of floating point. Unfortunately they're not any faster than floating point right now, so they're not integrated into `ABCNets.jl` yet.

