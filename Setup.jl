__precompile__()

module Setup

    using Zygote
    using CuArrays
    using CUDAnative
    using CUDAdrv
    using NNlib
    using Test
    using BenchmarkTools
    using Flux
    using Plots
    using Images
    using Colors
    using Flux.Data.MNIST
    using Statistics
    using Base.Iterators: repeated, partition
    using Printf
    using BSON

    Zygote.@adjoint gpu(a :: Array) = gpu(a), a̅ -> (cpu(a̅),)
    Zygote.refresh()

end