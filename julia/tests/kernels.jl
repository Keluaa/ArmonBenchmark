
import .Armon: @generic_kernel, @threads, @batch, @kernel, @index
import .Armon: KernelWithThreading, KernelWithoutThreading, KernelWithSIMD, KernelWithoutSIMD


@generic_kernel function saxpy_1D(α::T, x::V, y::V) where {T, V <: AbstractArray{T}}
    i = @index_1D_lin()
    y[i] += α * x[i]
end


@generic_kernel function saxpy_2D(α::T, x::V, y::V) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    y[i] += α * x[i]
end


@warn "kernels testing NYI"


@testset "Kernel" begin
    @testset begin
        # TODO: CPU / GPU compilation
        # TODO: with/without threads + with/without SIMD
        # TODO: equivalence 1D/2D (need correct indexing)
        # TODO: equivalence CPU/CPU+simd/CPU+threads/CPU+simd+threads/GPU
    end
end
