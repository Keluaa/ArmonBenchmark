
import .Armon: @generic_kernel


@generic_kernel function saxpy_1D(α::T, x::V, y::V) where {T, V <: AbstractArray{T}}
    i = @index_1D_lin()
    y[i] += α * x[i]
end


@generic_kernel function saxpy_2D(α::T, x::V, y::V) where {T, V <: AbstractArray{T}}
    i = @index_2D_lin()
    y[i] += α * x[i]
end


error("kernels testing NYI")


@testset "Kernel" begin
    @testset begin
        # CPU / GPU compilation
        # with/without threads + with/without SIMD
    end
end
