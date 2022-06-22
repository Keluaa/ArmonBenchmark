
using Test

include("armon_module_gpu_2D.jl")
using .Armon


domain_bounds = [(100, 100), (50, 100), (100, 50), (43, 137)]
CFD_tests = [:Sod, :Sod_y, :Sod_circ#= , :Bizarrium =#]
transposition = [false, true]
scheme = [:Godunov, :GAD_minmod]
axis_splitting = [:Sequential, :SequentialSym, :Strang]


default_params = (
    ieee_bits = 64,
    euler_projection = true, transpose_dims = false, axis_splitting = :Sequential,
    maxcycle = 100,
    silent = 5, 
    write_output = false,
    use_ccall = false, use_threading = false, 
    use_simd = false
)


combinaisons = Iterators.product(
    domain_bounds,
    CFD_tests,
    transposition,
    scheme,
    axis_splitting
)


function test_armon(use_gpu, domain, test, transpose, s, axis)
    try
        armon(ArmonParameters(;
            test = test,
            scheme = s,
            transpose_dims = transpose,
            nx = domain[1],
            ny = domain[2],
            use_gpu = use_gpu,
            axis_splitting = axis,
            default_params...
        ))
        return true
    catch e
        println("Test failed: ($(domain[1]), $(domain[2])), $test, transpose_dims=$transpose, scheme=$s, splitting=$axis")
        rethrow(e)
    end
end


function test_comp_cpu_gpu(domain, test, transpose, s, axis)
    try
        dt_cpu, _, _ = armon(ArmonParameters(;
            test = test,
            scheme = s,
            transpose_dims = transpose,
            nx = domain[1],
            ny = domain[2],
            use_gpu = false,
            axis_splitting = axis,
            default_params...
        ))
        dt_gpu, _, _ = armon(ArmonParameters(;
            test = test,
            scheme = s,
            transpose_dims = transpose,
            nx = domain[1],
            ny = domain[2],
            use_gpu = true,
            axis_splitting = axis,
            default_params...
        ))
        return isapprox(dt_cpu, dt_gpu; atol=1e-9)
    catch e
        println("Test failed: ($(domain[1]), $(domain[2])), $test, transpose_dims=$transpose, scheme=$s, splitting=$axis")
        rethrow(e)
    end
end


@testset "Test Examples" begin
    @testset "CPU" begin
        for params in combinaisons
            @test test_armon(false, params...)
        end
    end

    @testset "GPU" begin
        for params in combinaisons
            @test test_armon(true, params...)
        end
    end
end


@testset "Comparison CPU/GPU" begin
    for params in combinaisons
        @test test_comp_cpu_gpu(params...)
    end
end
