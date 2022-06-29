
using Test

include("armon_1D.jl")
using .Armon


cells_list = [10000, 12345]
CFD_tests = [:Sod#= , :Bizarrium =#]
scheme = [:Godunov, :GAD_minmod]


default_params = (
    ieee_bits = 64,
    euler_projection = true,
    maxcycle = 100,
    silent = 5, 
    write_output = false,
    use_ccall = false, use_threading = false, 
    use_simd = false
)


combinaisons = Iterators.product(
    cells_list,
    CFD_tests,
    scheme
)


function test_armon(use_gpu, cells, test, s)
    try
        armon(ArmonParameters(;
            test = test,
            scheme = s,
            nbcell = cells,
            use_gpu = use_gpu,
            default_params...
        ))
        return true
    catch e
        println("Test failed: $(cells), $test, scheme=$s")
        rethrow(e)
    end
end


function test_comp_cpu_gpu(cells, test, s)
    try
        dt_cpu, _, _ = armon(ArmonParameters(;
            test = test,
            scheme = s,
            nbcell = cells,
            use_gpu = false,
            default_params...
        ))
        dt_gpu, _, _ = armon(ArmonParameters(;
            test = test,
            scheme = s,
            nbcell = cells,
            use_gpu = true,
            default_params...
        ))
        return isapprox(dt_cpu, dt_gpu; atol=1e-9)
    catch e
        println("Test failed: ($(cells)), $test, scheme=$s")
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
