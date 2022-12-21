
using Printf
import .Armon: @i, @indexing_vars, ArmonData, init_test, time_loop

include("reference_data/reference_functions.jl")


function cmp_cpu_with_reference_for(type, test)
    ref_params = get_reference_params(test, type)
    dt, cycles, data = run_armon_reference(ref_params)
    ref_data = ArmonData(type, ref_params.nbcell, ref_params.comm_array_size)
    return compare_with_reference_data(ref_params, dt, cycles, data, ref_data)
end


@testset "Convergence" begin
    @testset "$test with $type" for type in (Float32, Float64), 
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        @test begin
            differences_count = cmp_cpu_with_reference_for(type, test)
            differences_count == 0
        end
    end
end
