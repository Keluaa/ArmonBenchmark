
using Printf
import .Armon: @i, @indexing_vars, ArmonData, init_test, time_loop


function cmp_cpu_with_reference(ref_params)
    dt, cycles, data = run_armon_reference(ref_params)
    ref_data = ArmonData(typeof(ref_params.Dt), ref_params.nbcell, ref_params.comm_array_size)
    return compare_with_reference_data(ref_params, dt, cycles, data, ref_data)
end


@testset "Convergence" begin
    @testset "$test with $type" for type in (Float32, Float64), 
                                    test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
        @test begin
            ref_params = get_reference_params(test, type)
            differences_count = cmp_cpu_with_reference(ref_params)
            differences_count == 0
        end
    end

    @testset "Single boundary condition per pass" begin
        @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium, :Sedov)
            @test begin
                ref_params = get_reference_params(test, Float64; single_comm_per_axis_pass=true)
                differences_count = cmp_cpu_with_reference(ref_params)
                differences_count == 0 
            end skip=true  # TODO: all tests are broken since the indexing is still not 100% correct
        end
    end
end
