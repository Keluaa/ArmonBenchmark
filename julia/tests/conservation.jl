
import .Armon: @i, @indexing_vars, ArmonData, init_test, time_loop, conservation_vars

include("reference_data/reference_functions.jl")


@testset "Conservation" begin
    @testset "$test" for test in (:Sod, :Sod_y, :Sod_circ)
        ref_params = get_reference_params(test, Float64)
        ref_params.maxcycle = 10000
        ref_params.maxtime = 10000

        data = ArmonData(Float64, ref_params.nbcell, ref_params.comm_array_size)
        init_test(ref_params, data)

        init_mass, init_energy = conservation_vars(ref_params, data)
        time_loop(ref_params, data, data)
        end_mass, end_energy = conservation_vars(ref_params, data)

        @test   init_mass ≈ end_mass    atol=1e-12
        @test init_energy ≈ end_energy  atol=1e-12
    end
end
