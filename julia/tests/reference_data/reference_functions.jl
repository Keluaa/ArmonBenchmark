
using Printf
using Test
import .Armon: @i, @indexing_vars, ArmonData, init_test, time_loop, TestCase


function get_reference_params(test::Symbol, type::Type)
    ArmonParameters(; 
        ieee_bits=sizeof(type)*8,
        test, scheme=:GAD, projection=:euler_2nd, riemann_limiter=:minmod,
        nghost=5, nx=100, ny=100, 
        cfl=0,
        maxcycle=1000, maxtime=0,
        silent=5, write_output=false, measure_time=false,
        use_MPI=false,
        single_comm_per_axis_pass=false, async_comms=false)
end


function run_armon_reference(ref_params::ArmonParameters{T}) where T
    data = ArmonData(T, ref_params.nbcell, ref_params.comm_array_size)
    init_test(ref_params, data)
    dt, cycles, _, _ = time_loop(ref_params, data, data)
    return dt, cycles, data
end


function get_reference_data_file_name(test::TestCase, type::Type)
    test_name = typeof(test).name.name
    return joinpath(@__DIR__, "ref_$(test_name)_$(sizeof(type)*8)bits.csv")
end


function write_reference_data(ref_params::ArmonParameters{T}, ref_file::IO, ref_data::ArmonData{V}, 
        dt::T, cycles::Int) where {T, V <: AbstractArray{T}}
    (; nx, ny) = ref_params
    @indexing_vars(ref_params)

    @printf(ref_file, "%.15f, %d\n", dt, cycles)
    
    vars_to_write = [ref_data.x, ref_data.y, ref_data.rho, ref_data.umat, ref_data.vmat, ref_data.pmat]
    
    for j in 1:ny
        for i in 1:nx
            idx = @i(i, j)
            @printf(ref_file, "%.15f", vars_to_write[1][idx])
            for var in vars_to_write[2:end]
                @printf(ref_file, ", %.15f", var[idx])
            end
            println(ref_file)
        end
        println(ref_file)
    end
end


function read_reference_data(ref_params::ArmonParameters{T}, ref_file::IO, ref_data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; nx, ny) = ref_params
    @indexing_vars(ref_params)

    ref_dt = parse(T, readuntil(ref_file, ','))
    ref_cycles = parse(Int, readuntil(ref_file, '\n'))

    vars_to_read = [ref_data.x, ref_data.y, ref_data.rho, ref_data.umat, ref_data.vmat, ref_data.pmat]
    
    for j in 1:ny
        for i in 1:nx
            idx = @i(i, j)
            for var in vars_to_read[1:end-1]
                var[idx] = parse(T, readuntil(ref_file, ','))
            end
            vars_to_read[end][idx] = parse(T, readuntil(ref_file, '\n'))
        end
    end

    return ref_dt, ref_cycles
end


function compare_with_reference_data(ref_params::ArmonParameters{T}, dt::T, cycles::Int, data::ArmonData{V}, ref_data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; nx, ny) = ref_params
    @indexing_vars(ref_params)
    ref_file_name = get_reference_data_file_name(ref_params.test, T)

    open(ref_file_name, "r") do ref_file
        ref_dt, ref_cycles = read_reference_data(ref_params, ref_file, ref_data)
        @test ref_dt ≈ dt atol=1e-13
        @test ref_cycles == cycles
    end

    differences_count = 0
    fields_to_compare = (:x, :y, :rho, :umat, :vmat, :pmat)
    for j in 1:ny
        row_range = @i(1,j):@i(nx,j)
        for field in fields_to_compare
            ref_row = @view getfield(ref_data, field)[row_range]
            cur_row = @view getfield(data, field)[row_range]
            diff_count = sum(.~ isapprox.(ref_row, cur_row; atol=1e-13))
            differences_count += diff_count
            (diff_count > 0) && @debug "Row $j has $diff_count differences in '$field' with the reference"
        end
    end

    return differences_count
end
