
using Printf
using Test
import .Armon: @i, @indexing_vars, ArmonData, TestCase, init_test, time_loop 
import .Armon: write_data_to_file, read_data_from_file


function get_reference_params(test::Symbol, type::Type; overriden_options...)
    ref_options = Dict(
        :ieee_bits => sizeof(type)*8,
        :test => test, :scheme => :GAD, :projection => :euler_2nd, :riemann_limiter => :minmod,
        :nghost => 5, :nx => 100, :ny => 100,
        :cfl => 0,
        :maxcycle => 1000, :maxtime => 0,  # Always run until reaching the default maximum time of the test
        :silent => 5, :write_output => false, :measure_time => false,
        :use_MPI => false,
        :single_comm_per_axis_pass => false, :async_comms => false
    )
    merge!(ref_options, overriden_options)
    ArmonParameters(; ref_options...)
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

    @printf(ref_file, "%#.15g, %d\n", dt, cycles)

    col_range = 1:ny
    row_range = 1:nx
    ref_params.output_precision = 15
    write_data_to_file(ref_params, ref_data, col_range, row_range, ref_file)
end


function read_reference_data(ref_params::ArmonParameters{T}, ref_file::IO, 
        ref_data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; nx, ny) = ref_params
    @indexing_vars(ref_params)

    ref_dt = parse(T, readuntil(ref_file, ','))
    ref_cycles = parse(Int, readuntil(ref_file, '\n'))

    col_range = 1:ny
    row_range = 1:nx
    read_data_from_file(ref_params, ref_data, col_range, row_range, ref_file)

    return ref_dt, ref_cycles
end


function compare_with_reference_data(ref_params::ArmonParameters{T}, dt::T, cycles::Int, 
        data::ArmonData{V}, ref_data::ArmonData{V}) where {T, V <: AbstractArray{T}}
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
