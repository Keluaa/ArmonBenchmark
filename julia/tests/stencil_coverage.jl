
cd(@__DIR__)

if !@isdefined(Armon)
    include("../Armon.jl")
end

using .Armon
using Test


include("../kernel_stencil.jl")


lin_idx(p, i, j) = p.index_start + j * p.idx_row + i * p.idx_col


kernel_calls = [
    (p) -> kernel_stencil(p, Armon.acoustic!; 
        type_args=Dict{Symbol, Any}(:V => AbstractArray)),
    (p) -> kernel_stencil(p, Armon.acoustic_GAD!; 
        type_args=Dict{Symbol, Any}(:T => Float64, :LimiterType => Armon.MinmodLimiter)),
    (p) -> kernel_stencil(p, Armon.update_perfect_gas_EOS!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.update_bizarrium_EOS!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.cell_update!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.euler_projection!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.first_order_euler_remap!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.second_order_euler_remap!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => p.row_length, :i_start => lin_idx(p, 0, 1)-p.row_length, :d => 1),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => p.row_length, :i_start => lin_idx(p, p.nx+1, 1)-p.row_length, :d => -1),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => 1, :i_start => lin_idx(p, 1, p.ny+1)-1, :d => -p.row_length),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => 1, :i_start => lin_idx(p, 1, 0)-1, :d => p.row_length),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.read_border_array!;
        args=Dict{Symbol, Any}(:side_length => 1),
        type_args=Dict{Symbol, Any}(:V => AbstractArray)),
    (p) -> kernel_stencil(p, Armon.write_border_array!;
        args=Dict{Symbol, Any}(:side_length => 1),
        type_args=Dict{Symbol, Any}(:V => AbstractArray)),
    (p) -> kernel_stencil(p, Armon.init_test;
        type_args=Dict{Symbol, Any}(:Test => typeof(p.test))),  
]

kernel_names = [
    "acoustic!",
    "acoustic_GAD!",
    "update_perfect_gas_EOS!",
    "update_bizarrium_EOS!",
    "cell_update!",
    "euler_projection!",
    "first_order_euler_remap!",
    "second_order_euler_remap!",
    "boundaryConditions!_left",
    "boundaryConditions!_right",
    "boundaryConditions!_top",
    "boundaryConditions!_bottom",
    "read_border_array!",
    "write_border_array!",
    "init_test",
]


function get_all_kernel_stencils()
    p = ArmonParameters(;
        :ieee_bits => sizeof(Float64)*8,
        :test => :Sod, :scheme => :GAD, :projection => :euler_2nd, :riemann_limiter => :minmod,
        :nghost => 5, :nx => 100, :ny => 100,
        :cfl => 0,
        :silent => 5, :write_output => false, :measure_time => false,
        :use_MPI => false,
        :single_comm_per_axis_pass => false, :async_comms => false
    )

    kernel_stencils = Dict{String, Dict{Symbol, Any}}()
    
    for (kernel_name, kernel_call) in zip(kernel_names, kernel_calls)
        arrays_stencils = kernel_call(p)
        if haskey(kernel_stencils, kernel_name)
            prev_arrays_stencils = kernel_stencils[kernel_name]
            for (array_label, array_stencil) in prev_arrays_stencils
                prev_arrays_stencils[array_label] = union(array_stencil, arrays_stencils[array_label])
            end
        else
            kernel_stencils[kernel_name] = arrays_stencils
        end
    end

    return kernel_stencils
end


function imprinting_data(params::ArmonParameters{T}) where T
    dims = (params.row_length, params.col_length)
    offsets = (-params.nghost, -params.nghost)
    return Armon.ArmonData{ImprintingArray{T, 2}}(
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}(dims, offsets),
        ImprintingArray{T, 2}((params.comm_array_size, 1))
    )
end


function build_kernel_hiearchy()
    params = ArmonParameters(;
        :ieee_bits => sizeof(Float64)*8,
        :test => :Sod, :scheme => :GAD, :projection => :euler_2nd, :riemann_limiter => :minmod,
        :nghost => 5, :nx => 1, :ny => 1,
        :cfl => 0,
        :silent => 5, :write_output => false, :measure_time => false,
        :use_MPI => false,
        :single_comm_per_axis_pass => false, :async_comms => false,
        :use_simd => false, :use_threading => false,
    )
    host_array = Vector{Float64}()

    dt = params.Dt

    steps = [
        (d, drs) -> (Armon.dtCFL, Armon.dtCFL(params, d, dt)),
        (d, drs) -> (Armon.update_EOS!, Armon.update_EOS!(params, d, Armon.full_domain(drs), :full)),
        (d, drs) -> (Armon.boundaryConditions!, Armon.boundaryConditions!(params, d, host_array, params.current_axis)),
        (d, drs) -> (Armon.numericalFluxes!, Armon.numericalFluxes!(params, d, dt, drs, :full)),
        (d, drs) -> (Armon.cellUpdate!, Armon.cellUpdate!(params, d, dt, drs)),
        (d, drs) -> (Armon.projection_remap!, Armon.projection_remap!(params, d, host_array, dt)),
    ]

    axes_stencils = Dict{Armon.Axis, Dict{Symbol, Armon.ArmonData}}()

    for axis in (Armon.X_axis, Armon.Y_axis)
        steps_stencil = Dict{Symbol, Armon.ArmonData}()
        axes_stencils[axis] = steps_stencil

        Armon.update_axis_parameters(params, axis)
        domain_ranges = Armon.compute_domain_ranges(params)

        for step in steps
            data = imprinting_data(params)
            func, _ = step(data, domain_ranges)
            step_name = typeof(func).name.name
            steps_stencil[step_name] = data
        end
    end

    for (axis, steps_stencil) in axes_stencils
        println("\n$axis:")
        for (step, data) in steps_stencil
            println("  $step:")
            for field in fieldnames(typeof(data))
                array = getproperty(data, field)
                !was_accessed(array) && continue
                println("     $field:\t", array)
            end
        end
    end

    axes_stencils
end


function print_data_stencil(data::Armon.ArmonData{<:ImprintingArray}; indent="")
    for field in fieldnames(typeof(data))
        array = getproperty(data, field)
        !was_accessed(array) && continue
        println("$indent$field:\t", array)
    end
end


function compute_kernel_ranges()
    params = ArmonParameters(;
        :ieee_bits => sizeof(Float64)*8,
        :test => :Sod, :scheme => :GAD, :projection => :euler_2nd, :riemann_limiter => :minmod,
        :nghost => 5, :nx => 1, :ny => 1,
        :cfl => 0,
        :silent => 5, :write_output => false, :measure_time => false,
        :use_MPI => false,
        :single_comm_per_axis_pass => false, :async_comms => false,
        :use_simd => false, :use_threading => false,
    )
    host_array = Vector{Float64}()

    dt = params.Dt

    steps = [
        # (d, drs) -> (Armon.dtCFL, Armon.dtCFL(params, d, dt)),
        (d, drs) -> (Armon.update_EOS!, Armon.update_EOS!(params, d, Armon.full_domain(drs), :full)),
        (d, drs) -> (Armon.boundaryConditions!, Armon.boundaryConditions!(params, d, host_array, params.current_axis)),
        (d, drs) -> (Armon.numericalFluxes!, Armon.numericalFluxes!(params, d, dt, drs, :full)),
        (d, drs) -> (Armon.cellUpdate!, Armon.cellUpdate!(params, d, dt, drs)),
        (d, drs) -> (Armon.projection_remap!, Armon.projection_remap!(params, d, host_array, dt)),
    ]

    for axis in (Armon.X_axis, Armon.Y_axis)
        Armon.update_axis_parameters(params, axis)
        domain_ranges = Armon.compute_domain_ranges(params)

        steps_range = Dict{Symbol, MVector{2, Tuple{Int, Int}}}()
        current_stencils = nothing

        println("\n$axis:")

        for step in Iterators.reverse(steps)
            data = imprinting_data(params)
            func, _ = step(data, domain_ranges)
            step_name = typeof(func).name.name

            step_range = MVector{2, Tuple{Int, Int}}((0, 0), (0, 0))

            if isnothing(current_stencils)
                # Init for the last step
                steps_range[step_name] = step_range
                current_stencils = data
                print_data_stencil(data; indent="    ")
                continue
            end

            for field in fieldnames(typeof(data))
                step_stencil = getproperty(data, field)
                current_stencil = getproperty(current_stencils, field)

                if was_read(current_stencil) && was_written(step_stencil)
                    # Extend the range of the current step so that it writes to all cells which are read
                    for d in Base.OneTo(2)
                        get_range = current_stencil.min_get[d]:current_stencil.max_get[d]

                        f_offset, l_offset = step_range[d]
                        set_range = (step_stencil.min_set[d] + f_offset):(step_stencil.max_set[d] + l_offset)

                        f_offset += min(0, first(get_range) - first(set_range))
                        l_offset += max(0, last(get_range) - last(set_range))
                        step_range[d] = (f_offset, l_offset)
                    end
                end
            end

            for field in fieldnames(typeof(data))
                step_stencil = getproperty(data, field)
                current_stencil = getproperty(current_stencils, field)

                f_offsets = first.(step_range)
                l_offsets = last.(step_range)
                step_stencil.min_get .+= f_offsets
                step_stencil.max_get .+= l_offsets
                step_stencil.min_set .+= f_offsets
                step_stencil.max_set .+= l_offsets

                # TODO: sets should cancel gets

                union!(current_stencil, step_stencil)
            end

            steps_range[step_name] = step_range

            println("  $step_name:")
            print_data_stencil(data; indent="    ")
        end

        display(steps_range)
        println()
    end
end
