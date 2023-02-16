
julia_script_path = joinpath(@__DIR__, "run_julia.jl")


struct JuliaParams <: BackendParams
    options::Tuple{Vector{String}, String, String}
    jl_places::String
    jl_proc_bind::String
    threads::Int
    ieee_bits::Int
    block_size::Int
    use_simd::Int
    dimension::Int
    async_comms::Bool
end


backend_disp_name(::JuliaParams) = "Julia"


function run_backend_msg(measure::MeasureParams, julia_params::JuliaParams, cluster_params::ClusterParams)
    """Running Julia with:
    - $(julia_params.threads) threads
    - threads binding: $(julia_params.jl_proc_bind), places: $(julia_params.jl_places)
    - $(julia_params.use_simd == 1 ? "with" : "without") SIMD
    - $(julia_params.dimension)D
    - $(julia_params.async_comms ? "a" : "")synchronous communications
    - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
    - with $(cluster_params.processes) processes on $(cluster_params.node_count) nodes ($(cluster_params.distribution) distribution)
   """
end


function iter_combinaisons(measure::MeasureParams, threads, ::Val{Julia})
    Iterators.map(
        params->JuliaParams(params...),
        Iterators.product(
            measure.armon_params,
            measure.jl_places,
            measure.jl_proc_bind,
            threads,
            measure.ieee_bits,
            measure.block_sizes,
            measure.use_simd,
            measure.dimension,
            measure.async_comms,
        )
    )
end


function run_backend(measure::MeasureParams, params::JuliaParams, cluster_params::ClusterParams, base_file_name::String)
    armon_options = [
        "julia", "-t", params.threads
    ]
    append!(armon_options, isempty(measure.node) ? julia_options_no_cluster : julia_options)
    push!(armon_options, julia_script_path)

    if measure.device == CUDA
        append!(armon_options, ["--gpu", "CUDA"])
    elseif measure.device == ROCM
        append!(armon_options, ["--gpu", "ROCM"])
    else
        # no option needed for CPU
    end

    if params.dimension == 1
        cells_list = measure.cells_list
    else
        cells_list = measure.domain_list
    end

    if measure.cst_cells_per_process
        # Scale the cells by the number of processes
        if params.dimension == 1
            cells_list .*= cluster_params.processes
        else
            # We need to distribute the factor along each axis, while keeping the divisibility of 
            # the cells count, since it will be divided by the number of processes along each axis.
            # Therefore we make the new values multiples of 64, but this is still not perfect.
            scale_factor = cluster_params.processes^(1/params.dimension)
            cells_list = cells_list .* scale_factor
            cells_list .-= [cells .% 64 for cells in cells_list]
            cells_list = Vector{Int}[convert.(Int, cells) for cells in cells_list]

            if any(any(cells .≤ 0) for cells in cells_list)
                error("Cannot scale the cell list by the number of processes: $cells_list")
            end
        end
    end

    if params.dimension == 1
        cells_list_str = join(cells_list, ',')
    else
        cells_list_str = join([join(string.(cells), ',') for cells in cells_list], ';')
    end

    append!(armon_options, armon_base_options)
    append!(armon_options, [
        "--dim", params.dimension,
        "--block-size", 256,
        "--use-simd", params.use_simd,
        "--ieee", 64,
        "--tests", join(measure.tests_list, ','),
        "--cells-list", cells_list_str,
        "--threads-places", params.jl_places,
        "--threads-proc-bind", params.jl_proc_bind,
        "--data-file", base_file_name,
        "--gnuplot-script", measure.gnuplot_script,
        "--repeats", measure.repeats,
        "--verbose", (measure.verbose ? 2 : 5),
        "--gnuplot-hist-script", measure.gnuplot_hist_script,
        "--time-histogram", measure.time_histogram,
        "--time-MPI-graph", measure.time_MPI_plot,
        "--gnuplot-MPI-script", measure.gnuplot_MPI_script,
        "--use-mpi", measure.use_MPI,
        "--limit-to-mem", measure.limit_to_max_mem,
        "--track-energy", measure.track_energy
    ])

    if params.dimension > 1
        append!(armon_options, [
            "--async-comms", params.async_comms,
            "--splitting", join(measure.axis_splitting, ','),
            "--flat-dims", measure.flatten_time_dims
        ])

        if isnothing(measure.process_grid_ratios)
            push!(armon_options, "--proc-grid", join([
                join(string.(process_grid), ',') 
                for process_grid in measure.process_grids
            ], ';'))
        else
            push!(armon_options, "--proc-grid-ratio", join([
                join(string.(ratio), ',')
                for ratio in measure.process_grid_ratios
                if check_ratio_for_grid(cluster_params.processes, ratio)
            ], ';'))
        end
    end

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end


function run_backend_reference(measure::MeasureParams, params::JuliaParams, cluster_params::ClusterParams)
    armon_options = [
        "julia", "-t", params.threads
    ]
    append!(armon_options, isempty(measure.node) ? julia_options_no_cluster : julia_options)
    push!(armon_options, julia_script_path)

    if measure.device == CUDA
        append!(armon_options, ["--gpu", "CUDA"])
    elseif measure.device == ROCM
        append!(armon_options, ["--gpu", "ROCM"])
    else
        # no option needed for CPU
    end

    if params.dimension == 1
        cells_list = [1000]
        cells_list_str = join(cells_list, ',')
    else
        cells_list = params.dimension == 2 ? [[360,360]] : [[60,60,60]]
        cells_list_str = join([join(string.(cells), ',') for cells in cells_list], ';')
    end

    append!(armon_options, armon_base_options)
    append!(armon_options, [
        "--dim", params.dimension,
        "--block-size", 256,
        "--use-simd", params.use_simd,
        "--ieee", 64,
        "--cycle", 1,
        "--tests", measure.tests_list[1],
        "--cells-list", cells_list_str,
        "--threads-places", params.jl_places,
        "--threads-proc-bind", params.jl_proc_bind,
        "--repeats", 1,
        "--verbose", 5,
        "--time-histogram", false,
        "--time-MPI-graph", false,
        "--use-mpi", measure.use_MPI
    ])

    if params.dimension > 1
        append!(armon_options, [
            "--async-comms", params.async_comms,
            "--splitting", measure.axis_splitting[1]
        ])

        if isnothing(measure.process_grid_ratios)
            push!(armon_options, "--proc-grid", join([
                join(string.(process_grid), ',') 
                for process_grid in measure.process_grids
            ], ';'))
        else
            push!(armon_options, "--proc-grid-ratio", join([
                join(string.(ratio), ',')
                for ratio in measure.process_grid_ratios
                if check_ratio_for_grid(cluster_params.processes, ratio)
            ], ';'))
        end
    end

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end


function build_data_file_base_name(measure::MeasureParams, processes::Int, distribution::String,
        node_count::Int, params::JuliaParams)
    name, legend = build_data_file_base_name(measure, processes, distribution, node_count, params.threads, 
                                             params.use_simd, params.dimension, Julia)

    if length(measure.jl_proc_bind) > 1
        name *= "_$(params.jl_proc_bind)"
        legend *= ", bind: $(params.jl_proc_bind)"
    end

    if length(measure.jl_places) > 1
        name *= "_$(params.jl_places)"
        legend *= ", places: $(params.jl_places)"
    end

    if length(measure.async_comms) > 1
        async_str = params.async_comms ? "async" : "sync"
        name *= "_$async_str"
        legend *= ", $async_str"
    end

    if !isempty(params.options[2])
        legend *= ", " * params.options[2]
    end

    if !isempty(params.options[3])
        name *= "_" * params.options[3]
    end

    return name * "_", legend
end

