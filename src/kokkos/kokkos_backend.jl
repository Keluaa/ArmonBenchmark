
kokkos_script_path = joinpath(@__DIR__, "run_kokkos.jl")


struct KokkosParams <: BackendParams
    options::Tuple{Vector{String}, String, String}
    omp_places::String
    omp_proc_bind::String
    threads::Int
    ieee_bits::Int
    use_simd::Int
    dimension::Int
    compiler::Compiler
end


backend_disp_name(::KokkosParams) = "C++ Kokkos"
backend_run_dir(::KokkosParams) = joinpath(@__DIR__, "../../kokkos")


function run_backend_msg(measure::MeasureParams, kokkos_params::KokkosParams, cluster_params::ClusterParams)
    """Running C++ Kokkos backend with:
     - $(kokkos_params.threads) threads
     - threads binding: $(kokkos_params.omp_proc_bind), places: $(kokkos_params.omp_places)
     - $(kokkos_params.use_simd == 1 ? "with" : "without") SIMD
     - $(kokkos_params.dimension)D
     - compiled with $(kokkos_params.compiler)
     - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
     - with $(cluster_params.processes) processes on $(cluster_params.node_count) nodes ($(cluster_params.distribution) distribution)
    """
end


function iter_combinaisons(measure::MeasureParams, threads, ::Val{Kokkos})
    Iterators.map(
        params->KokkosParams(params...),
        Iterators.product(
            measure.armon_params,
            measure.omp_places,
            measure.omp_proc_bind,
            threads,
            measure.ieee_bits,
            measure.use_simd,
            measure.dimension,
            measure.compilers,
        )
    )
end


function run_backend(measure::MeasureParams, params::KokkosParams, cluster_params::ClusterParams, base_file_name::String)
    if cluster_params.processes > 1
        error("The Kokkos backend doesn't support MPI yet")
    end

    if measure.limit_to_max_mem
        @warn "The Kokkos backend doesn't support the 'limit_to_max_mem'" maxlog=1
    end

    if measure.time_histogram
        @warn "The Kokkos backend doesn't support the 'time_histogram'" maxlog=1
    end

    armon_options = Any["julia"]
    append!(armon_options, isempty(measure.node) ? julia_options_no_cluster : julia_options)
    push!(armon_options, kokkos_script_path)

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
        "--use-simd", params.use_simd,
        "--ieee", params.ieee_bits,
        "--tests", join(measure.tests_list, ','),
        "--cells-list", cells_list_str,
        "--num-threads", params.threads,
        "--threads-places", params.omp_places,
        "--threads-proc-bind", params.omp_proc_bind,
        "--data-file", base_file_name,
        "--gnuplot-script", measure.gnuplot_script,
        "--repeats", measure.repeats,
        "--verbose", (measure.verbose ? 2 : 3),
        "--compiler", params.compiler,
        "--track-energy", measure.track_energy
    ])

    if params.dimension > 1
        append!(armon_options, [
            "--splitting", join(measure.axis_splitting, ',')
        ])
    end

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end


function run_backend_reference(measure::MeasureParams, params::KokkosParams, cluster_params::ClusterParams)
    if cluster_params.processes > 1
        error("The Kokkos backend doesn't support MPI yet")
    end

    if measure.limit_to_max_mem
        @warn "The Kokkos backend doesn't support the 'limit_to_max_mem'" maxlog=1
    end

    if measure.time_histogram
        @warn "The Kokkos backend doesn't support the 'time_histogram'" maxlog=1
    end

    armon_options = ["julia"]
    append!(armon_options, isempty(measure.node) ? julia_options_no_cluster : julia_options)
    push!(armon_options, kokkos_script_path)

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
        "--use-simd", params.use_simd,
        "--ieee", 64,
        "--cycle", 1,
        "--tests", measure.tests_list[1],
        "--cells-list", cells_list_str,
        "--threads-places", params.omp_places,
        "--threads-proc-bind", params.omp_proc_bind,
        "--repeats", 1,
        "--verbose", 5
    ])

    if params.dimension > 1
        append!(armon_options, [
            "--splitting", measure.axis_splitting[1]
        ])
    end

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end



function build_data_file_base_name(measure::MeasureParams, processes::Int, distribution::String,
        node_count::Int, params::KokkosParams)
    name, legend = build_data_file_base_name(measure, processes, distribution, node_count, params.threads, 
                                             params.use_simd, params.dimension, Kokkos)

    if length(measure.omp_proc_bind) > 1
        name *= "_$(params.omp_proc_bind)"
        legend *= ", bind: $(params.omp_proc_bind)"
    end

    if length(measure.omp_places) > 1
        name *= "_$(params.omp_places)"
        legend *= ", places: $(params.omp_places)"
    end

    if length(measure.compilers) > 1
        name *= "_$(params.compiler)"
        legend *= ", $(params.compiler)"
    end

    if !isempty(params.options[2])
        legend *= ", " * params.options[2]
    end

    if !isempty(params.options[3])
        name *= "_" * params.options[3]
    end

    return name * "_", legend
end
