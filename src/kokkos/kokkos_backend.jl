
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
backend_run_dir(::KokkosParams) = abspath(joinpath(@__DIR__, "../../kokkos"))


function run_backend_msg(measure::MeasureParams, params::KokkosParams, cluster::ClusterParams)
    """Running C++ Kokkos backend with:
     - $(params.threads) threads
     - threads binding: $(params.omp_proc_bind), places: $(params.omp_places)
     - $(params.use_simd == 1 ? "with" : "without") SIMD
     - $(params.dimension)D
     - compiled with $(params.compiler)
     - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
     - with $(cluster.processes) processes on $(cluster.node_count) nodes ($(cluster.distribution) distribution)
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


function run_backend(measure::MeasureParams, params::KokkosParams, cluster::ClusterParams, base_file_name::String)
    if cluster.processes > 1
        error("The Kokkos backend doesn't support MPI yet")
    end

    if measure.limit_to_max_mem
        @warn "The Kokkos backend doesn't support the 'limit_to_max_mem'" maxlog=1
    end

    if measure.time_histogram
        @warn "The Kokkos backend doesn't support the 'time_histogram'" maxlog=1
    end

    armon_options = Any["julia"]

    if !measure.make_sub_script
        push!(armon_options, "--color=yes")
    end

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
            cells_list .*= cluster.processes
        else
            # We need to distribute the factor along each axis, while keeping the divisibility of 
            # the cells count, since it will be divided by the number of processes along each axis.
            # Therefore we make the new values multiples of 64, but this is still not perfect.
            scale_factor = cluster.processes^(1/params.dimension)
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

    append!(armon_options, ARMON_BASE_OPTIONS)
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

    additional_options, _, _ = params.options
    append!(armon_options, additional_options)

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
