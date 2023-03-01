
cpp_script_path = joinpath(@__DIR__, "run_cpp.jl")


struct CppParams <: BackendParams
    options::Tuple{Vector{String}, String, String}
    omp_places::String
    omp_proc_bind::String
    threads::Int
    ieee_bits::Int
    use_simd::Int
    dimension::Int
    compiler::Compiler
end


backend_disp_name(::CppParams) = "C++ OpenMP"
backend_run_dir(::CppParams) = abspath(joinpath(@__DIR__, "../../cpp"))


function run_backend_msg(measure::MeasureParams, params::CppParams, cluster::ClusterParams)
    """Running C++ OpenMP backend with:
     - $(params.threads) threads
     - threads binding: $(params.omp_proc_bind), places: $(params.omp_places)
     - $(params.use_simd == 1 ? "with" : "without") SIMD
     - 1D
     - compiled with $(params.compiler)
     - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
     - with $(cluster.processes) processes on $(cluster.node_count) nodes ($(cluster.distribution) distribution)
    """
end


function iter_combinaisons(measure::MeasureParams, threads, ::Val{CPP})
    Iterators.map(
        params->CppParams(params...),
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


function run_backend(measure::MeasureParams, params::CppParams, cluster::ClusterParams, base_file_name::String)
    if cluster.processes > 1
        error("The C++ backend doesn't support MPI")
    end

    if measure.limit_to_max_mem
        @warn "The C++ backend doesn't support the 'limit_to_max_mem'" maxlog=1
    end

    if measure.time_histogram
        @warn "The C++ backend doesn't support the 'time_histogram'" maxlog=1
    end

    armon_options = Any["julia"]

    if !measure.make_sub_script
        push!(armon_options, "--color=yes")
    end

    push!(armon_options, cpp_script_path)

    if measure.device != CPU
        error("The C++ backend works only on the CPU")
    end

    if params.dimension == 1
        cells_list = measure.cells_list
    else
        error("The C++ backend works only in 1D")
    end

    if measure.cst_cells_per_process
        # Scale the cells by the number of processes
        cells_list .*= cluster.processes
    end

    cells_list_str = join(cells_list, ',')

    append!(armon_options, ARMON_BASE_OPTIONS)
    append!(armon_options, [
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
        "--compiler", params.compiler
    ])

    additional_options, _, _ = params.options
    append!(armon_options, additional_options)

    return armon_options
end


function build_data_file_base_name(measure::MeasureParams, processes::Int, distribution::String,
        node_count::Int, params::CppParams)
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
