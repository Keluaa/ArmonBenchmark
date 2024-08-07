
cpp_script_path = joinpath(@__DIR__, "run_cpp.jl")


struct CppParams <: BackendParams
    options::Tuple{Vector{String}, String, String}
    omp_places::String
    omp_proc_bind::String
    threads::Int
    use_simd::Int
    dimension::Int
    compiler::Compiler
end


backend_disp_name(::CppParams) = "C++ OpenMP"
backend_type(::CppParams) = CPP
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
            measure.use_simd,
            measure.dimension,
            measure.compilers,
        )
    )
end


function build_job_step(measure::MeasureParams,
        params::CppParams, cluster::ClusterParams,
        base_file_name::String, legend::String)

    if cluster.processes > 1
        error("The C++ backend doesn't support MPI")
    end

    if measure.time_histogram
        @warn "The C++ backend doesn't support the 'time_histogram'" maxlog=1
    end

    if measure.device != CPU
        error("The C++ backend works only on the CPU")
    end

    if params.dimension != 1
        error("The C++ backend works only in 1D")
    end

    if params.dimension == 1
        cells_list = [[cells] for cells in mesure.cells_list]
    else
        cells_list = measure.domain_list
    end

    if isnothing(measure.process_grid_ratios)
        proc_grids = measure.process_grids
    else
        ratios = filter(ratio -> check_ratio_for_grid(cluster.processes, ratio),
            measure.process_grid_ratios)
        proc_grids = split_N.(cluster.processes, ratios)
    end

    options = Dict{Symbol, Any}()
    options[:type] = measure.type
    options[:verbose] = measure.verbose ? 2 : 3
    options[:in_sub_script] = measure.make_sub_script
    options[:tests] = measure.tests_list
    options[:min_acquisition] = measure.min_acquisition_time

    perf_plot = PlotInfo(base_file_name * "%s.csv", measure.perf_plot, measure.gnuplot_script)
    time_hist = PlotInfo(base_file_name * "%s_hist.csv", measure.time_histogram, measure.gnuplot_hist_script)
    time_MPI  = PlotInfo(base_file_name * "%s_MPI_time.csv", measure.time_MPI_plot, measure.gnuplot_MPI_script)
    energy_plot = PlotInfo(base_file_name * "energy.csv", measure.energy_plot, measure.energy_script)

    return JobStep(
        cluster, params,
        measure.node, params.threads, params.dimension,
        cells_list, proc_grids, measure.repeats, measure.cycles,
        base_file_name, legend,
        perf_plot, time_hist, time_MPI, energy_plot,
        options
    )
end


function adjust_reference_job(step::JobStep, ::Val{CPP})
    step.options[:min_acquisition] = 0
end


function build_backend_command(step::JobStep, ::Val{CPP})
    armon_options = Any["julia"]

    if !(step.options[:in_sub_script])
        push!(armon_options, "--color=yes")
    end

    push!(armon_options, cpp_script_path)

    cells_list_str = join(step.cells_list, ',')

    append!(armon_options, ARMON_BASE_OPTIONS)
    append!(armon_options, [
        "--cycle", step.cycles,
        "--use-simd", step.backend.use_simd,
        "--ieee", step.options[:type] == Float64 ? "64" : "32",
        "--tests", join(step.options[:tests_list], ','),
        "--cells-list", cells_list_str,
        "--num-threads", step.backend.threads,
        "--threads-places", step.backend.omp_places,
        "--threads-proc-bind", step.backend.omp_proc_bind,
        "--min-acquisition-time", step.options[:min_acquisition],
        "--repeats", step.repeats,
        "--verbose", step.options[:verbose],
        "--compiler", step.backend.compiler
    ])

    if step.perf_plot.do_plot
        append!(armon_options, [
            "--data-file", step.base_file_name,
            "--gnuplot-script", step.perf_plot.plot_script
        ])
    end

    if step.energy_plot.do_plot
        append!(armon_options, [
            "--repeats-count-file", step.energy_plot.data_file * ".TMP"
        ])
    end

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

    return name, legend
end
