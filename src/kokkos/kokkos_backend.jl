
kokkos_script_path = joinpath(@__DIR__, "run_kokkos.jl")


struct KokkosParams <: BackendParams
    options::Tuple{Vector{String}, String, String}
    omp_places::String
    omp_proc_bind::String
    threads::Int
    ieee_bits::Int
    use_simd::Int
    use_md_iter::Int
    dimension::Int
    compiler::Compiler
end


backend_disp_name(::KokkosParams) = "C++ Kokkos"
backend_type(::KokkosParams) = Kokkos
backend_run_dir(::KokkosParams) = abspath(joinpath(@__DIR__, "../../kokkos"))


function run_backend_msg(measure::MeasureParams, params::KokkosParams, cluster::ClusterParams)
    """Running C++ Kokkos backend with:
     - $(params.threads) threads
     - threads binding: $(params.omp_proc_bind), places: $(params.omp_places)
     - $(params.use_simd == 1 ? "with" : "without") SIMD
     - $(params.use_md_iter == 0 ? "without" : params.use_md_iter == 1 ? "with 2D" : "with MD") iterations, with$(params.use_md_iter == 3 ? "" : "out") load balancing
     - $(params.dimension)D
     - compiled with $(params.compiler) with Kokkos v$(measure.kokkos_version)
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
            measure.use_md_iter,
            measure.dimension,
            measure.compilers,
        )
    )
end


function build_job_step(measure::MeasureParams,
        params::KokkosParams, cluster::ClusterParams,
        base_file_name::String, legend::String)

    if cluster.processes > 1
        error("The Kokkos backend doesn't support MPI yet")
    end

    if measure.time_histogram
        @warn "The Kokkos backend doesn't support the 'time_histogram'" maxlog=1
    end

    if params.dimension == 1
        cells_list = [[cells] for cells in measure.cells_list]
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
    options[:verbose] = measure.verbose ? 2 : 3
    options[:in_sub_script] = measure.make_sub_script
    options[:device] = measure.device
    options[:tests] = measure.tests_list
    options[:axis_splitting] = measure.axis_splitting
    options[:min_acquisition] = measure.min_acquisition_time
    options[:cmake_options] = measure.cmake_options
    options[:kokkos_version] = measure.kokkos_version

    perf_plot = PlotInfo(base_file_name * "%s_perf.csv", measure.perf_plot, measure.gnuplot_script)
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


function adjust_reference_job(step::JobStep, ::Val{Kokkos})
    step.options[:min_acquisition] = 0
end


function build_backend_command(step::JobStep, ::Val{Kokkos})
    armon_options = Any["julia"]

    if !(step.options[:in_sub_script])
        push!(armon_options, "--color=yes")
    end

    push!(armon_options, kokkos_script_path)

    if step.options[:device] == CUDA
        append!(armon_options, ["--gpu", "CUDA"])
    elseif step.options[:device] == ROCM
        append!(armon_options, ["--gpu", "ROCM"])
    else
        # no option needed for CPU
    end

    if step.dimension == 1
        cells_list_str = join(step.cells_list, ',')
    else
        cells_list_str = join([join(string.(cells), ',') for cells in step.cells_list], ';')
    end

    append!(armon_options, ARMON_BASE_OPTIONS)
    append!(armon_options, [
        "--dim", step.dimension,
        "--cycle", step.cycles,
        "--use-simd", step.backend.use_md_iter == 0 ? step.backend.use_simd : 0,
        "--use-2d-iter", step.backend.use_md_iter == 1,
        "--use-md-iter", step.backend.use_md_iter >= 2,
        "--balance-md-iter", step.backend.use_md_iter == 3,
        "--ieee", step.backend.ieee_bits,
        "--tests", join(step.options[:tests], ','),
        "--cells-list", cells_list_str,
        "--num-threads", step.backend.threads,
        "--threads-places", step.backend.omp_places,
        "--threads-proc-bind", step.backend.omp_proc_bind,
        "--min-acquisition-time", step.options[:min_acquisition],
        "--repeats", step.repeats,
        "--verbose", step.options[:verbose],
        "--compiler", step.backend.compiler,
        "--splitting", join(step.options[:axis_splitting], ','),
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

    cmake_options = step.options[:cmake_options]
    append!(armon_options, [
        "--extra-cmake-options", cmake_options
    ])

    append!(armon_options, [
        "--kokkos-version", step.options[:kokkos_version]
    ])

    additional_options, _, _ = step.backend.options
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

    if length(measure.use_md_iter) > 1
        if params.use_md_iter == 1
            name *= "_2d_iter"
            legend *= ", 2D iterations"
        elseif params.use_md_iter == 2
            name *= "_md_iter"
            legend *= ", MD iterations"
        elseif params.use_md_iter == 3
            name *= "_md_iter_balanced"
            legend *= ", balanced MD iterations"
        end
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
