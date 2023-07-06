
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
    use_kokkos::Bool
    kokkos_backends::String
end


backend_disp_name(::JuliaParams) = "Julia"
backend_type(::JuliaParams) = Julia
backend_run_dir(::JuliaParams) = abspath(joinpath(@__DIR__, "../../julia"))


function run_backend_msg(measure::MeasureParams, julia::JuliaParams, cluster::ClusterParams)
    if julia.use_kokkos
        return """Running Julia with:
        - $(julia.threads) threads
        - threads binding: $(julia.jl_proc_bind), places: $(julia.jl_places)
        - kokkos using $(julia.kokkos_backends)
        - $(julia.dimension)D
        - $(julia.async_comms ? "a" : "")synchronous communications
        - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
        - with $(cluster.processes) processes on $(cluster.node_count) nodes ($(cluster.distribution) distribution)
       """
    else
        return """Running Julia with:
        - $(julia.threads) threads
        - threads binding: $(julia.jl_proc_bind), places: $(julia.jl_places)
        - $(julia.use_simd == 1 ? "with" : "without") SIMD
        - $(julia.dimension)D
        - $(julia.async_comms ? "a" : "")synchronous communications
        - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
        - with $(cluster.processes) processes on $(cluster.node_count) nodes ($(cluster.distribution) distribution)
       """
    end
end


function iter_combinaisons(measure::MeasureParams, threads, ::Val{Julia})
    kokkos_backends = copy(measure.kokkos_backends)
    if false in measure.use_kokkos
        # The empty backend plus the filter prevents to go through all combinations of kokkos
        # backends when `use_kokkos=false`.
        push!(kokkos_backends, "")
        unique!(kokkos_backends)
    end

    Iterators.map(
        params->JuliaParams(params...),
        Iterators.filter(
            params -> params[10] != isempty(params[11]),  # If we use kokkos, we require a non empty backend, else we do
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
                measure.use_kokkos,
                kokkos_backends
            )
        )
    )
end


function build_job_step(measure::MeasureParams,
        params::JuliaParams, cluster::ClusterParams,
        base_file_name::String, legend::String)

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
    options[:verbose] = measure.verbose ? 2 : 5
    options[:in_sub_script] = measure.make_sub_script
    options[:device] = measure.device
    options[:tests] = measure.tests_list
    options[:axis_splitting] = measure.axis_splitting
    options[:use_MPI] = measure.use_MPI
    options[:limit_to_max_mem] = measure.limit_to_max_mem
    options[:min_acquisition] = measure.min_acquisition_time
    options[:cmake_options] = measure.cmake_options
    options[:kokkos_version] = measure.kokkos_version

    perf_plot = PlotInfo(base_file_name * "%s_perf.csv", measure.perf_plot, measure.gnuplot_script)
    time_hist = PlotInfo(base_file_name * "%s_hist.csv", measure.time_histogram, measure.gnuplot_hist_script)
    time_MPI  = PlotInfo(base_file_name * "%s_MPI_time.csv", measure.time_MPI_plot, measure.gnuplot_MPI_script)
    energy_plot = PlotInfo(base_file_name * "_energy.csv", measure.energy_plot, measure.energy_script)

    return JobStep(
        cluster, params,
        measure.node, params.threads, params.dimension,
        cells_list, proc_grids, measure.repeats, measure.cycles,
        base_file_name, legend,
        perf_plot, time_hist, time_MPI, energy_plot,
        options
    )
end


function adjust_reference_job(step::JobStep, ::Val{Julia})
    step.options[:min_acquisition] = 0
end


function build_backend_command(step::JobStep, ::Val{Julia})
    armon_options = [
        "julia", "-t", step.threads,
        "-O3", "--check-bounds=no",
        "--project=$(@__DIR__())"
    ]

    if !step.options[:in_sub_script]
        push!(armon_options, "--color=yes")
    end

    push!(armon_options, julia_script_path)

    if step.options[:device] == CUDA
        append!(armon_options, ["--gpu", "CUDA"])
    elseif step.options[:device] == ROCM
        append!(armon_options, ["--gpu", "ROCM"])
    else
        # no option needed for CPU
    end

    if step.dimension == 1
        cells_list_str = join(first.(step.cells_list), ',')
    else
        cells_list_str = join([join(string.(cells), ',') for cells in step.cells_list], ';')
    end

    append!(armon_options, ARMON_BASE_OPTIONS)
    append!(armon_options, [
        "--dim", step.dimension,
        "--cycle", step.cycles,
        "--block-size", step.backend.block_size,
        "--use-simd", step.backend.use_simd,
        "--ieee", step.backend.ieee_bits,
        "--tests", join(step.options[:tests], ','),
        "--cells-list", cells_list_str,
        "--threads-places", step.backend.jl_places,
        "--threads-proc-bind", step.backend.jl_proc_bind,
        "--repeats", step.repeats,
        "--verbose", step.options[:verbose],
        "--use-mpi", step.options[:use_MPI],
        "--limit-to-mem", step.options[:limit_to_max_mem],
        "--min-acquisition-time", step.options[:min_acquisition],
        "--async-comms", step.backend.async_comms,
        "--splitting", join(step.options[:axis_splitting], ','),
        "--proc-grid", join([join(grid, ',') for grid in step.proc_grids], ';'),
        "--use-kokkos", step.backend.use_kokkos
    ])

    if step.backend.use_kokkos
        push!(armon_options, "--kokkos-backends", step.backend.kokkos_backends)
        push!(armon_options, "--kokkos-version", options[:kokkos_version])

        if step.options[:in_sub_script]
            push!(armon_options, "--kokkos-build-dir", "€KOKKOS_BUILD_DIR")
        end

        if !isempty(step.options[:cmake_options])
            append!(armon_options, [
                "--cmake-options", step.options[:cmake_options]
            ])
        end
    end

    if step.perf_plot.do_plot || step.time_hist.do_plot || step.time_MPI.do_plot
        append!(armon_options, [
            "--data-file", step.base_file_name
        ])
    end

    if step.perf_plot.do_plot
        append!(armon_options, [
            "--gnuplot-script", step.perf_plot.plot_script
        ])
    end

    if step.time_hist.do_plot
        append!(armon_options, [
            "--gnuplot-hist-script", step.time_hist.plot_script,
            "--time-histogram", true
        ])
    end

    if step.time_MPI.do_plot
        append!(armon_options, [
            "--gnuplot-MPI-script", step.time_MPI.plot_script,
            "--time-MPI-graph", true
        ])
    end

    if step.energy_plot.do_plot
        append!(armon_options, [
            "--repeats-count-file", step.energy_plot.data_file * ".TMP"
        ])
    end

    additional_options, _, _ = step.backend.options
    append!(armon_options, additional_options)

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

    if length(measure.use_kokkos) > 1
        name *= "_" * (params.use_kokkos ? "" : "no_") * "kokkos"
        legend *= ", " * (params.use_kokkos ? "with" : "without") * " kokkos"
    end

    if length(measure.kokkos_backends) > 1 && params.use_kokkos
        name *= "_$(params.kokkos_backends)"
        legend *= " " * params.kokkos_backends
    end

    if !isempty(params.options[2])
        legend *= ", " * params.options[2]
    end

    if !isempty(params.options[3])
        name *= "_" * params.options[3]
    end

    return name, legend
end
