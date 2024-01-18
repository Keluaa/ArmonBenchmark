
function parse_measure_params(file_line_parser, script_dir)
    backends = [Julia]
    compilers = [GCC]
    device = CPU
    node = "a100"
    distributions = ["block"]
    processes = [1]
    node_count = [1]
    processes_per_node = 0
    max_time = 3600
    extra_modules = []

    make_sub_script = true
    one_job_per_cell = false
    one_script_per_step = false

    threads = [4]
    ieee_bits = [64]
    block_sizes = [128]
    use_simd = [true]
    jl_places = ["cores"]
    jl_proc_bind = ["close"]
    omp_places = ["cores"]
    omp_proc_bind = ["close"]
    dimension = [2]
    async_comms = [false]
    cells_list = "12.5e3, 25e3, 50e3, 100e3, 200e3, 400e3, 800e3, 1.6e6, 3.2e6, 6.4e6, 12.8e6, 25.6e6, 51.2e6, 102.4e6"
    domain_list = "100,100; 250,250; 500,500; 750,750; 1000,1000"
    process_grids = ["1,1"]
    process_grid_ratios = nothing
    tests_list = ["Sod"]
    axis_splitting = ["Sequential"]
    armon_params = [[
        "--write-output", "0",
        "--verbose", "2"
    ]]
    armon_params_legends = [""]
    armon_params_names = [""]
    use_MPI = true
    name = nothing
    repeats = 1
    cycles = 20
    log_scale = true
    error_bars = false
    plot_title = ""
    verbose = false
    use_max_threads = false
    process_scaling = false
    min_acquisition_time = 0
    use_kokkos = [false]
    use_md_iter = [0]
    cmake_options = ""
    kokkos_backends = ["Serial,OpenMP"]
    kokkos_version = "4.0.00"

    perf_plot = true
    gnuplot_script = nothing
    plot_file = nothing

    track_energy = false
    energy_references = 3
    energy_plot = false

    time_histogram = false
    flatten_time_dims = false

    time_MPI_plot = false

    last_i = 0
    for (i, line) in file_line_parser
        last_i = i
        line = chomp(line)
        if line == "-"; break; end    # End of this measure
        if line == ""; continue; end  # Empty line
        if startswith(line, '#'); continue; end # Comment
        if isnothing(findfirst('=', line))
            error("Missing '=' at line $(i)")
        end

        option, value = split(line, '='; limit=2) .|> strip
        if option == "backends"
            raw_backends = split(value, ',') .|> strip .|> lowercase
            backends = []
            for raw_backend in raw_backends
                if raw_backend == "julia"
                    push!(backends, Julia)
                elseif raw_backend == "kokkos"
                    push!(backends, Kokkos)
                elseif raw_backend == "cpp"
                    push!(backends, CPP)
                else
                    error("Unknown backend: $raw_backend, at line $i")
                end
            end
        elseif option == "device"
            if value == "CPU"
                device = CPU
            elseif value == "CUDA"
                device = CUDA
            elseif value == "ROCM"
                device = ROCM
            else
                error("Unknown device: $value, at line $i")
            end
        elseif option == "compilers"
            raw_compilers = split(value, ',') .|> strip .|> lowercase
            compilers = []
            for raw_compiler in raw_compilers
                if raw_compiler == "gcc"
                    push!(compilers, GCC)
                elseif raw_compiler == "clang"
                    push!(compilers, Clang)
                elseif raw_compiler == "icc"
                    push!(compilers, ICC)
                elseif raw_compiler == "aocc"
                    push!(compilers, AOCC)
                elseif raw_compiler == "icx"
                    push!(compilers, ICX)
                else
                    error("Unknown compiler: $raw_compiler, at line $i")
                end
            end
        elseif option == "node"
            node = value
        elseif option == "distributions"
            distributions = split(value, ',')
        elseif option == "processes"
            processes = parse.(Int, split(value, ','))
        elseif option == "node_count"
            node_count = parse.(Int, split(value, ','))
        elseif option == "processes_per_node"
            processes_per_node = parse(Int, value)
        elseif option == "max_time"
            max_time = Int(round(duration_from_string(value)))
        elseif option == "modules"
            extra_modules = split(value, ',') .|> strip
        elseif option == "make_sub_script"
            make_sub_script = parse(Bool, value)
        elseif option == "one_job_per_cell"
            one_job_per_cell = parse(Bool, value)
        elseif option == "one_script_per_step"
            one_script_per_step = parse(Bool, value)
        elseif option == "threads"
            threads = parse.(Int, split(value, ','))
        elseif option == "block_sizes"
            block_sizes = parse.(Int, split(value, ','))
        elseif option == "ieee_bits"
            ieee_bits = parse.(Int, split(value, ','))
        elseif option == "use_simd"
            use_simd = parse.(Int, split(value, ','))
        elseif option == "jl_places"
            jl_places = split(value, ',')
        elseif option == "jl_proc_bind"
            jl_proc_bind = split(value, ',')
        elseif option == "omp_places"
            omp_places = split(value, ',')
        elseif option == "omp_proc_bind"
            omp_proc_bind = split(value, ',')
        elseif option == "dim"
            dimension = parse.(Int, split(value, ','))
        elseif option == "async_comms"
            async_comms = parse.(Bool, split(value, ','))
        elseif option == "cells"
            cells_list = value
        elseif option == "domains"
            domain_list = value
        elseif option == "process_grids"
            process_grids = split(value, ';')
        elseif option == "process_grid_ratios"
            process_grid_ratios = split(value, ';')
        elseif option == "tests"
            tests_list = split(value, ',')
        elseif option == "splitting"
            axis_splitting = split(value, ',')
        elseif option == "armon"
            armon_params = split.(split(value, ';') .|> strip, ' ')
        elseif option == "legends"
            armon_params_legends = split(value, ';') .|> strip
        elseif option == "name_suffixes"
            armon_params_names = split(value, ';') .|> strip
        elseif option == "use_MPI"
            use_MPI = parse(Bool, value)
        elseif option == "name"
            name = value
        elseif option == "repeats"
            repeats = parse(Int, value)
        elseif option == "cycles"
            cycles = parse(Int, value)
        elseif option == "gnuplot"
            gnuplot_script = value
        elseif option == "plot"
            plot_file = value
        elseif option == "title"
            plot_title = value
        elseif option == "log_scale"
            log_scale = parse(Bool, value)
        elseif option == "error_bars"
            error_bars = parse(Bool, value)
        elseif option == "verbose"
            verbose = parse(Bool, value)
        elseif option == "use_max_threads"
            use_max_threads = parse(Bool, value)
        elseif option == "track_energy"
            track_energy = parse(Bool, value)
        elseif option == "energy_references"
            energy_references = parse(Int, value)
        elseif option == "process_scaling"
            process_scaling = parse(Bool, value)
        elseif option == "min_acquisition_time"
            min_acquisition_time = Int(round(duration_from_string(value)))
        elseif option == "energy_plot"
            energy_plot = parse(Bool, value)
        elseif option == "perf_plot"
            perf_plot = parse(Bool, value)
        elseif option == "time_hist"
            time_histogram = parse(Bool, value)
        elseif option == "flat_hist_dims"
            flatten_time_dims = parse(Bool, value)
        elseif option == "time_MPI_plot"
            time_MPI_plot = parse(Bool, value)
        elseif option == "use_kokkos"
            use_kokkos = parse.(Bool, split(value, ',') .|> strip)
        elseif option == "use_md_iter"
            # 0: no, 1: 2D, 2: MD, 3: MD+balancing
            use_md_iter = parse.(Int, split(value, ',') .|> strip)
        elseif option == "cmake_options"
            cmake_options = value
        elseif option == "kokkos_backends"
            kokkos_backends = split(value, ';') .|> strip
        elseif option == "kokkos_version"
            kokkos_version = value
        else
            error("Unknown option: $option, at line $i")
        end
    end

    # Post processing

    cells_list = convert.(Int, parse.(Float64, split(cells_list, ',')))

    domain_list = split(domain_list, ';')
    domain_list = [convert.(Int, parse.(Float64, split(cells_domain, ',')))
                   for cells_domain in domain_list]

    if !isnothing(process_grid_ratios)
        # Make sure that all ratios are compatible with all processes counts
        process_grid_ratios = [parse.(Int, split(ratio, ',')) for ratio in process_grid_ratios]
        process_grids = [[1, 1]] # Provide a dummy grid
    else
        # Use the explicitly defined process grid.
        process_grids = [parse.(Int, split(process_grid, ',')) for process_grid in process_grids]
    end

    if isnothing(name)
        error("Expected a name for the measurement at line ", last_i)
    end
    if isnothing(gnuplot_script)
        gnuplot_script = name * ".plot"
    end
    if isnothing(plot_file)
        # By default, same name as the plot script but without the extension
        plot_file = splitext(gnuplot_script)[1]
    end

    if !isnothing(process_grid_ratios) && any(dimension .== 1)
        error("'process_grid_ratio' is incompatible with 1D") 
    end

    if time_histogram && length(tests_list) > 1
        error("The histogram can only be made when there is only a single test to do")
    end

    if time_MPI_plot && !use_MPI
        error("Cannot make an MPI communications time graph without using MPI")
    end

    if length(armon_params) != length(armon_params_legends)
        error("Expected $(length(armon_params)) legends, got $(length(armon_params_legends))")
    end

    if length(armon_params) != length(armon_params_names)
        error("Expected $(length(armon_params)) names, got $(length(armon_params_names))")
    end

    if track_energy && !one_job_per_cell
        error("Cannot track energy for jobs with multiple cell domains per step. Use 'one_job_per_cell=true'")
    end

    if track_energy && !make_sub_script
        error("Cannot track energy outside of a submission script. `make_sub_script` should be `true`")
    end

    params_and_legends = collect(zip(armon_params, armon_params_legends, armon_params_names))

    rel_plot_scripts_dir = joinpath(".", PLOT_SCRIPTS_DIR_NAME)
    rel_plots_dir = joinpath(".", PLOTS_DIR_NAME)

    gnuplot_script = joinpath(rel_plot_scripts_dir, gnuplot_script)
    plot_file = joinpath(rel_plots_dir, plot_file)

    gnuplot_hist_script = joinpath(rel_plot_scripts_dir, name * "_hist.plot")
    hist_plot_file = joinpath(rel_plots_dir, name * "_hist")

    gnuplot_MPI_script = joinpath(rel_plot_scripts_dir, name * "_MPI_time.plot")
    time_MPI_plot_file = joinpath(rel_plots_dir, name * "_MPI_time")

    energy_script = joinpath(rel_plot_scripts_dir, name * "_energy.plot")
    energy_plot_file = joinpath(rel_plots_dir, name * "_energy")

    return MeasureParams(
        device, node, distributions, processes, node_count, processes_per_node, max_time, use_MPI,
        extra_modules,
        make_sub_script, one_job_per_cell, one_script_per_step,
        backends, compilers, threads, use_simd, jl_proc_bind, jl_places, omp_proc_bind, omp_places,
        dimension, async_comms, ieee_bits, block_sizes,
        use_kokkos, kokkos_backends, use_md_iter, cmake_options, kokkos_version,
        cycles, cells_list, domain_list, process_grids, process_grid_ratios, tests_list,
        axis_splitting, params_and_legends,
        name, script_dir, repeats, log_scale, error_bars, plot_title,
        verbose, use_max_threads,
        track_energy, energy_references, process_scaling, min_acquisition_time,
        perf_plot, gnuplot_script, plot_file,
        time_histogram, flatten_time_dims, gnuplot_hist_script, hist_plot_file,
        time_MPI_plot, gnuplot_MPI_script, time_MPI_plot_file,
        energy_plot, energy_script, energy_plot_file
    )
end


function parse_measure_script_file(file::IOStream, name::String, script_dir::String)
    measures::Vector{MeasureParams} = []
    file_line_parser = enumerate(eachline(file))
    while !eof(file)
        measure = try
            parse_measure_params(file_line_parser, script_dir)
        catch e
            println("Error while parsing measure $(length(measures)+1) of file '$name':")
            rethrow(e)
        end
        push!(measures, measure)
    end
    return measures
end


const USAGE = """
Usage: 
julia batch_measure.jl [--override-node=<node>,<new node>]
                       [--start-at=<measure index>]
                       [--do-only=<measures count>]
                       [--skip-first=<combinaison count>]
                       [--count=<combinaison count>]
                       [--no-overwrite]
                       [--no-plot-update]
                       [--step-scripts=true|false]
                       [--sub-now]
                       [--help|-h]
                       <measurement files>...'
"""


function parse_arguments()
    if length(ARGS) == 0
        error("Invalid number of arguments.\n" * USAGE)
    end

    batch_options = BatchOptions()

    node_overrides = Dict{String, String}()

    measures::Vector{MeasureParams} = []

    for arg in ARGS
        if (startswith(arg, "-"))
            # Batch parameter
            if (startswith(arg, "--override-node="))
                node, replacement_node = split(split(arg, '=')[2], ',')
                node_overrides[node] = replacement_node
            elseif (startswith(arg, "--start-at="))
                value = split(arg, '=')[2]
                batch_options.start_at = parse(Int, value)
            elseif (startswith(arg, "--skip-first="))
                value = split(arg, '=')[2]
                batch_options.skip_first = parse(Int, value)
            elseif (startswith(arg, "--do-only="))
                value = split(arg, '=')[2]
                batch_options.do_only = parse(Int, value)
            elseif (startswith(arg, "--count="))
                value = split(arg, '=')[2]
                batch_options.comb_count = parse(Int, value)
            elseif (startswith(arg, "--no-overwrite"))
                batch_options.no_overwrite = true
            elseif (startswith(arg, "--no-plot-update"))
                batch_options.no_plot_update = true
            elseif (startswith(arg, "--step-scripts="))
                value = split(arg, '=')[2]
                batch_options.one_script_per_step = parse(Bool, value)
            elseif (startswith(arg, "--sub-now"))
                batch_options.submit_now = true
            elseif arg == "--help" || arg == "-h"
                println(USAGE)
                exit(0)
            else
                error("Wrong batch option: " * arg * "\n" * USAGE)
            end
        else
            # Measure file
            script_file = open(arg, "r")
            script_dir = dirname(arg)
            append!(measures, parse_measure_script_file(script_file, arg, script_dir))
            close(script_file)
        end
    end

    for (node, replacement_node) in node_overrides
        for measure in measures
            measure.node == node && (measure.node = replacement_node)
        end
    end

    return measures, batch_options
end
