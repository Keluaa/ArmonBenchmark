
using Printf
using Dates

@enum Device CPU CUDA ROCM


mutable struct MeasureParams
    # ccc_mprun params
    device::Device
    node::String
    distributions::Vector{String}
    processes::Vector{Int}
    node_count::Vector{Int}
    max_time::Int
    use_MPI::Bool

    # Backend params
    threads::Vector{Int}
    use_simd::Vector{Int}
    jl_proc_bind::Vector{String}
    jl_places::Vector{String}
    dimension::Vector{Int}

    # Armon params
    cells_list::Vector{Int}
    domain_list::Vector{Vector{Int}}
    tests_list::Vector{String}
    transpose_dims::Vector{Bool}
    axis_splitting::Vector{String}
    common_armon_params::Vector{String}

    # Measurement params
    name::String
    repeats::Int
    gnuplot_script::String
    plot_file::String
    log_scale::Bool
    plot_title::String
    verbose::Bool

    # > Time histogram
    time_histogram::Bool
    flatten_time_dims::Bool
    gnuplot_hist_script::String
    hist_plot_file::String

    # > MPI communications time plot
    time_MPI_plot::Bool
    gnuplot_MPI_script::String
    time_MPI_plot_file::String
end


struct IntiParams
    processes::Int
    distribution::String
    node_count::Int
end


struct JuliaParams
    jl_places::String
    jl_proc_bind::String
    threads::Int
    use_simd::Int
    dimension::Int
end


no_inti_cmd(armon_options, nprocs) = `mpiexecjl -n $(nprocs) $(armon_options)`
inti_cmd(armon_options, inti_options) = `ccc_mprun $(inti_options) $(armon_options)`

julia_options = ["-O3", "--check-bounds=no", "--project"]
julia_options_no_inti = ["-O3", "--check-bounds=no"]
armon_base_options = [
    "--write-output", "0",
    "--verbose", "2"
]
max_inti_cores = 128  # Maximum number of cores in a node

required_modules = ["cuda", "rocm", "hwloc", "mpi"]

julia_script_path = "./julia/run_julia.jl"
julia_tmp_script_output_file = "./tmp_script_output.txt"

data_dir = "./data/"
plot_scripts_dir = "./plot_scripts/"
plots_dir = "./plots/"


base_gnuplot_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$(graph_file_name)'
set ylabel 'Giga Cells/sec'
set xlabel 'Cells count'
set title "$(title)"
set key $(legend_pos) top
$(log_scale ? "set logscale x" : "")
`echo "$(graph_file_name)" > ./plots/last_update`
plot """

base_gnuplot_histogram_script_commands(graph_file_name, title) = """
set terminal pdfcairo color size 10in, 6in
set output '$(graph_file_name)'
set ylabel 'Total loop time (%)'
set title "$(title)"
set key left top
set style fill solid 1.00 border 0
plot """

base_gnuplot_MPI_time_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$(graph_file_name)'
set ylabel 'Time [sec]'
set xlabel 'Cells count'
set title "$(title)"
set key $(legend_pos) top
set ytics nomirror
set mytics
set yrange [0:]
set y2tics
set my2tics
set y2range [0:]
set y2label 'Communication Time / Total Time [%]'
$(log_scale ? "set logscale x" : "")
`echo "$(graph_file_name)" > ./plots/last_update`
plot """

gnuplot_plot_command(data_file, legend_title, pt_index) = "'$(data_file)' w lp pt $(pt_index) title '$(legend_title)'"
gnuplot_hist_plot_command(data_file, legend_title, color_index) = "'$(data_file)' using 2: xtic(1) with histogram lt $(color_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_1(data_file, legend_title, pt_index) = "'$(data_file)' using 1:2 axis x1y1 w lp pt $(pt_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_2(data_file, legend_title, pt_index) = "'$(data_file)' using 1:(\$2/\$3*100) axis x1y2 w lp pt $(pt_index) dt 4 title '$(legend_title)'"


function parse_measure_params(file_line_parser)    
    device = CPU
    node = "a100"
    distributions = ["block"]
    processes = [1]
    node_count = [1]
    max_time = 3600  # 1h
    threads = [4]
    use_simd = [true]
    jl_places = ["cores"]
    jl_proc_bind = ["close"]
    dimension = [1]
    cells_list = "12.5e3, 25e3, 50e3, 100e3, 200e3, 400e3, 800e3, 1.6e6, 3.2e6, 6.4e6, 12.8e6, 25.6e6, 51.2e6, 102.4e6"
    domain_list = "100,100; 250,250; 500,500; 750,750; 1000,1000"
    tests_list = ["Sod"]
    transpose_dims = [false]
    axis_splitting = ["Sequential"]
    common_armon_params = [
        "--write-output", "0",
        "--verbose", "2",
        "--euler", "0"
    ]
    use_MPI = true
    name = nothing
    repeats = 1
    gnuplot_script = nothing
    plot_file = nothing
    log_scale = true
    plot_title = nothing
    verbose = false

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

        option, value = split(line, '=')
        if option == "device"
            if value == "CPU"
                device = CPU
            elseif value == "CUDA"
                device = CUDA
            elseif value == "ROCM"
                device = ROCM
            else
                error("Unknown device: $(value), at line $(i)")
            end
        elseif option == "node"
            node = value
        elseif option == "distributions"
            distributions = split(value, ',')
        elseif option == "processes"
            processes = parse.(Int, split(value, ','))
        elseif option == "node_count"
            node_count = parse.(Int, split(value, ','))
        elseif option == "max_time"
            max_time = parse(Int, value)
        elseif option == "threads"
            threads = parse.(Int, split(value, ','))
        elseif option == "use_simd"
            use_simd = parse.(Int, split(value, ','))
        elseif option == "jl_places"
            jl_places = split(value, ',')
        elseif option == "jl_proc_bind"
            jl_proc_bind = split(value, ',')
        elseif option == "dim"
            dimension = parse.(Int, split(value, ','))
        elseif option == "cells"
            cells_list = value
        elseif option == "domains"
            domain_list = value
        elseif option == "tests"
            tests_list = split(value, ',')
        elseif option == "transpose"
            transpose_dims = parse.(Bool, split(value, ','))
        elseif option == "splitting"
            axis_splitting = split(value, ',')
        elseif option == "armon"
            common_armon_params = split(value, ' ')
        elseif option == "use_MPI"
            use_MPI = parse(Bool, value)
        elseif option == "name"
            name = value
        elseif option == "repeats"
            repeats = parse(Int, value)
        elseif option == "gnuplot"
            gnuplot_script = value
        elseif option == "plot"
            plot_file = value
        elseif option == "title"
            plot_title = value
        elseif option == "log_scale"
            log_scale = parse(Bool, value)
        elseif option == "verbose"
            verbose = parse(Bool, value)
        elseif option == "time_hist"
            time_histogram = parse(Bool, value)
        elseif option == "flat_hist_dims"
            flatten_time_dims = parse(Bool, value)
        elseif option == "time_MPI_plot"
            time_MPI_plot = parse(Bool, value)
        else
            error("Unknown option: $(option), at line $(i)")
        end
    end

    # Post processing

    cells_list = convert.(Int, parse.(Float64, split(cells_list, ',')))

    domain_list = split(domain_list, ';')
    domain_list = [convert.(Int, parse.(Float64, split(cells_domain, ',')))
                   for cells_domain in domain_list]

    if isnothing(name)
        error("Expected a name for the measurement at line ", last_i)
    end
    if isnothing(gnuplot_script)
        gnuplot_script = name * ".plot"
    end
    if isnothing(plot_file)
        # By default, same name as the plot script but as a'.pdf' file
        plot_file = gnuplot_script[1:findlast('.', gnuplot_script)-1] * ".pdf"
    end
    if isnothing(plot_title)
        plot_title = "You forgot to add a title"
    end

    if time_histogram && length(tests_list) > 1
        error("The histogram can only be made when there is only a single test to do")
    end

    if time_MPI_plot && !use_MPI
        error("Cannot make an MPI communications time graph without using MPI")
    end

    mkpath(data_dir * name)
    gnuplot_script = plot_scripts_dir * gnuplot_script
    plot_file = plots_dir * plot_file

    gnuplot_hist_script = plot_scripts_dir * name * "_hist.plot"
    hist_plot_file = plots_dir * name * "_hist.pdf"

    gnuplot_MPI_script = plot_scripts_dir * name * "_MPI_time.plot"
    time_MPI_plot_file = plots_dir * name * "_MPI_time.pdf"

    return MeasureParams(device, node, distributions, processes, node_count, max_time, use_MPI,
        threads, use_simd, jl_proc_bind, jl_places, 
        dimension, cells_list, domain_list, tests_list, 
        transpose_dims, axis_splitting, common_armon_params,
        name, repeats, gnuplot_script, plot_file, log_scale, plot_title, verbose, 
        time_histogram, flatten_time_dims, gnuplot_hist_script, hist_plot_file,
        time_MPI_plot, gnuplot_MPI_script, time_MPI_plot_file)
end


function parse_measure_script_file(file::IOStream, name::String)
    measures::Vector{MeasureParams} = []
    file_line_parser = enumerate(eachline(file))
    while !eof(file)
        measure = try
            parse_measure_params(file_line_parser)
        catch e
            println("Error while parsing measure $(length(measures)+1) of file '$name':")
            rethrow(e)
        end
        push!(measures, measure)
    end
    return measures
end


function parse_arguments()
    if length(ARGS) == 0
        error("Invalid number of arguments. Usage: 'julia batch_measure.jl <script files>...'")
    end

    start_at = 1

    node_overrides = Dict{String, String}()

    measures::Vector{MeasureParams} = []

    for arg in ARGS
        if (startswith(arg, "--"))
            # Batch parameter
            if (startswith(arg, "--override-node="))
                node, replacement_node = split(split(arg, '=')[2], ',')
                node_overrides[node] = replacement_node
            elseif (startswith(arg, "--startat="))
                value = split(arg, '=')[2]
                start_at = parse(Int, value)
            else
                error("Wrong batch option: " * arg)
            end
        else
            # Measure file
            script_file = open(arg, "r")
            append!(measures, parse_measure_script_file(script_file, arg))
            close(script_file)
        end
    end

    for (node, replacement_node) in node_overrides
        for measure in measures
            measure.node == node && (measure.node = replacement_node)
        end
    end

    return measures, start_at
end


function build_inti_combinaisons(measure::MeasureParams)
    return Iterators.map(
        params->IntiParams(params...),
        Iterators.product(
            measure.processes,
            measure.distributions,
            measure.node_count
        )
    )
end


function parse_combinaisons(measure::MeasureParams)
    return Iterators.map(
        params->JuliaParams(params...),    
        Iterators.product(
            measure.jl_places,
            measure.jl_proc_bind,
            measure.threads,
            measure.use_simd,
            measure.dimension
        )
    )
end


function armon_combinaisons(measure::MeasureParams, dimension::Int)
    if dimension == 1
        return Iterators.product(
            measure.tests_list,
            [false],
            ["Sequential"]
        )
    else
        return Iterators.product(
            measure.tests_list,
            measure.transpose_dims,
            measure.axis_splitting
        )
    end
end


function build_armon_data_file_name(measure::MeasureParams, dimension::Int,
        base_file_name::String, legend_base::String,
        test::String, transpose_dims::Bool, axis_splitting::String)
    file_name = base_file_name * test
    if dimension == 1
        legend = "$test, $legend_base"
    else
        legend = test

        if length(measure.transpose_dims) > 1
            file_name *= transpose_dims ? "_transposed" : ""
            legend *= transpose_dims ? "ᵀ" : ""
        end

        if length(measure.axis_splitting) > 1
            file_name *= "_" * string(axis_splitting)
            legend *= ", " * string(axis_splitting)
        end

        legend *= ", " * legend_base
    end
    return file_name, legend
end


function run_backend(measure::MeasureParams, params::JuliaParams, base_file_name::String)
    armon_options = [
        "julia", "-t", params.threads
    ]
    append!(armon_options, isempty(measure.node) ? julia_options_no_inti : julia_options)
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

    if params.dimension == 1
        cells_list_str = join(cells_list, ',')
    else
        cells_list_str = join([join(string.(cells), ',') for cells in cells_list], ';')
    end

    append!(armon_options, armon_base_options)
    append!(armon_options, measure.common_armon_params)
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
        "--use-mpi", measure.use_MPI
    ])

    if params.dimension > 1
        append!(armon_options, [
            "--transpose", join(measure.transpose_dims, ','),
            "--splitting", join(measure.axis_splitting, ','),
            "--flat-dims", measure.flatten_time_dims
        ])
    end

    return armon_options
end


function build_data_file_base_name(measure::MeasureParams, 
        processes::Int, distribution::String, node_count::Int,
        threads::Int, use_simd::Int, dimension::Int)
    # Build a file name based on the measurement name and the parameters that don't have a single value
    name = data_dir * measure.name * "/"

    # Build a plot legend entry for the measurement
    legend = ""

    name *= string(measure.device)
    legend *= string(measure.device)

    name *= isempty(measure.node) ? "_local" : "_" * measure.node
    legend *= isempty(measure.node) ? ", local" : (measure.device != CPU ? ", " * measure.node : "")

    if length(measure.distributions) > 1
        name *= "_" * distribution
        legend *= ", " * distribution
    end

    if length(measure.processes) > 1
        name *= "_$(processes)proc"
        legend *= ", $(processes) processes"
    end

    if length(measure.node_count) > 1
        name *= "_$(node_count)nodes"
        legend *= ", $(node_count) nodes"
    end

    if length(measure.dimension) > 1
        name *= "_$(dimension)D"
        legend *= ", $(dimension)D"
    end

    if length(measure.use_simd) > 1
        name *= use_simd ? "_SIMD" : "_NO_SIMD"
        legend *= use_simd ? ", SIMD" : ""
    end

    if length(measure.threads) > 1
        name *= "_$(threads)td"
        legend *= ", $(threads) Threads"
    end

    return name, legend
end


function build_data_file_base_name_omp_params(name::String, legend::String, measure::MeasureParams,
        omp_schedule::String, omp_proc_bind::String, omp_places::String)
    if length(measure.omp_schedule) > 1
        name *= "_$(omp_schedule)"
        legend *= ", $(omp_schedule)"
    end

    if length(measure.omp_proc_bind) > 1
        name *= "_$(omp_proc_bind)"
        legend *= ", bind: $(omp_proc_bind)"
    end

    if length(measure.omp_places) > 1
        name *= "_$(omp_places)"
        legend *= ", places: $(omp_places)"
    end

    return name, legend
end


function build_data_file_base_name(measure::MeasureParams, processes::Int, distribution::String,
        node_count::Int, params::JuliaParams)
    name, legend = build_data_file_base_name(measure, processes, distribution, node_count, params.threads, 
                                             params.use_simd, params.dimension)

    if length(measure.jl_proc_bind) > 1
        name *= "_$(params.jl_proc_bind)"
        legend *= ", bind: $(params.jl_proc_bind)"
    end

    if length(measure.jl_places) > 1
        name *= "_$(params.jl_places)"
        legend *= ", places: $(params.jl_places)"
    end
    
    return name * "_", legend
end


function build_inti_options(measure::MeasureParams, inti_params::IntiParams, threads::Int)
    return [
        "-p", measure.node,
        "-N", inti_params.node_count,                  # Number of nodes to distribute the processes to
        "-n", inti_params.processes,                   # Number of processes
        "-E", "-m block:$(inti_params.distribution)",  # Threads distribution
        # Get the exclusive usage of the node, to make sure that Nvidia GPUs are accessible and to
        # further control threads/memory usage
        "-x",
        "-c", threads
    ]
end


function create_all_data_files_and_plot(measure::MeasureParams)
    plot_commands = []
    hist_commands = []
    plot_MPI_commands = []
    for inti_params in build_inti_combinaisons(measure)
        # Marker style for the plot
        point_type = 5
        
        for parameters in parse_combinaisons(measure)
            if parameters.threads * inti_params.processes > max_inti_cores * inti_params.node_count
                continue
            end

            base_file_name, legend_base = build_data_file_base_name(measure, inti_params.processes, inti_params.distribution, inti_params.node_count, parameters)
            dimension = parameters.dimension

            for (test, transpose_dims, axis_splitting) in armon_combinaisons(measure, dimension)
                data_file_name_base, legend = build_armon_data_file_name(measure, dimension, base_file_name, legend_base, test, transpose_dims, axis_splitting)
                
                legend = replace(legend, '_' => "\\_")  # '_' makes subscripts in gnuplot
                
                data_file_name = data_file_name_base * ".csv"
                open(data_file_name, "w") do _ end  # Create/Clear the file
                plot_cmd = gnuplot_plot_command(data_file_name, legend, point_type)
                push!(plot_commands, plot_cmd)

                if measure.time_histogram
                    hist_file_name = data_file_name_base * "_hist.csv"
                    open(hist_file_name, "w") do _ end  # Create/Clear the file
                    plot_cmd = gnuplot_hist_plot_command(hist_file_name, legend, point_type)
                    push!(hist_commands, plot_cmd)
                end

                if measure.time_MPI_plot
                    MPI_plot_file_name = data_file_name_base * "_MPI_time.csv"
                    open(MPI_plot_file_name, "w") do _ end  # Create/Clear the file
                    plot_cmd = gnuplot_MPI_plot_command_1(MPI_plot_file_name, legend, point_type)
                    push!(plot_MPI_commands, plot_cmd)
                    plot_cmd = gnuplot_MPI_plot_command_2(MPI_plot_file_name, 
                        measure.device == CPU ? ("(relative) " * legend) : (legend * " (relative)"), point_type)
                    push!(plot_MPI_commands, plot_cmd)
                end
            end
        end
    end

    # Create the gnuplot script. It will then be run at each new data point
    open(measure.gnuplot_script, "w") do gnuplot_script
        print(gnuplot_script, base_gnuplot_script_commands(measure.plot_file, measure.plot_title, 
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end

    if measure.time_histogram
        # Same for the histogram plot script
        open(measure.gnuplot_hist_script, "w") do gnuplot_script
            print(gnuplot_script, base_gnuplot_histogram_script_commands(measure.hist_plot_file, measure.plot_title))
            plot_cmd = join(hist_commands, ", \\\n     ")
            println(gnuplot_script, plot_cmd)
        end
    end

    if measure.time_MPI_plot
        # Same for the MPI plot script
        open(measure.gnuplot_MPI_script, "w") do gnuplot_script
            plot_title = measure.plot_title * ", MPI communications time"
            print(gnuplot_script, base_gnuplot_MPI_time_script_commands(measure.time_MPI_plot_file, plot_title,
                measure.log_scale, measure.device == CPU ? "right" : "left"))
            plot_cmd = join(plot_MPI_commands, ", \\\n     ")
            println(gnuplot_script, plot_cmd)
        end
    end
end


function run_measure(measure::MeasureParams, julia_params::JuliaParams, inti_params::IntiParams)
    if julia_params.threads * inti_params.processes > max_inti_cores * inti_params.node_count
        println("Skipping running $(inti_params.processes) Julia processes with $(julia_params.threads) threads on $(inti_params.node_count) nodes.")
        return
    end

    base_file_name, _ = build_data_file_base_name(measure, 
        inti_params.processes, inti_params.distribution, inti_params.node_count, julia_params)
    armon_options = run_backend(measure, julia_params, base_file_name)

    println("""Running Julia with:
 - $(julia_params.threads) threads
 - threads binding: $(julia_params.jl_proc_bind), places: $(julia_params.jl_places)
 - $(julia_params.use_simd == 1 ? "with" : "without") SIMD
 - $(julia_params.dimension)D
 - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
 - with $(inti_params.processes) processes on $(inti_params.node_count) nodes ($(inti_params.distribution) distribution)
""")

    if isempty(measure.node)
        cmd = no_inti_cmd(armon_options, inti_params.processes)
    else
        inti_options = build_inti_options(measure, inti_params, julia_params.threads)
        cmd = inti_cmd(armon_options, inti_options)
    end

    try
        run(cmd)
    catch e
        if isa(e, InterruptException)
            # The user pressed Crtl-C
            println("Interrupted at $(i)/$(length(measures))")
            return
        else
            rethrow(e)
        end
    end
end


function setup_env()
    # Make sure that the output folders exist
    mkpath(data_dir)
    mkpath(plot_scripts_dir)
    mkpath(plots_dir)

    # Are we in a login node?
    in_login_node = startswith(readchomp(`hostname`), "login")
    if in_login_node
        # Check if all of the required modules are loaded
        modules_list_raw = readchomp(`bash -c "module list"`)
        missing_modules = copy(required_modules)
        for module_name in eachmatch(r"\d+\)\s+([^\s]+)", modules_list_raw)
            for (i, missing_module) in enumerate(missing_modules)
                if startswith(module_name.captures[1], missing_module)
                    deleteat!(missing_modules, i)
                    break
                end
            end
        end

        if length(missing_modules) > 0
            println("Missing modules: ", missing_modules)
            error("Some modules are missing")
        end
    end
end


function main()
    measures, start_at = parse_arguments()

    Base.exit_on_sigint(false) # To be able to properly handle Crtl-C
    
    start_time = Dates.now()

    # Main loop, running in the login node, parsing through all measurments to do
    setup_env()
    for (i, measure) in Iterators.drop(enumerate(measures), start_at - 1)
        println(" ==== Measurement $(i)/$(length(measures)): $(measure.name) ==== ")

        if isempty(measure.node)
            @warn "Running outside of INTI: cannot control distribution type" maxlog=1
        end

        # Create the files and plot script once at the beginning
        create_all_data_files_and_plot(measure)

        # For each main parameter combinaison, run a job
        for julia_params in parse_combinaisons(measure)
            for inti_params in build_inti_combinaisons(measure)
                run_measure(measure, julia_params, inti_params)
            end
        end
    end

    end_time = Dates.now()
    duration = Dates.canonicalize(round(end_time - start_time, Dates.Second))
    println("Total time measurements time: ", duration)
end


main()
