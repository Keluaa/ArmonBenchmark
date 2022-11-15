
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
    create_sub_job_chain::Bool
    add_reference_job::Bool
    one_job_per_cell::Bool

    # Backend params
    threads::Vector{Int}
    use_simd::Vector{Int}
    jl_proc_bind::Vector{String}
    jl_places::Vector{String}
    dimension::Vector{Int}
    async_comms::Vector{Bool}
    jl_mpi_impl::Vector{String}

    # Armon params
    cells_list::Vector{Int}
    domain_list::Vector{Vector{Int}}
    process_grids::Vector{Vector{Int}}
    process_grid_ratios::Union{Nothing, Vector{Vector{Int}}}
    tests_list::Vector{String}
    transpose_dims::Vector{Bool}
    axis_splitting::Vector{String}
    armon_params::Vector{Tuple{Vector{String}, String, String}}  # Tuple: options, legend, name suffix

    # Measurement params
    name::String
    repeats::Int
    gnuplot_script::String
    plot_file::String
    log_scale::Bool
    error_bars::Bool
    plot_title::String
    verbose::Bool
    use_max_threads::Bool
    cst_cells_per_process::Bool
    limit_to_max_mem::Bool

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
    async_comms::Bool
    jl_mpi_impl::String
    options::Tuple{Vector{String}, String, String}
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
plots_update_file = plots_dir * "last_update"
sub_scripts_dir = "./sub_scripts/"
sub_scripts_output_dir = "./jobs_output/"


base_gnuplot_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Giga Cells/sec'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set yrange [0:]
$(log_scale ? "set logscale x" : "")
`echo "$graph_file_name" >> $plots_update_file`
plot """

base_gnuplot_histogram_script_commands(graph_file_name, title) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Total loop time (%)'
set title "$title"
set key left top
set style fill solid 1.00 border 0
set xtics rotate by 90 right
`echo "$graph_file_name" >> $plots_update_file`
plot """

base_gnuplot_MPI_time_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Communications Time [sec]'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set ytics nomirror
set mytics
set yrange [0:]
set y2tics
set my2tics
set y2range [0:]
set y2label 'Communication Time / Total Time [%]'
$(log_scale ? "set logscale x" : "")
`echo "$graph_file_name" >> $plots_update_file`
plot """

gnuplot_plot_command(data_file, legend_title, pt_index; mode="lp") = "'$(data_file)' w $(mode) pt $(pt_index) title '$(legend_title)'"
gnuplot_plot_command_errorbars(data_file, legend_title, pt_index) = gnuplot_plot_command(data_file, legend_title, pt_index; mode="yerrorlines")
gnuplot_hist_plot_command(data_file, legend_title, color_index) = "'$(data_file)' using 2: xtic(1) with histogram lt $(color_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_1(data_file, legend_title, color_index, pt_index) = "'$(data_file)' using 1:2 axis x1y1 w lp lc $(color_index) pt $(pt_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_2(data_file, legend_title, color_index, pt_index) = "'$(data_file)' using 1:(\$2/\$3*100) axis x1y2 w lp lc $(color_index) pt $(pt_index-1) dt 4 title '$(legend_title)'"


sub_script_content(job_name, index, partition, nodes, processes, cores_per_process, max_time, ref_command, commands, next_script) = """
#!/bin/bash
#MSUB -r $(job_name)_$index
#MSUB -o $(sub_scripts_output_dir)stdout_$(job_name)_$index.txt
#MSUB -e $(sub_scripts_output_dir)stderr_$(job_name)_$index.txt
#MSUB -q $partition
#MSUB -N $nodes
#MSUB -n $processes
#MSUB -c $cores_per_process
#MSUB -T $max_time
#MSUB -x
cd \${BRIDGE_MSUB_PWD}
module load $(join(required_modules, ' '))
$(isnothing(ref_command) ? "" : string(ref_command)[2:end-1])
$(join([string(cmd)[2:end-1] for cmd in commands], "\n"))
$(isnothing(ref_command) ? "" : string(ref_command)[2:end-1])
$(!isnothing(next_script) ? "ccc_msub $next_script" : "echo 'All done.'")
"""


function parse_measure_params(file_line_parser)    
    device = CPU
    node = "a100"
    distributions = ["block"]
    processes = [1]
    node_count = [1]
    max_time = 3600  # 1h
    create_sub_job_chain = false
    add_reference_job = false
    one_job_per_cell = false
    threads = [4]
    use_simd = [true]
    jl_places = ["cores"]
    jl_proc_bind = ["close"]
    dimension = [1]
    async_comms = [false]
    jl_mpi_impl = ["async"]
    cells_list = "12.5e3, 25e3, 50e3, 100e3, 200e3, 400e3, 800e3, 1.6e6, 3.2e6, 6.4e6, 12.8e6, 25.6e6, 51.2e6, 102.4e6"
    domain_list = "100,100; 250,250; 500,500; 750,750; 1000,1000"
    process_grids = ["1,1"]
    process_grid_ratios = nothing
    tests_list = ["Sod"]
    transpose_dims = [false]
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
    gnuplot_script = nothing
    plot_file = nothing
    log_scale = true
    error_bars = false
    plot_title = nothing
    verbose = false
    use_max_threads = false
    cst_cells_per_process = false
    limit_to_max_mem = false

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
        elseif option == "create_sub_job_chain"
            create_sub_job_chain = parse(Bool, value)
        elseif option == "add_reference_job"
            add_reference_job = parse(Bool, value)
        elseif option == "one_job_per_cell"
            one_job_per_cell = parse(Bool, value)
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
        elseif option == "async_comms"
            async_comms = parse.(Bool, split(value, ','))
        elseif option == "jl_mpi_impl"
            jl_mpi_impl = split(value, ',')
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
        elseif option == "transpose"
            transpose_dims = parse.(Bool, split(value, ','))
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
        elseif option == "cst_cells_per_process"
            cst_cells_per_process = parse(Bool, value)
        elseif option == "limit_to_max_mem"
            limit_to_max_mem = parse(Bool, value)
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
        # By default, same name as the plot script but as a'.pdf' file
        plot_file = gnuplot_script[1:findlast('.', gnuplot_script)-1] * ".pdf"
    end
    if isnothing(plot_title)
        plot_title = "You forgot to add a title"
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

    params_and_legends = collect(zip(armon_params, armon_params_legends, armon_params_names))

    mkpath(data_dir * name)
    gnuplot_script = plot_scripts_dir * gnuplot_script
    plot_file = plots_dir * plot_file

    gnuplot_hist_script = plot_scripts_dir * name * "_hist.plot"
    hist_plot_file = plots_dir * name * "_hist.pdf"

    gnuplot_MPI_script = plot_scripts_dir * name * "_MPI_time.plot"
    time_MPI_plot_file = plots_dir * name * "_MPI_time.pdf"

    return MeasureParams(device, node, distributions, processes, node_count, max_time, use_MPI,
        create_sub_job_chain, add_reference_job, one_job_per_cell,
        threads, use_simd, jl_proc_bind, jl_places, 
        dimension, async_comms, jl_mpi_impl, cells_list, domain_list, process_grids, process_grid_ratios, tests_list, 
        transpose_dims, axis_splitting, params_and_legends,
        name, repeats, gnuplot_script, plot_file, log_scale, error_bars, plot_title, verbose, use_max_threads, 
        cst_cells_per_process, limit_to_max_mem,
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


const USAGE = """
Usage: 
julia batch_measure.jl [--override-node=<node>,<new node>]
                       [--start-at=<measure index>]
                       [--do-only=<measures count>]
                       [--skip-first=<combinaison count>]
                       [--count=<combinaison count>]
                       [--help]
                       <script files>...'
"""


function parse_arguments()
    if length(ARGS) == 0
        error("Invalid number of arguments.\n" * USAGE)
    end

    start_at = 1
    skip_first = 0
    comb_count = typemax(Int)
    do_only = typemax(Int)

    node_overrides = Dict{String, String}()

    measures::Vector{MeasureParams} = []

    for arg in ARGS
        if (startswith(arg, "--"))
            # Batch parameter
            if (startswith(arg, "--override-node="))
                node, replacement_node = split(split(arg, '=')[2], ',')
                node_overrides[node] = replacement_node
            elseif (startswith(arg, "--start-at="))
                value = split(arg, '=')[2]
                start_at = parse(Int, value)
            elseif (startswith(arg, "--skip-first="))
                value = split(arg, '=')[2]
                skip_first = parse(Int, value)
            elseif (startswith(arg, "--do-only="))
                value = split(arg, '=')[2]
                do_only = parse(Int, value)
            elseif (startswith(arg, "--count="))
                value = split(arg, '=')[2]
                comb_count = parse(Int, value)
            elseif arg == "--help" || arg == "-h"
                println(USAGE)
            else
                error("Wrong batch option: " * arg * "\n" * USAGE)
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

    return measures, start_at, do_only, skip_first, comb_count
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


function parse_combinaisons(measure::MeasureParams, inti_params::IntiParams)
    if measure.use_max_threads
        process_per_node = inti_params.processes ÷ inti_params.node_count
        threads_per_process = max_inti_cores ÷ process_per_node
        return Iterators.map(
            params->JuliaParams(params...),
            Iterators.product(
                measure.jl_places,
                measure.jl_proc_bind,
                [threads_per_process],
                measure.use_simd,
                measure.dimension,
                measure.async_comms,
                measure.jl_mpi_impl
            )
        )
    else
        return Iterators.map(
            params->JuliaParams(params...),    
            Iterators.product(
                measure.jl_places,
                measure.jl_proc_bind,
                measure.threads,
                measure.use_simd,
                measure.dimension,
                measure.async_comms,
                measure.jl_mpi_impl,
                measure.armon_params,
            )
        )
    end
end


function armon_combinaisons(measure::MeasureParams, dimension::Int)
    if dimension == 1
        return Iterators.product(
            measure.tests_list,
            [false],
            ["Sequential"],
            [[1, 1]],
            [nothing]
        )
    else
        return Iterators.product(
            measure.tests_list,
            measure.transpose_dims,
            measure.axis_splitting,
            isnothing(measure.process_grid_ratios) ? measure.process_grids : [nothing],
            isnothing(measure.process_grid_ratios) ? [nothing] : measure.process_grid_ratios
        )
    end
end


function check_ratio_for_grid(n_proc, ratios)
    (rpx, rpy) = ratios
    r = rpx / rpy
    try
        px = convert(Int, √(n_proc * r)) 
        py = convert(Int, √(n_proc / r))
    catch
        return false
    end
    return true
end


function process_ratio_to_grid(n_proc, ratios)
    (rpx, rpy) = ratios
    r = rpx / rpy
    px = convert(Int, √(n_proc * r)) 
    py = convert(Int, √(n_proc / r))
    return px, py
end


function build_armon_data_file_name(measure::MeasureParams, dimension::Int,
        base_file_name::String, legend_base::String,
        test::String, transpose_dims::Bool, axis_splitting::String, process_grid::Vector{Int})
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
            file_name *= "_" * axis_splitting
            legend *= ", " * axis_splitting
        end

        if length(measure.process_grids) > 1
            grid_str = join(process_grid, 'x')
            file_name *= "_pg=$grid_str"
            legend *= ", process grid: $grid_str"
        end

        legend *= ", " * legend_base
    end
    return file_name, legend
end


function run_backend(measure::MeasureParams, params::JuliaParams, inti_params::IntiParams, base_file_name::String)
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

    if measure.cst_cells_per_process
        # Scale the cells by the number of processes
        if params.dimension == 1
            cells_list .*= inti_params.processes
        else
            # We need to distribute the factor along each axis, while keeping the divisibility of 
            # the cells count, since it will be divided by the number of processes along each axis.
            # Therefore we make the new values multiples of 64, but this is still not perfect.
            scale_factor = inti_params.processes^(1/params.dimension)
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
        "--limit-to-mem", measure.limit_to_max_mem
    ])

    if params.dimension > 1
        append!(armon_options, [
            "--async-comms", params.async_comms,
            "--mpi-impl", params.jl_mpi_impl,
            "--transpose", join(measure.transpose_dims, ','),
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
                if check_ratio_for_grid(inti_params.processes, ratio)
            ], ';'))
        end
    end

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end


function run_backend_reference(measure::MeasureParams, params::JuliaParams, inti_params::IntiParams)
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
            "--mpi-impl", params.jl_mpi_impl,
            "--transpose", measure.transpose_dims[1],
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
                if check_ratio_for_grid(inti_params.processes, ratio)
            ], ';'))
        end
    end

    additionnal_options, _ = params.options
    append!(armon_options, additionnal_options)

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

    if length(measure.async_comms) > 1
        async_str = params.async_comms ? "async" : "sync"
        name *= "_$async_str"
        legend *= ", $async_str"
    end

    if length(measure.jl_mpi_impl) > 1
        name *= "_$(params.jl_mpi_impl)"
        legend *= ", MPI $(params.jl_mpi_impl)"
    end

    if !isempty(params.options[2])
        legend *= ", " * params.options[2]
    end

    if !isempty(params.options[3])
        name *= "_" * params.options[3]
    end
    
    return name * "_", legend
end


function build_inti_options(measure::MeasureParams, inti_params::IntiParams, threads::Int)
    if measure.create_sub_job_chain
        # The rest of the parameters are put in the job submission script
        return [
            "-E", "-m block:$(inti_params.distribution)"
        ]
    else
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
end


function create_all_data_files_and_plot(measure::MeasureParams, skip_first::Int)
    plot_commands = []
    hist_commands = []
    plot_MPI_commands = []
    color_index = 1
    comb_i = 0
    for inti_params in build_inti_combinaisons(measure)
        # Marker style for the plot
        point_type = 5
        
        for parameters in parse_combinaisons(measure, inti_params)
            comb_i += 1
            erase_files = comb_i > skip_first
            
            if parameters.threads * inti_params.processes > max_inti_cores * inti_params.node_count
                continue
            end

            base_file_name, legend_base = build_data_file_base_name(measure, inti_params.processes, inti_params.distribution, inti_params.node_count, parameters)
            dimension = parameters.dimension

            for (test, transpose_dims, axis_splitting, process_grid, process_grid_ratio) in armon_combinaisons(measure, dimension)
                if dimension > 1 && isnothing(process_grid)
                    if check_ratio_for_grid(inti_params.processes, process_grid_ratio)
                        px, py = process_ratio_to_grid(inti_params.processes, process_grid_ratio)
                    else
                        continue  # This ratio is incompatible with this process count
                    end
                    process_grid = Int[px, py]
                end
                
                data_file_name_base, legend = build_armon_data_file_name(measure, dimension, base_file_name, legend_base, test, transpose_dims, axis_splitting, process_grid)
                
                legend = replace(legend, '_' => "\\_")  # '_' makes subscripts in gnuplot
                
                data_file_name = data_file_name_base * ".csv"
                erase_files && (open(data_file_name, "w") do _ end)  # Create/Clear the file
                if measure.error_bars
                    plot_cmd = gnuplot_plot_command_errorbars(data_file_name, legend, point_type)
                else
                    plot_cmd = gnuplot_plot_command(data_file_name, legend, point_type)
                end
                push!(plot_commands, plot_cmd)

                if measure.time_histogram
                    hist_file_name = data_file_name_base * "_hist.csv"
                    erase_files && (open(hist_file_name, "w") do _ end)  # Create/Clear the file
                    plot_cmd = gnuplot_hist_plot_command(hist_file_name, legend, point_type)
                    push!(hist_commands, plot_cmd)
                end

                if measure.time_MPI_plot
                    MPI_plot_file_name = data_file_name_base * "_MPI_time.csv"
                    erase_files && (open(MPI_plot_file_name, "w") do _ end)  # Create/Clear the file
                    plot_cmd = gnuplot_MPI_plot_command_1(MPI_plot_file_name, legend, color_index, point_type)
                    push!(plot_MPI_commands, plot_cmd)
                    plot_cmd = gnuplot_MPI_plot_command_2(MPI_plot_file_name, 
                        measure.device == CPU ? ("(relative) " * legend) : (legend * " (relative)"), color_index, point_type)
                    push!(plot_MPI_commands, plot_cmd)
                    color_index += 1  # Each line and its relative counterpart will have the same color
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
        # Same for the MPI plot script
        open(measure.gnuplot_MPI_script, "w") do gnuplot_script
            plot_title = measure.plot_title * ", MPI communications time"
            print(gnuplot_script, base_gnuplot_MPI_time_script_commands(measure.time_MPI_plot_file, plot_title,
                measure.log_scale, measure.device == CPU ? "right" : "left"))
            plot_cmd = join(plot_MPI_commands, ", \\\n     ")
            println(gnuplot_script, plot_cmd)
        end
    end
end


function run_measure(measure::MeasureParams, julia_params::JuliaParams, inti_params::IntiParams, i::Int)
    if julia_params.threads * inti_params.processes > max_inti_cores * inti_params.node_count
        println("Skipping running $(inti_params.processes) Julia processes with $(julia_params.threads) threads on $(inti_params.node_count) nodes.")
        return nothing, nothing
    end

    if !isnothing(measure.process_grid_ratios) && !any(map(Base.Fix1(check_ratio_for_grid, inti_params.processes), measure.process_grid_ratios))
        println("Skipping running $(inti_params.processes) Julia processes since none of the given grid ratios can entirely divide $(inti_params.processes)")
        return nothing, nothing
    end

    base_file_name, _ = build_data_file_base_name(measure, 
        inti_params.processes, inti_params.distribution, inti_params.node_count, julia_params)
    armon_options = run_backend(measure, julia_params, inti_params, base_file_name)

    println("""Running Julia with:
 - $(julia_params.threads) threads
 - threads binding: $(julia_params.jl_proc_bind), places: $(julia_params.jl_places)
 - $(julia_params.use_simd == 1 ? "with" : "without") SIMD
 - $(julia_params.dimension)D
 - $(julia_params.async_comms ? "a" : "")synchronous communications
 - MPI $(julia_params.jl_mpi_impl) implementation
 - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
 - with $(inti_params.processes) processes on $(inti_params.node_count) nodes ($(inti_params.distribution) distribution)
""")

    if measure.create_sub_job_chain
        ref_cmd = nothing
        if measure.add_reference_job
            ref_armon_options = run_backend_reference(measure, julia_params, inti_params)
            if isempty(measure.node)
                ref_cmd = no_inti_cmd(ref_armon_options, inti_params.processes)
            else
                inti_options = build_inti_options(measure, inti_params, julia_params.threads)
                ref_cmd = inti_cmd(ref_armon_options, inti_options)
            end
        end
        
        if measure.one_job_per_cell
            # Split '--cells-list' into their own commands
            i = findfirst(v -> v == "--cells-list", armon_options)
            cells_list = split(armon_options[i+1], julia_params.dimension == 1 ? "," : ";")
            
            cmds = Cmd[]
            for cells in cells_list
                armon_options[i+1] = cells
                if isempty(measure.node)
                    cmd = no_inti_cmd(armon_options, inti_params.processes)
                else
                    inti_options = build_inti_options(measure, inti_params, julia_params.threads)
                    cmd = inti_cmd(armon_options, inti_options)
                end
                push!(cmds, cmd)
            end
        else
            if isempty(measure.node)
                cmd = no_inti_cmd(armon_options, inti_params.processes)
            else
                inti_options = build_inti_options(measure, inti_params, julia_params.threads)
                cmd = inti_cmd(armon_options, inti_options)
            end

            cmds = [cmd]
        end

        return cmds, ref_cmd
    else
        if isempty(measure.node)
            cmd = no_inti_cmd(armon_options, inti_params.processes)
        else
            inti_options = build_inti_options(measure, inti_params, julia_params.threads)
            cmd = inti_cmd(armon_options, inti_options)
        end
    end

    try
        println("Waiting for job to start...")
        run(cmd)
    catch e
        if isa(e, InterruptException)
            # The user pressed Crtl-C
            println("Interrupted at measure n°$(i)")
        else
            rethrow(e)
        end
    end

    return nothing, nothing
end


job_command_type = Tuple{MeasureParams, IntiParams, JuliaParams, Vector{Cmd}, Union{Nothing, Cmd}}


function make_submission_scripts(job_commands::Vector{job_command_type})
    # Remove all previous scripts (if any)
    rm_sub_scripts_files = "rm -f $(sub_scripts_dir)sub_*.sh"
    run(`bash -c $rm_sub_scripts_files`)

    # Save each command to a submission script, which will launch the next command after the job
    # completes.
    for (i, (measure, inti_params, julia_params, commands, ref_command)) in enumerate(job_commands)
        sub_script_name = sub_scripts_dir * measure.name * "_$i.sh"
        open(sub_script_name, "w") do sub_script_file
            if i < length(job_commands)
                next_job_name = job_commands[i+1][1].name
                next_job_file_name = sub_scripts_dir * next_job_name * "_$(i+1).sh"
            else
                next_job_file_name = nothing
            end

            print(sub_script_file, 
                sub_script_content(measure.name, i, measure.node, 
                    inti_params.node_count, inti_params.processes, julia_params.threads, measure.max_time,
                    ref_command, commands,
                    next_job_file_name))
        end
        println("Created job submission script $sub_script_name")
    end
end


function setup_env()
    # Make sure that the output folders exist
    mkpath(data_dir)
    mkpath(plot_scripts_dir)
    mkpath(plots_dir)
    mkpath(sub_scripts_dir)
    mkpath(sub_scripts_output_dir)

    # Clear the plots update file
    run(`truncate -s 0 $plots_update_file`)

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
    measures, start_at, do_only, skip_first, comb_count = parse_arguments()

    Base.exit_on_sigint(false) # To be able to properly handle Crtl-C
    
    start_time = Dates.now()

    job_commands = job_command_type[]

    # Main loop, running in the login node, parsing through all measurments to do
    setup_env()

    measures_to_do = Iterators.take(Iterators.drop(enumerate(measures), start_at - 1), do_only)
    if length(measures_to_do) == 0
        println("Nothing to do.")
        return
    end

    for (i, measure) in measures_to_do
        println(" ==== Measurement $(i)/$(length(measures)): $(measure.name) ==== ")

        if isempty(measure.node)
            @warn "Running outside of INTI: cannot control distribution type" maxlog=1
        end

        # Create the files and plot script once at the beginning
        create_all_data_files_and_plot(measure, i == start_at ? skip_first : 0)  # TODO : take into account 'comb_count', here the other runs after will still have their data overwritten

        # For each main parameter combinaison, run a job
        comb_i = 0
        comb_c = 0
        for inti_params in build_inti_combinaisons(measure)
            for julia_params in parse_combinaisons(measure, inti_params)
                comb_i += 1
                if i == start_at && comb_i <= skip_first
                    continue
                elseif comb_c > comb_count
                    @goto end_loop
                end

                commands, ref_command = run_measure(measure, julia_params, inti_params, i)

                if measure.create_sub_job_chain && !isnothing(commands)
                    push!(job_commands, (measure, inti_params, julia_params, commands, ref_command))
                end

                comb_c += 1
            end
        end
    end

    @label end_loop

    if !isempty(job_commands)
        make_submission_scripts(job_commands)
    else
        end_time = Dates.now()
        duration = Dates.canonicalize(round(end_time - start_time, Dates.Second))
        println("Total time measurements time: ", duration)
    end
end


main()
