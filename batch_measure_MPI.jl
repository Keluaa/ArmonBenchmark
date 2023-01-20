
using Printf
using Dates

@enum Device CPU CUDA ROCM
@enum Backend Julia Kokkos CPP
@enum Compiler GCC Clang ICC


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
    backends::Vector{Backend}
    compilers::Vector{Compiler}
    threads::Vector{Int}
    use_simd::Vector{Int}
    jl_proc_bind::Vector{String}
    jl_places::Vector{String}
    omp_proc_bind::Vector{String}
    omp_places::Vector{String}
    dimension::Vector{Int}
    async_comms::Vector{Bool}
    ieee_bits::Vector{Int}
    block_sizes::Vector{Int}

    # Armon params
    cells_list::Vector{Int}
    domain_list::Vector{Vector{Int}}
    process_grids::Vector{Vector{Int}}
    process_grid_ratios::Union{Nothing, Vector{Vector{Int}}}
    tests_list::Vector{String}
    axis_splitting::Vector{String}
    armon_params::Vector{Tuple{Vector{String}, String, String}}  # Tuple: options, legend, name suffix

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
    track_energy::Bool

    # > Time histogram
    time_histogram::Bool
    flatten_time_dims::Bool
    gnuplot_hist_script::String
    hist_plot_file::String

    # > MPI communications time plot
    time_MPI_plot::Bool
    gnuplot_MPI_script::String
    time_MPI_plot_file::String

    # > Energy plot
    energy_plot::Bool
    energy_plot_reps::Bool
    energy_script::String
    energy_plot_file::String
end


struct ClusterParams
    processes::Int
    distribution::String
    node_count::Int
end


abstract type BackendParams end 


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


no_cluster_cmd(armon_options, nprocs) = `mpiexecjl -n $(nprocs) $(armon_options)`
cluster_cmd(armon_options, cluster_options) = `ccc_mprun $(cluster_options) $(armon_options)`

julia_options = ["-O3", "--check-bounds=no", "--project"]
julia_options_no_cluster = ["-O3", "--check-bounds=no"]
armon_base_options = [
    "--write-output", "0",
    "--verbose", "2"
]
max_node_cores = 128  # Maximum number of cores in a node

required_modules = ["cuda", #="rocm",=# "hwloc", "mpi"]

julia_script_path = "./julia/run_julia.jl"
kokkos_script_path = "./kokkos/run_kokkos.jl"
cpp_script_path = "./cpp/run_cpp.jl"

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


base_gnuplot_energy_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Energy Consumption [J]'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set yrange [0:]
$(log_scale ? "set logscale x" : "")
`echo "$graph_file_name" >> $plots_update_file`
plot """

gnuplot_plot_command(data_file, legend_title, pt_index; mode="lp") = "'$(data_file)' w $(mode) pt $(pt_index) title '$(legend_title)'"
gnuplot_plot_command_errorbars(data_file, legend_title, pt_index) = gnuplot_plot_command(data_file, legend_title, pt_index; mode="yerrorlines")
gnuplot_hist_plot_command(data_file, legend_title, color_index) = "'$(data_file)' using 2: xtic(1) with histogram lt $(color_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_1(data_file, legend_title, color_index, pt_index) = "'$(data_file)' using 1:2 axis x1y1 w lp lc $(color_index) pt $(pt_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_2(data_file, legend_title, color_index, pt_index) = "'$(data_file)' using 1:(\$2/\$3*100) axis x1y2 w lp lc $(color_index) pt $(pt_index-1) dt 4 title '$(legend_title)'"
gnuplot_energy_plot_command(data_file, legend_title, color_index, pt_index; mode="lp") = "'$(data_file)' using 1:2:3 w $(mode) lc $(color_index) pt $(pt_index) t '$(legend_title)'"
gnuplot_energy_plot_command_errorbars(data_file, legend_title, color_index, pt_index) = gnuplot_energy_plot_command(data_file, legend_title, color_index, pt_index; mode="yerrorlines")
gnuplot_energy_vals_plot_command(data_file, legend_title, color_index, pt_index, rep) = "'$(data_file)' u 1:$(rep+3) w lp lc $(color_index) dt 3 pt $(pt_index) t '$(legend_title)'"

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
    backends = [Julia]
    compilers = [GCC]
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
    ieee_bits = [64]
    block_sizes = [1024]
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
    gnuplot_script = nothing
    plot_file = nothing
    log_scale = true
    error_bars = false
    plot_title = nothing
    verbose = false
    use_max_threads = false
    cst_cells_per_process = false
    limit_to_max_mem = false

    track_energy = false
    energy_plot = false
    energy_plot_reps = false

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

        option, value = split(line, '=') .|> strip
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
        elseif option == "jl_mpi_impl"
            @warn "'jl_mpi_impl' option is ignored" maxlog=1
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
            @warn "'transpose' option is ignored" maxlog=1
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
        elseif option == "track_energy"
            track_energy = parse(Bool, value)
        elseif option == "energy_plot"
            energy_plot = parse(Bool, value)
        elseif option == "energy_plot_reps"
            energy_plot_reps = parse(Bool, value)
        elseif option == "time_hist"
            time_histogram = parse(Bool, value)
        elseif option == "flat_hist_dims"
            flatten_time_dims = parse(Bool, value)
        elseif option == "time_MPI_plot"
            time_MPI_plot = parse(Bool, value)
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

    energy_script = plot_scripts_dir * name * "_Energy.plot"
    energy_plot_file = plots_dir * name * "_Energy.pdf"

    return MeasureParams(device, node, distributions, processes, node_count, max_time, use_MPI,
        create_sub_job_chain, add_reference_job, one_job_per_cell,
        backends, compilers, threads, use_simd, jl_proc_bind, jl_places, omp_proc_bind, omp_places,
        dimension, async_comms, ieee_bits, block_sizes,
        cells_list, domain_list, process_grids, process_grid_ratios, tests_list, 
        axis_splitting, params_and_legends,
        name, repeats, gnuplot_script, plot_file, log_scale, error_bars, plot_title, verbose,
        use_max_threads, cst_cells_per_process, limit_to_max_mem, track_energy,
        time_histogram, flatten_time_dims, gnuplot_hist_script, hist_plot_file,
        time_MPI_plot, gnuplot_MPI_script, time_MPI_plot_file,
        energy_plot, energy_plot_reps, energy_script, energy_plot_file)
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
                       [--help|-h]
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
        if (startswith(arg, "-"))
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
                exit(0)
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


function build_cluster_combinaisons(measure::MeasureParams)
    return Iterators.map(
        params->ClusterParams(params...),
        Iterators.product(
            measure.processes,
            measure.distributions,
            measure.node_count
        )
    )
end


function parse_combinaisons(measure::MeasureParams, cluster_params::ClusterParams, backend::Backend)
    if measure.use_max_threads
        process_per_node = cluster_params.processes ÷ cluster_params.node_count
        threads_per_process = max_node_cores ÷ process_per_node
        threads = [threads_per_process]
    else
        threads = measure.threads
    end

    if backend == Julia
        return Iterators.map(
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
    elseif backend == Kokkos
        return Iterators.map(
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
    elseif backend == CPP
        return Iterators.map(
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
    else
        error("Unknown backend: $backend")
    end
end


function armon_combinaisons(measure::MeasureParams, dimension::Int)
    if dimension == 1
        return Iterators.product(
            measure.tests_list,
            ["Sequential"],
            [[1, 1]],
            [nothing]
        )
    else
        return Iterators.product(
            measure.tests_list,
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
        test::String, axis_splitting::String, process_grid::Vector{Int})
    file_name = base_file_name * test
    if dimension == 1
        legend = "$test, $legend_base"
    else
        legend = test

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


function run_backend(measure::MeasureParams, params::CppParams, cluster_params::ClusterParams, base_file_name::String)
    if cluster_params.processes > 1
        error("The C++ backend doesn't support MPI")
    end

    if measure.limit_to_max_mem
        @warn "The C++ backend doesn't support the 'limit_to_max_mem'" maxlog=1
    end

    if measure.time_histogram
        @warn "The C++ backend doesn't support the 'time_histogram'" maxlog=1
    end

    armon_options = Any["julia"]
    append!(armon_options, isempty(measure.node) ? julia_options_no_cluster : julia_options)
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
        cells_list .*= cluster_params.processes
    end

    cells_list_str = join(cells_list, ',')

    append!(armon_options, armon_base_options)
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
        "--compiler", params.compiler,
        "--track-energy", measure.track_energy
    ])

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end

# TODO : now with "--track-energy" the references might not be needed anymore

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


function run_backend_reference(measure::MeasureParams, params::CppParams, cluster_params::ClusterParams)
    if cluster_params.processes > 1
        error("The C++ backend doesn't support MPI")
    end

    if measure.limit_to_max_mem
        @warn "The C++ backend doesn't support the 'limit_to_max_mem'" maxlog=1
    end

    if measure.time_histogram
        @warn "The C++ backend doesn't support the 'time_histogram'" maxlog=1
    end

    armon_options = ["julia"]
    append!(armon_options, isempty(measure.node) ? julia_options_no_cluster : julia_options)
    push!(armon_options, cpp_script_path)

    if measure.device != CPU
        error("The C++ backend works only on the CPU")
    end

    if params.dimension == 1
        cells_list = [1000]
        cells_list_str = join(cells_list, ',')
    else
        error("The C++ backend works only in 1D")
    end

    append!(armon_options, armon_base_options)
    append!(armon_options, [
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

    additionnal_options, _, _ = params.options
    append!(armon_options, additionnal_options)

    return armon_options
end


function build_data_file_base_name(measure::MeasureParams, 
        processes::Int, distribution::String, node_count::Int,
        threads::Int, use_simd::Int, dimension::Int, backend::Backend)
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
        name *= use_simd == 1 ? "_SIMD" : "_NO_SIMD"
        legend *= use_simd == 1 ? ", SIMD" : ""
    end

    if length(measure.threads) > 1
        name *= "_$(threads)td"
        legend *= ", $(threads) Threads"
    end

    if length(measure.backends) > 1
        name *= "_$(backend)"
        legend *= ", $(backend)"
    end

    return name, legend
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


function build_cluster_options(measure::MeasureParams, cluster_params::ClusterParams, threads::Int)
    if measure.create_sub_job_chain
        # The rest of the parameters are put in the job submission script
        return [
            "-E", "-m block:$(cluster_params.distribution)"
        ]
    else
        return [
            "-p", measure.node,
            "-N", cluster_params.node_count,                  # Number of nodes to distribute the processes to
            "-n", cluster_params.processes,                   # Number of processes
            "-E", "-m block:$(cluster_params.distribution)",  # Threads distribution
            # Get the exclusive usage of the node, to make sure that Nvidia GPUs are accessible and to
            # further control threads/memory usage
            "-x",
            "-c", threads
        ]
    end
end


function create_all_data_files_and_plot(measure::MeasureParams, skip_first::Int, comb_count::Int)
    plot_commands = []
    hist_commands = []
    plot_MPI_commands = []
    plot_energy_commands = []
    color_index = 1
    comb_i = 0
    comb_c = 0
    for cluster_params in build_cluster_combinaisons(measure)
        # Marker style for the plot
        point_type = 5

        for backend in measure.backends, parameters in parse_combinaisons(measure, cluster_params, backend)
            comb_i += 1
            do_combinaison = comb_i > skip_first && comb_c <= comb_count

            if parameters.threads * cluster_params.processes > max_node_cores * cluster_params.node_count
                do_combinaison && (comb_c += 1)
                continue
            end

            base_file_name, legend_base = build_data_file_base_name(measure, cluster_params.processes, cluster_params.distribution, cluster_params.node_count, parameters)
            dimension = parameters.dimension

            for (test, axis_splitting, process_grid, process_grid_ratio) in armon_combinaisons(measure, dimension)
                if dimension > 1 && isnothing(process_grid)
                    if check_ratio_for_grid(cluster_params.processes, process_grid_ratio)
                        px, py = process_ratio_to_grid(cluster_params.processes, process_grid_ratio)
                    else
                        continue  # This ratio is incompatible with this process count
                    end
                    process_grid = Int[px, py]
                end

                data_file_name_base, legend = build_armon_data_file_name(measure, dimension, base_file_name, legend_base, test, axis_splitting, process_grid)

                legend = replace(legend, '_' => "\\_")  # '_' makes subscripts in gnuplot

                data_file_name = data_file_name_base * ".csv"
                do_combinaison && (open(data_file_name, "w") do _ end)  # Create/Clear the file
                if measure.error_bars
                    plot_cmd = gnuplot_plot_command_errorbars(data_file_name, legend, point_type)
                else
                    plot_cmd = gnuplot_plot_command(data_file_name, legend, point_type)
                end
                push!(plot_commands, plot_cmd)

                if measure.time_histogram
                    hist_file_name = data_file_name_base * "_hist.csv"
                    do_combinaison && (open(hist_file_name, "w") do _ end)  # Create/Clear the file
                    plot_cmd = gnuplot_hist_plot_command(hist_file_name, legend, point_type)
                    push!(hist_commands, plot_cmd)
                end

                if measure.time_MPI_plot
                    MPI_plot_file_name = data_file_name_base * "_MPI_time.csv"
                    do_combinaison && (open(MPI_plot_file_name, "w") do _ end)  # Create/Clear the file
                    plot_cmd = gnuplot_MPI_plot_command_1(MPI_plot_file_name, legend, color_index, point_type)
                    push!(plot_MPI_commands, plot_cmd)
                    plot_cmd = gnuplot_MPI_plot_command_2(MPI_plot_file_name, 
                        measure.device == CPU ? ("(relative) " * legend) : (legend * " (relative)"), color_index, point_type)
                    push!(plot_MPI_commands, plot_cmd)
                end

                if measure.energy_plot
                    energy_plot_file_name = data_file_name_base * "_Energy.csv"
                    do_combinaison && (open(energy_plot_file_name, "w") do _ end)  # Create/Clear the file
                    if measure.error_bars
                        plot_cmd = gnuplot_energy_plot_command_errorbars(data_file_name, legend, color_index, point_type)
                    else
                        plot_cmd = gnuplot_energy_plot_command(data_file_name, legend, color_index, point_type)
                    end
                    push!(plot_energy_commands, plot_cmd)
                    measure.energy_plot_reps && for r in measure.repeats
                        plot_cmd = gnuplot_energy_vals_plot_command(data_file_name, legend * " - $r", color_index, point_type, r)
                        push!(plot_energy_commands, plot_cmd)
                    end
                end

                if measure.time_MPI_plot || measure.energy_plot
                    color_index += 1  # Each line and its relative counterpart(s) will have the same color
                end
            end

            do_combinaison && (comb_c += 1)
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

    if measure.energy_plot
        # Same for the energy plot
        open(measure.energy_script, "w") do gnuplot_script
            plot_title = measure.plot_title * ", Energy consumption"
            print(gnuplot_script, base_gnuplot_energy_script_commands(measure.energy_plot_file, plot_title,
                measure.log_scale, measure.device == CPU ? "right" : "left"))
            plot_cmd = join(plot_energy_commands, ", \\\n     ")
            println(gnuplot_script, plot_cmd)
        end
    end
end


backend_disp_name(::JuliaParams) = "Julia"
backend_disp_name(::KokkosParams) = "C++ Kokkos"
backend_disp_name(::CppParams) = "C++ OpenMP"


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


function run_backend_msg(measure::MeasureParams, cpp_params::CppParams, cluster_params::ClusterParams)
    """Running C++ OpenMP backend with:
     - $(cpp_params.threads) threads
     - threads binding: $(cpp_params.omp_proc_bind), places: $(cpp_params.omp_places)
     - $(cpp_params.use_simd == 1 ? "with" : "without") SIMD
     - 1D
     - compiled with $(cpp_params.compiler)
     - on $(string(measure.device)), node: $(isempty(measure.node) ? "local" : measure.node)
     - with $(cluster_params.processes) processes on $(cluster_params.node_count) nodes ($(cluster_params.distribution) distribution)
    """
end


function run_measure(measure::MeasureParams, backend_params::BackendParams, cluster_params::ClusterParams, i::Int)
    backend_name = backend_disp_name(backend_params)
    num_threads = backend_params.threads

    if num_threads * cluster_params.processes > max_node_cores * cluster_params.node_count
        println("Skipping running $(cluster_params.processes) $(backend_name) processes with $(num_threads) threads on $(cluster_params.node_count) nodes.")
        return nothing, nothing
    end

    if !isnothing(measure.process_grid_ratios) && !any(map(Base.Fix1(check_ratio_for_grid, cluster_params.processes), measure.process_grid_ratios))
        println("Skipping running $(cluster_params.processes) $(backend_name) processes since none of the given grid ratios can entirely divide $(cluster_params.processes)")
        return nothing, nothing
    end

    base_file_name, _ = build_data_file_base_name(measure, 
        cluster_params.processes, cluster_params.distribution, cluster_params.node_count, backend_params)
    armon_options = run_backend(measure, backend_params, cluster_params, base_file_name)

    println(run_backend_msg(measure, backend_params, cluster_params))

    if measure.create_sub_job_chain
        ref_cmd = nothing
        if measure.add_reference_job
            ref_armon_options = run_backend_reference(measure, backend_params, cluster_params)
            if isempty(measure.node)
                ref_cmd = no_cluster_cmd(ref_armon_options, cluster_params.processes)
            else
                cluster_options = build_cluster_options(measure, cluster_params, num_threads)
                ref_cmd = cluster_cmd(ref_armon_options, cluster_options)
            end
        end

        if measure.one_job_per_cell
            # Split '--cells-list' into their own commands
            i = findfirst(v -> v == "--cells-list", armon_options)
            cells_list = split(armon_options[i+1], backend_params.dimension == 1 ? "," : ";")

            cmds = Cmd[]
            for cells in cells_list
                armon_options[i+1] = cells
                if isempty(measure.node)
                    cmd = no_cluster_cmd(armon_options, cluster_params.processes)
                else
                    cluster_options = build_cluster_options(measure, cluster_params, num_threads)
                    cmd = cluster_cmd(armon_options, cluster_options)
                end
                push!(cmds, cmd)
            end
        else
            if isempty(measure.node)
                cmd = no_cluster_cmd(armon_options, cluster_params.processes)
            else
                cluster_options = build_cluster_options(measure, cluster_params, num_threads)
                cmd = cluster_cmd(armon_options, cluster_options)
            end

            cmds = [cmd]
        end

        return cmds, ref_cmd
    else
        if isempty(measure.node)
            cmd = no_cluster_cmd(armon_options, cluster_params.processes)
        else
            cluster_options = build_cluster_options(measure, cluster_params, num_threads)
            cmd = cluster_cmd(armon_options, cluster_options)
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


job_command_type = Tuple{MeasureParams, ClusterParams, BackendParams, Vector{Cmd}, Union{Nothing, Cmd}}


function make_submission_scripts(job_commands::Vector{job_command_type})
    # Remove all previous scripts (if any)
    rm_sub_scripts_files = "rm -f $(sub_scripts_dir)sub_*.sh"
    run(`bash -c $rm_sub_scripts_files`)

    # Save each command to a submission script, which will launch the next command after the job
    # completes.
    for (i, (measure, cluster_params, backend_params, commands, ref_command)) in enumerate(job_commands)
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
                    cluster_params.node_count, cluster_params.processes, backend_params.threads, measure.max_time,
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

    # Clear the plots update file
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
            @warn "Running outside of a cluster: cannot control distribution type" maxlog=1
        end

        # Create the files and plot script once at the beginning
        create_all_data_files_and_plot(measure, i == start_at ? skip_first : 0, comb_count)

        # For each main parameter combinaison, run a job
        comb_i = 0
        comb_c = 0
        for cluster_params in build_cluster_combinaisons(measure)
            for backend in measure.backends, backend_params in parse_combinaisons(measure, cluster_params, backend)
                comb_i += 1
                if i == start_at && comb_i <= skip_first
                    continue
                elseif comb_c > comb_count
                    @goto end_loop
                end

                commands, ref_command = run_measure(measure, backend_params, cluster_params, i)

                if measure.create_sub_job_chain && !isnothing(commands)
                    push!(job_commands, (measure, cluster_params, backend_params, commands, ref_command))
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
