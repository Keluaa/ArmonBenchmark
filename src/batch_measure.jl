
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


no_cluster_cmd(armon_options, nprocs) = `mpiexecjl -n $(nprocs) $(armon_options)`
cluster_cmd(armon_options, cluster_options) = `ccc_mprun $(cluster_options) $(armon_options)`

required_modules = ["cuda", #="rocm",=# "hwloc", "mpi"]

project_dir = joinpath(@__DIR__, "..")
data_dir = joinpath(project_dir, "data")
plot_scripts_dir = joinpath(project_dir, "plot_scripts")
plots_dir = joinpath(project_dir, "plots")
plots_update_file = joinpath(plots_dir, "last_update")
sub_scripts_dir = joinpath(project_dir, "sub_scripts")
sub_scripts_output_dir = joinpath(project_dir, "jobs_output")

julia_options = ["-O3", "--check-bounds=no", "--project=$project_dir"]
julia_options_no_cluster = ["-O3", "--check-bounds=no"]
armon_base_options = [
    "--write-output", "0",
    "--verbose", "2"
]
max_node_cores = 128  # Maximum number of cores in a node


abstract type BackendParams end 


include(joinpath(@__DIR__, "gnuplot_commands.jl"))
include(joinpath(@__DIR__, "measure_file_parsing.jl"))
include(joinpath(@__DIR__, "julia/julia_backend.jl"))
include(joinpath(@__DIR__, "kokkos/kokkos_backend.jl"))
include(joinpath(@__DIR__, "cpp/cpp_backend.jl"))


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

    return iter_combinaisons(measure, threads, Val(backend))
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

# TODO : now with "--track-energy" the references might not be needed anymore

function build_data_file_base_name(measure::MeasureParams, 
        processes::Int, distribution::String, node_count::Int,
        threads::Int, use_simd::Int, dimension::Int, backend::Backend)
    # Build a file name based on the measurement name and the parameters that don't have a single value
    name = joinpath(data_dir, measure.name) * "/"

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
                    energy_plot_file_name = data_file_name_base * "_ENERGY.csv"
                    do_combinaison && (open(energy_plot_file_name, "w") do _ end)  # Create/Clear the file
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
    create_plot_file(measure, plot_commands)
    measure.time_histogram && create_histogram_plot_file(measure, hist_commands)
    measure.time_MPI_plot && create_MPI_time_plot_file(measure, plot_MPI_commands)
    measure.energy_plot && create_energy_plot_file(measure, plot_energy_commands)
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


!isinteractive() && main()
