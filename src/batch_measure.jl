
using Printf
using Dates


@enum Device CPU CUDA ROCM
@enum Backend Julia Kokkos CPP
@enum Compiler GCC Clang ICC


mutable struct BatchOptions
    start_at::Int
    do_only::Int
    skip_first::Int
    comb_count::Int
    no_overwrite::Bool
    no_plot_update::Bool
end


function BatchOptions()
    BatchOptions(0, typemax(Int), 0, typemax(Int), false, false)
end


mutable struct MeasureParams
    # Slurm params
    device::Device
    node::String
    distributions::Vector{String}
    processes::Vector{Int}
    node_count::Vector{Int}
    max_time::Int
    use_MPI::Bool

    # Job params
    make_sub_script::Bool
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
    script_dir::String
    repeats::Int
    log_scale::Bool
    error_bars::Bool
    plot_title::String
    verbose::Bool
    use_max_threads::Bool
    cst_cells_per_process::Bool
    limit_to_max_mem::Bool
    track_energy::Bool
    energy_references::Int

    # > Performance plot
    perf_plot::Bool
    gnuplot_script::String
    plot_file::String

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
    energy_script::String
    energy_plot_file::String
end


struct ClusterParams
    processes::Int
    distribution::String
    node_count::Int
end


struct JobStep
    armon_options::Vector{String}
    node_partition::String
    cluster::ClusterParams
    threads::Int
    dimension::Int
    base_file_name::String
    legend::String
end


abstract type BackendParams end


const REQUIRED_MODULES = ["cuda", #= "rocm", =# "hwloc", "mpi"]


const PROJECT_DIR = joinpath(@__DIR__, "..")

const DATA_DIR_NAME         = "data"
const PLOT_SCRIPTS_DIR_NAME = "plot_scripts"
const PLOTS_DIR_NAME        = "plots"
const JOB_SCRIPS_DIR_NAME   = "jobs_scripts"
const JOBS_OUTPUT_DIR_NAME  = "jobs_output"

const ADD_ENERGY_SCRIPT_PATH = joinpath(@__DIR__, "add_energy_data_point.sh")


const ARMON_BASE_OPTIONS = [
    "--write-output", "0"
    "--verbose", "2"
]

const ENERGY_ACCOUNTING_OPTIONS = [
    "--noheader", "--noconvert", "-P", "-o", "ConsumedEnergyRaw"
]


include(joinpath(@__DIR__, "gnuplot_commands.jl"))
include(joinpath(@__DIR__, "measure_file_parsing.jl"))

include(joinpath(@__DIR__, "julia/julia_backend.jl"))
include(joinpath(@__DIR__, "kokkos/kokkos_backend.jl"))
include(joinpath(@__DIR__, "cpp/cpp_backend.jl"))

#
# Parameters combinaisons parsing
#

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


struct ArmonCombinaison
    test::String
    axis_splitting::String
    process_grid::Union{Nothing, Vector{Int}}
    process_grid_ratios::Union{Nothing, Vector{Int}}
end


function armon_combinaisons(measure::MeasureParams, dimension::Int)
    if dimension == 1
        return Iterators.map(
            p -> ArmonCombinaison(p...),
            Iterators.product(
                measure.tests_list,
                ["Sequential"],
                [[1, 1]],
                [nothing]
            )
        )
    else
        return Iterators.map(
            p -> ArmonCombinaison(p...),
            Iterators.product(
                measure.tests_list,
                measure.axis_splitting,
                isnothing(measure.process_grid_ratios) ? measure.process_grids : [nothing],
                isnothing(measure.process_grid_ratios) ? [nothing] : measure.process_grid_ratios
            )
        )
    end
end

#
# Process grids
#

"""
    split_N(N, R)

Splits the number `N` into a list of `length(R)` integers scaled by `R` such that their product is N

Throws `InexactError` if the ratios cannot divide `N` evenly.

```jldoctest
julia> r = split_N(48, (1, 2, 3))
3-element Vector{Int64}:
 2
 4
 6

julia> prod(r)
48

julia> split_N(64, (1, 16))
2-element Vector{Int64}:
  2
 32
```
"""
function split_N(N::Int, R)
    d = length(R)
    n = zeros(Int, d)
    R_prod = prod(R)
    for (i, r) in enumerate(R)
        other_r = R_prod / r
        r_i = r^(d-1) / other_r
        n_i = (N * r_i)^(1/d)
        try
            n_i = convert(Int, round(n_i; sigdigits=10))
        catch e
            if e isa InexactError
                println("The ratios $R cannot divide $N")
            end
            rethrow(e)
        end
        n[i] = n_i
    end
    return n
end


function check_ratio_for_grid(n_proc, ratios)
    return try
        split_N(n_proc, ratios)
        true
    catch
        false
    end
end


function get_process_grid(cluster::ClusterParams, armon_params::ArmonCombinaison)
    if isnothing(armon_params.process_grid)
        if check_ratio_for_grid(cluster.processes, armon_params.process_grid_ratios)
            return split_N(cluster.processes, armon_params.process_grid_ratios)
        else
            return nothing
        end
    else
        return armon_params.process_grid
    end
end

#
# Job steps and submission script
#

# TODO: replace by a Slurm submission script (#SBATCH instead of #MSUB)
job_script_header(
    job_name, job_work_dir, job_stdout_file, job_stderr_file,
    job_partition, job_nodes, job_processes, job_cores, job_time_limit
) = """
#!/bin/bash
#MSUB -r $(job_name)
#MSUB -o $(job_stdout_file)
#MSUB -e $(job_stderr_file)
#MSUB -q $(job_partition)
#MSUB -N $(job_nodes)
#MSUB -n $(job_processes)
#MSUB -c $(job_cores)
#MSUB -T $(job_time_limit)
#MSUB -x

cd $(job_work_dir)
module load $(join(REQUIRED_MODULES, ' '))
"""


function job_step_command_args(step::JobStep; in_sub_script=true)
    cluster = step.cluster
    args = [
        "ccc_mprun",  # TODO: replace by srun, + options
        "-N", cluster.node_count,
        "-n", cluster.processes,
        "-E", "-m block:$(cluster.distribution)",
        "-c", step.threads
    ]
    if !in_sub_script
        append!(args, [
            "-p", step.node_partition,
            "-x"
        ])
    end
end


add_energy_data_command(step_idx, step_cells, file_var) = 
    `sacct \$SACCT_OPTS -j \${SLURM_JOB_ID}.$step_idx | \$ADD_ENERGY_SCRIPT \$$file_var $step_cells`


options_to_str(opts::Vector{String}) = cmd_to_string(Cmd(opts))
cmd_to_string(cmd::Cmd) = string(cmd)[2:end-1]


function command_for_step(step::JobStep; in_sub_script=true)
    options = job_step_command_args(step; in_sub_script)
    append!(options, step.armon_options)
    return options_to_str(options)
end


function make_reference_job_from(step::JobStep)
    ref_step = deepcopy(step)

    i_cells = findfirst(v -> v == "--cells-list", ref_step.armon_options)

    # 60 cells per process in each direction
    i_grid = findfirst(v -> v == "--proc-grid", ref_step.armon_options)
    if isnothing(i_grid)
        if ref_step.dimension > 1
            error("Missing '--proc-grid' in arguments list. Cannot make a reference job.")
        end
        cells = ref_step.cluster.processes * 60
        ref_step.armon_options[i_cells] = string(cells)
    else
        proc_grid = ref_step.armon_options[i_grid+1]
        proc_grid = parse.(Int, split(proc_grid, ','))
        cells = proc_grid .* 60
        ref_step.armon_options[i_cells] = join(cells, ',')
    end

    return ref_step, prod(cells)
end


function create_sub_script(measure::MeasureParams, steps::Vector{JobStep}; header="")
    script_path = joinpath(measure.script_dir, JOB_SCRIPS_DIR_NAME)
    open(script_path, "w") do script
        job_work_dir = PROJECT_DIR
        job_stdout_file = joinpath(measure.script_dir, JOBS_OUTPUT_DIR_NAME, "stdout_$(measure.name).txt")
        job_stderr_file = joinpath(measure.script_dir, JOBS_OUTPUT_DIR_NAME, "stderr_$(measure.name).txt")

        job_processes = maximum(steps) do step
            step.cluster.processes
        end

        println(script, job_script_header(
            measure.name,
            job_work_dir, job_stdout_file, job_stderr_file,
            measure.node, measure.node_count, job_processes, 1,
            measure.max_time
        ))

        if measure.track_energy
            # Energy consumption tracking variables
            energy_ref_data = joinpath(measure.script_dir, DATA_DIR_NAME, measure.name * "_energy_ref.csv")
            energy_data = joinpath(measure.script_dir, DATA_DIR_NAME, measure.name * "_energy.csv")
            println(script, "SACCT_OPTS=\"", options_to_str(ENERGY_ACCOUNTING_OPTIONS), '"')
            println(script, "ADD_ENERGY_SCRIPT=\"", ADD_ENERGY_SCRIPT_PATH, '"')            
            println(script, "ENERGY_REF_DATA=\"", energy_ref_data, '"')
            println(script, "ENERGY_DATA=\"", energy_data, '"')
        end

        !isempty(header) && println(script, '\n', header, '\n')

        ref_count = 0
        if measure.track_energy
            # Energy consumption references
            ref_step, ref_cells = make_reference_job_from(first(steps))

            println(script, "# Initial warmup")
            println(script, command_for_step(ref_step))
            println(script)
            ref_count += 1

            println(script, "# Energy references")
            for _ in measure.energy_references
                println(script, command_for_step(ref_step))
                println(script, cmd_to_string(add_energy_data_command(ref_count, ref_cells, "ENERGY_REF_DATA")))
                ref_count += 1
            end
            println(script)
        end

        for (step_idx, step) in enumerate(steps)
            append_to_sub_script(script, measure, step, ref_count + step_idx - 1)
        end
    end
end


function append_to_sub_script(script::IO, measure::MeasureParams, step::JobStep, step_idx::Int)
    if measure.one_job_per_cell
        # Extract the cells-list option
        i = findfirst(v -> v == "--cells-list", step.armon_options)
        popat!(step.armon_options, i)
        cells_list = popat!(step.armon_options, i)

        # Create a variable holding the common arguments
        println(script, "OPTIONS=\"", options_to_str(step.armon_options), '"')

        step_cmd = options_to_str(job_step_command_args(step)) * " \$OPTIONS"

        # Split each domain in '--cells-list' into their own command
        cells_list = split(cells_list, step.dimension == 1 ? "," : ";")
        for cells in cells_list
            println(script, step_cmd, " --cells-list ", cells)
            if track_energy
                job_cells = prod(parse.(Int, split(cells, ',')))
                println(script, cmd_to_string(add_energy_data_command(step_idx, job_cells, "ENERGY_DATA")))
            end
        end
    else
        println(script, command_for_step(step))
    end
end


function run_job_steps(steps::Vector{JobStep})
    for step in steps
        try
            cmd = command_for_step(step; in_sub_script=false)
            println("Waiting for job to start...")
            run(cmd)
        catch e
            if isa(e, InterruptException)
                # The user pressed Crtl-C
                println("Interrupted")
                return
            else
                rethrow(e)
            end
        end
    end
end

#
# Measure steps and output files
#

function build_job_step(measure::MeasureParams, backend::BackendParams, cluster::ClusterParams)
    backend_name = backend_disp_name(backend)
    num_threads = backend.threads

    if num_threads * cluster.processes > max_node_cores * cluster.node_count
        println("Skipping running $(cluster.processes) $backend_name processes with $num_threads \
                 threads on $(cluster.node_count) nodes.")
        return nothing
    end

    if !isnothing(measure.process_grid_ratios) && 
            !any(map(Base.Fix1(check_ratio_for_grid, cluster.processes), measure.process_grid_ratios))
        println("Skipping running $(cluster.processes) $backend_name processes since none of the \
                 given grid ratios can entirely divide $(cluster.processes)")
        return nothing
    end

    base_file_name, legend = build_data_file_base_name(measure, cluster.processes, cluster.distribution,
        cluster.node_count, backend_params)
    armon_options = run_backend(measure, backend, cluster, base_file_name)

    dimension = backend.dimension

    return JobStep(armon_options, measure.node, cluster, num_threads, dimension, base_file_name, legend)
end


function ensure_dirs_exist(measure::MeasureParams)
    for dir_name in [
                DATA_DIR_NAME,
                PLOT_SCRIPTS_DIR_NAME,
                PLOTS_DIR_NAME,
                JOB_SCRIPS_DIR_NAME,
                JOBS_OUTPUT_DIR_NAME
            ]
        dir_path = joinpath(measure.script_dir, dir_name)
        mkpath(dir_path)
    end
end


function create_all_data_files(measure::MeasureParams, steps::Vector{JobStep};
        no_overwrite=false, no_plot_update=false)

    point_type = 5  # Marker style for the plot
    color_index = 1
    error_bars = measure.error_bars

    perf_plot_cmds = []
    time_hist_cmds = []
    MPI_time_cmds = []
    energy_plot_cmds = []

    ensure_dirs_exist(measure)

    if measure.energy_plot
        ref_file_name = joinpath(measure.script_dir, DATA_DIR_NAME, measure.name * "_energy_ref.csv")
        !no_overwrite && open(identity, ref_file_name, "w")

        file_name = joinpath(measure.script_dir, DATA_DIR_NAME, measure.name * "_energy.csv")
        !no_overwrite && open(identity, file_name, "w")
        legend = replace(measure.name, '_' => "\\_")
        push!(energy_plot_cmds, gp_energy_plot_cmd(file_name, legend, color_index, point_type))
    end

    for step in steps
        for armon_params in armon_combinaisons(measure, step.dimension)
            process_grid = get_process_grid(step.cluster, armon_params)
            isnothing(process_grid) && continue

            legend = replace(step.legend, '_' => "\\_")  # '_' makes subscripts in gnuplot
            incr_color = false

            base_file_path = joinpath(measure.script_dir, DATA_DIR_NAME, step.base_file_name)

            if measure.perf_plot
                file_name = base_file_path * ".csv"
                !no_overwrite && open(identity, file_name, "w")
                push!(perf_plot_cmds, gp_perf_plot_cmd(file_name, legend, point_type; error_bars))
            end

            if measure.time_histogram
                file_name = base_file_path* "_hist.csv"
                !no_overwrite && open(identity, file_name, "w")
                push!(time_hist_cmds, gp_hist_plot_cmd(file_name, legend, point_type))
            end

            if measure.time_MPI_plot
                file_name = base_file_path * "_MPI_time.csv"
                !no_overwrite && open(identity, file_name, "w")
                push!(MPI_time_cmds, gp_MPI_time_cmd(file_name, legend, color_index, point_type))
                percent_legend = measure.device == CPU ? ("(relative) " * legend) : (legend * " (relative)")
                push!(MPI_time_cmds, gp_MPI_percent_cmd(file_name, percent_legend, color_index, point_type))
                incr_color = true
            end

            if incr_color
                color_index += 1  # Each line will have a unique color
            end
        end
    end

    if !no_plot_update
        measure.perf_plot      && create_plot_file(measure, perf_plot_cmds)
        measure.time_histogram && create_histogram_plot_file(measure, time_hist_cmds)
        measure.time_MPI_plot  && create_MPI_time_plot_file(measure, MPI_time_cmds)
        measure.energy_plot    && create_energy_plot_file(measure, energy_plot_cmds, ref_file_name)
    end
end


function build_all_steps_of_measure(measure::MeasureParams, first_measure::Bool, skip_first::Int, comb_count::Int)
    steps = JobStep[]

    comb_i = 0
    comb_c = 0
    for cluster_params in build_cluster_combinaisons(measure)
        for backend in measure.backends, backend_params in parse_combinaisons(measure, cluster_params, backend)
            comb_i += 1
            if first_measure && comb_i <= skip_first
                continue
            elseif comb_c > comb_count
                return steps, true
            end

            job_step = build_job_step(measure, backend_params, cluster_params)
            push!(steps, job_step)

            comb_c += 1
        end
    end

    return steps, false
end

#
# Measurements
#

function do_measures(measures::Vector{MeasureParams}, batch_options::BatchOptions)
    measures_to_do = enumerate(measures)
    measures_to_do = Iterators.drop(measures_to_do, batch_options.start_at - 1)
    measures_to_do = Iterators.take(measures_to_do, batch_options.do_only)
    if length(measures_to_do) == 0
        println("Nothing to do.")
        return
    end

    no_overwrite = batch_options.no_overwrite
    no_plot_update = batch_options.no_plot_update
    first_measure = true
    for (i, measure) in measures_to_do
        println(" ==== Measurement $(i)/$(length(measures)): $(measure.name) ==== ")

        job_steps = build_all_steps_of_measure(measure, first_measure, 
            batch_options.skip_first, batch_options.comb_count)
        first_measure = false

        if isempty(job_steps)
            println("Nothing to do.")
            continue
        end

        create_all_data_files(measure, steps; no_overwrite, no_plot_update)

        if measure.make_sub_script
            create_sub_script(measure, steps;
                header=" ==== Measurement $(i)/$(length(measures)): $(measure.name) ====")
        else
            run_job_steps(steps)
        end
    end
end


function main()
    measures, batch_options = parse_arguments()

    Base.exit_on_sigint(false) # To be able to properly handle Crtl-C
    start_time = Dates.now()

    do_measures(measures, batch_options)

    if any(.! getproperty.(measures, :make_sub_script))
        end_time = Dates.now()
        duration = Dates.canonicalize(round(end_time - start_time, Dates.Second))
        println("Total time measurements time: ", duration)
    end
end


!isinteractive() && main()
