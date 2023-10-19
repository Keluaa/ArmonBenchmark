
using Printf
using Dates


@enum Device CPU CUDA ROCM
@enum Backend Julia Kokkos CPP
@enum Compiler GCC Clang ICC AOCC ICX


mutable struct BatchOptions
    start_at::Int
    do_only::Int
    skip_first::Int
    comb_count::Int
    no_overwrite::Bool
    no_plot_update::Bool
    submit_now::Bool
    one_script_per_step::Bool
end


function BatchOptions()
    BatchOptions(1, typemax(Int), 0, typemax(Int), false, false, false, false)
end


mutable struct MeasureParams
    # Slurm params
    device::Device
    node::String
    distributions::Vector{String}
    processes::Vector{Int}
    node_count::Vector{Int}
    processes_per_node::Int
    max_time::Int
    use_MPI::Bool

    # Job params
    make_sub_script::Bool
    one_job_per_cell::Bool
    one_script_per_step::Bool

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
    use_kokkos::Vector{Bool}
    kokkos_backends::Vector{String}
    use_md_iter::Vector{Int}
    cmake_options::String
    kokkos_version::String

    # Armon params
    cycles::Int
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
    track_energy::Bool
    energy_references::Int
    process_scaling::Bool
    min_acquisition_time::Int

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


mutable struct ClusterParams
    processes::Int
    distribution::String
    node_count::Int
end


abstract type BackendParams end


mutable struct PlotInfo
    data_file::String
    do_plot::Bool
    plot_script::String
end


mutable struct JobStep
    cluster::ClusterParams
    backend::BackendParams

    node_partition::String
    threads::Int
    dimension::Int
    cells_list::Vector{Vector{Int}}
    proc_grids::Vector{Vector{Int}}
    repeats::Int
    cycles::Int

    base_file_name::String
    legend::String

    perf_plot::PlotInfo
    time_hist::PlotInfo
    time_MPI::PlotInfo
    energy_plot::PlotInfo

    options::Dict{Symbol, Any}
end


const REQUIRED_MODULES = ["cuda", #= "rocm", =# "hwloc", "mpi", "cmake/3.22.2"]


const PROJECT_DIR = joinpath(@__DIR__, "..")

const DATA_DIR_NAME         = "data"
const PLOT_SCRIPTS_DIR_NAME = "plot_scripts"
const PLOTS_DIR_NAME        = "plots"
const JOB_SCRIPS_DIR_NAME   = "jobs_scripts"
const JOBS_OUTPUT_DIR_NAME  = "jobs_output"

const ADD_ENERGY_SCRIPT_PATH = joinpath(@__DIR__, "add_energy_data_point.sh")
const RECENT_JOBS_FILE = joinpath(@__DIR__, "recent_jobs.txt")

const DEFAULT_MAX_NODE_CORES = 128

const ARMON_BASE_OPTIONS = [
    "--write-output", "0",
    "--verbose", "2"
]

const ENERGY_ACCOUNTING_OPTIONS = [
    "--noheader", "--noconvert", "-P", "-o", "ConsumedEnergyRaw"
]


include(joinpath(@__DIR__, "common_utils.jl"))
include(joinpath(@__DIR__, "gnuplot_commands.jl"))
include(joinpath(@__DIR__, "measure_file_parsing.jl"))

include(joinpath(@__DIR__, "julia/julia_backend.jl"))
include(joinpath(@__DIR__, "kokkos/kokkos_backend.jl"))
include(joinpath(@__DIR__, "cpp/cpp_backend.jl"))

#
# Parameters combinaisons parsing
#

function build_cluster_combinaisons(measure::MeasureParams)
    if measure.processes_per_node > 0
        node_count = [0]
    else
        node_count = measure.node_count
    end

    return Iterators.map(
        params->ClusterParams(params...),
        Iterators.product(
            measure.processes,
            measure.distributions,
            node_count
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
            error("The ratios $R cannot divide $N")
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
# Slurm interaction
#

const IS_SLURM_AVAILABLE = try read(`type sinfo`); true catch; false end
const CACHED_PARTITIONS_INFO = Dict{String, Dict}()

function get_partition_info(partition::String)
    !IS_SLURM_AVAILABLE && error("Slurm is not available in this shell")

    if haskey(CACHED_PARTITIONS_INFO, partition)
        return CACHED_PARTITIONS_INFO[partition]
    end

    info_str = readchomp(`sinfo --exact --noheader --partition $partition -o '%R, %D, %c, %X, %Y, %Z'`)
    info_str = split.(split(info_str, '\n'), ", ")

    if length(info_str) > 1
        @info "Partition '$partition' has more than one configuration. We choose the one with the most nodes."
        sort!(info_str; by=(i) -> parse(Int, i[2]))
    end

    info_str = first(info_str)

    infos = Dict{Symbol, Int}()
    infos[:nodes]   = parse(Int, info_str[2])
    infos[:cpus]    = parse(Int, info_str[3])
    infos[:sockets] = parse(Int, info_str[4])
    infos[:cores]   = parse(Int, info_str[5])
    infos[:threads] = parse(Int, info_str[6])

    CACHED_PARTITIONS_INFO[partition] = infos
    return infos
end


function get_max_cores_of_partition(partition::String)
    if IS_SLURM_AVAILABLE
        return get_partition_info(partition)[:cpus]
    else
        @warn "Slurm is unavailable, using $DEFAULT_MAX_NODE_CORES as the maximum number of cores in a node" maxlog=1
        return DEFAULT_MAX_NODE_CORES
    end
end


function submit_script(script_path)
    output = readchomp(`ccc_msub $script_path`)  # TODO: replace by `sbatch`
    println(script_path, " => ", output)
    job_id = match(r"\d+$", output)
    return isnothing(job_id) ? job_id : job_id.match
end


scratch_dir() = get(ENV, "CCCSCRATCHDIR", tempdir())  # TODO: replace by a local preference

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
        "ccc_mprun",  # TODO: replace by srun, + options (try with '-v' to see the options)
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
    return args
end


add_energy_data_command(step_idx, file_var, tmp_file_var) = 
    "sleep 1 && sacct \$SACCT_OPTS -j \${SLURM_JOB_ID}.$step_idx | \$ADD_ENERGY_SCRIPT \$$file_var \$$tmp_file_var"


build_backend_command(step::JobStep) = build_backend_command(step, Val(backend_type(step.backend)))


cmd_to_string(cmd::Cmd) = string(cmd)[2:end-1]


function options_to_str(opts::Vector)
    cmd_str = cmd_to_string(`$opts`)
    return replace(cmd_str, '€' => '$')  # Workaround to build commands using variables
end


function command_for_step(step::JobStep; in_sub_script=true)
    options = job_step_command_args(step; in_sub_script)
    cmd_str = options_to_str(options)
    if in_sub_script
        cmd_str *= " \\\n    "
    end
    cmd_str *= options_to_str(build_backend_command(step))
    return cmd_str
end


function energy_ref_file(step::JobStep)
    ref_file_name, ext = splitext(step.energy_plot.data_file)
    return ref_file_name * "_ref" * ext
end


function make_reference_job_from(step::JobStep)
    # A reference job is only used to measure the energy consumption overhead, therefore it should
    # only use a very limited number of cells, while using very similar code paths with the real
    # steps.
    ref_step = deepcopy(step)

    # 60 cells per process in each direction
    cells = first(ref_step.proc_grids) .* 60
    ref_step.cells_list = [cells]

    ref_step.perf_plot.do_plot = false
    ref_step.time_hist.do_plot = false
    ref_step.time_MPI.do_plot = false

    adjust_reference_job(ref_step, Val(backend_type(ref_step.backend)))

    return ref_step
end


function create_sub_script(
    measure::MeasureParams, steps::Vector{JobStep};
    header="", name_suffix="", step_count=length(steps), step_idx_offset=0
)
    script_name = measure.name * name_suffix * ".sh"
    script_path = joinpath(measure.script_dir, JOB_SCRIPS_DIR_NAME, script_name)

    open(script_path, "w") do script
        job_work_dir = abspath(measure.script_dir)
        job_output_file_name = "$(measure.name)$(name_suffix)_%I"  # '%I' is replaced by the job ID
        job_stdout_file = joinpath(job_work_dir, JOBS_OUTPUT_DIR_NAME, job_output_file_name * "_stdout.txt") |> abspath
        job_stderr_file = joinpath(job_work_dir, JOBS_OUTPUT_DIR_NAME, job_output_file_name * "_stderr.txt") |> abspath

        job_processes = maximum(steps) do step
            step.cluster.processes
        end
        job_nodes = maximum(steps) do step
            step.cluster.node_count
        end

        println(script, job_script_header(
            measure.name,
            job_work_dir, job_stdout_file, job_stderr_file,
            measure.node, job_nodes, job_processes, 1,
            measure.max_time
        ))

        if measure.track_energy
            # Energy consumption tracking variables
            println(script, "SACCT_OPTS=\"", options_to_str(ENERGY_ACCOUNTING_OPTIONS), '"')
            println(script, "ADD_ENERGY_SCRIPT=\"", ADD_ENERGY_SCRIPT_PATH, '"')
        end

        if true in measure.use_kokkos
            # Create a tmp directory for all sources for this job
            println(script, "KOKKOS_BUILD_DIR=\"", joinpath(scratch_dir(), "kokkos_build_\${SLURM_JOB_ID}"), '"')
        end

        # Put the header in a `echo` call
        !isempty(header) && println(script, "\necho -e \"", replace(header, '\n' => "\n#"), "\"")

        step_idx = 0
        if measure.track_energy
            # Julia warmup
            ref_step = make_reference_job_from(first(steps))
            println(script, "\n# Initial warmup")
            println(script, command_for_step(ref_step))
            step_idx += 1
        end

        for (i_step, step) in enumerate(steps)
            step_file = basename(step.base_file_name)
            step_file = replace(step_file, r"_+$" => "")  # Remove tailling '_'
            i_step += step_idx_offset
            println(script, "\necho \"== Step $i_step/$step_count: $step_file ==\"")
            step_idx = append_to_sub_script(script, measure, step, step_idx)
        end

        println(script, "\necho \"== All done ==\"")

        if true in measure.use_kokkos
            # Clear the tmp directory
            println(script, "\nrm -rf \$KOKKOS_BUILD_DIR")
        end
    end

    println("Created submission script '$script_name'")

    return script_path
end


function append_to_sub_script(script::IO, measure::MeasureParams, step::JobStep, step_idx::Int)
    if measure.one_job_per_cell
        if measure.track_energy
            # Add the data file variables
            energy_ref_data = energy_ref_file(step)
            energy_data = step.energy_plot.data_file
            println(script, "ENERGY_REF_DATA=\"", energy_ref_data, '"')
            println(script, "ENERGY_DATA=\"", energy_data, '"')
            println(script, "TMP_STEP_DATA=\"", energy_data, ".TMP\"")

            if step.energy_plot.do_plot
                println(script, "ENERGY_PLOT_SCRIPT=\"", step.energy_plot.plot_script, '"')
            end

            println(script, "\n# Remove any potential leftovers")
            println(script, "rm -f \$TMP_STEP_DATA")

            # Add as many energy references as needed
            ref_step = make_reference_job_from(step)
            println(script, "\n# Energy references")
            println(script, "OPTIONS=\"", options_to_str(build_backend_command(ref_step)), '"')
            ref_step_cmd = options_to_str(job_step_command_args(ref_step)) * " \$OPTIONS"
            for _ in 1:measure.energy_references
                println(script, ref_step_cmd)
                println(script, add_energy_data_command(step_idx, "ENERGY_REF_DATA", "TMP_STEP_DATA"))
                step_idx += 1
            end

            println(script, "\n# Dummy for first measurement")
            ref_step.cells_list = [first(step.cells_list)]
            println(script, command_for_step(ref_step))
            step_idx += 1

            println(script, "\n# Energy measurements")
        end

        options = build_backend_command(step)

        # Remove the list of cells
        i_cells = findfirst(==("--cells-list"), options)
        popat!(options, i_cells + 1)
        popat!(options, i_cells)

        # Create a variable holding the common arguments
        println(script, "OPTIONS=\"", options_to_str(options), '"')

        step_cmd = options_to_str(job_step_command_args(step)) * " \$OPTIONS"

        # Split each domain in '--cells-list' into its own command
        for cells in step.cells_list
            println(script, step_cmd, " --cells-list ", join(cells, ','))
            if measure.track_energy
                print(script, add_energy_data_command(step_idx, "ENERGY_DATA", "TMP_STEP_DATA"))
                if step.energy_plot.do_plot
                    println(script, " && gnuplot \$ENERGY_PLOT_SCRIPT")
                else
                    println(script)
                end
            end
            step_idx += 1
        end
    else
        println(script, command_for_step(step))
        step_idx += 1
    end

    return step_idx
end


function create_script_for_each_step(measure::MeasureParams, steps::Vector{JobStep}; header="")
    step_count = length(steps)
    paths = []
    for (i_step, step) in enumerate(steps)
        name_suffix = basename(step.base_file_name) |> splitext |> first
        name_suffix = "_" * replace(name_suffix, r"_+$" => "")
        script_path = create_sub_script(measure, [step];
            header, name_suffix, step_count, step_idx_offset=i_step-1)
        push!(paths, script_path)
    end
    return paths
end


function run_job_steps(measure::MeasureParams, steps::Vector{JobStep})
    for step in steps
        try
            cmd = command_for_step(step; in_sub_script=false)
            println("Waiting for job to start...")
            run(Cmd(cmd; dir=measure.script_dir))
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
    max_node_cores = get_max_cores_of_partition(measure.node)

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

    base_file_name, legend = build_data_file_base_name(measure,
        cluster.processes, cluster.distribution, cluster.node_count, backend)
    base_file_name = joinpath(".", DATA_DIR_NAME, measure.name, base_file_name)

    job_step = build_job_step(measure, backend, cluster, base_file_name, legend)

    if measure.process_scaling
        processes_idx = findfirst(==(cluster.processes), measure.processes)
        job_step.cells_list = job_step.cells_list[processes_idx:processes_idx]
    end

    return job_step
end


function build_armon_data_file_name(measure::MeasureParams,
        test::String, axis_splitting::String, process_grid::Vector{Int})
    file_name_extra = [""]
    legend_extra = [""]

    if length(measure.tests_list) > 1
        push!(file_name_extra, test)
        push!(legend_extra, test)
    end

    if length(measure.axis_splitting) > 1
        push!(file_name_extra, string(axis_splitting))
        push!(legend_extra, string(axis_splitting))
    end

    if length(measure.process_grids) > 1
        grid_str = join(process_grid, '×')
        push!(file_name_extra, "_pg=" * grid_str)
        push!(legend_extra, "process grid: " * grid_str)
    end

    return join(file_name_extra, '_'), join(legend_extra, ", ")
end


function build_data_file_base_name(measure::MeasureParams, 
        processes::Int, distribution::String, node_count::Int,
        threads::Int, use_simd::Int, dimension::Int, backend::Backend)
    # Build a file name based on the measurement name and the parameters that don't have a single
    # value, with a matching legend entry
    name = string(measure.device)
    legend = string(measure.device)

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

    if length(measure.node_count) > 1 || (measure.processes_per_node > 0 && length(measure.processes) > 1)
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


function ensure_dirs_exist(measure::MeasureParams)
    for dir_name in [
                joinpath(DATA_DIR_NAME, measure.name),
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

    point_type = 5  # Marker style for the line in the plot
    color_index = 1  # Index in the colormap
    error_bars = measure.error_bars

    perf_plot_cmds = []
    time_hist_cmds = []
    MPI_time_cmds = []
    energy_plot_cmds = []
    energy_plot_refs = []

    ensure_dirs_exist(measure)

    for step in steps
        if step.energy_plot.do_plot
            ref_file_name = energy_ref_file(step)
            ref_file_path = joinpath(measure.script_dir, ref_file_name)
            !no_overwrite && open(identity, ref_file_path, "w")

            ref_idx = length(energy_plot_refs) + 1
            push!(energy_plot_refs, gp_energy_ref_cmd(ref_file_name, ref_idx))

            file_name = step.energy_plot.data_file
            file_path = joinpath(measure.script_dir, file_name)
            !no_overwrite && open(identity, file_path, "w")
            legend = replace(step.legend, '_' => "\\_")
            push!(energy_plot_cmds, gp_energy_plot_cmd(file_name, legend,
                color_index, point_type, ref_idx, step.cycles))
        end

        for armon_params in armon_combinaisons(measure, step.dimension)
            process_grid = get_process_grid(step.cluster, armon_params)
            isnothing(process_grid) && continue

            incr_color = false

            file_name_extra, legend = build_armon_data_file_name(measure,
                armon_params.test, armon_params.axis_splitting, process_grid)

            legend = step.legend * legend
            legend = replace(legend, '_' => "\\_")  # '_' alone makes a subscript in gnuplot

            if step.perf_plot.do_plot
                file_name = Printf.format(Printf.Format(step.perf_plot.data_file), file_name_extra)
                file_path = joinpath(measure.script_dir, file_name)
                !no_overwrite && open(identity, file_path, "w")
                push!(perf_plot_cmds, gp_perf_plot_cmd(file_name, legend, point_type; error_bars))
            end

            if step.time_hist.do_plot
                file_name = Printf.format(Printf.Format(step.time_hist.data_file), file_name_extra)
                file_path = joinpath(measure.script_dir, file_name)
                !no_overwrite && open(identity, file_path, "w")
                push!(time_hist_cmds, gp_hist_plot_cmd(file_name, legend, point_type))
            end

            if step.time_MPI.do_plot
                file_name = Printf.format(Printf.Format(step.time_MPI.data_file), file_name_extra)
                file_path = joinpath(measure.script_dir, file_name)
                !no_overwrite && open(identity, file_path, "w")
                raw_legend = measure.device == CPU ? ("(time) " * legend) : (legend * " (time)")
                push!(MPI_time_cmds, gp_MPI_time_cmd(file_name, raw_legend, color_index, point_type))
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
        measure.energy_plot    && create_energy_plot_file(measure, energy_plot_cmds, energy_plot_refs)
    end
end


function build_all_steps_of_measure(measure::MeasureParams, first_measure::Bool, skip_first::Int, comb_count::Int)
    steps = JobStep[]

    comb_i = 0
    comb_c = 0
    for cluster_params in build_cluster_combinaisons(measure)
        if measure.processes_per_node > 0
            cluster_params.node_count = cld(cluster_params.processes, measure.processes_per_node)
        end

        for backend in measure.backends, backend_params in parse_combinaisons(measure, cluster_params, backend)
            comb_i += 1
            if first_measure && comb_i <= skip_first
                continue
            elseif comb_c > comb_count - 1
                return steps, true
            end

            job_step = build_job_step(measure, backend_params, cluster_params)
            !isnothing(job_step) && push!(steps, job_step)

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

    sub_scripts = []

    no_overwrite = batch_options.no_overwrite
    no_plot_update = batch_options.no_plot_update
    first_measure = true
    for (i, measure) in measures_to_do
        println(" ==== Measurement $(i)/$(length(measures)): $(measure.name) ==== ")

        one_script_per_step = batch_options.one_script_per_step || measure.one_script_per_step

        job_steps, comb_end = build_all_steps_of_measure(measure, first_measure, 
            batch_options.skip_first, batch_options.comb_count)
        first_measure = false

        if isempty(job_steps)
            println("Nothing to do.")
            comb_end && return
            continue
        end

        create_all_data_files(measure, job_steps; no_overwrite, no_plot_update)

        if measure.make_sub_script
            header = "==== Measurement $(i)/$(length(measures)): $(measure.name) ===="
            if one_script_per_step
                script_paths = create_script_for_each_step(measure, job_steps; header)
                append!(sub_scripts, script_paths)
            else
                script_path = create_sub_script(measure, job_steps; header)
                push!(sub_scripts, script_path)
            end
        else
            run_job_steps(measure, job_steps)
        end

        comb_end && return
    end

    if batch_options.submit_now
        # We submit the scripts once all of them have been created, in case there is any errors for
        # one of them
        open(RECENT_JOBS_FILE, "w") do recent_jobs
            for sub_script in sub_scripts
                job_id = submit_script(sub_script)
                println(recent_jobs, job_id)
            end
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
