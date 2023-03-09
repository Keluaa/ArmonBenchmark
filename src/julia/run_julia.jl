
using Printf
using Statistics
using TimerOutputs


include(joinpath(@__DIR__, "omp_simili.jl"))
include(joinpath(@__DIR__, "..", "common_utils.jl"))

armon_path = joinpath(@__DIR__, "../../julia/")

#
# Arguments parsing
#

scheme = :GAD_minmod
riemann = :acoustic
riemann_limiter = :minmod
nghost = 2
cfl = 0.
Dt = 0.
maxtime = 0.0
maxcycle = 500
projection = :euler
cst_dt = false
dt_on_even_cycles = false
ieee_bits = 64
silent = 2
output_file = "output"
write_output = false
write_ghosts = false
write_slices = false
use_threading = true
use_simd = true
use_gpu = false
block_size = 1024
gpu = :CUDA
threads_places = :cores
threads_proc_bind = :close
dimension = 2
output_precision = nothing
compare = false
compare_ref = false
comparison_tol = 1e-10

axis_splitting = []
tests = []
cells_list = []

limit_to_max_mem = false

use_MPI = true
verbose_MPI = false
file_MPI_dump = ""
proc_domains = [(1, 1)]
proc_grid_ratios = nothing
reorder_grid = true
async_comms = false

measure_time = true

base_file_name = ""
gnuplot_script = ""
gnuplot_hist_script = ""
gnuplot_MPI_script = ""
MPI_time_plot = false
time_histogram = false
repeats = 1
no_precompilation = false


dim_index = findfirst(x->x=="--dim", ARGS)
if !isnothing(dim_index)
    dimension = parse(Int, ARGS[dim_index+1])
    dimension != 2 && error("Only 2D is supported for now")
    deleteat!(ARGS, [dim_index, dim_index+1])
end


i = 1
while i <= length(ARGS)
    arg = ARGS[i]

    # Solver params
    if arg == "-s"
        global scheme = Symbol(replace(ARGS[i+1], '-' => '_'))
        global i += 1
    elseif arg == "--ieee"
        global ieee_bits = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--cycle"
        global maxcycle = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--riemann"
        global riemann = Symbol(replace(ARGS[i+1], '-' => '_'))
        global i += 1
    elseif arg == "--riemann-limiter"
        global riemann_limiter = Symbol(replace(ARGS[i+1], '-' => '_'))
        global i += 1
    elseif arg == "--time"
        global maxtime = parse(Float64, ARGS[i+1])
        global i += 1
    elseif arg == "--cfl"
        global cfl = parse(Float64, ARGS[i+1])
        global i += 1
    elseif arg == "--dt"
        global Dt = parse(Float64, ARGS[i+1])
        global i += 1
    elseif arg == "--projection"
        global projection = Symbol(ARGS[i+1])
        global i += 1
    elseif arg == "--cst-dt"
        global cst_dt = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--dt-even-cycles"
        global dt_on_even_cycles = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--nghost"
        global nghost = parse(Int, ARGS[i+1])
        global i += 1

    # Solver output params
    elseif arg == "--verbose"
        global silent = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--output-file"
        global output_file = ARGS[i+1]
        global i += 1
    elseif arg == "--write-output"
        global write_output = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--write-ghosts"
        global write_ghosts = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--write-slices"
        global write_slices = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--output-digits"
        global output_precision = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--compare"
        global compare = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--compare-ref"
        global compare_ref = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--comparison-tolerance"
        global comparison_tol = parse(Float64, ARGS[i+1])
        global i += 1

    # Multithreading params
    elseif arg == "--use-threading"
        global use_threading = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-simd"
        global use_simd = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-std-threads"
        use_std_threads = parse(Bool, ARGS[i+1])
        if use_std_threads
            ENV["USE_STD_LIB_THREADS"] = "true"
        else
            ENV["USE_STD_LIB_THREADS"] = "false"
        end
        global i += 1
    elseif arg == "--threads-places"
        global threads_places = Symbol(ARGS[i+1])
        global i += 1
    elseif arg == "--threads-proc-bind"
        global threads_proc_bind = Symbol(ARGS[i+1])
        global i += 1

    # GPU params
    elseif arg == "--gpu"
        global gpu = Symbol(uppercase(ARGS[i+1]))
        global use_gpu = true
        global i += 1
    elseif arg == "--block-size"
        global block_size = parse(Int, ARGS[i+1])
        global i += 1

    # 2D only params
    elseif arg == "--splitting"
        global axis_splitting = Symbol.(split(ARGS[i+1], ','))
        global i += 1

    # List params
    elseif arg == "--tests"
        global tests = Symbol.(split(ARGS[i+1], ','))
        global i += 1
    elseif arg == "--cells-list"
        domains_list = split(ARGS[i+1], ';')
        domains_list = split.(domains_list, ',')
        domains_list_t = Vector{Tuple{Int, Int}}(undef, length(domains_list))
        for (i, domain) in enumerate(domains_list)
            domains_list_t[i] = Tuple(convert.(Int, parse.(Float64, domain)))
        end
        global cells_list = domains_list_t
        global i += 1

    # MPI params
    elseif arg == "--use-mpi"
        global use_MPI = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--verbose-mpi"
        global verbose_MPI = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--file-mpi-dump"
        global file_MPI_dump = ARGS[i+1]
        global i += 1
    elseif arg == "--proc-grid"
        raw_proc_domain_list = split(ARGS[i+1], ';')
        raw_proc_domain_list = split.(raw_proc_domain_list, ',')
        proc_domain_list_t = Vector{NTuple{2, Int}}(undef, length(raw_proc_domain_list))
        for (i, proc_domain) in enumerate(raw_proc_domain_list)
            proc_domain_list_t[i] = Tuple(parse.(Int, proc_domain))
        end
        global proc_domains = proc_domain_list_t
        global i += 1
    elseif arg == "--proc-grid-ratio"
        raw_proc_grid_ratios = split(ARGS[i+1], ';')
        raw_proc_grid_ratios = split.(raw_proc_grid_ratios, ',')
        proc_grid_ratios_t = Vector{NTuple{2, Int}}(undef, length(raw_proc_grid_ratios))
        for (i, grid_ratio) in enumerate(raw_proc_grid_ratios)
            proc_grid_ratios_t[i] = Tuple(parse.(Int, grid_ratio))
        end
        global proc_grid_ratios = proc_grid_ratios_t
        global i += 1
    elseif arg == "--reorder-grid"
        global reorder_grid = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--async-comms"
        global async_comms = parse(Bool, ARGS[i+1])
        global i += 1

    # Measurements params
    elseif arg == "--measure-time"
        global measure_time = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--repeats"
        global repeats = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--limit-to-mem"
        global limit_to_max_mem = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--no-precomp"
        global no_precompilation = parse(Bool, ARGS[i+1])
        global i += 1

    # Measurement output params
    elseif arg == "--data-file"
        global base_file_name = ARGS[i+1]
        global i += 1
    elseif arg == "--gnuplot-script"
        global gnuplot_script = ARGS[i+1]
        global i += 1
    elseif arg == "--gnuplot-hist-script"
        global gnuplot_hist_script = ARGS[i+1]
        global i += 1
    elseif arg == "--time-histogram"
        global time_histogram = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--time-MPI-graph"
        global MPI_time_plot = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--gnuplot-MPI-script"
        global gnuplot_MPI_script = ARGS[i+1]
        global i += 1
    else
        println("Wrong option: ", arg)
        exit(1)
    end
    global i += 1
end

if MPI_time_plot && !measure_time
    @error "'--time-MPI-graph true' needs '--measure-time true'"
end

#
# Muting mechanism for the 'no CUDA capable device detected' error of CUDA.jl 
#

using Logging

struct MuteCUDANoDevice <: AbstractLogger
    global_logger::AbstractLogger
end

Logging.min_enabled_level(logger::MuteCUDANoDevice) = Logging.min_enabled_level(logger.global_logger)
Logging.handle_message(logger::MuteCUDANoDevice, args...; kwargs...) = Logging.handle_message(logger.global_logger, args...; kwargs...)

function Logging.shouldlog(logger::MuteCUDANoDevice, level, _module, group, id)
    if level >= Logging.Error && group === :initialization && id === :CUDA_215238c1
        # Ignore the error message at https://github.com/JuliaGPU/CUDA.jl/blob/master/src/initialization.jl#L92
        return false
    end
    return Logging.shouldlog(logger.global_logger, level, _module, group, id)
end

muted_cuda_logger = MuteCUDANoDevice(Base.global_logger())
prev_global_logger = Base.global_logger(muted_cuda_logger)

#
# GPU selection
#

if use_gpu
    # If SLURM is used to dispatch jobs, we can use the local ID of the process to uniquely 
    # assign the GPUs to each process.
    gpu_index = parse(Int, get(ENV, "SLURM_LOCALID", "-1"))
    if gpu_index == -1
        @warn "SLURM_LOCALID is not defined. GPU device defaults to 0. All processes on the same \
               node will use the same GPU." maxlog=1
        gpu_index = 0
    end

    # We must select a GPU before initializing MPI
    if use_MPI && gpu == :ROCM
        @warn "ROCM is, for now, not MPI aware"
        # TODO: select ROC device in the same way as for CUDA
    else
        using CUDA
        gpu_index %= CUDA.ndevices()  # In case we want more processes than GPUs
        CUDA.device!(gpu_index)
    end
end

#
# MPI initialization
#

using MPI
MPI.Init()

rank = MPI.Comm_rank(MPI.COMM_WORLD)
global_size = MPI.Comm_size(MPI.COMM_WORLD)
is_root = rank == 0
if is_root
    println("Using MPI with $global_size processes")
    loading_start_time = time_ns()
end

# Create a communicator for each node of the MPI world
node_local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, rank)

# Get the rank and the number of processes running on the same node
local_rank = MPI.Comm_rank(node_local_comm)
local_size = MPI.Comm_size(node_local_comm)

# Pin the threads on the node with no overlap with the other processes running on the same node
thread_offset = local_rank * Threads.nthreads()
omp_bind_threads(thread_offset, threads_places, threads_proc_bind)

if verbose_MPI || !isempty(file_MPI_dump)
    # Call 'MPI_Get_processor_name', in order to get the name of the node on which the current 
    # process is running.
    raw_node_name = Vector{UInt8}(undef, MPI.API.MPI_MAX_PROCESSOR_NAME)
    len = Ref{Cint}()
    MPI.API.MPI_Get_processor_name(raw_node_name, len)
    node_name = unsafe_string(pointer(raw_node_name), len[])

    using ThreadPinning: threadinfo, getcpuids

    if !isempty(file_MPI_dump)
        is_root && println("Writing MPI processes info to $file_MPI_dump...")
        # Write the debug info to the file, ordered by process rank
        mpi_info_file = MPI.File.open(MPI.COMM_WORLD, file_MPI_dump; write=true)

        info_line_length = 300
        cores_line_length = 10+512
        proc_offset = info_line_length + cores_line_length + 2

        if use_gpu
            info_line = @sprintf("%4d: local %-2d/%2d (gpu %s) in node %s", rank, local_rank+1, local_size, string(CUDA.uuid(CUDA.device()))[1:8], node_name)
        else
            info_line = @sprintf("%4d: local %-2d/%2d in node %s", rank, local_rank+1, local_size, node_name)
        end
        
        cores_line = Vector{String}()
        push!(cores_line, " - cores: ")
        for tid in getcpuids()
            push!(cores_line, @sprintf("%4d", tid))
        end
        cores_line = reduce(*, cores_line)

        info_line = @sprintf("%-300s\n", info_line[1:min(length(info_line), info_line_length)])
        cores_line = @sprintf("%-522s\n", cores_line[1:min(length(cores_line), cores_line_length)])
        proc_info_lines = info_line * cores_line

        MPI.File.write_at_all(mpi_info_file, proc_offset * rank, proc_info_lines)
    end

    if verbose_MPI
        is_root && println("Processes info:")
        # Print the debug info in order, one process at a time
        for i in 1:global_size
            if i == rank+1
                if use_gpu
                    @printf(" - %-4d: local %-2d/%2d (gpu %s) in node %s\n", rank, local_rank+1, local_size, string(CUDA.uuid(CUDA.device()))[1:8], node_name)
                else
                    @printf(" - %-4d: local %-2d/%2d in node %s\n", rank, local_rank+1, local_size, node_name)
                end
                threadinfo(; color=false, blocksize=64)
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end
end

use_gpu && is_root && @info "Using $(gpu == :ROCM ? "ROCm" : "CUDA") GPU"

#
# Armon loading
#

using Armon

Base.global_logger(prev_global_logger)


function build_params(test, domain; 
        ieee_bits, riemann, scheme, cfl, Dt, cst_dt, dt_on_even_cycles, 
        axis_splitting, maxtime, maxcycle, 
        silent, output_file, write_output,
        use_threading, use_simd, use_gpu, 
        use_MPI, px, py, reorder_grid, async_comms)
    ArmonParameters(;
        ieee_bits, riemann, scheme, projection, riemann_limiter,
        nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
        test=test, nx=domain[1], ny=domain[2],
        axis_splitting, 
        maxtime, maxcycle, silent, output_file, write_output, write_ghosts, write_slices, output_precision, 
        measure_time,
        use_threading, use_simd, use_gpu, device=gpu, block_size, use_MPI, px, py, 
        reorder_grid, async_comms,
        compare, is_ref=compare_ref, comparison_tolerance=comparison_tol)
end


function flatten_timer(timer::TimerOutput, prefix = "")
    if timer.name != "root" 
        timer.name = prefix * timer.name
        prefix = timer.name * "."
    else
        prefix = ""
    end
    for inner_timer in values(timer.inner_timers)
        flatten_timer(inner_timer, prefix)
    end
end


function timer_to_table(io::IO, timer::TimerOutput)
    flat_timer = deepcopy(timer)
    flatten_timer(flat_timer)
    flat_timer = TimerOutputs.flatten(flat_timer)
    for timer in sort!(collect(values(flat_timer.inner_timers)); by=x->TimerOutputs.sortf(x, :name))
        println(io, timer.name, ", ",
                    timer.accumulated_data.time, ", ",
                    timer.accumulated_data.ncalls, ", ",
                    timer.accumulated_data.allocs)
    end
end


function merge_time_contribution(timer_1, timer_2)
    if any(isnothing, (timer_1, timer_2))
        # Return the first non-nothing timer, or nothing 
        return something(timer_1, timer_2, Some(nothing))
    end
    return merge!(timer_1, timer_2)
end


function get_gpu_max_mem()
    if gpu == :ROCM
        is_root && @warn "AMDGPU.jl doesn't know how much memory there is. Using the a default value of 32GB" maxlog=1
        # TODO: max memory is available in the latest versions of AMDGPU.jl 
        return 32*10^9
    else
        total_mem = CUDA.Mem.info()[2]
        return total_mem
    end
end


get_cpu_max_mem() = Int(Sys.total_memory())


function run_armon(params::ArmonParameters)
    vals_cells_per_sec = Vector{Float64}(undef, repeats)
    total_time_contrib = nothing
    total_cycles = 0

    for i in 1:repeats
        stats = armon(params)

        vals_cells_per_sec[i] = stats.giga_cells_per_sec
        total_cycles += stats.cycles

        total_time_contrib = merge_time_contribution(total_time_contrib, stats.timer)
    end

    return total_cycles, vals_cells_per_sec, total_time_contrib
end


function do_measure(data_file_name, test, cells, splitting)
    params = build_params(test, cells; 
        ieee_bits, riemann, scheme, cfl, 
        Dt, cst_dt, dt_on_even_cycles=false, axis_splitting=splitting,
        maxtime, maxcycle, silent, output_file, write_output,
        use_threading, use_simd, use_gpu,
        use_MPI=false, px=1, py=1,
        reorder_grid=false, async_comms=false
    )

    @printf(" - %-4s %-14s %11g cells (%5gx%-5g): ", 
        string(test), string(splitting), cells[1] * cells[2], cells[1], cells[2])

    if limit_to_max_mem
        max_mem = params.use_gpu ? get_gpu_max_mem() : get_cpu_max_mem()
        max_mem *= 0.95  # Leave 5% of memory available to account for the Julia runtime, OS, etc...

        variables_count = fieldcount(Armon.ArmonData)
        data_size = variables_count * params.nbcell * sizeof(typeof(params.Dt))

        if data_size > max_mem
            @printf("skipped because of memory: %.1f > %.1f GB\n", data_size / 10^9, max_mem / 10^9)
            return nothing
        end
    end

    time_start = time_ns()
    cycles, vals_cells_per_sec, time_contrib = run_armon(params)
    time_end = time_ns()

    duration = (time_end - time_start) / 1.0e9

    mean_cells_per_sec = mean(vals_cells_per_sec)
    std_cells_per_sec = std(vals_cells_per_sec; corrected=true, mean=mean_cells_per_sec)

    @printf("%8.3f ± %3.1f Giga cells/sec %s\n", mean_cells_per_sec, std_cells_per_sec, 
        get_duration_string(duration))

    if !isempty(data_file_name)
        # Append the result to the data file
        open(data_file_name, "a") do data_file
            println(data_file, "$(prod(cells)), $mean_cells_per_sec, $std_cells_per_sec")
        end
    end

    return time_contrib
end


function do_measure_MPI(data_file_name, MPI_time_file_name, test, cells, splitting, px, py)
    params = build_params(test, cells;
        ieee_bits, riemann, scheme, cfl,
        Dt, cst_dt, dt_on_even_cycles, axis_splitting=splitting,
        maxtime, maxcycle, silent, output_file, write_output,
        use_threading, use_simd, use_gpu,
        use_MPI, px, py,
        reorder_grid, async_comms
    )

    if is_root
        @printf(" - (%2dx%-2d) %-4s %-14s %11g cells (%5gx%-5g): ",
            px, py, string(test), string(splitting), prod(cells), cells[1], cells[2])
    end

    if limit_to_max_mem
        max_mem = params.use_gpu ? get_gpu_max_mem() : get_cpu_max_mem()
        max_mem *= 0.95  # Leave 5% of memory available to account for the Julia runtime, MPI, OS, etc...

        variables_count = fieldcount(Armon.ArmonData)
        data_size = variables_count * params.nbcell * sizeof(typeof(params.Dt))

        # TODO : Account for resource usage overlap, for GPUs and multi-socket nodes

        if data_size > max_mem
            is_root && @printf("skipped because of memory: %.1f > %.1f GB\n", 
                data_size / 10^9 * params.proc_size, max_mem / 10^9 * params.proc_size)
            return nothing
        end
    end

    time_start = time_ns()
    _, vals_cells_per_sec, time_contrib = run_armon(params)
    MPI.Barrier(MPI.COMM_WORLD)
    time_end = time_ns()

    duration = (time_end - time_start) / 1.0e9

    # Gather the throughput and energy measurements on the root process
    merged_vals_cells_per_sec = MPI.Gather(vals_cells_per_sec, 0, MPI.COMM_WORLD)
    total_cells_per_sec = MPI.Reduce(sum(vals_cells_per_sec) / repeats, MPI.Op(+, Float64; iscommutative=true), 0, MPI.COMM_WORLD)

    # Gather the total MPI communication time
    MPI_time = 0
    total_cycle_time = 0
    if !isnothing(time_contrib)
        flat_time_contrib = TimerOutputs.flatten(time_contrib)
        if haskey(flat_time_contrib, "MPI")
            MPI_time += TimerOutputs.time(flat_time_contrib["MPI"])
        end
        if haskey(flat_time_contrib, "time_step AllReduce")
            MPI_time += TimerOutputs.time(flat_time_contrib["time_step AllReduce"])
        end

        total_cycle_time = TimerOutputs.time(flat_time_contrib["solver_cycle"])
    end
    total_MPI_time = MPI.Reduce(MPI_time / repeats, MPI.SUM, 0, MPI.COMM_WORLD)

    !is_root && return time_contrib  # Only the root process does the output

    if length(merged_vals_cells_per_sec) > 1
        std_cells_per_sec = std(merged_vals_cells_per_sec; corrected=true) * sqrt(params.proc_size)
    else
        std_cells_per_sec = 0
    end

    @printf("%8.3f ± %4.2f Giga cells/sec", total_cells_per_sec, std_cells_per_sec)

    # Append the result to the data file
    if !isempty(data_file_name)
        open(data_file_name, "a") do data_file
            println(data_file, "$(prod(cells)), $total_cells_per_sec, $std_cells_per_sec")
        end
    end

    total_MPI_time /= params.proc_size

    if !isempty(MPI_time_file_name)
        open(MPI_time_file_name, "a") do data_file
            println(data_file, "$(prod(cells)), $total_MPI_time, $total_cycle_time")
        end
    end

    if MPI_time_plot
        @printf(", %4.1f%% of MPI time, ", round(total_MPI_time / total_cycle_time * 100; digits=1))
    end

    @printf(" %s\n", get_duration_string(duration))

    return time_contrib
end


function process_ratio_to_grid(n_proc, ratios)
    (rpx, rpy) = ratios
    r = rpx / rpy
    # In theory the ratios have been pre-checked so that those conversions don't throw InexactError
    px = convert(Int, √(n_proc * r))
    py = convert(Int, √(n_proc / r))
    return px, py
end


function update_plot(gnuplot_script)
    if !isempty(gnuplot_script)
        # We redirect the output of gnuplot to null so that there is no warning messages displayed
        run(pipeline(`gnuplot $(gnuplot_script)`, stdout=devnull, stderr=devnull))
    end
end


if !isnothing(proc_grid_ratios)
    # Convert the ratios to process grids
    proc_domains = map(Base.Fix1(process_ratio_to_grid, global_size), proc_grid_ratios)
end


if is_root
    loading_end_time = time_ns()
    @printf("Loading time: %3.1f sec\n", (loading_end_time - loading_start_time) / 1e9)
    if !no_precompilation
        println("Compiling...")
        compile_start_time = time_ns()
    end
end


!no_precompilation && for test in tests
    # We redirect stdout so that in case 'silent < 5', output functions are pre-compiled and so they 
    # don't influence the timing results.
    # 'devnull' is not used here since 'println' and others will not go through their normal code paths.
    dummy_pipe = Pipe()
    redirect_stdout(dummy_pipe) do
        run_armon(build_params(test, (240*proc_domains[1][1], 240*proc_domains[1][2]);
            ieee_bits, riemann, scheme, cfl, 
            Dt, cst_dt, dt_on_even_cycles, axis_splitting=axis_splitting[1], 
            maxtime, maxcycle=10, silent, output_file, write_output=false,
            use_threading, use_simd, 
            use_gpu, use_MPI, px=proc_domains[1][1], py=proc_domains[1][2], 
            reorder_grid, async_comms
        ))
    end
end


if is_root && !no_precompilation
    compile_end_time = time_ns()
    @printf(" (time: %3.1f sec)\n", (compile_end_time - compile_start_time) / 1e9)
end


for test in tests, splitting in axis_splitting, (px, py) in proc_domains
    if isempty(base_file_name)
        data_file_name = ""
        hist_file_name = ""
        MPI_time_file_name = ""
    else
        data_file_name = base_file_name

        if length(tests) > 1
            data_file_name *= "_" * string(test)
        end

        if length(axis_splitting) > 1
            data_file_name *= "_" * string(splitting)
        end

        if length(proc_domains) > 1
            data_file_name *= "_pg=$(px)x$(py)"
        end

        hist_file_name = data_file_name * "_hist.csv"
        MPI_time_file_name = data_file_name * "_MPI_time.csv"
        data_file_name *= "_perf.csv"

        data_dir = dirname(data_file_name)
        if !isdir(data_dir)
            mkpath(data_dir)
        end
    end

    total_time_contrib = nothing

    for cells in cells_list
        if use_MPI
            time_contrib = do_measure_MPI(data_file_name, MPI_time_file_name, test, cells, splitting, px, py)
        else
            time_contrib = do_measure(data_file_name, test, cells, splitting)
        end

        if is_root
            total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)

            update_plot(gnuplot_script)
            update_plot(gnuplot_MPI_script)
        end
    end

    if time_histogram && is_root && !isempty(hist_file_name) && !isnothing(total_time_contrib)
        open(hist_file_name, "a") do data_file
            timer_to_table(data_file, total_time_contrib)
        end

        update_plot(gnuplot_hist_script)
    end
end
