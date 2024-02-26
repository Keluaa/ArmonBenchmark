
using Printf
using Statistics
using Preferences
using TimerOutputs


include(joinpath(@__DIR__, "openmp_utils.jl"))
include(joinpath(@__DIR__, "../common_utils.jl"))

armon_path = joinpath(@__DIR__, "../../julia/")

armon_uuid = Base.UUID("b773a7a3-b593-48d6-82f6-54bf745a629b")

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
use_kokkos = false
cmake_options = []
kokkos_backends = [:Serial, :OpenMP]
kokkos_build_dir = missing
kokkos_version = nothing
use_md_iter = 0
print_kokkos_threads_affinity = false
block_size = 1024
gpu = :CUDA
threads_places = :cores
threads_proc_bind = :close
skip_cpuids = []
dimension = 2
output_precision = nothing
compare = false
compare_ref = false
comparison_tol = 1e-10
check_result = false

axis_splitting = []
tests = []
cells_list = []

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
min_acquisition_time = 0.0
manual_mem_management = false
repeats_count_file = ""


dim_index = findfirst(x->x=="--dim", ARGS)
if !isnothing(dim_index)
    dimension = parse(Int, ARGS[dim_index+1])
    dimension != 2 && error("Only 2D is supported for now")
    deleteat!(ARGS, [dim_index, dim_index+1])
end


default_preferences = Dict(
    "use_std_lib_threads" => Preferences.load_preference(armon_uuid, "use_std_lib_threads", false),
    "use_fast_math" => Preferences.load_preference(armon_uuid, "use_fast_math", true),
    "use_inbounds" => Preferences.load_preference(armon_uuid, "use_inbounds", true),
)

preferences = deepcopy(default_preferences)


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
    elseif arg == "--check-result"
        global check_result = parse(Bool, ARGS[i+1])
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
        preferences["use_std_lib_threads"] = use_std_threads
        global i += 1
    elseif arg == "--threads-places"
        global threads_places = Symbol(ARGS[i+1])
        global i += 1
    elseif arg == "--threads-proc-bind"
        global threads_proc_bind = Symbol(ARGS[i+1])
        global i += 1
    elseif arg == "--skip-cpuids"
        global skip_cpuids = parse.(Int, split(ARGS[i+1], ',') .|> strip)
        global i += 1

    # GPU params
    elseif arg == "--gpu"
        global gpu = Symbol(uppercase(ARGS[i+1]))
        global use_gpu = true
        global i += 1
    elseif arg == "--block-size"
        global block_size = parse(Int, ARGS[i+1])
        global i += 1

    # Kokkos params
    elseif arg == "--use-kokkos"
        global use_kokkos = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--cmake-options"
        global cmake_options = split(ARGS[i+1], ';') .|> strip
        global i += 1
    elseif arg == "--kokkos-backends"
        global kokkos_backends = split(ARGS[i+1], ',') .|> strip .|> Symbol
        global i += 1
    elseif arg == "--kokkos-build-dir"
        global kokkos_build_dir = ARGS[i+1]
        global i += 1
    elseif arg == "--print-kokkos-thread-affinity"
        global print_kokkos_threads_affinity = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--kokkos-version"
        global kokkos_version = ARGS[i+1]
        global i += 1
    elseif arg == "--use-md-iter"
        global use_md_iter = parse(Int, ARGS[i+1])
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
    elseif arg == "--no-precomp"
        global no_precompilation = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--min-acquisition-time"
        global min_acquisition_time = parse(Float64, ARGS[i+1])
        global i += 1
    elseif arg == "--manual-mem-management"
        global manual_mem_management = parse(Bool, ARGS[i+1])
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
    elseif arg == "--repeats-count-file"
        global repeats_count_file = ARGS[i+1]
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

min_acquisition_time *= 1e9  # sec to ns


if preferences != default_preferences
    # TODO: add a dummy project to the scratch dir of the job, add its path to `LOAD_PATH` and modify
    # the preferences from there
    # Preferences.set_preferences!(armon_uuid, preferences...; force=true)
    error("This job requires changing the preferences, which isn't SLURM or MPI-safe for now.")
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

# If the MPI.jl config is wrong, Julia might only detect 1 MPI process
max_processes = maximum(prod, proc_domains)
max_processes > global_size && error("Not enough processes ($global_size), need at most $max_processes")

# Create a communicator for each node of the MPI world
node_local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, rank)

# Get the rank and the number of processes running on the same node
local_rank = MPI.Comm_rank(node_local_comm)
local_size = MPI.Comm_size(node_local_comm)

# Pin the threads on the node with no overlap with the other processes running on the same node.
# Pinning the Julia threads this way will prevent OpenMP threads from being correctly setup.
thread_offset = local_rank * Threads.nthreads()
if !(use_kokkos && :OpenMP in kokkos_backends)
    omp_bind_threads(thread_offset, threads_places, threads_proc_bind; skip_cpus=skip_cpuids)
else
    OMP_NUM_THREADS, OMP_PLACES, OMP_PROC_BIND = build_omp_vars(
        thread_offset, threads_places, threads_proc_bind; skip_cpus=skip_cpuids)
    ENV["OMP_PLACES"] = OMP_PLACES
    ENV["OMP_PROC_BIND"] = OMP_PROC_BIND
    ENV["OMP_NUM_THREADS"] = OMP_NUM_THREADS

    if print_kokkos_threads_affinity
        println("Kokkos threads bind with:")
        println("OMP_PLACES = ", ENV["OMP_PLACES"])
        println("OMP_PROC_BIND = ", ENV["OMP_PROC_BIND"])
        println("OMP_NUM_THREADS = ", ENV["OMP_NUM_THREADS"])
        ENV["OMP_DISPLAY_ENV"] = true
        ENV["OMP_DISPLAY_AFFINITY"] = true
    end
end

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

#
# GPU selection
#

if use_gpu && !use_kokkos
    # Use the local MPI rank to dispatch GPUs among local ranks
    gpu_index = local_rank

    if gpu === :ROCM
        using AMDGPU
    elseif gpu === :CUDA
        using CUDA
    end

    if use_MPI && gpu === :ROCM
        gpu_index %= length(AMDGPU.devices())
        AMDGPU.device_id!(gpu_index + 1)
    elseif gpu === :CUDA
        gpu_index %= CUDA.ndevices()
        CUDA.device!(gpu_index)
    end
end

use_gpu && is_root && @info "Using $(gpu == :ROCM ? "ROCm" : "CUDA") GPU"

#
# Kokkos initialization
#

if use_kokkos
    using Kokkos

    if is_root
        backends = [getproperty(Kokkos, backend) for backend in kokkos_backends]
        Kokkos.set_backends(backends)
        Kokkos.set_build_dir(kokkos_build_dir)
        Kokkos.set_cmake_options(cmake_options)
        !isnothing(kokkos_version) && Kokkos.set_kokkos_version(kokkos_version)
        Kokkos.load_wrapper_lib()  # All compilation (if any) of the C++ wrapper happens here
    else
        Kokkos.set_build_dir(kokkos_build_dir; local_only=true)
    end

    MPI.Barrier(MPI.COMM_WORLD)

    !is_root && Kokkos.load_wrapper_lib(; no_compilation=true, no_git=true)
    Kokkos.initialize(map_device_id_by=:mpi_rank)

    if print_kokkos_threads_affinity
        println("Kokkos max threads: ", Kokkos.BackendFunctions.omp_get_max_threads())
        println("Kokkos OpenMP threads affinity:")
        println(Kokkos.BackendFunctions.omp_capture_affinity())
        threadinfo(; color=false)
    end
end

#
# Armon loading
#

using Armon

Base.global_logger(prev_global_logger)


function check_preferences(expected_prefs)
    uuid = Base.PkgId(Armon).uuid
    loaded_prefs = Base.get_preferences(uuid)
    ok = true
    for (k, v) in expected_prefs
        if !haskey(loaded_prefs, k) || loaded_prefs[k] != v
            @warn "Wrong preference for $k: expected $v, got $(loaded_prefs[k])"
            ok = false
        end
    end
    return ok
end
!check_preferences(preferences) && exit(1)


function build_params(test, domain; 
        ieee_bits, riemann, scheme, cfl, Dt, cst_dt, dt_on_even_cycles, 
        axis_splitting, maxtime, maxcycle, 
        silent, output_file, write_output,
        use_threading, use_simd, use_gpu, 
        use_MPI, px, py, reorder_grid, async_comms)
    options = (;
        ieee_bits, riemann, scheme, projection, riemann_limiter,
        nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
        test=test, nx=domain[1], ny=domain[2],
        axis_splitting, 
        maxtime, maxcycle, silent, output_file, write_output, write_ghosts, write_slices, output_precision, 
        measure_time,
        use_threading, use_simd, use_gpu, device=gpu, block_size,
        use_kokkos,
        use_MPI, px, py,
        reorder_grid, async_comms,
        compare, is_ref=compare_ref, comparison_tolerance=comparison_tol,
        check_result
    )

    if use_kokkos
        options = (; options..., use_md_iter, cmake_options)
    end

    ArmonParameters(; options...)
end


function flatten_timer(timer::TimerOutput, prefix = "")
    # Transforms the name of the timer "B" in `to["A"]["B"]` to "A.B", and do same for all of its children
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
    if gpu === :ROCM
        return AMDGPU.Runtime.Mem.info()[2]
    else
        total_mem = CUDA.Mem.info()[2]
        return total_mem
    end
end


get_cpu_max_mem() = Int(Sys.total_memory())


function continue_acquisition(current_repeats, acquisition_start, for_precompilation)
    current_repeats < repeats && return true
    for_precompilation && return false

    if is_root
        min_time_reached = (time_ns() - acquisition_start) >= min_acquisition_time
    else
        min_time_reached = true
    end

    if use_MPI
        min_time_reached = MPI.Bcast(min_time_reached, 0, MPI.COMM_WORLD)
    end

    return !min_time_reached
end


function free_memory_if_needed(params::ArmonParameters, manual_mem_management, current_repeats)
    # This is mainly here for GPU memory management, which is handled manually by the underlying
    # packages. Julia sees GPU arrays as a single pointer on CPU RAM, which makes it difficult to be
    # freed automatically.

    mem_info = Armon.memory_info(params)
    mem_required = Armon.memory_required(params)

    if params.use_MPI && !params.use_gpu
        # Account for sharing the RAM of the CPU
        mem_required *= local_size
    end

    overhead_max = 1e9  # 1GB of overhead max. MPI buffers shouldn't reach this much memory usage

    if manual_mem_management
        # Minimize the number of GC passes by estimating how many repeats we can fit on the device.
        # The GC always runs before the first mesurement.
        gc_freq = max(1, floor(Int, mem_info.total / (mem_required * 1.50)) - 1)
        if current_repeats % gc_freq == 0
            @warn "Manual GC pass (every $gc_freq repeats for this measurement)"
            GC.gc(true)
        end
    elseif mem_info.free < mem_info.total * 0.05 || mem_info.free < (mem_required * 1.05 + overhead_max)
        GC.gc(true)
    end

    if mem_required + min(overhead_max, mem_required * 0.05) > mem_info.total
        if is_root
            println("skipped because of memory requirements")
            req_gb   = @sprintf("%.2f GB", mem_required   / 1e9)
            total_gb = @sprintf("%.2f GB", mem_info.total / 1e9)
            @warn "The device has $total_gb of memory in total, but $req_gb are needed. \
                Accounting for overhead, the solver might not be able to allocate all data." maxlog=1
        end
        return true
    end

    return false
end


function run_armon(params::ArmonParameters; for_precompilation=false)
    vals_cells_per_sec = Vector{Float64}()
    total_time_contrib = nothing
    current_repeats = 0
    acquisition_start = time_ns()

    while continue_acquisition(current_repeats, acquisition_start, for_precompilation)
        free_memory_if_needed(params, manual_mem_management, current_repeats) && break

        stats = armon(params)

        push!(vals_cells_per_sec, stats.giga_cells_per_sec)

        total_time_contrib = merge_time_contribution(total_time_contrib, stats.timer)

        current_repeats += 1
    end

    return current_repeats, vals_cells_per_sec, total_time_contrib
end


is_first_measure = true
function print_first_params(params)
    if is_first_measure
        println(stderr, "First params:")
        println(stderr, params)
        global is_first_measure = false
    end
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

    print_first_params(params)

    @printf(" - ")
    length(tests) > 1          && @printf("%-4s ", string(test))
    length(axis_splitting) > 1 && @printf("%-14s ", string(splitting))
    @printf("%11g cells (%5gx%-5g): ", prod(cells), cells[1], cells[2])

    time_start = time_ns()
    actual_repeats, vals_cells_per_sec, time_contrib = run_armon(params)
    time_end = time_ns()

    if actual_repeats == 0
        # Measure skipped because not enough memory is available
        return nothing, nothing
    end

    duration = (time_end - time_start) / 1.0e9

    mean_cells_per_sec = mean(vals_cells_per_sec)
    std_cells_per_sec = std(vals_cells_per_sec; corrected=true, mean=mean_cells_per_sec)

    @printf("%8.3f ± %3.1f Giga cells/sec %s", mean_cells_per_sec, std_cells_per_sec, 
        get_duration_string(duration))

    if actual_repeats != repeats
        println(" ($actual_repeats repeats)")
    else
        println()
    end

    if !isempty(data_file_name)
        # Append the result to the data file
        open(data_file_name, "a") do data_file
            println(data_file, "$(prod(cells)), $mean_cells_per_sec, $std_cells_per_sec, $actual_repeats")
        end
    end

    return time_contrib, actual_repeats
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
        print_first_params(params)
        @printf(" - (%2dx%-2d) ", px, py)
        length(tests) > 1          && @printf("%-4s ", string(test))
        length(axis_splitting) > 1 && @printf("%-14s ", string(splitting))
        @printf("%11g cells (%5gx%-5g): ", prod(cells), cells[1], cells[2])
    end

    time_start = time_ns()
    actual_repeats, vals_cells_per_sec, time_contrib = run_armon(params)
    MPI.Barrier(MPI.COMM_WORLD)
    time_end = time_ns()

    if actual_repeats == 0
        # Measure skipped because not enough memory is available
        return nothing, nothing
    end

    duration = (time_end - time_start) / 1.0e9

    # Gather the throughput and energy measurements on the root process
    merged_vals_cells_per_sec = MPI.Gather(vals_cells_per_sec, 0, MPI.COMM_WORLD)
    total_cells_per_sec = MPI.Reduce(sum(vals_cells_per_sec) / actual_repeats, MPI.Op(+, Float64; iscommutative=true), 0, MPI.COMM_WORLD)

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

        total_cycle_time /= actual_repeats
        MPI_time /= actual_repeats
    end
    total_MPI_time = MPI.Reduce(MPI_time, MPI.SUM, 0, MPI.COMM_WORLD)

    !is_root && return time_contrib, nothing  # Only the root process does the output

    if length(merged_vals_cells_per_sec) > 1
        std_cells_per_sec = std(merged_vals_cells_per_sec; corrected=true) * sqrt(params.proc_size)
    else
        std_cells_per_sec = 0
    end

    @printf("%8.3f ± %4.2f Giga cells/sec", total_cells_per_sec, std_cells_per_sec)

    # Append the result to the data file
    if !isempty(data_file_name)
        open(data_file_name, "a") do data_file
            println(data_file, "$(prod(cells)), $total_cells_per_sec, $std_cells_per_sec, $actual_repeats")
        end
    end

    total_MPI_time /= params.proc_size

    if !isempty(MPI_time_file_name)
        open(MPI_time_file_name, "a") do data_file
            println(data_file, "$(prod(cells)), $total_MPI_time, $total_cycle_time, $actual_repeats")
        end
    end

    if MPI_time_plot
        @printf(", %4.1f%% of MPI time, ", round(total_MPI_time / total_cycle_time * 100; digits=1))
    end

    @printf(" %s", get_duration_string(duration))

    println(" ($actual_repeats repeats)")

    return time_contrib, actual_repeats
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
        try
            # We redirect the output of gnuplot to null so that there is no warning messages displayed
            run(pipeline(`gnuplot $(gnuplot_script)`, stdout=devnull, stderr=devnull))
        catch e
            println(stderr, "Gnuplot error: ", e)
        end
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
    out_pipe = Pipe()
    try
        redirect_stdout(out_pipe) do
            run_armon(build_params(test, (240*proc_domains[1][1], 240*proc_domains[1][2]);
                ieee_bits, riemann, scheme, cfl, 
                Dt, cst_dt, dt_on_even_cycles, axis_splitting=axis_splitting[1], 
                maxtime, maxcycle=10, silent, output_file, write_output=false,
                use_threading, use_simd, 
                use_gpu, use_MPI, px=proc_domains[1][1], py=proc_domains[1][2], 
                reorder_grid, async_comms
            ); for_precompilation=true)
        end
    catch e
        println("Precompilation error!")
        close(out_pipe.in)
        out = String(read(out_pipe))
        !isempty(out) && println(out)
        rethrow(e)
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
            time_contrib, actual_repeats = do_measure_MPI(
                    data_file_name, MPI_time_file_name, test, cells, splitting, px, py)
        else
            time_contrib, actual_repeats = do_measure(data_file_name, test, cells, splitting)
        end

        if is_root
            total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)

            update_plot(gnuplot_script)
            update_plot(gnuplot_MPI_script)

            if !isempty(repeats_count_file)
                open(repeats_count_file, "w") do file
                    println(file, prod(cells), ",", actual_repeats)
                end
            end
        end
    end

    if time_histogram && is_root && !isempty(hist_file_name) && !isnothing(total_time_contrib)
        open(hist_file_name, "a") do data_file
            timer_to_table(data_file, total_time_contrib)
        end

        update_plot(gnuplot_hist_script)
    end
end


if use_kokkos
    GC.gc(true)  # Ensure that all views have been finalized
    Kokkos.finalize()
end
