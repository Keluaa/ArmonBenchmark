
using Printf
using Statistics

include("omp_simili.jl")
include("vtune_lib.jl")
using .VTune

scheme = :GAD_minmod
riemann = :acoustic
riemann_limiter = :minmod
iterations = 4
nghost = 2
cfl = 0.6
Dt = 0.
maxtime = 0.0
maxcycle = 500
euler_projection = true
projection = :euler
cst_dt = false
dt_on_even_cycles = false
ieee_bits = 64
silent = 2
output_file = "output"
write_output = false
write_ghosts = false
use_ccall = false
use_threading = true
use_simd = true
interleaving = false
use_gpu = false
block_size = 1024
gpu = :CUDA
threads_places = :cores
threads_proc_bind = :close
dimension = 2
use_temp_vars_for_projection = false
output_precision = 6
compare = false
compare_ref = false
comparison_tol = 1e-10

transpose_dims = []
axis_splitting = []
tests = []
cells_list = []

limit_to_max_mem = false

use_MPI = false
verbose_MPI = false
file_MPI_dump = ""
proc_domains = [(1, 1)]
proc_grid_ratios = nothing
single_comm_per_axis_pass = false
reorder_grid = true
async_comms = false
mpi_impl = :async  # :sync, :async, :transpose

measure_time = true
measure_hw_counters = false
hw_counters_options = "(cache-misses,cache-references,L1-dcache-loads,L1-dcache-load-misses)," *
    #= "(LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores)," * =#
    "(dTLB-loads,dTLB-load-misses)," *
    "(cpu-cycles,instructions,branch-instructions)"

base_file_name = ""
gnuplot_script = ""
gnuplot_hist_script = ""
gnuplot_MPI_script = ""
time_histogram = false
flatten_time_dims = false
time_MPI_graph = false
repeats = 1


dim_index = findfirst(x->x=="--dim", ARGS)
if !isnothing(dim_index)
    dimension = parse(Int, ARGS[dim_index+1])
    if dimension ∉ (1, 2)
        error("Unexpected dimension: $dimension")
    end
    deleteat!(ARGS, [dim_index, dim_index+1])
end


i = 1
while i <= length(ARGS)
    arg = ARGS[i]

    # Solver params
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
    elseif arg == "--euler"
        global euler_projection = parse(Bool, ARGS[i+1])
        global projection = :euler
        global i += 1
    elseif arg == "--projection"
        global projection = Symbol(ARGS[i+1])
        global euler_projection = (projection != :none)
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

    # Solver output params
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
    elseif arg == "--output-digits"
        global output_precision = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--compare"
        global compare = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--compare-ref"
        global compare_ref = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--comparision-tolerance"
        global comparison_tol = parse(Float64, ARSG[i+1])
        global i += 1

    # Multithreading params
    elseif arg == "--use-threading"
        global use_threading = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-simd"
        global use_simd = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-ccall"
        global use_ccall = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--interleaving"
        global interleaving = parse(Bool, ARGS[i+1])
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

    # GPU params
    elseif arg == "--use-gpu"
        global use_gpu = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--gpu"
        global gpu = Symbol(uppercase(ARGS[i+1]))
        global i += 1
        if gpu == :ROCM
            ENV["USE_ROCM_GPU"] = "true"
        elseif gpu == :CUDA
            ENV["USE_ROCM_GPU"] = "false"
        else
            error("Unknown gpu: " * gpu)
        end
        global use_gpu = true
    elseif arg == "--block-size"
        global block_size = parse(Int, ARGS[i+1])
        global i += 1
        ENV["GPU_BLOCK_SIZE"] = block_size

    # 1D only params
    elseif arg == "--iterations"
        global iterations = parse(Int, ARGS[i+1])
        global i += 1

    # 2D only params
    elseif arg == "--transpose"
        if dimension != 2
            error("'--transpose' is 2D only")
        end
        global transpose_dims = parse.(Bool, split(ARGS[i+1], ','))
        global i += 1
    elseif arg == "--splitting"
        if dimension != 2
            error("'--splitting' is 2D only")
        end
        global axis_splitting = Symbol.(split(ARGS[i+1], ','))
        global i += 1
    elseif arg == "--flat-dims"
        if dimension != 2
            error("'--flat-dims' is 2D only")
        end
        global flatten_time_dims = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--temp-vars-transpose"
        global use_temp_vars_for_projection = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--continuous-ranges"
        ENV["USE_CONTINUOUS_RANGES"] = parse(Bool, ARGS[i+1])
        global i += 1

    # List params
    elseif arg == "--tests"
        global tests = Symbol.(split(ARGS[i+1], ','))
        global i += 1
    elseif arg == "--cells-list"
        if dimension == 1
            list = split(ARGS[i+1], ',')
            global cells_list = convert.(Int, parse.(Float64, list))
        else
            domains_list = split(ARGS[i+1], ';')
            domains_list = split.(domains_list, ',')
            domains_list_t = Vector{Tuple{Int, Int}}(undef, length(domains_list))
            for (i, domain) in enumerate(domains_list)
                domains_list_t[i] = Tuple(convert.(Int, parse.(Float64, domain)))
            end
            global cells_list = domains_list_t
        end
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
    elseif arg == "--single-comm"
        global single_comm_per_axis_pass = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--reorder-grid"
        global reorder_grid = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--async-comms"
        global async_comms = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--mpi-impl"
        global mpi_impl = Symbol(ARGS[i+1])
        global i += 1

    # Measurements params
    elseif arg == "--measure-time"
        global measure_time = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--measure-hw-counters"
        global measure_hw_counters = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--hw-counters"
        global hw_counters_options = ARGS[i+1]
        global i += 1
    elseif arg == "--repeats"
        global repeats = parse(Int, ARGS[i+1])
        global i += 1
    elseif arg == "--limit-to-mem"
        global limit_to_max_mem = parse(Bool, ARGS[i+1])
        global i += 1

    # Measurement output params
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
        global time_MPI_graph = parse(Bool, ARGS[i+1])
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


if use_MPI
    if use_gpu
        # We must select a GPU before initializing MPI
        if gpu == :ROCM
            error("ROCM is, for now, not MPI aware")
        end

        # If SLURM is used to dispatch jobs, we can use the local ID of the process to uniquely 
        # assign the GPUs to each process.
        gpu_index = parse(Int, get(ENV, "SLURM_LOCALID", "-1"))
        if gpu_index == -1
            @warn "SLURM_LOCALID is not defined. GPU device defaults to 0. All processes on the same node will use the same GPU." maxlog=1
            gpu_index = 0
        end

        using CUDA
        gpu_index %= CUDA.ndevices()  # In case we want more processes than GPUs
        CUDA.device!(gpu_index)
    end

    using MPI
    MPI.Init()

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    global_size = MPI.Comm_size(MPI.COMM_WORLD)
    is_root = rank == 0
    if is_root
        println("Using MPI with $global_size processes")
        loading_start_time = time_ns()
    end

    if dimension == 1
        include("armon_1D_MPI.jl")
    elseif mpi_impl == :sync
        include("armon_2D_MPI.jl")
    elseif mpi_impl == :async
        include("armon_2D_MPI_async.jl")
    elseif mpi_impl == :transpose
        include("armon_2D_MPI_transposition.jl")
    else
        error("Unknown Julia 2D MPI implementation: $mpi_impl")
    end
    using .Armon

    # Create a communicator for each node of the MPI world
    node_local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.COMM_TYPE_SHARED, rank)

    # Get the rank and the number of processes running on the same node
    local_rank = MPI.Comm_rank(node_local_comm)
    local_size = MPI.Comm_size(node_local_comm)

    # Pin the threads on the node with no overlap with the other processes running on the same node
    thread_offset = local_rank * Threads.nthreads()
    omp_bind_threads(thread_offset, threads_places, threads_proc_bind)

    if verbose_MPI || !isempty(file_MPI_dump)
        # Call 'MPI_Get_processor_name', which is not exposed by MPI.jl, in order to get the name of
        # the node on which the current process is running.
        raw_node_name = Vector{UInt8}(undef, 256)  # MPI_MAX_PROCESSOR_NAME == 256
        len = Ref{Cint}()
        #= MPI.@mpichk =# ccall((:MPI_Get_processor_name, MPI.libmpi), Cint, (Ptr{Cuchar}, Ptr{Cint}), raw_node_name, len)
        node_name = unsafe_string(pointer(raw_node_name), len[])

        using ThreadPinning  # To use threadinfo() and getcpuids()
        
        if !isempty(file_MPI_dump)
            is_root && println("Writing MPI process info to $file_MPI_dump...")
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

    if dimension == 1
        function build_params(test, cells; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid, async_comms)
            return ArmonParameters(; ieee_bits, test, riemann, scheme, iterations, nghost, nbcell=cells, cfl, Dt, euler_projection,
                cst_dt, maxtime, maxcycle, silent, write_output,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, use_MPI)
        end
    elseif mpi_impl == :async
        function build_params(test, domain; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid, async_comms)
            return ArmonParameters(; ieee_bits, riemann, scheme, projection, riemann_limiter,
                nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
                test=test, nx=domain[1], ny=domain[2],
                transpose_dims, axis_splitting, 
                maxtime, maxcycle, silent, output_file, write_output, write_ghosts, output_precision, 
                measure_time,
                use_ccall, use_threading, use_simd, use_gpu, device=gpu, block_size, use_MPI, px, py, 
                single_comm_per_axis_pass, reorder_grid, async_comms,
                compare, is_ref=compare_ref, comparison_tolerance=comparison_tol)
        end
    elseif mpi_impl == :transpose
        function build_params(test, domain; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid, async_comms)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
                test=test, nx=domain[1], ny=domain[2],
                euler_projection, transpose_dims, axis_splitting, 
                maxtime, maxcycle, silent, output_file, write_output, write_ghosts,
                measure_time, measure_hw_counters, hw_counters_options, hw_counters_output=hw_counters_output_file,
                use_ccall, use_threading, use_simd, use_gpu, use_MPI, px, py, 
                single_comm_per_axis_pass, reorder_grid, async_comms, use_temp_vars_for_projection)
        end
    elseif mpi_impl == :sync
        function build_params(test, domain; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims,
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file,
                use_ccall, use_threading, use_simd, interleaving, use_gpu,
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid, async_comms)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
                test=test, nx=domain[1], ny=domain[2],
                euler_projection, transpose_dims=false, axis_splitting,
                maxtime, maxcycle, silent, output_file, write_output, write_ghosts, 
                measure_time, measure_hw_counters, hw_counters_options, hw_counters_output=hw_counters_output_file,
                use_ccall, use_threading, use_simd, use_gpu, use_MPI, px, py,
                single_comm_per_axis_pass, reorder_grid)
        end
    end
else
    loading_start_time = time_ns()
    
    if dimension == 1
        include("armon_1D.jl")
    else
        include("armon_2D.jl")
    end
    using .Armon

    global_size = 1
    is_root = true

    if !use_gpu
        omp_bind_threads(threads_places, threads_proc_bind)
    end

    if dimension == 1
        function build_params(test, cells; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid, async_comms)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, 
                test=test, nbcell=cells,
                euler_projection, maxtime, maxcycle, silent, output_file, write_output,
                use_ccall, use_threading, use_simd, interleaving, use_gpu)
        end
    else
        function build_params(test, domain; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file,
                use_ccall, use_threading, use_simd, interleaving, use_gpu,
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid, async_comms)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, 
                test=test, nx=domain[1], ny=domain[2],
                euler_projection, transpose_dims, axis_splitting, 
                maxtime, maxcycle, silent, output_file, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, use_gpu)
        end
    end
end


if use_gpu && is_root
    if gpu == :ROCM
        println("Using ROCM GPU")
    else
        println("Using CUDA GPU")
    end
end


if dimension == 1
    transpose_dims = [false]
    axis_splitting = [:Sequential]
end


function merge_time_contribution(time_contrib_1, time_contrib_2)
    if isnothing(time_contrib_1)
        return time_contrib_2
    elseif isnothing(time_contrib_2)
        return time_contrib_1
    end

    return map((e, f) -> (e.first => (e.second + f.second)), time_contrib_1, time_contrib_2)
end


function reduce_time_contribution(time_contrib_1, time_contrib_2)
    if isnothing(time_contrib_1)
        return time_contrib_2
    elseif isnothing(time_contrib_2)
        return time_contrib_1
    end

    return map((e, f) -> (e.first => (min(e.second, f.second))), time_contrib_1, time_contrib_2)
end


MIN_TIME_CONTRIB = parse(Bool, get(ENV, "GET_MIN_TIME", "false"))
if MIN_TIME_CONTRIB
    is_root && @info "Time contribution will be the minimum of all repeats" logmax=1
end


function get_gpu_max_mem()
    if gpu== :ROCM
        is_root && @warn "AMDGPU.jl doesn't know how much memory there is. Using the a default value of 32GB" maxlog=1
        return 32*10^9
    else
        total_mem = CUDA.Mem.info()[2]
        return total_mem
    end
end


get_cpu_max_mem() = Int(Sys.total_memory())


function get_duration_string(duration_sec::Float64)
    hours = floor(duration_sec / 3600)
    duration_sec -= hours * 3600

    minutes = floor(duration_sec / 60)
    duration_sec -= minutes * 60

    seconds = floor(duration_sec)
    duration_sec -= seconds

    ms = floor(duration_sec * 1000)

    str = ""
    print_next = false
    for (duration, unit, force_print) in ((hours, 'h', false), (minutes, 'm', false), (seconds, 's', true))
        if print_next
            str *= @sprintf("%02d", duration) * unit
        elseif duration > 0 || force_print
            str *= @sprintf("%2d", duration) * unit
            print_next = true
        else
            str *= "   "
        end
    end
    
    if hours == 0 && minutes == 0
        str *= @sprintf("%03dms", ms)
    end

    return str
end


function run_armon(params::ArmonParameters, with_profiling::Bool)
    vals_cells_per_sec = Vector{Float64}(undef, repeats)
    total_time_contrib = nothing
    total_cycles = 0
    mean_async_efficiency = 0

    for i in 1:repeats
        with_profiling && @resume_profiling()
        
        if mpi_impl == :async
            _, cycles, cells_per_sec, time_contrib, async_efficiency = armon(params)
        else
            _, cycles, cells_per_sec, time_contrib = armon(params)
            async_efficiency = 0.
        end

        with_profiling && @pause_profiling()
        vals_cells_per_sec[i] = cells_per_sec
        total_cycles += cycles
        mean_async_efficiency += async_efficiency
        
        if MIN_TIME_CONTRIB
            total_time_contrib = reduce_time_contribution(total_time_contrib, time_contrib)
        else
            total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)
        end
    end

    mean_async_efficiency /= repeats
    
    return total_cycles, vals_cells_per_sec, total_time_contrib, mean_async_efficiency
end


function do_measure(data_file_name, hw_c_file_name, test, cells, transpose, splitting)
    params = build_params(test, cells; 
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, dt_on_even_cycles=false, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
        maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file=hw_c_file_name,
        use_ccall, use_threading, use_simd,
        interleaving, use_gpu,
        use_MPI=false, px=0, py=0,
        single_comm_per_axis_pass=false, reorder_grid=false, async_comms=false
    )

    if dimension == 1
        @printf(" - %s, %11g cells: ", test, cells)
    else
        @printf(" - %-4s %-14s %11g cells (%5gx%-5g): ", 
            string(test) * (transpose ? "ᵀ" : ""),
            string(splitting), cells[1] * cells[2], cells[1], cells[2])
    end

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
    cycles, vals_cells_per_sec, time_contrib, async_efficiency = run_armon(params, true)
    time_end = time_ns()

    duration = (time_end - time_start) / 1.0e9

    mean_cells_per_sec = mean(vals_cells_per_sec)
    std_cells_per_sec = std(vals_cells_per_sec; corrected=true, mean=mean_cells_per_sec)

    @printf("%8.3f ± %3.1f Giga cells/sec %s\n", cells_per_sec, std_cells_per_sec, get_duration_string(duration))

    # Append the result to the data file
    if !isempty(data_file_name)
        open(data_file_name, "a") do data_file
            if dimension == 1
                println(data_file, cells, ", ", cells_per_sec, ", ", std_cells_per_sec)
            else
                println(data_file, cells[1] * cells[2], ", ", cells_per_sec, ", ", std_cells_per_sec)
            end
        end
    end

    return time_contrib
end


function do_measure_MPI(data_file_name, comm_file_name, hw_c_file_name, test, cells, transpose, splitting, px, py)
    if is_root
        if dimension == 1
            @printf(" - %s, %11g cells: ", test, cells)
        else
            @printf(" - (%2dx%-2d) %-4s %-14s %11g cells (%5gx%-5g): ", 
                px, py,
                string(test) * (transpose ? "ᵀ" : ""),
                string(splitting), cells[1] * cells[2], cells[1], cells[2])
        end
    end

    params = build_params(test, cells; 
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
        maxtime, maxcycle, silent, output_file, write_output, hw_counters_output_file=hw_c_file_name,
        use_ccall, use_threading, use_simd,
        interleaving, use_gpu,
        use_MPI, px, py,
        single_comm_per_axis_pass, reorder_grid, async_comms
    )

    if limit_to_max_mem
        max_mem = params.use_gpu ? get_gpu_max_mem() : get_cpu_max_mem()
        max_mem *= 0.95  # Leave 5% of memory available to account for the Julia runtime, MPI, OS, etc...

        variables_count = fieldcount(Armon.ArmonData)
        data_size = variables_count * params.nbcell * sizeof(typeof(params.Dt))

        # TODO : Account for ressource usage overlap, for GPUs and multi-socket nodes
        
        if data_size > max_mem
            is_root && @printf("skipped because of memory: %.1f > %.1f GB\n", data_size / 10^9 * params.proc_size, max_mem / 10^9 * params.proc_size)
            return nothing
        end
    end

    time_start = time_ns()

    cycles, vals_cells_per_sec, time_contrib, async_efficiency = run_armon(params, true)

    MPI.Barrier(MPI.COMM_WORLD)
    time_end = time_ns()

    duration = (time_end - time_start) / 1.0e9

    # Merge the cells throughput and the time distribution of all processes in one reduction.
    # Since 'time_contrib' is an array of pairs, it is not a bits type. We first convert the values
    # to an array of floats, and then rebuild the array of pairs using the one of the root process.
    time_contrib_vals = last.(time_contrib)
    merged_time_contrib_vals = MPI.Reduce(time_contrib_vals, MPI.Op(+, Float64; iscommutative=true), 0, MPI.COMM_WORLD)
 
    # Gather the throughput measurements on the root process
    merged_vals_cells_per_sec = MPI.Gather(vals_cells_per_sec, 0, MPI.COMM_WORLD)
    total_cells_per_sec = MPI.Reduce(sum(vals_cells_per_sec) / repeats, MPI.Op(+, Float64; iscommutative=true), 0, MPI.COMM_WORLD)

    if is_root
        if cycles <= 5
            println("not enough cycles ($cycles), cannot get an accurate measurement")
            return time_contrib
        end

        if length(merged_vals_cells_per_sec) > 1
            std_cells_per_sec = std(merged_vals_cells_per_sec; corrected=true) * sqrt(params.proc_size)
        else
            std_cells_per_sec = 0
        end

        @printf("%8.3f ± %4.2f Giga cells/sec", total_cells_per_sec, std_cells_per_sec)

        # Append the result to the data file
        if !isempty(data_file_name)
            open(data_file_name, "a") do data_file
                if dimension == 1
                    println(data_file, cells, ", ", total_cells_per_sec, ", ", std_cells_per_sec)
                else
                    println(data_file, cells[1] * cells[2], ", ", total_cells_per_sec, ", ", std_cells_per_sec)
                end
            end
        end
        
        # Rebuild the time contribution array
        for (i, (step_label, _)) in enumerate(time_contrib)
            time_contrib[i] = step_label => merged_time_contrib_vals[i]
        end

        if time_MPI_graph
            # Sum the time of each MPI communications. Time positions with a label ending with '_MPI'
            # count as time spent in MPI, and this time is part of a step not ending with MPI.
            total_time = 0.
            total_MPI_time = 0.
            for (step_label, step_time) in time_contrib
                if endswith(step_label, "_MPI")
                    total_MPI_time += step_time
                end
                total_time += step_time
            end
            
            # ns to sec
            total_time /= 1e9
            total_MPI_time /= 1e9

            # Append the result to the data file
            if !isempty(comm_file_name)
                open(comm_file_name, "a") do data_file
                    if dimension == 1
                        println(data_file, cells, ", ", total_MPI_time, ", ", total_time)
                    else
                        println(data_file, cells[1] * cells[2], ", ", total_MPI_time, ", ", total_time)
                    end
                end
            end

            @printf(", %5.1f%% of MPI time", total_MPI_time / total_time * 100)

            if mpi_impl == :async
                @printf(" (async: %5.1f%%)", async_efficiency * 100)
            end
        end

        @printf(" %s\n", get_duration_string(duration))
    end

    return time_contrib
end


function process_ratio_to_grid(n_proc, ratios)
    (rpx, rpy) = ratios
    r = rpx / rpy
    # In theory the ratios have been pre-checked so that those convertions don't throw InexactError
    px = convert(Int, √(n_proc * r))
    py = convert(Int, √(n_proc / r))
    return px, py
end


if !isnothing(proc_grid_ratios)
    # Convert the ratios to process grids
    proc_domains = map(Base.Fix1(process_ratio_to_grid, global_size), proc_grid_ratios)
end


if is_root
    loading_end_time = time_ns()
    @printf("Loading time: %3.1f sec\n", (loading_end_time - loading_start_time) / 1e9)
    println("Compiling...")
    compile_start_time = time_ns()
end

for test in tests, transpose in transpose_dims
    # We redirect stdout so that in case 'silent < 5', output functions are pre-compiled and so they 
    # don't influence the timing results.
    # 'devnull' is not used here since 'println' and others will not go through their normal code paths.
    dummy_pipe = Pipe()
    redirect_stdout(dummy_pipe) do
        run_armon(build_params(test, dimension == 1 ? 10000 : (240*proc_domains[1][1], 240*proc_domains[1][2]);
            ieee_bits, riemann, scheme, iterations, cfl, 
            Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims=transpose, axis_splitting=axis_splitting[1], 
            maxtime, maxcycle=10, silent, output_file, write_output=false, hw_counters_output_file="", use_ccall, use_threading, use_simd, 
            interleaving, use_gpu, use_MPI, px=proc_domains[1][1], py=proc_domains[1][2], 
            single_comm_per_axis_pass, reorder_grid, async_comms
        ), false)
    end
end

if is_root
    compile_end_time = time_ns()
    @printf(" (time: %3.1f sec)\n", (compile_end_time - compile_start_time) / 1e9)
end


for test in tests, transpose in transpose_dims, splitting in axis_splitting, (px, py) in proc_domains
    if isempty(base_file_name)
        data_file_name = ""
        hist_file_name = ""
        comm_file_name = ""
        hw_c_file_name = ""
    elseif dimension == 1
        data_file_name = base_file_name * string(test) * ".csv"
        hist_file_name = base_file_name * string(test) * "_hist.csv"
        comm_file_name = base_file_name * string(test) * "_MPI_time.csv"
        hw_c_file_name = base_file_name * string(test) * "_hw_counters.csv"
    else
        data_file_name = base_file_name * string(test)

        if length(transpose_dims) > 1
            data_file_name *= transpose ? "_transposed" : ""
        end

        if length(axis_splitting) > 1
            data_file_name *= "_" * string(splitting)
        end

        if length(proc_domains) > 1
            data_file_name *= "_pg=$(px)x$(py)"
        end

        hist_file_name = data_file_name * "_hist.csv"
        comm_file_name = data_file_name * "_MPI_time.csv"
        hw_c_file_name = data_file_name * "_hw_counters.csv"
        data_file_name *= ".csv"
    end

    total_time_contrib = nothing

    for cells in cells_list
        if use_MPI
            time_contrib = do_measure_MPI(data_file_name, comm_file_name, hw_c_file_name, test, cells, transpose, splitting, px, py)
        else
            time_contrib = do_measure(data_file_name, hw_c_file_name, test, cells, transpose, splitting)
        end

        if is_root
            total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)

            if !isempty(gnuplot_script)
                # We redirect the output of gnuplot to null so that there is no warning messages displayed
                run(pipeline(`gnuplot $(gnuplot_script)`, stdout=devnull, stderr=devnull))
            end

            if time_MPI_graph && !isempty(gnuplot_MPI_script)
                # Same for the MPI time graph
                run(pipeline(`gnuplot $(gnuplot_MPI_script)`, stdout=devnull, stderr=devnull))
            end
        end
    end

    if time_histogram && is_root && !isempty(hist_file_name)
        open(hist_file_name, "a") do data_file
            for (label, time) in total_time_contrib
                label = replace(label, '_' => "\\\\_", '!' => "")
                println(data_file, '"', label, "\", ", time / length(cells_list))
            end
        end

        if !isempty(gnuplot_hist_script)
            # Update the histogram
            run(pipeline(`gnuplot $(gnuplot_hist_script)`, stdout=devnull, stderr=devnull))
        end
    end
end
