
using Printf

include("omp_simili.jl")

scheme = :GAD_minmod
riemann = :acoustic
iterations = 4
nghost = 2
cfl = 0.6
Dt = 0.
maxtime = 0.0
maxcycle = 500
euler_projection = true
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
threads_places = :cores
threads_proc_bind = :close
dimension = 1

transpose_dims = []
axis_splitting = []
tests = []
cells_list = []

use_MPI = false
verbose_MPI = false
file_MPI_dump = ""
proc_domains = [(1, 1)]
proc_grid_ratios = nothing
single_comm_per_axis_pass = false
reorder_grid = true
async_comms = false

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
    elseif arg == "--verbose"
        global silent = parse(Int, ARGS[i+1])
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
    elseif arg == "--output-file"
        global output_file = ARGS[i+1]
        global i += 1
    elseif arg == "--write-output"
        global write_output = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--write-ghosts"
        global write_ghosts = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-ccall"
        global use_ccall = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-threading"
        global use_threading = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-simd"
        global use_simd = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--interleaving"
        global interleaving = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--use-gpu"
        global use_gpu = parse(Bool, ARGS[i+1])
        global i += 1
    elseif arg == "--euler"
        global euler_projection = parse(Bool, ARGS[i+1])
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

    # Additionnal params
    elseif arg == "--gpu"
        gpu = ARGS[i+1]
        global i += 1
        if gpu == "ROCM"
            ENV["USE_ROCM_GPU"] = "true"
        elseif gpu == "CUDA"
            ENV["USE_ROCM_GPU"] = "false"
        else
            println("Unknown gpu: ", gpu)
            exit(1)
        end
        global use_gpu = true
    elseif arg == "--block-size"
        block_size = parse(Int, ARGS[i+1])
        global i += 1
        ENV["GPU_BLOCK_SIZE"] = block_size
    elseif arg == "--repeats"
        global repeats = parse(Int, ARGS[i+1])
        global i += 1
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
    else
        println("Wrong option: ", arg)
        exit(1)
    end
    global i += 1
end


if use_MPI
    if use_gpu
        # We must select a GPU before initializing MPI
        if parse(Bool, get(ENV, "USE_ROCM_GPU", "false"))
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
        gpu_index %= CUDA.ndevices()  # In case we want more processes than GPUs
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
    else
        if async_comms
            println("Using async comms")
            include("armon_2D_MPI_async.jl")
        else
            include("armon_2D_MPI.jl")
        end
    end
    using .Armon

    # Create a communicator for each node of the MPI world
    node_local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.MPI_COMM_TYPE_SHARED, rank)

    # Get the rank and the number of processes running on the same node
    local_rank = MPI.Comm_rank(node_local_comm)
    local_size = MPI.Comm_size(node_local_comm)

    # Pin the threads on the node with no overlap with the other processes running on the same node
    thread_offset = local_rank * Threads.nthreads()
    omp_bind_threads(thread_offset, threads_places, threads_proc_bind)

    if verbose_MPI || !isempty(file_MPI_dump)
        # Call 'MPI_Get_processor_name', which is not exposed by MPI.jl, in order to get the name of
        # the node on which the current process is running.
        raw_node_name = Vector{UInt8}(undef, 256)  # MPI_MAX_PROCESSOR_NAME == 256
        len = Ref{Cint}()
        #= MPI.@mpichk =# ccall((:MPI_Get_processor_name, MPI.libmpi), Cint, (Ptr{Cuchar}, Ptr{Cint}), raw_node_name, len)
        node_name = unsafe_string(pointer(raw_node_name), len[])

        using ThreadPinning  # To use threadinfo() and getcpuids()
        
        if !isempty(file_MPI_dump)
            is_root && println("Writing MPI process info to $file_MPI_dump...")
            # Write the debug info to the file, ordered by process rank
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
            # Print the debug info in order, one process at a time
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
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, 
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, 
                test=test, nbcell=cells,
                euler_projection, maxtime, maxcycle, silent, output_file, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, use_MPI)
        end
    else
        function build_params(test, domain; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, 
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
                test=test, nx=domain[1], ny=domain[2],
                euler_projection, transpose_dims, axis_splitting, 
                maxtime, maxcycle, silent, output_file, write_output, write_ghosts,
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
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, 
                use_ccall, use_threading, use_simd, interleaving, use_gpu, 
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, 
                test=test, nbcell=cells,
                euler_projection, maxtime, maxcycle, silent, output_file, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, interleaving, use_gpu)
        end
    else
        function build_params(test, domain; 
                ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims, 
                axis_splitting, maxtime, maxcycle, silent, output_file, write_output, 
                use_ccall, use_threading, use_simd, interleaving, use_gpu,
                use_MPI, px, py, single_comm_per_axis_pass, reorder_grid)
            return ArmonParameters(; ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, 
                test=test, nx=domain[1], ny=domain[2],
                euler_projection, transpose_dims, axis_splitting, 
                maxtime, maxcycle, silent, output_file, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, use_gpu)
        end
    end
end



if use_gpu && is_root
    if parse(Bool, get(ENV, "USE_ROCM_GPU", "false"))
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


function run_armon(params::ArmonParameters)
    total_cells_per_sec = 0
    total_time_contrib = nothing
    total_cycles = 0

    for _ in 1:repeats
        _, cycles, cells_per_sec, time_contrib = armon(params)
        total_cells_per_sec += cells_per_sec
        total_cycles += cycles
        total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)
    end
    
    return total_cycles, total_cells_per_sec / repeats, total_time_contrib
end


function do_measure(data_file_name, test, cells, transpose, splitting)
    params = build_params(test, cells; 
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, dt_on_even_cycles=false, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
        maxtime, maxcycle, silent, output_file, write_output, use_ccall, use_threading, use_simd,
        interleaving, use_gpu,
        use_MPI=false, px=0, py=0,
        single_comm_per_axis_pass=false, reorder_grid=false
    )

    if dimension == 1
        @printf(" - %s, %11g cells: ", test, cells)
    else
        @printf(" - %-4s %-14s %11g cells (%5gx%-5g): ", 
            string(test) * (transpose ? "ᵀ" : ""),
            string(splitting), cells[1] * cells[2], cells[1], cells[2])
    end

    cycles, cells_per_sec, time_contrib = run_armon(params)

    @printf("%6.3f Giga cells/sec\n", cells_per_sec)

    # Append the result to the data file
    if !isempty(data_file_name)
        open(data_file_name, "a") do data_file
            if dimension == 1
                println(data_file, cells, ", ", cells_per_sec)
            else
                println(data_file, cells[1] * cells[2], ", ", cells_per_sec)
            end
        end
    end

    return time_contrib
end


function do_measure_MPI(data_file_name, comm_file_name, test, cells, transpose, splitting, px, py)
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

    time_start = time_ns()

    cycles, cells_per_sec, time_contrib = run_armon(build_params(test, cells; 
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, dt_on_even_cycles, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
        maxtime, maxcycle, silent, output_file, write_output, use_ccall, use_threading, use_simd,
        interleaving, use_gpu,
        use_MPI, px, py,
        single_comm_per_axis_pass, reorder_grid
    ))
    
    MPI.Barrier(MPI.COMM_WORLD)
    time_end = time_ns()

    # Merge the cells throughput and the time distribution of all processes in one reduction.
    # Since 'time_contrib' is an array of pairs, it is not a bits type. We first convert the values
    # to an array of floats, and then rebuild the array of pairs using the one of the root process.
    time_contrib_vals = Vector{Float64}(undef, length(time_contrib) + 1)
    time_contrib_vals[1:end-1] .= last.(time_contrib)
    time_contrib_vals[end] = cells_per_sec
    merged_time_contrib_vals = MPI.Reduce(time_contrib_vals, MPI.Op(+, Float64; iscommutative=true), 0, MPI.COMM_WORLD)
 
    if is_root
        if cycles <= 5
            println("not enough cycles ($cycles), cannot get an accurate measurement")
            return time_contrib
        end

        total_cells_per_sec = merged_time_contrib_vals[end]

        @printf("%6.3f Giga cells/sec", total_cells_per_sec)

        # Append the result to the data file
        if !isempty(data_file_name)
            open(data_file_name, "a") do data_file
                if dimension == 1
                    println(data_file, cells, ", ", total_cells_per_sec)
                else
                    println(data_file, cells[1] * cells[2], ", ", total_cells_per_sec)
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
                else
                    total_time += step_time
                end
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

            # TODO : add the total time (hh:mm:ss format)
            @printf(", %5.1f%% of MPI time\n", total_MPI_time / total_time * 100)
        else
            print("\n")
        end
    end

    return time_contrib
end


function process_ratio_to_grid(n_proc, ratios)
    (rpx, rpy) = ratios
    r = rpx / rpy
    # In theory the ratios have been pre-checked so that those convertions don't throw InexactError
    px = convert(Int, √(n_proc * r))
    py = convert(Int, √(n_proc / r))
    return px, py
end


if !isnothing(proc_grid_ratios)
    # Convert the ratios to process grids
    proc_domains = map(Base.Fix1(process_ratio_to_grid, global_size), proc_grid_ratios)
end


if is_root
    loading_end_time = time_ns()
    @printf("Loading time: %3.1f sec\n", (loading_end_time - loading_start_time) / 1e9)
    println("Compiling...")
    compile_start_time = time_ns()
end

for test in tests, transpose in transpose_dims
    run_armon(build_params(test, dimension == 1 ? 10000 : (10, 10);
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, dt_on_even_cycles=false, euler_projection, transpose_dims=transpose, axis_splitting=axis_splitting[1], 
        maxtime, maxcycle=1, silent=5, output_file, write_output=false, use_ccall, use_threading, use_simd, 
        interleaving, use_gpu, use_MPI=false, px=1, py=1, single_comm_per_axis_pass, reorder_grid=false))
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
    elseif dimension == 1
        data_file_name = base_file_name * string(test) * ".csv"
        hist_file_name = base_file_name * string(test) * "_hist.csv"
        comm_file_name = base_file_name * string(test) * "_MPI_time.csv"
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
        data_file_name *= ".csv"
    end

    total_time_contrib = nothing

    for cells in cells_list
        if use_MPI
            time_contrib = do_measure_MPI(data_file_name, comm_file_name, test, cells, transpose, splitting, px, py)
        else
            time_contrib = do_measure(data_file_name, test, cells, transpose, splitting)
        end

        if is_root
            # total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)

            if !isempty(gnuplot_script)
                # We redirect the output of gnuplot to null so that there is no warning messages displayed
                run(pipeline(`gnuplot $(gnuplot_script)`, stdout=devnull, stderr=devnull))
            end

            if !isempty(gnuplot_MPI_script)
                # Same for the MPI time graph
                run(pipeline(`gnuplot $(gnuplot_MPI_script)`, stdout=devnull, stderr=devnull))
            end
        end
    end

    if time_histogram && is_root && !isempty(hist_file_name)
        if flatten_time_dims
            flat_time_contrib = nothing
            for axis_time_contrib in total_time_contrib
                flat_time_contrib = merge_time_contribution(flat_time_contrib, axis_time_contrib)
            end
            total_time_contrib = flat_time_contrib
        elseif dimension == 2
            # Append the axis name at the beginning of each label and merge into a single list
            axes = ["X ", "Y "]
            merged_time_contrib = []
            for (axis, axis_time_contrib) in enumerate(total_time_contrib)
                axis_time_contrib = collect(axis_time_contrib)
                map!((e) -> (axes[axis] * e.first => e.second), axis_time_contrib, axis_time_contrib)
                append!(merged_time_contrib, axis_time_contrib)
            end
            total_time_contrib = sort(merged_time_contrib)
        end

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
