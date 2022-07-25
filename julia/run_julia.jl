
using Printf
using ArgParse
using CUDA
using MPI
using ThreadPinning


include("omp_simili.jl")


mutable struct RunArguments
    dim::Int
    scheme::Symbol
    riemann::Symbol
    iterations::Int
    ieee_bits::Int
    maxcycle::Int
    nghost::Int
    cfl::Float64
    Dt::Float64
    cst_dt::Bool
    maxtime::Float64
    euler_projection::Bool
    
    axis_splitting::Vector{Symbol}
    transpose_dims::Vector{Bool}
    dt_on_even_cycles::Bool

    tests::Vector{Symbol}
    cells_list::Vector{Any}
    repeats::Int

    verbose::Int
    write_output::Bool
    write_ghosts::Bool
    data_file::String
    time_histogram::Bool
    time_MPI_graph::Bool
    gnuplot_script::String
    gnuplot_hist_script::String
    gnuplot_MPI_script::String
    flatten_time_dims::Bool

    use_ccall::Bool
    use_simd::Bool
    
    use_threading::Bool
    interleaving::Bool
    use_std_lib_threads::Bool
    threads_places::Symbol
    threads_proc_bind::Symbol

    use_gpu::Bool
    gpu::String
    block_size::Int

    use_mpi::Bool
    verbose_MPI::Bool
    file_MPI_dump::String
    proc_grid::Vector{Tuple{Int, Int}}
    proc_grid_ratios::Vector{Tuple{Int, Int}}
    single_comm_per_axis_pass::Bool
    reorder_grid::Bool
end


function RunArguments(args::Dict{Symbol, Any})
    fields = Vector{Any}(undef, fieldcount(RunArguments))
    for (i, field) in enumerate(fieldnames(RunArguments))
        if !haskey(args, field)
            error("Missing '$field' in arguments dict")
        end
        fields[i] = args[field]
    end
    return RunArguments(fields...)
end


# Default copy method
function Base.copy(p::RunArguments)
    return RunArguments([getfield(p, k) for k in fieldnames(RunArguments)]...)
end

#
# Argument parsing
# 

function ArgParse.parse_item(::Symbol, x::AbstractString)
    return Symbol(replace(x, '-' => '_'))
end


function ArgParse.parse_item(::Vector{Symbol}, x::AbstractString)
    return Symbol.(split(replace(x, '-' => '_'), ','))
end


function ArgParse.parse_item(::Vector{Bool}, x::AbstractString)
    return parse.(Bool, split(x, ','))
end


function ArgParse.parse_item(::Vector{NTuple{2, Int}}, x::AbstractString)
    domains = split(x, ';')
    domains = split.(domains, ',')
    domains_tuples = Vector{NTuple{2, Int}}(undef, length(domains))
    for (i, domain) in enumerate(domains)
        domains_tuples[i] = Tuple(parse.(Int, domain))
    end
    return domains_tuples
end


function parse_all_arguments(raw_arguments::Vector{String})
    settings = ArgParseSettings("""Main Julia Armon runner."""; prog="Armon", autofix_names=true)

    add_arg_group(settings, "Solver options")
    @add_arg_table settings begin
        "--dim"
            help = "Dimension of the solver"
            arg_type = Int
            default = 1
        "--scheme", "-s"
            help = "Riemann solver scheme. Possible values: 'Godunov', 'GAD-minmod', 'GAD-superbee', 'GAD-no-limiter'"
            arg_type = Symbol
            default = :GAD_minmod
        "--riemann"
            help = "Type of riemann solver"
            arg_type = Symbol
            default = :acoustic
        "--ieee"
            help = "Precision (64: Float64, 32: Float32)"
            arg_type = Int
            default = 64
            dest_name = :ieee_bits
        "--cycle"
            help = "Maximum number of cycles"
            arg_type = Int
            default = 500
            dest_name = :maxcycle
        "--nghost"
            help = "Number of ghost cells around the sub-domains"
            arg_type = Int
            range_tester = x -> x > 0
            default = 2
        "--cfl"
            help = "CFL value to use"
            arg_type = Float64
            default = 0.6
        "--dt"
            help = "Initial time step. If set to 0, the best value for the test will be chosen."
            arg_type = Float64
            default = 0.
            dest_name = :Dt
        "--cst-dt"
            help = "Use a constant time step (the value of --dt)"
            arg_type = Bool
            default = false
        "--time"
            help = "Maximum time of the simulation. If set to 0, the best value for the test will be chosen."
            arg_type = Float64
            default = 0.
        "--euler"
            help = "Enables a projection step after the lagrangian solver, to keep a regular cartesian mesh."
            arg_type = Bool
            default = true
            dest_name = :euler_projection
    end

    add_arg_group(settings, "2D Solver options")
    @add_arg_table settings begin
        "--splitting"
            help = "List of options. Axis splitting method: 'Sequential', 'SequentialSym', 'Strang'"
            arg_type = Vector{Symbol}
            default = Symbol[:Sequential]
            dest_name = :axis_splitting
        "--transpose"
            help = "List of options. If true, most arrays will be transposed between each axis pass."
            arg_type = Vector{Bool}
            default = Bool[false]
            dest_name = :transpose_dims
        "--dt-even-cycles"
            help = "Compute the time step only on even cycles"
            arg_type = Bool
            default = false
            dest_name = :dt_on_even_cycles
    end

    add_arg_group(settings, "Measure options")
    @add_arg_table settings begin
        "--tests"
            help = "List of options. Tests to use: 'Sod', 'Bizarrium', additionnally in 2D: 'Sod_y', 'Sod_circ'"
            arg_type = Vector{Symbol}
            required = true
        "--cells-list"
            help = "List of options. In 1D, the number of cells is delimited by ','. In 2D, the domains are separeated by ';' and the dimensions by ','"
            arg_type = String
            required = true
        "--repeats"
            help = "Repeats each measure X times."
            arg_type = Int
            default = 1
    end

    add_arg_group(settings, "Output options")
    @add_arg_table settings begin
        "--verbose"
            help = "Verbosity level: 0-max, 5-silent"
            arg_type = Int
            default = 2
        "--write-output"
            help = "If the result should be written to a file"
            arg_type = Bool
            default = false
        "--write-ghosts"
            help = "If the ghost cells should be included in the output file"
            arg_type = Bool
            default = false
        "--data-file"
            help = "Base file name to which the cells throughputs will be written to. The other plot outputs will use the same base name."
            arg_type = String
            default = ""
        "--time-histogram"
            help = "Enables the histogram output, which details the time spent on each step of the solver."
            arg_type = Bool
            default = false
        "--time-MPI-graph"
            help = "Enables the MPI communication time output, which details the time spent in MPI calls."
            arg_type = Bool
            default = false
        "--gnuplot-script"
            help = "If given, this script will be called after each measurement to update the plot in real time."
            arg_type = String
            default = ""
        "--gnuplot-hist-script"
            help = "Same as '--gnuplot-script', but for the histogram."
            arg_type = String
            default = ""
        "--gnuplot-MPI-script"
            help = "Same as '--gnuplot-script', but for the MPI time."
            arg_type = String
            default = ""
        "--flat-dims"
            help = "(2D only) If true, the axes of the histogram will be flattened."
            arg_type = Bool
            default = false
            dest_name = :flatten_time_dims
    end
    
    add_arg_group(settings, "Performance options")
    @add_arg_table settings begin
        "--use-ccall"
            help = "Replaces the core functions of the solver by calls to a C library. Only for 1D."
            arg_type = Bool
            default = false
        "--use-simd"
            help = "Enables SIMD in loops."
            arg_type = Bool
            default = true
    end

    add_arg_group(settings, "Multi-threading options")
    @add_arg_table settings begin
        "--use-threading"
            help = "Enables multithreading."
            arg_type = Bool
            default = true
        "--interleaving"
            help = "Changes the thread scheduling of the iterations of multi-threaded loops."
            arg_type = Bool
            default = false
        "--use-std-threads"
            help = "If true, switches from Polyester.jl threading to Threads.@threads threading. Much slower."
            arg_type = Bool
            default = false
            dest_name = :use_std_lib_threads
        "--threads-places"
            help = "Almost equivalent to the OpenMP option OMP_PLACES. Possible values: 'threads', 'cores', 'sockets', 'numa'"
            arg_type = Symbol
            default = :cores
        "--threads-proc-bind"
            help = "Almost equivalent to the OpenMP option OMP_PROC_BIND. Possible values: 'false', 'compact', 'close', 'spread'"
            arg_type = Symbol
            default = :close
    end

    add_arg_group(settings, "GPU options")
    @add_arg_table settings begin
        "--use-gpu"
            help = "Replaces all data handling functions by GPU kernels."
            arg_type = Bool
            default = false
        "--gpu"
            help = "Which GPU backend to use ('CUDA' or 'ROCM'). If set, enables '--use-gpu' by default."
            arg_type = String
            default = ""
        "--block-size"
            help = "Sets the block size of the GPU kernels."
            arg_type = Int
            default = 256
    end

    add_arg_group(settings, "MPI options")
    @add_arg_table settings begin
        "--use-mpi"
            help = "Enables MPI. The mesh will be splitted in sub-domains."
            arg_type = Bool
            default = false
        "--verbose-mpi"
            help = "Prints process (and cores) information at startup."
            arg_type = Bool
            default = false
        "--file-mpi-dump"
            help = "Prints process (and cores) information to the given file."
            arg_type = String
            default = ""
        "--proc-grid"
            help = "(2D only) List of values. The dimensions of the cartesian grid the processes will be distributed onto."
            arg_type = Vector{NTuple{2, Int}}
            default = Vector{NTuple{2, Int}}[(1, 1)]
        "--proc-grid-ratio"
            help = "(2D only, overriddes '--proc-grid') List of values. Distributes the processes on the grid with the given ratios."
            arg_type = Vector{NTuple{2, Int}}
            default = Vector{NTuple{2, Int}}[]
        "--single-comm"
            help = "(2D only) Removes one MPI communication before the eulerrian projection step by computing on one layer of ghost cells. Requires one more ghost cell."
            arg_type = Bool
            default = true
            dest_name = :single_comm_per_axis_pass
        "--reorder-grid"
            help = "(2D only) Reorders the ranks of the processes. Passed to MPI_Cart_create."
            arg_type = Bool
            default = true
    end


    parsed_args = parse_args(raw_arguments, settings; as_symbols=true)
    print(parsed_args)

    # Handle special cases

    if parsed_args["dim"] == 1
        list = split(parsed_args["cells-list"], ',')
        cells_list = convert.(Int, parse.(Float64, list))
    else
        cells_list = ArgParse.parse_item(Vector{NTuple{2, Int}}, parsed_args["cells-list"])
    end

    parsed_args[:iterations] = 0  # Unused option

    return RunArguments(parsed_args)
end

#
# Environment setup (vars, MPI, threads...)
#

function setup_run_armon(args::RunArguments)
    # Set environment variables needed at compile time
    if args.gpu == "ROCM"
        ENV["USE_ROCM_GPU"] = "true"
    elseif args.gpu == "CUDA"
        ENV["USE_ROCM_GPU"] = "false"
    else
        error("Unknown GPU: $(args.gpu)")
    end

    ENV["GPU_BLOCK_SIZE"] = args.block_size
    ENV["USE_STD_LIB_THREADS"] = args.use_std_lib_threads

    loading_start_time = time_ns()
    if args.use_mpi
        is_root = setup_runs_with_MPI(args)
    else
        is_root = setup_runs(args)
    end
    loading_end_time = time_ns()

    if is_root
        @printf("Loading time: %3.1f sec\n", (loading_end_time - loading_start_time) / 1e9)
    end

    return is_root
end


function setup_runs(args::RunArguments)
    if args.dim == 1
        include("armon_1D.jl")
    else
        include("armon_2D.jl")
    end

    if !args.use_gpu
        omp_bind_threads(args.threads_places, args.threads_proc_bind)
    end

    return true
end


function setup_runs_with_MPI(args::RunArguments)
    if !MPI.Initialized()
        MPI.Init()
    end

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    global_size = MPI.Comm_size(MPI.COMM_WORLD)
    is_root = rank == 0
    if is_root
        println("Using MPI with $global_size processes")
    end

    if args.dim == 1
        include("armon_1D_MPI.jl")
    else
        include("armon_2D_MPI.jl")
    end
    
    # Create a communicator for each node of the MPI world
    node_local_comm = MPI.Comm_split_type(MPI.COMM_WORLD, MPI.MPI_COMM_TYPE_SHARED, rank)

    # Get the rank and the number of processes running on the same node
    local_rank = MPI.Comm_rank(node_local_comm)
    local_size = MPI.Comm_size(node_local_comm)

    # Pin the threads on the node with no overlap with the other processes running on the same node
    thread_offset = local_rank * Threads.nthreads()
    omp_bind_threads(thread_offset, args.threads_places, args.threads_proc_bind)

    if args.verbose_MPI || !isempty(args.file_MPI_dump)
        print_MPI_info(args, is_root, rank, global_size, local_rank, local_size)
    end

    if !isempty(args.proc_grid_ratios)
        # Convert the ratios to process grids
        args.proc_grid = map(Base.Fix1(process_ratio_to_grid, global_size), args.proc_grid_ratios)
    end

    return is_root
end


function print_MPI_info(args::RunArguments, is_root::Bool, 
        rank::Int, global_size::Int, 
        local_rank::Int, local_size::Int)
    # Call 'MPI_Get_processor_name', which is not exposed by MPI.jl, in order to get the name of
    # the node on which the current process is running.
    raw_node_name = Vector{UInt8}(undef, 256)  # MPI_MAX_PROCESSOR_NAME == 256
    len = Ref{Cint}()
    #= MPI.@mpichk =# ccall((:MPI_Get_processor_name, MPI.libmpi), Cint, (Ptr{Cuchar}, Ptr{Cint}), raw_node_name, len)
    node_name = unsafe_string(pointer(raw_node_name), len[])

    if !isempty(args.file_MPI_dump)
        is_root && println("Writing MPI process info to $(args.file_MPI_dump)...")
        # Write the debug info to the file, ordered by process rank
        mpi_info_file = MPI.File.open(MPI.COMM_WORLD, args.file_MPI_dump; write=true)

        info_line_length = 300
        cores_line_length = 10+512
        proc_offset = info_line_length + cores_line_length + 2

        if args.use_gpu
            # Add the first 8 hex digits of the GPU UUID
            info_line = @sprintf("%4d: local %-2d/%2d (gpu %s) in node %s",
                rank, local_rank+1, local_size, string(CUDA.uuid(CUDA.device()))[1:8], node_name)
        else
            info_line = @sprintf("%4d: local %-2d/%2d in node %s", 
                rank, local_rank+1, local_size, node_name)
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

    if args.verbose_MPI
        is_root && println("Processes info:")
        # Print the debug info in order, one process at a time
        for i in 1:global_size
            if i == rank+1
                if args.use_gpu
                    @printf(" - %-4d: local %-2d/%2d (gpu %s) in node %s\n", 
                        rank, local_rank+1, local_size, string(CUDA.uuid(CUDA.device()))[1:8], node_name)
                else
                    @printf(" - %-4d: local %-2d/%2d in node %s\n", 
                        rank, local_rank+1, local_size, node_name)
                end
                threadinfo(; color=false, blocksize=64)
            end
            MPI.Barrier(MPI.COMM_WORLD)
        end
    end
end


function process_ratio_to_grid(n_proc, ratios)
    (rpx, rpy) = ratios
    r = rpx / rpy
    # In theory the ratios have been pre-checked so that those convertions don't throw InexactError
    px = convert(Int, √(n_proc * r))
    py = convert(Int, √(n_proc / r))
    return px, py
end

# 
# Armon parameters building and running
#

function build_params(args::RunArguments, test::Symbol, transpose::Symbol, splitting::Symbol, cells::Any, px::Int, py::Int)
    (; ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, euler_projection, 
       maxtime, maxcycle, verbose, write_output, write_ghosts, use_ccall, use_threading, use_simd, 
       interleaving, use_gpu, use_MPI, single_comm_per_axis_pass, reorder_grid, dim) = args
    if dim == 1
        if use_mpi
            return Armon.ArmonParameters(; 
                ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, 
                test=test, nbcell=cells,
                euler_projection, maxtime, maxcycle, silent=verbose, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, interleaving, use_gpu, use_MPI)
        else
            return Armon.ArmonParameters(; 
                ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, 
                test=test, nbcell=cells,
                euler_projection, maxtime, maxcycle, silent=verbose, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, interleaving, use_gpu)
        end
    else
        if use_mpi
            return Armon.ArmonParameters(; 
                ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, dt_on_even_cycles,
                test=test, nx=cells[1], ny=cells[2],
                euler_projection, transpose_dims=transpose, axis_splitting=splitting, 
                maxtime, maxcycle, silent=verbose, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, use_gpu, use_MPI, px, py, 
                single_comm_per_axis_pass, reorder_grid)
        else
            return Armon.ArmonParameters(; 
                ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, 
                test=test, nx=cells[1], ny=cells[2],
                euler_projection, transpose_dims=transpose, axis_splitting=splitting, 
                maxtime, maxcycle, silent=verbose, write_output, write_ghosts,
                use_ccall, use_threading, use_simd, use_gpu)
        end
    end
end


function merge_time_contribution(time_contrib_1, time_contrib_2)
    if isnothing(time_contrib_1)
        return time_contrib_2
    elseif isnothing(time_contrib_2)
        return time_contrib_1
    end

    return map((e, f) -> (e.first => (e.second + f.second)), time_contrib_1, time_contrib_2)
end


function run_armon(params)
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

#
# Precompilation
#

function args_for_precompilation(args::RunArguments)
    changed_args = Dict{Symbol, Any}(
        :dt_on_even_cycles => false,
        :maxcycle => 1,
        :verbose => 5,
        :write_output => false
    )

    new_fields = [haskey(changed_args, k) ? changed_args[k] : getfield(args, k)
                  for k in fieldnames(RunArguments{T})]
    return RunArguments(new_fields...)
end


function precompile_armon(args::RunArguments, is_root::Bool)
    if is_root
        println("Compiling...")
        compile_start_time = time_ns()
    end

    precompile_args = args_for_precompilation(args)
    cells = args.dim == 1 ? 10000 : (10, 10)

    for test in tests, transpose in transpose_dims
        run_armon(build_params(precompile_args, test, transpose, :Sequential, cells, 1, 1))
    end
    
    if is_root
        compile_end_time = time_ns()
        @printf(" (time: %3.1f sec)\n", (compile_end_time - compile_start_time) / 1e9)
    end
end

#
# Measurements and plots
#

function do_measure(args::RunArguments, data_file_name::String, 
        test::Symbol, cells::Int, transpose::Bool, splitting::Symbol)
    params = build_params(args, test, transpose, splitting, cells, 1, 1)

    if args.dim == 1
        @printf(" - %s, %11g cells: ", test, cells)
    else
        @printf(" - %-4s %-14s %11g cells (%5gx%-5g): ", 
            string(test) * (transpose ? "ᵀ" : ""),
            string(splitting), prod(cells), cells[1], cells[2])
    end

    cycles, cells_per_sec, time_contrib = run_armon(params)

    @printf("%6.3f Giga cells/sec\n", cells_per_sec)

    # Append the result to the data file
    if !isempty(data_file_name)
        open(data_file_name, "a") do data_file
            if args.dim == 1
                println(data_file, cells, ", ", cells_per_sec)
            else
                println(data_file, prod(cells), ", ", cells_per_sec)
            end
        end
    end

    return time_contrib
end


function do_measure_MPI(args::RunArguments, data_file_name::String, comm_file_name::String, 
        test::Symbol, cells::Tuple{Int, Int}, transpose::Bool, splitting::Symbol, px::Int, py::Int,
        is_root::Bool)
    if is_root
        if args.dim == 1
            @printf(" - %s, %11g cells: ", test, cells)
        else
            @printf(" - (%2dx%-2d) %-4s %-14s %11g cells (%5gx%-5g): ", 
                px, py,
                string(test) * (transpose ? "ᵀ" : ""),
                string(splitting), prod(cells), cells[1], cells[2])
        end
    end

    params = build_params(args, test, transpose, splitting, cells, px, py)
    cycles, cells_per_sec, time_contrib = run_armon(params)

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
                    println(data_file, prod(cells), ", ", total_cells_per_sec)
                end
            end
        end
        
        # Rebuild the time contribution array
        for (i, (step_label, _)) in enumerate(time_contrib)
            time_contrib[i] = step_label => merged_time_contrib_vals[i]
        end

        if args.time_MPI_graph
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
                        println(data_file, prod(cells), ", ", total_MPI_time, ", ", total_time)
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


function build_files_names(args::RunArguments, test::Symbol, transpose::Bool, splitting::Symbol, px::Int, py::Int)
    base_file_name = args.data_file

    if isempty(base_file_name)
        data_file_name = ""
        hist_file_name = ""
        comm_file_name = ""
    elseif args.dim == 1
        data_file_name = base_file_name * string(test) * ".csv"
        hist_file_name = base_file_name * string(test) * "_hist.csv"
        comm_file_name = base_file_name * string(test) * "_MPI_time.csv"
    else
        data_file_name = base_file_name * string(test)

        if length(args.transpose_dims) > 1
            data_file_name *= transpose ? "_transposed" : ""
        end

        if length(args.axis_splitting) > 1
            data_file_name *= "_" * string(splitting)
        end

        if length(args.proc_grid) > 1
            data_file_name *= "_pg=$(px)x$(py)"
        end

        hist_file_name = data_file_name * "_hist.csv"
        comm_file_name = data_file_name * "_MPI_time.csv"
        data_file_name *= ".csv"
    end

    return data_file_name, hist_file_name, comm_file_name
end


function update_plots(args::RunArguments)
    if !isempty(args.gnuplot_script)
        # We redirect the output of gnuplot to null so that there is no warning messages displayed
        run(pipeline(`gnuplot $(args.gnuplot_script)`, stdout=devnull, stderr=devnull))
    end

    if !isempty(args.gnuplot_MPI_script)
        # Same for the MPI time graph
        run(pipeline(`gnuplot $(args.gnuplot_MPI_script)`, stdout=devnull, stderr=devnull))
    end
end


function build_time_histogram(args::RunArguments, hist_file_name::String, total_time_contrib)
    if isempty(hist_file_name) || isnothing(total_time_contrib)
        return
    end

    if args.flatten_time_dims
        flat_time_contrib = nothing
        for axis_time_contrib in total_time_contrib
            flat_time_contrib = merge_time_contribution(flat_time_contrib, axis_time_contrib)
        end
        total_time_contrib = flat_time_contrib
    elseif args.dim == 2
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
            println(data_file, '"', label, "\", ", time / length(args.cells_list))
        end
    end

    if !isempty(args.gnuplot_hist_script)
        # Update the histogram plot
        run(pipeline(`gnuplot $(args.gnuplot_hist_script)`, stdout=devnull, stderr=devnull))
    end
end


function run_measurements(args::RunArguments, is_root::Bool)
    for test in tests, transpose in transpose_dims, splitting in axis_splitting, (px, py) in proc_domains
        data_file_name, hist_file_name, comm_file_name = build_files_names(args, test, transpose, splitting, px, py)
    
        total_time_contrib = nothing
    
        for cells in cells_list
            if use_MPI
                time_contrib = do_measure_MPI(data_file_name, comm_file_name, test, cells, transpose, splitting, px, py)
            else
                time_contrib = do_measure(data_file_name, test, cells, transpose, splitting)
            end
    
            if is_root
                # total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)
                update_plots(args)
            end
        end
    
        if time_histogram && is_root
            build_time_histogram(args, hist_file_name, total_time_contrib)
        end
    end    
end


if isinteractive()
    println("""
    Usage:
     - build arguments:       args = parse_all_arguments(raw_args)
     - setup the environment: is_root = setup_run_armon(args)
     - (optional) precompile: precompile_armon(args, is_root)
     - run:                   run_measurements(args, is_root)
    """)
else
    args = parse_all_arguments(ARGS)
    is_root = setup_run_armon(args)
    precompile_armon(args, is_root)
    run_measurements(args, is_root)
end
