
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
euler_projection = false
cst_dt = false
ieee_bits = 64
silent = 2
write_output = false
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

base_file_name = ""
gnuplot_script = ""
gnuplot_hist_script = ""
time_histogram = false
flatten_time_dims = false
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
        global scheme=Symbol(replace(ARGS[i+1], '-' => '_'))
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
    elseif arg == "--write-output"
        global write_output = parse(Bool, ARGS[i+1])
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
    using MPI
    MPI.Init()
    is_root = MPI.Comm_rank(MPI.COMM_WORLD) == 0
    if is_root
        println("Using MPI with $(MPI.Comm_size(MPI.COMM_WORLD)) processes")
        println("Loading...")
    end

    if dimension == 1
        include("armon_1D_MPI.jl")
    else
        # include("armon_2D_MPI.jl")
        error("2D MPI NYI")
    end
else
    println("Loading...")

    if dimension == 1
        include("armon_1D.jl")
    else
        include("armon_2D.jl")
    end
    is_root = true
end
using .Armon


if !use_gpu
    # TODO : make sure that threads of different processes but on the same node don't pin their threads on the same cores
    is_root && @warn "Thread pinning might be problematic with MPI"
    omp_bind_threads(threads_places, threads_proc_bind)
end


if dimension == 1
    transpose_dims = [false]
    axis_splitting = [:Sequential]

    function build_params(test, cells; 
            ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, euler_projection, transpose_dims, 
            axis_splitting, maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, interleaving, use_gpu)
        return ArmonParameters(; ieee_bits, riemann, scheme, nghost, iterations, cfl, Dt, cst_dt, 
            test=test, nbcell=cells,
            euler_projection, maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, interleaving, use_gpu)
    end
else
    function build_params(test, domain; 
            ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, euler_projection, transpose_dims, 
            axis_splitting, maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, interleaving, use_gpu)
        return ArmonParameters(; ieee_bits, riemann, scheme, nghost, cfl, Dt, cst_dt, 
            test=test, nx=domain[1], ny=domain[2],
            euler_projection, transpose_dims, axis_splitting, 
            maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, use_gpu)
    end
end


function merge_time_contribution(time_contrib_1, time_contrib_2)
    if isnothing(time_contrib_1)
        return time_contrib_2
    elseif isnothing(time_contrib_2)
        return time_contrib_1
    end

    if dimension == 1
        return map((e, f) -> (e.first => (e.second + f.second)), time_contrib_1, time_contrib_2)
    else
        return map.((e, f) -> (e.first => (e.second + f.second)), time_contrib_1, time_contrib_2)
    end
end


function run_armon(params::ArmonParameters)
    total_cells_per_sec = 0
    total_time_contrib = nothing

    for _ in 1:repeats
        _, cells_per_sec, time_contrib = armon(params)
        total_cells_per_sec += cells_per_sec
        total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)
    end
    
    return total_cells_per_sec / repeats, total_time_contrib
end


function do_measure(data_file_name, test, cells, transpose, splitting)
    params = build_params(test, cells; 
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
        maxtime, maxcycle, silent, write_output, use_ccall, use_threading, use_simd,
        interleaving, use_gpu
    )

    if dimension == 1
        @printf(" - %s, %10g cells: ", test, cells)
    else
        @printf(" - %-4s %-14s %10g cells (%5gx%-5g): ", 
            string(test) * (transpose ? "ᵀ" : ""),
            string(splitting), cells[1] * cells[2], cells[1], cells[2])
    end

    cells_per_sec, time_contrib = run_armon(params)

    @printf("%.2g Giga cells/sec\n", cells_per_sec)

    # Append the result to the data file
    open(data_file_name, "a") do data_file
        if dimension == 1
            println(data_file, cells, ", ", cells_per_sec)
        else
            println(data_file, cells[1] * cells[2], ", ", cells_per_sec)
        end
    end

    return time_contrib
end


function do_measure_MPI(data_file_name, test, cells, transpose, splitting)
    if is_root
        if dimension == 1
            @printf(" - %s, %10g cells: ", test, cells)
        else
            @printf(" - %-4s %-14s %10g cells (%5gx%-5g): ", 
                string(test) * (transpose ? "ᵀ" : ""),
                string(splitting), cells[1] * cells[2], cells[1], cells[2])
        end
    end

    cells_per_sec, time_contrib = run_armon(build_params(test, cells; 
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
        maxtime, maxcycle, silent, write_output, use_ccall, use_threading, use_simd,
        interleaving, use_gpu
    ))

    is_root && @printf("%.2g Giga cells/sec\n", cells_per_sec)

    # Merge the results of all processes
    cells_per_sec = MPI.Reduce(cells_per_sec, MPI.Op(+, typeof(cells_per_sec)), 0, MPI.COMM_WORLD)

    # Merge the time distribution of all processes
    # TODO
    #time_contrib = MPI.Reduce(time_contrib, MPI.Op(merge_time_contribution; iscommutative=true), 0, MPI.COMM_WORLD)

    if is_root
        # Append the result to the data file
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


is_root && println("Compiling...")
for test in tests, transpose in transpose_dims
    run_armon(build_params(test, dimension == 1 ? 10000 : (10, 10);
        ieee_bits, riemann, scheme, iterations, cfl, 
        Dt, cst_dt, euler_projection, transpose_dims=transpose, axis_splitting=axis_splitting[1], 
        maxtime, maxcycle=1, silent=5, write_output=false, use_ccall, use_threading, use_simd, 
        interleaving, use_gpu))
end


for test in tests, transpose in transpose_dims, splitting in axis_splitting
    if dimension == 1
        data_file_name = base_file_name * string(test) * ".csv"
        hist_file_name = base_file_name * string(test) * "_hist.csv"
    else
        data_file_name = base_file_name * string(test)
        hist_file_name = base_file_name * string(test) 

        if length(transpose_dims) > 1
            data_file_name *= transpose ? "_transposed" : ""
            hist_file_name *= transpose ? "_transposed" : ""
        end

        if length(axis_splitting) > 1
            data_file_name *= "_" * string(splitting)
            hist_file_name *= "_" * string(splitting)
        end

        data_file_name *= ".csv"
        hist_file_name *= "_hist.csv"
    end

    total_time_contrib = nothing

    for cells in cells_list
        if use_MPI
            time_contrib = do_measure_MPI(data_file_name, test, cells, transpose, splitting)
        else
            time_contrib = do_measure(data_file_name, test, cells, transpose, splitting)
        end

        if is_root
            total_time_contrib = merge_time_contribution(total_time_contrib, time_contrib)

            if !isempty(gnuplot_script)
                # We redirect the output of gnuplot to null so that there is no warning messages displayed
                run(pipeline(`gnuplot $(gnuplot_script)`, stdout=devnull, stderr=devnull))
            end
        end
    end

    if time_histogram && is_root
        if flatten_time_dims
            flat_time_contrib = nothing
            for axis_time_contrib in total_time_contrib
                flat_time_contrib = merge_time_contribution(flat_time_contrib, axis_time_contrib)
            end
            total_time_contrib = flat_time_contrib
        elseif dimension == 2
            # Append the axis name at the beginning of each label and merge into a single list
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
