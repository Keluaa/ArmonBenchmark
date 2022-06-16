
using Printf

include("omp_simili.jl")

scheme = :GAD_minmod
riemann = :acoustic
iterations = 4
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

base_file_name = ""
gnuplot_script = ""
gnuplot_hist_script = ""
time_histogram = false
flatten_time_dims = false


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


if !use_gpu
    omp_bind_threads(threads_places, threads_proc_bind)
end


println("Working in $(dimension)D")
println("Loading...")
if dimension == 1
    include("armon_module_gpu.jl")
else
    include("armon_module_gpu_2D.jl")
end
using .Armon


if dimension == 1
    transpose_dims = [false]
    axis_splitting = [:Sequential]

    function build_params(test, cells; 
            ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, euler_projection, transpose_dims, 
            axis_splitting, maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, interleaving, use_gpu)
        return ArmonParameters(; ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, 
            test=test, nbcell=cells,
            euler_projection, maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, interleaving, use_gpu)
    end

    function run_armon(params::ArmonParameters)
        return armon(params)
    end
else
    function build_params(test, domain; 
            ieee_bits, riemann, scheme, iterations, cfl, Dt, cst_dt, euler_projection, transpose_dims, 
            axis_splitting, maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, interleaving, use_gpu)
        return ArmonParameters(; ieee_bits, riemann, scheme, cfl, Dt, cst_dt, 
            test=test, nx=domain[1], ny=domain[2],
            euler_projection, transpose_dims, axis_splitting, 
            maxtime, maxcycle, silent, write_output, 
            use_ccall, use_threading, use_simd, use_gpu)
    end

    function run_armon(params::ArmonParameters)
        _, cells_per_sec, time_contrib = armon(params)
        return cells_per_sec, time_contrib
    end
end


println("Compiling...")
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
        if dimension == 1
            @printf(" - %s, %10g cells: ", test, cells)
        else
            @printf(" - %-4s %-14s %10g cells (%5gx%-5g): ", 
                string(test) * (transpose ? "ᵀ" : ""),
                string(splitting), cells[1] * cells[2], cells[1], cells[2])
        end

        cells_per_sec, time_contrib = -1., 0.
        while cells_per_sec < 0
            cells_per_sec, time_contrib = run_armon(build_params(test, cells; 
                ieee_bits, riemann, scheme, iterations, cfl, 
                Dt, cst_dt, euler_projection, transpose_dims=transpose, axis_splitting=splitting,
                maxtime, maxcycle, silent, write_output, use_ccall, use_threading, use_simd,
                interleaving, use_gpu))
            if cells_per_sec < 0
                print(" (negative throughput, restarting...) ")
            end
        end
 
        @printf("%.2g Giga cells/sec\n", cells_per_sec)

        # Append the result to the data file
        open(data_file_name, "a") do data_file
            if dimension == 1
                println(data_file, cells, ", ", cells_per_sec)
            else
                println(data_file, cells[1] * cells[2], ", ", cells_per_sec)
            end
        end

        if !isempty(gnuplot_script)
            # We redirect the output of gnuplot to null so that there is no warning messages displayed
            run(pipeline(`gnuplot $(gnuplot_script)`, stdout=devnull, stderr=devnull))
        end

        if isnothing(total_time_contrib)
            total_time_contrib = time_contrib
        else
            if dimension == 1
                total_time_contrib = map((e, e′) -> (e.first => (e.second + e′.second)), total_time_contrib, time_contrib)
            else
                total_time_contrib = map.((e, e′) -> (e.first => (e.second + e′.second)), total_time_contrib, time_contrib)
            end
        end
    end

    if time_histogram
        if flatten_time_dims
            flat_time_contrib = nothing
            for axis_time_contrib in total_time_contrib
                if isnothing(flat_time_contrib)
                    flat_time_contrib = axis_time_contrib
                else
                    flat_time_contrib = map((e, e′) -> (e.first => (e.second + e′.second)), flat_time_contrib, axis_time_contrib)
                end
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
