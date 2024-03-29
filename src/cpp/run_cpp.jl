
using Printf
using Statistics


include(joinpath(@__DIR__, "..", "common_utils.jl"))


mutable struct CppOptions
    scheme::String
    riemann_limiter::String
    nghost::Int
    cfl::Float64
    Dt::Float64
    maxtime::Float64
    maxcycle::Int
    projection::String
    cst_dt::Bool
    ieee_bits::Int
    silent::Int
    output_file::String
    write_output::Bool
    use_simd::Bool
    use_std_vector::Bool
    num_threads::Int
    threads_places::String
    threads_proc_bind::String
    tests::Vector{String}
    cells_list::Vector{Int}
    base_file_name::String
    gnuplot_script::String
    repeats::Int
    min_acquisition_time::Int
    repeats_count_file::String
    compiler::String
end


project_dir = joinpath(@__DIR__, "../../cpp")
exe_name = "armon.exe"
exe_path = joinpath(project_dir, "$exe_name")
last_compile_options_file = joinpath(project_dir, "last_compile_options.txt")
default_make_options = ["--no-print-directory"]


function CppOptions(;
        scheme = "GAD", riemann_limiter = "minmod", projection = "euler",
        nghost = 2, cfl = 0.6, Dt = 0., maxtime = 0.,
        maxcycle = 500, cst_dt = false, ieee_bits = 64,
        silent = 2, output_file = "output", write_output = false,
        use_simd = true, use_std_vector = false,
        num_threads = 1, threads_places = "cores", threads_proc_bind = "close",
        tests = [], cells_list = [],
        base_file_name = "", gnuplot_script = "", repeats = 1, min_acquisition_time = 0,
        repeats_count_file = "",
        compiler = "clang")
    return CppOptions(
        scheme, riemann_limiter,
        nghost, cfl, Dt, maxtime, maxcycle,
        projection, cst_dt, ieee_bits,
        silent, output_file, write_output,
        use_simd, use_std_vector,
        num_threads, threads_places, threads_proc_bind,
        tests, cells_list,
        base_file_name, gnuplot_script, repeats, min_acquisition_time,
        repeats_count_file,
        compiler
    )
end


function parse_arguments(args::Vector{String})
    options = CppOptions()

    i = 1
    while i <= length(args)
        arg = args[i]

        # Solver params
        if arg == "-s"
            options.scheme = replace(args[i+1], '-' => '_')
            i += 1
        elseif arg == "--ieee"
            options.ieee_bits = parse(Int, args[i+1])
            i += 1
        elseif arg == "--cycle"
            options.maxcycle = parse(Int, args[i+1])
            i += 1
        elseif arg == "--riemann-limiter"
            options.riemann_limiter = replace(args[i+1], '-' => '_')
            i += 1
        elseif arg == "--time"
             options.maxtime = parse(Float64, args[i+1])
            i += 1
        elseif arg == "--cfl"
            options.cfl = parse(Float64, args[i+1])
            i += 1
        elseif arg == "--dt"
            options.Dt = parse(Float64, args[i+1])
            i += 1
        elseif arg == "--projection"
            options.projection = args[i+1]
            i += 1
        elseif arg == "--cst-dt"
            options.cst_dt = parse(Bool, args[i+1])
            i += 1
        elseif arg == "--nghost"
            options.nghost = parse(Int, args[i+1])
            i += 1

        # Solver output params
        elseif arg == "--verbose"
            options.silent = parse(Int, args[i+1])
            i += 1
        elseif arg == "--output-file"
            options.output_file = args[i+1]
            i += 1
        elseif arg == "--write-output"
            options.write_output = parse(Bool, args[i+1])
            i += 1

        # Multithreading params
        elseif arg == "--use-simd"
            options.use_simd = parse(Bool, args[i+1])
            i += 1
        elseif arg == "--num-threads"
            options.num_threads = parse(Int, args[i+1])
            i += 1
        elseif arg == "--threads-places"
            options.threads_places = args[i+1]
            i += 1
        elseif arg == "--threads-proc-bind"
            options.threads_proc_bind = args[i+1]
            i += 1

        # List params
        elseif arg == "--tests"
            options.tests = split(args[i+1], ',')
            i += 1
        elseif arg == "--cells-list"
            list = split(args[i+1], ',')
            options.cells_list = convert.(Int, parse.(Float64, list))
            i += 1

        # Measurements params
        elseif arg == "--repeats"
            options.repeats = parse(Int, args[i+1])
            i += 1
        elseif arg == "--min-acquisition-time"
            options.min_acquisition_time = parse(Int, args[i+1]) * 1e9
            i += 1

        # Measurement output params
        elseif arg == "--data-file"
            options.base_file_name = args[i+1]
            i += 1
        elseif arg == "--gnuplot-script"
            options.gnuplot_script = args[i+1]
            i += 1
        elseif arg == "--repeats-count-file"
            options.repeats_count_file = args[i+1]
            i += 1

        # C++ backend options
        elseif arg == "--compiler"
            options.compiler = args[i+1]
            i += 1
        elseif arg == "--use-std-vector"
            options.use_std_vector = parse(Bool, args[i+1])
            i += 1

        else
            error("Wrong option: ", arg)
        end

        i += 1
    end

    return options
end


function cpu_has_avx512()
    try
        read(`grep avx512 /proc/cpuinfo`)
        true
    catch
        false
    end
end


function run_cmd_print_on_error(cmd::Cmd)
    # Run a command without printing its output to stdout
    # However if there is an error we print the command's output
    # A temporary file is used to achieve that
    mktemp() do _, file
        try 
            run(pipeline(cmd, stdout=file#= , stderr=file =#))
        catch
            flush(file)
            seekstart(file)
            println("ERROR:\n", read(file, String))
            rethrow()
        end
    end
end


function check_if_recompile_needed(new_options)
    !isfile(last_compile_options_file) && return true
    last_options = collect(eachline(last_compile_options_file))
    return length(last_options) != length(new_options) || any(last_options .!= new_options)
end


function write_compile_options_file(make_options)
    open(last_compile_options_file, "w") do file
        foreach(x -> println(file, x), make_options)
    end
end


function compile_backend(options::CppOptions)
    # TODO: put compilation files in a tmp dir in the script dir named with the job ID
    make_options = []
    append!(make_options, default_make_options)

    use_simd = convert(Int, options.use_simd)
    use_threading = convert(Int, options.num_threads > 1)
    use_single_precision = convert(Int, options.ieee_bits == 32)
    use_std_vector = convert(Int, options.use_std_vector)

    append!(make_options, [
        "use_simd=$use_simd",
        "use_threading=$use_threading",
        "use_single_precision=$use_single_precision",
        "use_std_vector=$use_std_vector",
    ])

    if options.use_simd
        has_avx512 = cpu_has_avx512()
        push!(make_options, "has_avx512=$has_avx512")
    end

    if options.compiler == "clang"
        compiler = "use_clang=1"
    elseif options.compiler == "icc"
        compiler = "use_icc=1"
    elseif options.compiler == "dpcpp"
        compiler = "use_dpcpp=1"
    else
        compiler = "use_gcc=1"
    end
    push!(make_options, compiler)

    if check_if_recompile_needed(make_options)
        # Make sure the exe is deleted before compiling
        run_cmd_print_on_error(Cmd(`make clean`; dir=project_dir))
        run_cmd_print_on_error(Cmd(`make $make_options $exe_name`; dir=project_dir))
        write_compile_options_file(make_options)
    end
end


function setup_env(options::CppOptions)
    ENV["OMP_PLACES"] = options.threads_places
    ENV["OMP_PROC_BIND"] = options.threads_proc_bind
    ENV["OMP_NUM_THREADS"] = options.num_threads
    if haskey(ENV, "KMP_AFFINITY")
        # Prevent Intel's variables from interfering with ours
        delete!(ENV, "KMP_AFFINITY")
    end
end


function build_args_list(options::CppOptions)
    scheme = options.scheme
    projection = lowercase(options.projection)

    if scheme == "GAD"
        if lowercase(options.riemann_limiter) != "minmod"
            error("Unsupported limiter for GAD scheme: $(options.riemann_limiter)")
        end
        scheme = "GAD-minmod"
    end

    if projection == "none"
        projection = 0
    elseif projection == "euler"
        projection = 1
    else
        error("Unsupported projection: $projection")
    end

    return [
        "-s", scheme,
        "--cfl", options.cfl,
        "--dt", options.Dt,
        "--time", options.maxtime,
        "--cycle", options.maxcycle,
        "--euler", projection,
        "--nghost", options.nghost,
        "--cst-dt", Int(options.cst_dt),
        "--write-output", Int(options.write_output),
        "--output", options.output_file,
        "--verbose", options.silent
    ]
end


function get_run_command(args)
    return Cmd(`$exe_path $args`; dir=project_dir)
end


function run_and_parse_output(cmd::Cmd, verbose::Bool, repeats::Int, min_acquisition_time::Int)
    if verbose
        println(cmd)
    end

    vals_cells_per_sec = Vector{Float64}()

    total_repeats = 0
    acquisition_start = time_ns()

    while total_repeats < repeats || (time_ns() - acquisition_start) < min_acquisition_time
        output = read(cmd, String)

        mega_cells_per_sec_raw = match(r"Cells/sec:\s*\K[0-9\.]+", output)

        if isnothing(mega_cells_per_sec_raw)
            error("Command failed, wrong output: ", cmd, "\nOutput:\n", output)
        elseif verbose
            println(output)
        end

        mega_cells_per_sec = parse(Float64, mega_cells_per_sec_raw.match)
        giga_cells_per_sec = mega_cells_per_sec / 1e3
        push!(vals_cells_per_sec, giga_cells_per_sec)

        total_repeats += 1
    end

    return total_repeats, vals_cells_per_sec
end


function run_armon(options::CppOptions, verbose::Bool)
    compile_backend(options)
    setup_env(options)
    base_args = build_args_list(options)

    for test in options.tests
        if isempty(options.base_file_name)
            data_file_name = ""
        else
            data_file_name = options.base_file_name

            if length(options.tests) > 1
                data_file_name *= "_" * test
            end

            data_file_name *= "_perf.csv"

            data_dir = dirname(data_file_name)
            if !isdir(data_dir)
                mkpath(data_dir)
            end
        end

        for cells in options.cells_list
            args = Any[
                "-t", test,
                "--cells", cells
            ]
            append!(args, base_args)

            @printf(" - ")
            length(options.tests) > 1 && @printf("%s, ", test)
            @printf("%11g cells: ", prod(cells))
    
            run_cmd = get_run_command(args)

            time_start = time_ns()
            actual_repeats, repeats_cells_throughput = run_and_parse_output(run_cmd, verbose, options.repeats, options.min_acquisition_time)
            time_end = time_ns()

            duration = (time_end - time_start) / 1.0e9

            total_cells_per_sec = mean(repeats_cells_throughput)
            std_cells_per_sec = length(repeats_cells_throughput) > 1 ? std(repeats_cells_throughput; corrected=true) : 0

            @printf("%8.3f ± %4.2f Giga cells/sec %s", total_cells_per_sec, std_cells_per_sec,
                get_duration_string(duration))

            if actual_repeats != options.repeats
                println(" ($actual_repeats repeats)")
            else
                println()
            end

            if !isempty(data_file_name)
                # Append the result to the output file
                open(data_file_name, "a") do file
                    println(file, prod(cells), ", ", total_cells_per_sec, std_cells_per_sec, actual_repeats)
                end
            end

            if !isempty(options.gnuplot_script)
                # Update the plot
                # We redirect the output of gnuplot to null so that there is no warning messages displayed
                run(pipeline(`gnuplot $(options.gnuplot_script)`, stdout=devnull, stderr=devnull))
            end

            if !isempty(options.repeats_count_file)
                open(options.repeats_count_file, "w") do file
                    println(file, prod(cells), ",", actual_repeats)
                end
            end
        end
    end
end


if !isinteractive()
    cpp_options = parse_arguments(ARGS)
    run_armon(cpp_options, false)
end
