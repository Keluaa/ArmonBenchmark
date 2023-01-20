
using Printf
using Statistics


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
    compiler::String
    track_energy_consumption::Bool
end


project_dir = @__DIR__()
exe_name = "armon.exe"
exe_path = project_dir * "/$exe_name"
last_compile_options_file = project_dir * "/last_compile_options.txt"
run_dir = project_dir * "/../data"
default_make_options = ["--no-print-directory"]


function CppOptions(;
        scheme = "GAD", riemann_limiter = "minmod", projection = "euler",
        nghost = 2, cfl = 0.6, Dt = 0., maxtime = 0.,
        maxcycle = 500, cst_dt = false, ieee_bits = 64,
        silent = 2, output_file = "output", write_output = false,
        use_simd = true, use_std_vector = false,
        num_threads = 1, threads_places = "cores", threads_proc_bind = "close",
        tests = [], cells_list = [],
        base_file_name = "", gnuplot_script = "", repeats = 1,
        compiler = "clang",
        track_energy_consumption = false)
    return CppOptions(
        scheme, riemann_limiter,
        nghost, cfl, Dt, maxtime, maxcycle,
        projection, cst_dt, ieee_bits,
        silent, output_file, write_output,
        use_simd, use_std_vector,
        num_threads, threads_places, threads_proc_bind,
        tests, cells_list,
        base_file_name, gnuplot_script, repeats,
        compiler,
        track_energy_consumption
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
        elseif arg == "--track-energy"
            options.track_energy_consumption = parse(Bool, args[i+1])
            i += 1

        # Measurement output params
        elseif arg == "--data-file"
            options.base_file_name = args[i+1]
            i += 1
        elseif arg == "--gnuplot-script"
            options.gnuplot_script = args[i+1]
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


function get_current_energy_consumed()
    job_id = get(ENV, "SLURM_JOBID", "")
    if isempty(job)
        @warn "SLURM_JOBID is not defined, cannot get the energy consumption" maxlog=1
        return 0
    end

    format = "jobid,ConsumedEnergyRaw"
    slurm_cmd = `sstat -j $job_id -a -P -o $format`
    output = read(slurm_cmd, String)

    parsed = output |> strip |> split
    parsed = map(step -> split(step, '|'), parsed)
    current_job_step = last(parsed)

    length(current_job_step) != 2 && error("Expected two columns in the output. Output:\n$output\n")

    return parse(Int, last(current_job_step))  # In Joules
end


function run_and_parse_output(cmd::Cmd, verbose::Bool, repeats::Int, track_energy::Bool)
    if verbose
        println(cmd)
    end

    total_giga_cells_per_sec = 0
    energy_consumed = zeros(Int, repeats)
    prev_energy = track_energy ? get_current_energy_consumed() : 0

    for _ in 1:repeats
        output = read(cmd, String)

        if track_energy
            current_energy = get_current_energy_consumed()
            energy_consumed[i] = current_energy - prev_energy
            prev_energy = current_energy
        end

        mega_cells_per_sec_raw = match(r"Cells/sec:\s*\K[0-9\.]+", output)

        if isnothing(mega_cells_per_sec_raw)
            error("Command failed, wrong output: ", cmd, "\nOutput:\n", output)
        elseif verbose
            println(output)
        end

        mega_cells_per_sec = parse(Float64, mega_cells_per_sec_raw.match)
        giga_cells_per_sec = mega_cells_per_sec / 1e3
        total_giga_cells_per_sec += giga_cells_per_sec
    end

    total_giga_cells_per_sec /= repeats
    return total_giga_cells_per_sec, energy_consumed
end


function run_armon(options::CppOptions, verbose::Bool)
    compile_backend(options)
    setup_env(options)
    base_args = build_args_list(options)

    for test in options.tests
        if isempty(options.base_file_name)
            data_file_name = ""
        else
            data_file_name = options.base_file_name * test
            ergy_file_name = data_file_name * "_ENERGY.csv"
            data_file_name *= ".csv"
        end

        for cells in options.cells_list
            args = Any[
                "-t", test,
                "--cells", cells
            ]
            append!(args, base_args)
    
            @printf(" - %s, %11g cells: ", test, cells[1])
    
            run_cmd = get_run_command(args)
            cells_throughput, repeats_energy_consumed = run_and_parse_output(run_cmd, verbose, options.repeats, options.track_energy_consumption)

            mean_energy_consumed = mean(repeats_energy_consumed)

            if length(repeats_energy_consumed) > 1
                std_energy_consumed = std(repeats_energy_consumed; corrected=true)
            else
                std_energy_consumed = 0
            end

            @printf("%8.2f Giga cells/sec\n", cells_throughput)
            
            if !isempty(data_file_name)
                # Append the result to the output file
                open(data_file_name, "a") do file
                    println(file, cells, ", ", cells_throughput)
                end
            end

            if options.track_energy_consumption && !isempty(energy_file_name)
                open(energy_file_name, "a") do file
                    println(file, cells, ", ", mean_energy_consumed, ", ", std_energy_consumed, ", ",
                        join(repeats_energy_consumed, ", "))
                end
            end
            
            if !isempty(options.gnuplot_script)
                # Update the plot
                # We redirect the output of gnuplot to null so that there is no warning messages displayed
                run(pipeline(`gnuplot $(options.gnuplot_script)`, stdout=devnull, stderr=devnull))
            end
        end
    end
end


if !isinteractive()
    cpp_options = parse_arguments(ARGS)
    run_armon(cpp_options, false)
end
