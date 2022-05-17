
using Printf

@enum Device CPU CUDA ROCM
@enum Backend CPP Kokkos Julia
@enum Compiler GCC Clang ICC


mutable struct MeasureParams
    # ccc_mprun params
    device::Device
    node::String
    distributions::Vector{String}
    processes::Vector{Int}
    max_time::Int

    # Backend params
    backends::Vector{Backend}
    threads::Vector{Int}
    block_sizes::Vector{Int}
    ieee_bits::Vector{Int}
    use_simd::Vector{Int}
    compilers::Vector{Compiler}
    omp_schedule::Vector{String}
    omp_proc_bind::Vector{String}
    omp_places::Vector{String}

    # Armon params
    cells_list::Vector{Int}
    tests_list::Vector{String}
    common_armon_params::Vector{String}

    # Measurement params
    name::String
    gnuplot_script::String
    plot_file::String
    log_scale::Bool
    plot_title::String
    verbose::Bool
end


no_inti_cmd(measure_index, inti_index) = `julia $(PROGRAM_FILE) $(ARGS[1]) $(measure_index) $(inti_index)`
inti_cmd(options, measure_index, inti_index) = `ccc_mprun $(options) julia $(PROGRAM_FILE) $(ARGS[1]) $(measure_index) $(inti_index)`
gnuplot_cmd(plot_file) = `gnuplot $(plot_file)`

julia_options = ["-O3", "--check-bounds=no"]
armon_base_options = [
    "--write-output", "0",
    "--verbose", "2"
]
min_inti_cores = 4  # Minimun number of cores which will be allocated for each INTI job

base_make_options = "-j4"

julia_script_path = "./julia/run_julia.jl"
julia_tmp_script_output_file = "./tmp_script_output.txt"
cpp_exe_path = "./cpp/armon.exe"
cpp_make_dir = "./cpp/"
cpp_make_target = "armon.exe"
kokkos_make_dir = "./kokkos/"

data_dir = "./data/"
plot_scripts_dir = "./plot_scripts/"
plots_dir = "./plots/"


base_gnuplot_script_commands(graph_file_name, title, log_scale) = """
set terminal pdfcairo color size 10in, 6in
set output '$(graph_file_name)'
set ylabel 'Giga Cells/sec'
set xlabel 'Cells count'
set title "$(title)"
set key left top
$(log_scale ? "set logscale x" : "")
plot """

gnuplot_plot_command(data_file, legend_title, pt_index) = "'$(data_file)' w lp pt $(pt_index) title '$(legend_title)'"


function parse_measure_params(file_line_parser)    
    device = CPU
    node = "a100-bxi"
    distributions = ["block"]
    processes = [1]
    max_time = 3600  # 1h
    backends = [Julia]
    threads = [4]
    block_sizes = [256]
    ieee_bits = [64]
    use_simd = [true]
    compilers = [GCC]
    omp_schedule = ["static"]
    omp_proc_bind = ["spread"]
    omp_places = ["cores"]
    cells_list = [12.5e3, 25e3, 50e3, 100e3, 200e3, 400e3, 800e3, 1.6e6, 3.2e6, 6.4e6, 12.8e6, 25.6e6, 51.2e6, 102.4e6]
    tests_list = ["Sod"]
    common_armon_params = [
        "--write-output", "0",
        "--verbose", "2",
        "--euler", "0"
    ]
    name = nothing
    gnuplot_script = nothing
    plot_file = nothing
    log_scale = true
    plot_title = nothing
    verbose = false

    last_i = 0
    for (i, line) in file_line_parser
        last_i = i
        line = chomp(line)
        if line == "-"; break; end    # End of this measure
        if line == ""; continue; end  # Empty line
        if startswith(line, '#'); continue; end # Comment
        if isnothing(findfirst('=', line))
            error("Missing '=' at line $(i)")
        end

        option, value = split(line, '=')
        if option == "device"
            if value == "CPU"
                device = CPU
            elseif value == "CUDA"
                device = CUDA
            elseif value == "ROCM"
                device = ROCM
            else
                error("Unknown device: $(value), at line $(i)")
            end
        elseif option == "node"
            node = value
        elseif option == "distributions"
            distributions = split(value, ',')
        elseif option == "processes"
            processes = parse.(Int, split(value, ','))
        elseif option == "max_time"
            max_time = parse(Int, value)
        elseif option == "backends"
            backends = []
            for backend in split(value, ',')
                backend = lowercase(backend)
                if backend == "julia"
                    push!(backends, Julia)
                elseif backend == "kokkos"
                    push!(backends, Kokkos)
                elseif backend == "cpp"
                    push!(backends, CPP)
                else
                    error("Unknown backend: $(backend), at line $(i)")
                end
            end            
        elseif option == "threads"
            threads = parse.(Int, split(value, ','))
        elseif option == "block_sizes"
            block_sizes = parse.(Int, split(value, ','))
        elseif option == "ieee_bits"
            ieee_bits = parse.(Int, split(value, ','))
        elseif option == "use_simd"
            use_simd = parse.(Int, split(value, ','))
        elseif option == "compilers"
            compilers = []
            for compiler in split(value, ',')
                compiler = lowercase(compiler)
                if compiler == "gcc"
                    push!(compilers, GCC)
                elseif compiler == "clang"
                    push!(compilers, Clang)
                elseif compiler == "icc"
                    push!(compilers, ICC)
                else
                    error("Unknown compiler: $(compiler), at line $(i)")
                end
            end
        elseif option == "omp_schedule"
            omp_schedule = strip.(split(value, ';'))
        elseif option == "omp_proc_bind"
            omp_proc_bind = strip.(split(value, ';'))
        elseif option == "omp_places"
            omp_places = strip.(split(value, ';'))
        elseif option == "cells"
            cells_list = parse.(Float64, split(value, ','))
        elseif option == "tests"
            tests_list = split(value, ',')
        elseif option == "armon"
            common_armon_params = split(value, ' ')
        elseif option == "name"
            name = value
        elseif option == "gnuplot"
            gnuplot_script = value
        elseif option == "plot"
            plot_file = value
        elseif option == "title"
            plot_title = value
        elseif option == "log_scale"
            log_scale = parse(Bool, value)
        elseif option == "verbose"
            verbose = parse(Bool, value)
        else
            error("Unknown option: $(option), at line $(i)")
        end
    end

    # Post processing
    cells_list = convert.(Int, cells_list)

    if isnothing(name)
        error("Expected a name for the measurement at line ", last_i)
    end
    if isnothing(gnuplot_script)
        gnuplot_script = name * ".plot"
    end
    if isnothing(plot_file)
        # By default, same name as the plot script but as a'.pdf' file
        plot_file = gnuplot_script[1:findlast('.', gnuplot_script)-1] * ".pdf"
    end
    if isnothing(plot_title)
        plot_title = "You forgot to add a title"
    end

    mkpath(data_dir * name)
    gnuplot_script = plot_scripts_dir * gnuplot_script
    plot_file = plots_dir * plot_file

    return MeasureParams(device, node, distributions, processes, max_time,
        backends, threads, block_sizes, ieee_bits, use_simd, compilers,
        omp_schedule, omp_proc_bind, omp_places,
        cells_list, tests_list, common_armon_params,
        name, gnuplot_script, plot_file, log_scale, plot_title, verbose)
end


function parse_measure_script_file(file::IOStream)
    measures::Vector{MeasureParams} = []
    file_line_parser = enumerate(eachline(file))
    while !eof(file)
        measure = parse_measure_params(file_line_parser)
        push!(measures, measure)
    end
    return measures
end


function parse_arguments()
    if !(1 <= length(ARGS) <= 4)
        error("Invalid number of arguments. Usage: 'julia batch_measure.jl <script file>'")
    end

    measure_index = 0
    inti_index = 0

    script_file = open(ARGS[1], "r")
    measures = parse_measure_script_file(script_file)        
    close(script_file)

    if length(ARGS) > 1
        measure_index = parse(Int, ARGS[2])
        inti_index = parse(Int, ARGS[3])
    end

    return measures, measure_index, inti_index
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


function build_inti_combinaisons(measure::MeasureParams, inti_index::Int)
    options = [measure.processes, measure.distributions]
    return Iterators.drop(Iterators.product(options...), inti_index)
end


function parse_combinaisons(measure::MeasureParams, backend::Backend)
    if backend == Julia
        return Iterators.product(
            measure.threads,
            measure.block_sizes,
            measure.use_simd,
            measure.ieee_bits
        )
    elseif backend == CPP
        return Iterators.product(
            measure.threads,
            measure.omp_schedule,
            measure.omp_proc_bind,
            measure.omp_places,
            measure.use_simd,
            measure.ieee_bits,
            measure.compilers
        )
    elseif backend == Kokkos
        return Iterators.product(
            measure.threads,
            measure.omp_schedule,
            measure.omp_proc_bind,
            measure.omp_places,
            measure.block_sizes,
            measure.use_simd,
            measure.ieee_bits,
            measure.compilers
        )
    end
end


function recompile_cpp(block_size::Int, ieee_bits::Int, use_simd::Int, compiler::Compiler)
    make_options = [
        "--quiet",  # Don't print unuseful info
        "use_simd=$(use_simd)",
        "use_single_precision=$(ieee_bits == 32)",
    ]

    print("Recompiling the C++ code with: block_size=$(block_size), ieee_bits=$(ieee_bits), use_simd=$(use_simd), ")
    if compiler == GCC
        println("using GCC")
        # the makefile uses gcc by default
    elseif compiler == Clang
        println("using Clang")
        push!(options, "use_clang=1")
    elseif compiler == ICC
        println("using ICC")
        push!(options, "use_icc=1")
    else
        error("Wrong compiler")
    end

    run_cmd_print_on_error(Cmd(`make $(base_make_options) --quiet clean`, dir=cpp_make_dir))
    run_cmd_print_on_error(Cmd(`make $(base_make_options) $(make_options) $(cpp_make_target)`, dir=cpp_make_dir))

    return cpp_exe_path
end


function recompile_kokkos(device::Device, block_size::Int, ieee_bits::Int, use_simd::Int, compiler::Compiler)
    # error("TODO : block_size, compiler")
    # TODO : block_size
    # TODO : compiler

    make_options = [
        "--quiet",    # Don't print unuseful info
        "use_omp=1",  # Always enable OpenMP for Kokkos, since it can still help a bit for GPUs in the host side
        "use_simd=$(use_simd)",
        "use_single=$(convert(Int, ieee_bits == 32))"
    ]

    println("Recompiling the C++ Kokkos code with: block_size=$(block_size), ieee_bits=$(ieee_bits), use_simd=$(use_simd)")

    if device == CPU
        make_target = "build-omp"
        make_run_target = "run-omp"
    elseif device == CUDA
        make_target = "build-cuda"
        make_run_target = "run-cuda"
    elseif device == ROCM
        make_target = "build-hip"
        make_run_target = "run-hip"
    else
        error("Wrong device")
    end

    run_cmd_print_on_error(Cmd(`make $(base_make_options) $(make_options) $(make_target)`, dir=kokkos_make_dir))

    return make_run_target
end


function maybe_recompile(device::Device, block_size::Int, use_simd::Int, ieee_bits::Int, compiler::Compiler, backend::Backend, prev_params)
    # If 'prev_params' is empty, we are at the first iteration and we force a recompilation, to make sure that the version of the program is correct
    if length(prev_params) > 0
        # Check if the parameters changed enough so that a recompilation is needed
        prev_block_size, prev_use_simd, prev_ieee_bits, prev_compiler, prev_backend, prev_exe_path = prev_params
        need_compilation = prev_backend != backend ||
                           prev_compiler != compiler ||
                           prev_use_simd != use_simd ||
                           prev_ieee_bits != ieee_bits ||
                           prev_block_size != block_size

        if !need_compilation
            return prev_exe_path, prev_params
        end
    end

    if backend == CPP
        exe_path = recompile_cpp(block_size, ieee_bits, use_simd, compiler)
    elseif backend == Kokkos
        exe_path = recompile_kokkos(device, block_size, ieee_bits, use_simd, compiler)
    else
        error("Unknown backend")
    end

    return exe_path, (block_size, use_simd, ieee_bits, compiler, backend, exe_path)
end


function maybe_recompile(device::Device, _, use_simd::Int, ieee_bits::Int, compiler::Compiler, backend::Backend, prev_params)
    # Overload used by the C++ backend
    return maybe_recompile(device, 0, use_simd, ieee_bits, compiler, backend, prev_params)
end


function run_julia(measure::MeasureParams, threads::Int, block_size::Int, use_simd::Int, ieee_bits::Int, base_file_name::String)
    armon_options = []

    if measure.device == CUDA
        append!(armon_options, ["--gpu", "CUDA"])
    elseif measure.device == ROCM
        append!(armon_options, ["--gpu", "ROCM"])
    else
        # no option needed for CPU
    end

    append!(armon_options, armon_base_options)
    append!(armon_options, measure.common_armon_params)
    append!(armon_options, [
        "--block-size", block_size,
        "--use-simd", use_simd,
        "--ieee", ieee_bits,
        "--tests", join(measure.tests_list, ','),
        "--cells-list", join(string.(measure.cells_list), ','),
        "--data-file", base_file_name,
        "--gnuplot-script", measure.gnuplot_script,
        "--verbose", (measure.verbose ? 2 : 5)
    ])

    println("Running Julia with: $(threads) threads, $(block_size) block size, $(ieee_bits) bits, $(use_simd == 1 ? "with" : "without") SIMD")

    run(`julia -t $(threads) $(julia_options) $(julia_script_path) $(armon_options)`)
end


function get_cpp_run_cmd(exe_path, threads, omp_schedule, omp_proc_bind, omp_places, armon_options)
    ENV["OMP_NUM_THREADS"] = threads
    ENV["OMP_SCHEDULE"] = omp_schedule
    ENV["OMP_PROC_BIND"] = omp_proc_bind
    ENV["OMP_PLACES"] = omp_places
    cmd = `$(exe_path) $(armon_options)`
    return cmd
end


function get_kokkos_run_cmd(exe_path, threads, omp_schedule, omp_proc_bind, omp_places, armon_options)
    # For Kokkos, the exe_path is not a path but a make target that compiles (if needed) the exe then runs it with the arguments given.
    ENV["OMP_SCHEDULE"] = omp_schedule
    ENV["OMP_PROC_BIND"] = omp_proc_bind
    ENV["OMP_PLACES"] = omp_places
    args = "--kokkos-threads=$(threads) $(join(string.(armon_options), ' '))"
    cmd = Cmd(`make $(base_make_options) $(exe_path) args=\"$(args)\"`, dir=kokkos_make_dir)
    return cmd
end


function run_and_parse_output(cmd, verbose)
    if verbose
        println(cmd)
    end

    output = read(cmd, String)

    mega_cells_per_sec_raw = match(r"Cells/sec:\s*\K[0-9\.]+", output)

    if isnothing(mega_cells_per_sec_raw)
        println("Command failed, wrong output: ", cmd, "\nOutput:\n", output)
        error("Wrong output")
    elseif verbose
        println(output)
    end

    mega_cells_per_sec = parse(Float64, mega_cells_per_sec_raw.match)
    giga_cells_per_sec = mega_cells_per_sec / 1e3

    return giga_cells_per_sec
end


function run_armon(measure::MeasureParams, backend::Backend, threads::Int, omp_schedule::String, omp_proc_bind::String, omp_places::String, exe_path::String, base_file_name::String)
    println("Running $(backend == CPP ? "C++" : "Kokkos") with: $(threads) threads, binding: $(omp_proc_bind), places: $(omp_places)")

    for test in measure.tests_list
        data_file_name = base_file_name * test * ".csv"

        for cells in measure.cells_list
            armon_options = [
                "-t", test,
                "--cells", cells
            ]
            append!(armon_options, armon_base_options)
            append!(armon_options, measure.common_armon_params)
    
            # Do the measurement
            if backend == CPP
                cmd = get_cpp_run_cmd(exe_path, threads, omp_schedule, omp_proc_bind, omp_places, armon_options)
            elseif backend == Kokkos
                cmd = get_kokkos_run_cmd(exe_path, threads, omp_schedule, omp_proc_bind, omp_places, armon_options)
            else
                error("Wrong backend")
            end

            @printf(" - %s, %-10g cells: ", test, cells)

            cells_throughput = run_and_parse_output(cmd, measure.verbose)

            @printf("%.2f Giga cells/sec\n", cells_throughput)
    
            # Append the result to the output file
            open(data_file_name, "a") do file
                println(file, cells, ", ", cells_throughput)
            end

            # Update the plot
            # We redirect the output of gnuplot to null so that there is no warning messages displayed
            run(pipeline(`gnuplot $(measure.gnuplot_script)`, stdout=devnull, stderr=devnull))
        end
    end
end


function build_data_file_base_name(measure::MeasureParams, 
        processes::Int, distribution::String, 
        threads::Int, omp_schedule::String, omp_proc_bind::String, omp_places::String, 
        block_size::Int, use_simd::Int, ieee_bits::Int, 
        compiler::Compiler, backend::Backend)
    # Build a file name based on the measurement name and the parameters that don't have a single value
    name = data_dir * measure.name * "/"

    # Build a plot legend entry for the measurement
    legend = ""

    if measure.device == CPU
        name *= "CPU"
        legend *= "CPU"
    elseif measure.device == CUDA
        name *= "CUDA"
        legend *= "CUDA"
    elseif measure.device == ROCM
        name *= "ROCM"
        legend *= "ROCM"
    end

    name *= "_" * measure.node
    legend *= isempty(measure.node) ? "" : ", " * measure.node

    if length(measure.distributions) > 1
        name *= "_" * distribution
        legend *= ", " * distribution
    end

    if length(measure.processes) > 1
        name *= "_$(processes)proc"
        legend *= ", $(processes) processes"
    end

    if length(measure.backends) > 1
        if backend == CPP
            name *= "_CPP"
            legend *= ", CPP"
        elseif backend == Kokkos
            name *= "_Kokkos"
            legend *= ", Kokkos"
        elseif backend == Julia
            name *= "_Julia"
            legend *= ", Julia"
        end
    end

    if length(measure.compilers) > 1
        if compiler == GCC
            name *= "_GCC"
            legend *= " GCC"
        elseif compiler == Clang
            name *= "_Clang"
            legend *= " Clang"
        elseif compiler == ICC
            name *= "_ICC"
            legend *= " ICC"
        end
    end

    if length(measure.ieee_bits) > 1
        name *= "_$(ieee_bits)bits"
        legend *= ", $(ieee_bits)-bit"
    end

    if length(measure.use_simd) > 1
        name *= use_simd ? "_SIMD" : "_NO_SIMD"
        legend *= use_simd ? ", SIMD" : ""
    end

    if length(measure.block_sizes) > 1
        name *= "_$(block_size)bs"
        legend *= ", $(block_size) Block size"
    end

    if length(measure.threads) > 1
        name *= "_$(threads)td"
        legend *= ", $(threads) Threads"
    end

    if backend == CPP || backend == Kokkos
        if length(measure.omp_schedule) > 1
            name *= "_$(omp_schedule)"
            legend *= ", $(omp_schedule)"
        end
    
        if length(measure.omp_proc_bind) > 1
            name *= "_$(omp_proc_bind)"
            legend *= ", $(omp_proc_bind)"
        end
    
        if length(measure.omp_places) > 1
            name *= "_$(omp_places)"
            legend *= ", places: $(omp_places)"
        end
    end

    return name * "_", legend
end


function build_data_file_base_name(measure::MeasureParams, processes, distribution, threads, omp_schedule, omp_proc_bind, omp_places, use_simd, ieee_bits, compiler, backend)
    # Overload used for the C++'s backend parameters
    return build_data_file_base_name(measure, processes, distribution, threads, omp_schedule, omp_proc_bind, omp_places, 0, use_simd, ieee_bits, compiler, backend)
end


function build_data_file_base_name(measure::MeasureParams, processes, distribution, threads, block_size, use_simd, ieee_bits, backend)
    # Overload used for the Julia's backend parameters
    return build_data_file_base_name(measure, processes, distribution, threads, "", "", "", block_size, use_simd, ieee_bits, GCC, backend)
end


function create_all_data_files_and_plot(measure::MeasureParams)
    plot_commands = []
    for (processes, distribution) in build_inti_combinaisons(measure, 0)
        for backend in measure.backends
            # Marker style for the plot
            if backend == CPP
                point_type = 7
            elseif backend == Julia
                point_type = 5
            elseif backend == Kokkos
                point_type = 9
            else
                point_type = 1
            end

            for parameters in parse_combinaisons(measure, backend)
                base_file_name, legend_base = build_data_file_base_name(measure, processes, distribution, parameters..., backend)
                for test in measure.tests_list
                    data_file_name = base_file_name * test * ".csv"
                    open(data_file_name, "w") do _ end  # Create/Clear the file
                    legend = "$(test), $(legend_base)"
                    plot_cmd = gnuplot_plot_command(data_file_name, legend, point_type)
                    push!(plot_commands, plot_cmd)
                end
            end
        end
    end

    # Create the gnuplot script. It will then be run at each new data point
    open(measure.gnuplot_script, "w") do gnuplot_script
        print(gnuplot_script, base_gnuplot_script_commands(measure.plot_file, measure.plot_title, measure.log_scale))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end
end


function run_measure(measure::MeasureParams, inti_index::Int)
    # We are on a compute node, and there is no way to change the ccc_mprun parameters here
    # Therefore 'processes' and 'distribution' are fixed here. They are changed by an outer loop.
    ((processes, distribution), _) = iterate(build_inti_combinaisons(measure, inti_index))

    prev_params = ()

    for backend in measure.backends
        for parameters in parse_combinaisons(measure, backend)
            base_file_name, _ = build_data_file_base_name(measure, processes, distribution, parameters..., backend)
            if backend == Julia
                run_julia(measure, parameters..., base_file_name)
            else
                exe_path, prev_params = maybe_recompile(measure.device, parameters[end-3:end]..., backend, prev_params)
                run_armon(measure, backend, parameters[1:4]..., exe_path, base_file_name)
            end
        end
    end
end


function setup_env()
    # Environment variables needed to configure ROCM so that hipcc works correctly
    ENV["PATH"] *= ":/opt/rocm-4.5.0/hip/lib:/opt/rocm-4.5.0/hsa/lib"
    ENV["ROCM_PATH"] = "/opt/rocm-4.5.0"
    ENV["HIP_CLANG"] = "/opt/rocm-4.5.0/llvm/bin"
    ENV["HSA_PATH"] = "/opt/rocm-4.5.0/hsa"

    # Make sure that the output folders exist
    mkpath(data_dir)
    mkpath(plot_scripts_dir)
    mkpath(plots_dir)
end


function main()
    measures, measure_index, inti_index = parse_arguments()
    
    if measure_index != 0
        # Sub loop, running inside of a compute node
        run_measure(measures[measure_index], inti_index)
        return
    end

    # Main loop, running in the login node, parsing through all measurments to do
    setup_env()
    for (i, measure) in enumerate(measures)
        println(" ==== Measurement $(i)/$(length(measures)): $(measure.name) ==== ")

        # Create the files and plot script once at the beginning
        create_all_data_files_and_plot(measure)
        inti_index = 0

        # For each 'number of processes' and 'threads distribution' combinaison, create a new job
        for (processes, distribution) in build_inti_combinaisons(measure, inti_index)
            processes == 1 || error("Running multiple processes at once is not yet implemented")  # TODO

            if isempty(measure.node)
                println("Running outside of INTI: cannot control processes count and distribution type")
                cmd = no_inti_cmd(i, inti_index)
            else
                # Launch a new INTI job on a compute node
                inti_options = [
                    "-p", measure.node,
                    "-n", processes,                   # Number of processes
                    "-E", "-m block:$(distribution)",  # Threads distribution
                    # Allocate for the maximum number of threads needed
                    # To make sure that there is enough memory available, there is a minimum number of core allocated.
                    "-c", max(maximum(measure.threads), min_inti_cores)
                ]
                cmd = inti_cmd(inti_options, i, inti_index)
                println("Starting INTI job: ", cmd)
            end

            run(cmd)
            inti_index += 1
        end
    end
end


main()
