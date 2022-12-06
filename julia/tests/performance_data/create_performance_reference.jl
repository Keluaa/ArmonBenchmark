
include("../../armon_2D_MPI_async.jl")
using .Armon
using Dates


include("performance_functions.jl")


function create_solver_performance_data(device_type, perf_data, memory_available)
    for (size_label, size) in ((:small, SMALL_SIZE), (:big, BIG_SIZE))
        solver_data = Dict()

        for test in (:Sod_circ, :Bizarrium)
            params = get_performance_params(test, Float64, device_type, size)
            not_enough_mem = has_enough_memory_for(params, memory_available)

            if !not_enough_mem
                println("Measuring the solver's performance for $(size)x$(size) cells with the $test \
                         test case...")
                cells_per_sec = measure_solver_performance(params)
                solver_data[test] = cells_per_sec
                println("Result: ", round(cells_per_sec * 1e3, digits=5), " Mega cells/sec")
            end
        end
        
        if !isempty(solver_data)
            perf_data[size_label][:solver] = solver_data
        end
    end
end


function create_kernels_performance_data(device_type, perf_data, memory_available)
    for (size_label, size) in ((:small, SMALL_SIZE), (:big, BIG_SIZE))
        params = get_performance_params(:Sod, Float64, device_type, size)
        skip, data = setup_kernel_tests(params, memory_available)
        skip && continue

        kernels_data = Dict()

        println("Measuring each kernel's performance for $(size)x$(size) cells with the Sod \
                 test case...")

        for kernel_info in KERNELS
            (kernel_name, single_row, kernel_lambda) = kernel_info
            kernel_time = measure_kernel_performance(params, data, kernel_lambda, single_row)
            kernels_data[kernel_name] = kernel_time
            println(" - $kernel_name: ", round(kernel_time * 1e6, digits=1), " µs")
        end

        perf_data[size_label][:kernels] = kernels_data
    end
end


function init_perf_dict(device_type)
    device_data = get_device_info(device_type)
    device_hash = hash(device_data)

    if device_type == :GPU
        device_data[:host] = Sys.cpu_info()[1].model
    end

    device_data[:performance] = perf_data = Dict()
    perf_data[:small] = Dict()
    perf_data[:big] = Dict()

    # Add the current commit hash and the date to the measurement
    device_data[:commit] = readchomp(`git rev-parse HEAD`)
    device_data[:date] = Dates.format(Dates.now(), DateFormat("dd-mm-yyyy"))

    return device_hash, device_data
end


function warmup_GPU(memory_available)
    println("Warming up the GPU...")
    size = SMALL_SIZE
    for test in (:Sod_circ, :Bizarrium)
        params = get_performance_params(test, Float64, :GPU, size)
        not_enough_mem = has_enough_memory_for(params, memory_available)

        if !not_enough_mem
            println("Warmup with $(size)x$(size) cells with the $test test case...")
            cells_per_sec = measure_solver_performance(params)
            println("Result: ", round(cells_per_sec * 1e3, digits=5), " Mega cells/sec")
        end
    end
end


function create_performance_data(device_types)
    setup_cpu_threads()
    device_info = get_device_info(:CPU)
    check_julia_options(device_info)

    for device_type in device_types
        data = read_performance_data(device_type)

        device_hash, device_data = init_perf_dict(device_type)
        memory_available = device_data[:memory] * 1e9
        perf_data = device_data[:performance]

        device_type == :GPU && warmup_GPU(memory_available)

        println("Creating performance refenrence data for '$(device_data[:name])'")

        create_solver_performance_data(device_type, perf_data, memory_available)
        create_kernels_performance_data(device_type, perf_data, memory_available)

        data[repr(device_hash)] = device_data
        write_performance_data(device_type, data)
    end
end


if isinteractive()
    # Only run automatically through the terminal
elseif isempty(ARGS)
    create_performance_data((:CPU, :GPU))
else
    device_types = ARGS .|> uppercase .|> Symbol |> unique
    create_performance_data(device_types)
end
