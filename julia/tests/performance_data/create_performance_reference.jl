
include("../../armon_2D_MPI_async.jl")
using .Armon


include("performance_functions.jl")


function create_solver_performance_data(device_type, perf_data, memory_available)
    for (size_label, size) in ((:small, SMALL_SIZE), (:big, BIG_SIZE))
        solver_data = Dict()

        for test in (:Sod_circ, :Bizarrium)
            params = get_performance_params(test, Float64, device_type, size)
            not_enough_mem = has_enough_memory_for(params, memory_available)

            if !not_enough_mem
                cells_per_sec = measure_solver_performance(params)
                solver_data[test] = cells_per_sec
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

        for kernel_info in KERNELS
            (kernel_name, single_row, kernel_lambda) = kernel_info
            kernel_time = measure_kernel_performance(params, data, kernel_lambda, single_row)
            kernels_data[kernel_name] = kernel_time
        end

        perf_data[size_label][:kernels] = kernels_data
    end
end


function init_perf_dict(device_type)
    device_data = get_device_info(device_type)
    device_hash = hash(device_data)

    device_data[:performance] = perf_data = Dict()
    perf_data[:small] = Dict()
    perf_data[:big] = Dict()

    return device_hash, device_data
end


function create_performance_data()
    for device_type in (:CPU, :GPU)
        data = read_performance_data(device_type)

        device_hash, device_data = init_perf_dict(device_type)
        memory_available = device_data[:memory] * 1e9
        perf_data = device_data[:performance]

        create_solver_performance_data(device_type, perf_data, memory_available)
        create_kernels_performance_data(device_type, perf_data, memory_available)

        data[repr(device_hash)] = device_data
        write_performance_data(device_type, data)
    end
end


check_julia_options()
create_performance_data()
