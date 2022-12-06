
using Printf

include("performance_data/performance_functions.jl")


function get_device_perf_info(device_type)
    device_info = get_device_info(device_type)
    device_perf_data = get_perf_data_for_device(device_type, device_info)
    unknown_device = isnothing(device_perf_data)
    if unknown_device
        device_info_list = map(p -> join(p, ": "), collect(pairs(device_info)))
        device_info_str = join(device_info_list, "\n")
        device_info_hash = repr(hash(device_info))
        @warn "There is no reference results for this $device_type.\n\
               Device info:\n$device_info_str\n\
               hash: $device_info_hash" maxlog=1
    end
    memory_available = device_info[:memory] * 1e9
    return device_perf_data, memory_available
end


function get_solver_ref_perf(device_perf_data, test, size)
    isnothing(device_perf_data) && return true, nothing
    size_key = size == SMALL_SIZE ? "small" : "big"
    return false, device_perf_data[size_key]["solver"][String(test)]
end


function get_kernels_ref_perf(device_perf_data, size)
    isnothing(device_perf_data) && return true, nothing
    size_key = size == SMALL_SIZE ? "small" : "big"
    return false, device_perf_data[size_key]["kernels"]
end


function encouragements(name, context, new_perf, ref_perf, is_time)
    # Display the message only for improvements bigger than 5%
    is_big_improvement = if is_time
        # Time should decrease (+ 2µs of absolute (in)tolerance)
        new_perf < ref_perf && !isapprox(new_perf, ref_perf; atol=3e-6, rtol=0.05)
    else
        # Performance must increase
        new_perf > ref_perf && !isapprox(new_perf, ref_perf; rtol=0.05)
    end

    if is_big_improvement
        gain = abs(new_perf - ref_perf) / ref_perf
        gain = round(gain * 100; digits=1)

        unit = is_time ? "sec" : "Gc/sec"

        new_str = @sprintf("%.3g %s", new_perf, unit)
        ref_str = @sprintf("%.3g %s", ref_perf, unit)

        @info "The performance of the $name improved by $gain% for $context ($ref_str → $new_str)!\n\
               Remember to update the refenrence data."
    end

    return true
end


function test_solver_performance(params, ref_cells_per_sec, memory_available, missing_ref_data, context_str)
    not_enough_mem = has_enough_memory_for(params, memory_available)    
    skip_test = not_enough_mem || missing_ref_data
    
    @test begin
        cells_per_sec = measure_solver_performance(params)
        ok = cells_per_sec > ref_cells_per_sec || isapprox(cells_per_sec, ref_cells_per_sec; rtol=0.05)
        ok && encouragements("solver", context_str, cells_per_sec, ref_cells_per_sec, false)
    end skip=skip_test
end


function test_kernels_performance(params, data, kernels_perf_data, skip_test, context_str)
    @testset "$(kernel_info[1])" for kernel_info in KERNELS
        (kernel_name, single_row, kernel_lambda) = kernel_info

        if !isnothing(kernels_perf_data)
            ref_time = get(kernels_perf_data, kernel_name, nothing)
            unknown_kernel = isnothing(ref_time)
            unknown_kernel && @warn "Missing '$kernel_name' kernel entry in the performance data for this device"
        else
            ref_time = nothing
            unknown_kernel = true
        end

        @test begin
            kernel_time = measure_kernel_performance(params, data, kernel_lambda, single_row)
            ok = kernel_time < ref_time || isapprox(ref_time, kernel_time; atol=2e-6, rtol=0.05)
            ok && encouragements("'$kernel_name' kernel", context_str, kernel_time, ref_time, true)
        end skip=(skip_test || unknown_kernel)
    end
end


@testset "Performance" begin

    check_julia_options(get_device_info(:CPU))
    setup_cpu_threads()

    # Entire solver
    @testset "Solver" begin
        @testset "$device_type" for device_type in (:CPU, :GPU)
            device_perf_data, device_memory = get_device_perf_info(device_type)
            @testset "$test $(@sprintf("%.g", size^2))" for test in (:Sod_circ, :Bizarrium), 
                                                            size in (SMALL_SIZE, BIG_SIZE)
                context_str = "$device_type with $test and $(@sprintf("%.g", size^2)) cells"
                params = get_performance_params(test, Float64, device_type, size)
                missing_ref_data, ref_cells_per_sec = get_solver_ref_perf(device_perf_data, test, size)
                test_solver_performance(params, ref_cells_per_sec, device_memory, missing_ref_data, context_str)
            end
        end
    end

    # Kernel by kernel
    @testset "Kernels" begin
        @testset "$device_type $(@sprintf("%.g", size^2))" for device_type in (:CPU, :GPU), 
                                                               size in (SMALL_SIZE, BIG_SIZE)
            context_str = "$device_type with $(@sprintf("%.g", size^2)) cells"
            device_perf_data, device_memory = get_device_perf_info(device_type)

            params = get_performance_params(:Sod, Float64, device_type, size)
            missing_ref_data, kernels_perf_data = get_kernels_ref_perf(device_perf_data, size)
            skip_test = missing_ref_data

            if skip_test
                data = nothing
            else
                skip_test, data = setup_kernel_tests(params, device_memory)
            end
            
            test_kernels_performance(params, data, kernels_perf_data, skip_test, context_str)
        end
    end

    println("done with the performance")
end
