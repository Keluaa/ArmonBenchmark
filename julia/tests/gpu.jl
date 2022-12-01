
using CUDA
using AMDGPU
using CUDAKernels
using ROCKernels

import .Armon: data_to_gpu, data_from_gpu

include("reference_data/reference_functions.jl")


function run_armon_gpu_reference(ref_params::ArmonParameters{T}, device_array) where T
    data = ArmonData(T, ref_params.nbcell, ref_params.comm_array_size)
    init_test(ref_params, data)
    gpu_data = data_to_gpu(data, device_array)
    dt, cycles, _, _ = time_loop(ref_params, gpu_data, data)
    data_from_gpu(data, gpu_data)
    return dt, cycles, data
end


@testset "GPU" begin
    no_cuda = !CUDA.has_cuda_gpu()
    no_rocm = !AMDGPU.has_rocm_gpu()
    
    @testset "CUDA" begin
        @testset "$test with $type" for type in (Float32, Float64), test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium)
            @test begin
                ref_params = get_reference_params(test, type)
                ref_params.use_gpu = true
                ref_params.device = CUDADevice()
                dt, cycles, data = run_armon_gpu_reference(ref_params, CuArray)
                ref_data = ArmonData(type, ref_params.nbcell, ref_params.comm_array_size)
                compare_with_reference_data(ref_params, dt, cycles, data, ref_data)
            end skip=no_cuda
        end
    end

    @testset "ROCm" begin
        @testset "$test with $type" for type in (Float32, Float64), test in (:Sod, :Sod_y, :Sod_circ, :Bizarrium)
            @test begin
                ref_params = get_reference_params(test, type)
                ref_params.use_gpu = true
                ref_params.device = ROCDevice()
                dt, cycles, data = run_armon_gpu_reference(ref_params, ROCArray)
                ref_data = ArmonData(type, ref_params.nbcell, ref_params.comm_array_size)
                compare_with_reference_data(ref_params, dt, cycles, data, ref_data)
            end skip=no_rocm
        end
    end
end
