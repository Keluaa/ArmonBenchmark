
using CUDA
using AMDGPU
using Hwloc


function get_available_gpu()
    if CUDA.has_cuda_gpu()
        return :CUDA
    elseif AMDGPU.has_rocm_gpu()
        return :ROCM
    else
        error("No compatible GPU detected.")
    end
end


function get_gpu_info()
    gpu_type = get_available_gpu()

    if gpu_type == :CUDA
        device = CUDA.device()
        type = "CUDA"
        name = CUDA.name(device)
        memory = CUDA.totalmem(device) |> Int
    elseif gpu_type == :ROCM
        agent = AMDGPU.get_default_agent()
        type = "ROCm"
        name = AMDGPU.get_name(agent)
        memory = maximum(AMDGPU.pool_size.(mem_pools(agent))) |> Int
    end

    memory = round(memory / 1e9; digits=1)

    return Dict(
        :type   => type,
        :name   => name,
        :memory => memory  # GB
    )
end


function get_cpu_info()
    info = Sys.cpu_info()[1]

    name = info.model
    freq = info.speed
    memory = Sys.total_memory() / 1e9
    cores = num_physical_cores()
    numa_nodes = num_numa_nodes()
    sockets = num_packages()
    cache_sizes = string(cachesize())[2:end-1]

    memory = round(memory; digits=1)

    return Dict(
        :name        => name,
        :freq        => freq,    # MHz
        :memory      => memory,  # GB
        :cores       => cores,
        :numa_nodes  => numa_nodes,
        :sockets     => sockets,
        :cache_sizes => cache_sizes,
    )
end


function get_device_info(device_type::Symbol)
    if device_type == :CPU
        return get_cpu_info()
    elseif device_type == :GPU
        return get_gpu_info()
    else
        error("Unknown device: $device_type")
    end
end
