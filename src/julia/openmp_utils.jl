
using ThreadPinning


function binding_iterator_close(info::ThreadPinning.SysInfo, places::Vector{Vector{Int}};
    hyperthreads=false, skip_cpus=[]
)
    cpuids = Iterators.flatten(places)
    if hyperthreads || !info.hyperthreading
        return Iterators.filter(cpuid -> !(cpuid in skip_cpus), cpuids)
    else
        return Iterators.filter(cpuid -> !info.ishyperthread[cpuid+1] && !(cpuid in skip_cpus), cpuids)
    end
end


function binding_iterator_spread(info::ThreadPinning.SysInfo, places::Vector{Vector{Int}};
    hyperthreads=false, skip_cpus=[]
)
    if !all(length.(places) .== length(places[1]))
        error("The given places must all have the same length")
    end

    place_count = length(places)
    total = length(places[1]) * place_count

    spread_idx = Base.Fix2(divrem, place_count)
    i_to_cpuid = (i) -> places[i[2]+1][i[1]+1]
    idx_to_cpuid = i_to_cpuid ∘ spread_idx

    if hyperthreads || !info.hyperthreading
        return (cpuid for cpuid in Iterators.map(idx_to_cpuid, 0:total-1) if !(cpuid in skip_cpus))
    else
        return (cpuid for cpuid in Iterators.map(idx_to_cpuid, 0:total-1) if !info.ishyperthread[cpuid+1] && !(cpuid in skip_cpus))
    end
end


function omp_bind_threads(offset::Int, places::Symbol, bind::Symbol;
    hyperthreading=false, skip_cpus=[]
)
    if bind == Symbol("false")
        return
    end

    info = ThreadPinning.sysinfo()

    if places === :threads
        hyperthreading = true
        cpuid_places = [info.cpuids]
    elseif places === :cores
        cpuid_places = [info.cpuids]
    elseif places === :sockets
        cpuid_places = info.cpuids_sockets
    elseif places === :numa_domains || places === :numa
        cpuid_places = info.cpuids_numa
    else
        error("Unknown threads place: $places")
    end

    if bind === :compact || bind === :close
        cpuid_it = binding_iterator_close(info, cpuid_places; hyperthreads=hyperthreading, skip_cpus)
    elseif bind === :spread
        cpuid_it = binding_iterator_spread(info, cpuid_places; hyperthreads=hyperthreading, skip_cpus)
    else
        error("Unknown thread binding: $bind")
    end

    cpuid_it = Iterators.drop(cpuid_it, offset)
    cpuid_it = Iterators.take(cpuid_it, Threads.nthreads())

    pinned_count = 0
    for (tid, cpuid) in enumerate(cpuid_it)
        pinthread(tid, cpuid)
        pinned_count += 1
    end

    if pinned_count < Threads.nthreads()
        error("Could only pin $pinned_count out of $(Threads.nthreads()) threads")
    end
end


function build_omp_vars(offset::Int, places::Symbol, bind::Symbol;
    hyperthreading=false, skip_cpus=[]
)
    info = ThreadPinning.sysinfo()

    OMP_NUM_THREADS = string(Threads.nthreads())

    if places === :threads
        OMP_PLACES = "threads"
        hyperthreading = true
        cpuid_places = [info.cpuids]
    elseif places === :cores
        OMP_PLACES = "cores"
        cpuid_places = [info.cpuids]
    elseif places === :sockets
        OMP_PLACES = "sockets"
        cpuid_places = info.cpuids_sockets
    elseif places === :numa_domains || places === :numa
        OMP_PLACES = "numa"  # not always supported by the OpenMP implementation
        cpuid_places = info.cpuids_numa
    else
        error("Unknown threads place: $places")
    end

    if bind === :compact || bind === :close
        OMP_PROC_BIND = "close"
        cpuid_it = binding_iterator_close(info, cpuid_places; hyperthreads=hyperthreading, skip_cpus)
    elseif bind === :spread
        OMP_PROC_BIND = "spread"
        cpuid_it = binding_iterator_spread(info, cpuid_places; hyperthreads=hyperthreading, skip_cpus)
    elseif bind === Symbol("false")
        OMP_PROC_BIND = "false"
        return OMP_NUM_THREADS, OMP_PLACES, OMP_PROC_BIND
    end

    cpuid_it = Iterators.drop(cpuid_it, offset)
    cpuid_it = Iterators.take(cpuid_it, Threads.nthreads())

    OMP_PROC_BIND = "close"  # There is only a single place, the other settings are irrelevant 
    OMP_PLACES = []
    pinned_count = 0
    for cpuid in cpuid_it
        push!(OMP_PLACES, cpuid)
        pinned_count += 1
    end
    OMP_PLACES = join(map(p -> "{$p}", OMP_PLACES), ',')

    if pinned_count < Threads.nthreads()
        error("Could only pin $pinned_count out of $(Threads.nthreads()) threads")
    end

    return OMP_NUM_THREADS, OMP_PLACES, OMP_PROC_BIND
end
