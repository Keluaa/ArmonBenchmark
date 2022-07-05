
using ThreadPinning


function omp_bind_threads(places::Symbol, bind::Symbol)
    all_cores_requested = length(sysinfo().cpuids) == Threads.nthreads()

    if bind == Symbol("false")
        return
    end

    if Threads.nthreads() > 64
        # See https://github.com/JuliaSIMD/Polyester.jl/issues/83
        @warn "Polyester.jl doesn't support more than 64 threads." maxlog=1
    end

    if all_cores_requested
        pinthreads(:compact)
    elseif places == :sockets
        pinthreads(:sockets)
    elseif bind == :close
        if places == :threads
            pinthreads(:compact, hyperthreads=true)
        elseif places == :cores
            pinthreads(:compact)
        elseif places == :numa_domains
            pinthreads(:compact)
        else
            error("Unknown threads place: " * string(places))
        end
    elseif bind == :spread
        if places == :threads
            pinthreads(:spread)
        elseif places == :cores
            pinthreads(:spread)
        elseif places == :numa_domains
            pinthreads(:numa)
        else
            error("Unknown threads place: " * string(places))
        end
    elseif bind == :primary || bind == :master
        if places == :threads
            pinthreads(:firstn, hyperthreads=true)
        elseif places == :cores
            pinthreads(:firstn)
        elseif places == :numa_domains
            pinthreads(:firstn)
        else
            error("Unknown threads place: " * string(places))
        end
    else
        error("Unknown binding policy: '" * string(bind) * "'")
    end
end


function binding_iterator_close(info::ThreadPinning.SysInfo, places::Vector{Vector{Int}}; hyperthreads=false)
    cpuids = Iterators.flatten(places)
    if hyperthreads || !info.hyperthreading
        return cpuids
    else
        return Iterators.filter(cpuid -> !info.ishyperthread[cpuid+1], cpuids)
    end
end


function binding_iterator_spread(info::ThreadPinning.SysInfo, places::Vector{Vector{Int}}; hyperthreads=false)
    if !all(length.(places) .== length(places[1]))
        error("The given places must all have the same length")
    end

    place_count = length(places)
    total = length(places[1]) * place_count

    spread_idx = Base.Fix2(divrem, place_count)
    i_to_cpuid = (i) -> places[i[2]+1][i[1]+1]
    idx_to_cpuid = i_to_cpuid ∘ spread_idx

    if hyperthreads || !info.hyperthreading
        return (cpuid for cpuid in Iterators.map(idx_to_cpuid, 0:total-1))
    else
        return (cpuid for cpuid in Iterators.map(idx_to_cpuid, 0:total-1) if !info.ishyperthread[cpuid+1])
    end
end


function omp_bind_threads(offset::Int, places::Symbol, bind::Symbol; hyperthreading=false)
    if bind == Symbol("false")
        return
    end

    if Threads.nthreads() > 64
        # See https://github.com/JuliaSIMD/Polyester.jl/issues/83
        @warn "Polyester.jl doesn't support more than 64 threads." maxlog=1
    end

    info = sysinfo()
    
    if places == :threads
        hyperthreading = true
        cpuid_places = [info.cpuids]
    elseif places == :cores
        cpuid_places = [info.cpuids]
    elseif places == :sockets
        cpuid_places = info.cpuids_sockets
    elseif places == :numa_domains || places == :numa
        cpuid_places = info.cpuids_numa
    else
        error("Unknown threads place: $places")
    end

    if bind == :compact || bind == :close
        cpuid_it = binding_iterator_close(info, cpuid_places; hyperthreads=hyperthreading)
    elseif bind == :spread
        cpuid_it = binding_iterator_spread(info, cpuid_places; hyperthreads=hyperthreading)
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
