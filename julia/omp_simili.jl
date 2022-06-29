
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
