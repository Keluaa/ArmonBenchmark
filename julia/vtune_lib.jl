module VTune

const IntelITT_path = get(ENV, "IntelITT_PATH", "")
const use_ITT = !isempty(IntelITT_path)
const use_MPI_profiling = parse(Bool, get(ENV, "ARMON_VTUNE_MPI", "false"))
const use_VTune = parse(Bool, get(ENV, "ARMON_VTUNE_PROFILE", "false"))

if use_VTune
    if !use_MPI_profiling && !use_ITT
        error("Path to the IntelITT dir not defined in 'IntelITT_PATH' env var, this is required for VTune profiling. Clone it from https://github.com/mchristianl/IntelITT.jl")
    elseif use_MPI_profiling && use_ITT
        error("Exactly one of 'IntelITT_PATH' or 'ARMON_VTUNE_MPI' should be set.")
    end

    if use_ITT
        include(joinpath(IntelITT_path, "src/IntelITT.jl"))
        using .IntelITT
    else
        using MPI
    end
end


export @resume_profiling, @pause_profiling, @perf_task


function _enter_task(domain_ptr, task_str_ptr)
    @static if use_VTune && use_ITT
        IntelITT.__itt_task_begin(domain_ptr, IntelITT.__itt_null, IntelITT.__itt_null, task_str_ptr)
    end
end


function _exit_task(domain_ptr)
    @static if use_VTune && use_ITT
        IntelITT.__itt_task_end(domain_ptr)
    end
end


function _resume_profiling()
    @static if !use_VTune
        # do nothing
    elseif use_MPI_profiling
        ccall((:MPI_Pcontrol, MPI.libmpi), Cint, (Cint,), 1)
    else
        IntelITT.__itt_resume()
    end
end


function _pause_profiling()
    @static if !use_VTune
        # do nothing
    elseif use_MPI_profiling
        ccall((:MPI_Pcontrol, MPI.libmpi), Cint, (Cint,), 0)
    else
        IntelITT.__itt_pause()
    end
end


macro perf_task(domain, task_name, expr)
    if !use_VTune || !use_ITT
        return esc(quote $expr end)
    else
        #domain_hndl = Ptr{IntelITT.___itt_domain}(IntelITT.__itt_domain_create(string(domain)))
        #task_str_hndl = Ptr{IntelITT.___itt_domain}(IntelITT.__itt_string_handle_create(string(task_name)))
        
        return esc(quote
            _domain_hndl   = Ptr{VTune.IntelITT.___itt_domain}(VTune.IntelITT.__itt_domain_create($(string(domain))))
            _task_str_hndl = Ptr{VTune.IntelITT.___itt_domain}(VTune.IntelITT.__itt_string_handle_create($(string(task_name))))
            VTune._enter_task(_domain_hndl, _task_str_hndl)
            $(expr)
            VTune._exit_task(_domain_hndl)
        end)
    end
end


macro resume_profiling()
    if !use_VTune return end

    return esc(quote 
        VTune._resume_profiling()
    end)
end


macro pause_profiling()
    if !use_VTune return end

    return esc(quote
        VTune._pause_profiling()
    end)
end


end
