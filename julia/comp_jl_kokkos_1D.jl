
using Printf
using PrettyTables
using Libdl


const enable_NaN_tracking = false
if enable_NaN_tracking
    include("NaN_detector.jl")
else
    const NaNflag = false
    function reset_nan_flag() end
end


include("armon_1D.jl")


kokkos_comp_lib_path = "../kokkos/cmake-build-complib-serial/src/libcmp_cpp_jl_1D"
flt_t = Float64


lib_ptr = nothing
sym_get_data = nothing
sym_step_init_loop = nothing
sym_step_dtCFL = nothing
sym_step_boundary_conditions = nothing
sym_step_numerical_fluxes = nothing
sym_step_cell_update = nothing
sym_step_euler_projection = nothing
sym_step_update_EOS = nothing
sym_step_end_loop = nothing
sym_init_cmp = nothing
sym_init_test = nothing
sym_end_cmp = nothing


function load_lib()
    global sym_get_data = Libdl.dlsym(lib_ptr, :get_data)
    global sym_step_init_loop = Libdl.dlsym(lib_ptr, :step_init_loop)
    global sym_step_dtCFL = Libdl.dlsym(lib_ptr, :step_dtCFL)
    global sym_step_boundary_conditions = Libdl.dlsym(lib_ptr, :step_boundary_conditions)
    global sym_step_numerical_fluxes = Libdl.dlsym(lib_ptr, :step_numerical_fluxes)
    global sym_step_cell_update = Libdl.dlsym(lib_ptr, :step_cell_update)
    global sym_step_euler_projection = Libdl.dlsym(lib_ptr, :step_euler_projection)
    global sym_step_update_EOS = Libdl.dlsym(lib_ptr, :step_update_EOS)
    global sym_step_end_loop = Libdl.dlsym(lib_ptr, :step_end_loop)
    global sym_init_cmp = Libdl.dlsym(lib_ptr, :init_cmp)
    global sym_init_test = Libdl.dlsym(lib_ptr, :init_test)
    global sym_end_cmp = Libdl.dlsym(lib_ptr, :end_cmp)
end


function wrap_data(size::Int)::Armon.ArmonData{Vector{flt_t}}
    d_ptr = ccall(sym_get_data, Ptr{Ptr{flt_t}}, ())
    return Armon.ArmonData(
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  1), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  2), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  3), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  4), size),
        Vector{flt_t}(undef, 0),  # vmat is not present on the Kokkos backend
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  5), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  6), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  7), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  8), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr,  9), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 10), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 11), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 12), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 13), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 14), size),
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 15), size),
        Vector{flt_t}(undef, 0),  # vmat is not present on the Kokkos backend
        unsafe_wrap(Vector{flt_t}, unsafe_load(d_ptr, 16), size),
    )
end


function Base.copy(d::Armon.ArmonData{V}) where {T, V <: AbstractVector{T}}
    return Armon.ArmonData([copy(getfield(d, k)) for k in fieldnames(Armon.ArmonData{V})]...)
end


function disp(params, label, array)
    (; nghost, nbcell) = params
    println(label)
    pretty_table(array; 
        noheader = true, 
        highlighters = (Highlighter((d,i) -> (nghost ≤ i ≤ nbcell + nghost), 
                                   foreground=:light_blue),
                        Highlighter((d,i) -> (abs(d[i]) > 1. || !isfinite(d[i])), 
                                   foreground=:red)))
end


function disp_all(params::Armon.ArmonParameters{T}, label::String, data::Armon.ArmonData{V},
        member_list::Vector{Symbol}) where {T, V <: AbstractArray{T}}
    println(" == $label == ")
    for member in member_list
        disp(params, member, getfield(data, member))
    end
end


function check_for_NaNs(label::String, data::Armon.ArmonData{V}) where {T, V <: AbstractArray{T}}
    found_NaNs = false
    for name in fieldnames(Armon.ArmonData{V})
        if any(isnan, getfield(data, name))
            !found_NaNs && println("Found NaNs at $label")
            println("NaN in $name")
            found_NaNs = true
        end
    end
    return found_NaNs
end


function diff_data(label::String, params::Armon.ArmonParameters{T}, 
        dd::Armon.ArmonData{V}, dg::Armon.ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; row_length, nghost, nbcell) = params

    different = false

    for name in fieldnames(Armon.ArmonData{V})
        name in (:gmat, :vmat, :tmp_vrho) && continue

        cpu_val = getfield(dd, name)
        gpu_val = getfield(dg, name)

        diff_mask = .~ isapprox.(cpu_val, gpu_val; atol=1e-10)
        diff_count = sum(diff_mask)

        if diff_count > 0
            !different && println("At $label:")
            different = true
            println("$diff_count differences found in $name")
            if diff_count < 40
                for idx in 1:nbcell
                    !diff_mask[idx] && continue
                    i, j = ((idx-1) % row_length) + 1 - nghost, ((idx-1) ÷ row_length) + 1 - nghost
                    @printf(" - %5d (%3d,%3d): %10.5g ≢ %10.5g (%10.5g)\n", idx, i, j, cpu_val[idx], gpu_val[idx], cpu_val[idx] - gpu_val[idx])
                end
            end
        end
    end

    return different
end


function comp_cpu_gpu_loop(
        params_cpu::Armon.ArmonParameters{T}, data_cpu::Armon.ArmonData{V}, 
        data_gpu::Armon.ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, euler_projection) = params_cpu

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    ccall(sym_step_init_loop, Cvoid, ())
    
    while t < maxtime && cycle < maxcycle
        Armon.boundaryConditions!(params_cpu, data_cpu)
        ccall(sym_step_boundary_conditions, Cvoid, ())
        diff_data("boundaryConditions", params_cpu, data_cpu, data_gpu) && return cycle

        c_dt = Armon.dtCFL(params_cpu, data_cpu, dta)
        g_dt = ccall(sym_step_dtCFL, flt_t, ())
        diff_data("dtCFL", params_cpu, data_cpu, data_gpu) && return cycle

        if !isfinite(c_dt) || c_dt <= 0.
            println("Invalid c_dt: $(c_dt)")
            return cycle
        end
        if !isfinite(g_dt) || g_dt <= 0.
            println("Invalid g_dt: $(g_dt)")
            return cycle
        end
        if !isapprox(c_dt, g_dt)
            println("dt too different: $c_dt != $g_dt")
            return cycle
        end

        Armon.numericalFluxes!(params_cpu, data_cpu, c_dt)
        ccall(sym_step_numerical_fluxes, Cvoid, ())
        diff_data("numericalFluxes!", params_cpu, data_cpu, data_gpu) && return cycle

        Armon.cellUpdate!(params_cpu, data_cpu, c_dt)
        ccall(sym_step_cell_update, Cvoid, ())
        diff_data("cellUpdate!", params_cpu, data_cpu, data_gpu) && return cycle

        if euler_projection
            Armon.first_order_euler_remap!(params_cpu, data_cpu, c_dt)
            ccall(sym_step_euler_projection, Cvoid, ())
            diff_data("first_order_euler_remap!", params_cpu, data_cpu, data_gpu) && return cycle
        end

        Armon.update_EOS!(params_cpu, data_cpu)
        ccall(sym_step_update_EOS, Cvoid, ())
        diff_data("update_EOS!", params_cpu, data_cpu, data_gpu) && return cycle
        
        ccall(sym_step_end_loop, Cvoid, ())

        dt = c_dt

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle
end


function diff_time_loop(params::Armon.ArmonParameters{T}) where T
    if (lib_ptr ≠ nothing)
        Libdl.dlclose(lib_ptr)
    end
    global lib_ptr = Libdl.dlopen(kokkos_comp_lib_path)
    load_lib()

    
    params.use_gpu = false

    data = Armon.ArmonData(T, params.nbcell + 2 * params.nghost)
    Armon.init_test(params, data)
    data.tmp_rho .= 0.
    data.tmp_urho .= 0.
    data.tmp_vrho .= 0.
    data.tmp_Erho .= 0.

    c_test = begin
        if params.test == :Sod; 0
        elseif params.test == :Bizarrium; 1
        else -1
        end
    end

    c_scheme = begin
        if params.scheme == :Godunov; 0
        elseif params.scheme == :GAD_minmod; 1
        else -1
        end
    end

    ccall(sym_init_cmp, Cvoid, 
        (Cint, Cint, Cint, Cint, flt_t, flt_t, Cuchar, Cuchar),
        c_test, c_scheme, params.nghost, params.nbcell, params.cfl, params.Dt,
        params.cst_dt, params.euler_projection)
    
    ccall(sym_init_test, Cvoid, ())
    #=
    k_data = wrap_data(params.nbcell + params.nghost * 2)

    params_cpu = copy(params)
    
    if diff_data("init", params_cpu, data, k_data)
        ccall(sym_end_cmp, Cvoid, ())

        k_data_cpy = copy(k_data)

        println("Stopped at init")

        Libdl.dlclose(lib_ptr)
        global lib_ptr = nothing

        return params_cpu, data, k_data_cpy 
    end

    cycle = comp_cpu_gpu_loop(params_cpu, data, k_data)
    
    println("Completed $cycle cycles.")

    k_data_cpy = copy(k_data)

    ccall(sym_end_cmp, Cvoid, ())

    Libdl.dlclose(lib_ptr)
    global lib_ptr = nothing

    return params_cpu, data, k_data_cpy
    =#
end
