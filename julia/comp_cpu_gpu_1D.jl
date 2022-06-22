
using Printf
using PrettyTables


const enable_NaN_tracking = false
if enable_NaN_tracking
    include("NaN_detector.jl")
else
    const NaNflag = false
    function reset_nan_flag() end
end

include("armon_module_gpu.jl")


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


function all_data_from_gpu(dg_tmp::Armon.ArmonData{V}, dg::Armon.ArmonData{W}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    Armon.data_from_gpu(dg_tmp, dg)
    copyto!(dg_tmp.ustar_1, dg.ustar_1)
    copyto!(dg_tmp.pstar_1, dg.pstar_1)
    copyto!(dg_tmp.tmp_rho, dg.tmp_rho)
    copyto!(dg_tmp.tmp_urho, dg.tmp_urho)
    copyto!(dg_tmp.tmp_vrho, dg.tmp_vrho)
    copyto!(dg_tmp.tmp_Erho, dg.tmp_Erho)
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
        dd::Armon.ArmonData{V}, dg::Armon.ArmonData{W}, 
        dg_tmp::Armon.ArmonData{V}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    if W != V
        all_data_from_gpu(dg_tmp, dg)
    end

    different = false

    for name in fieldnames(Armon.ArmonData{V})
        name == :gmat && continue

        cpu_val = getfield(dd, name)
        gpu_val = getfield(dg_tmp, name)

        diff_mask = .~ isapprox.(cpu_val, gpu_val; atol=1e-10)
        diff_count = sum(diff_mask)

        if diff_count > 0
            !different && println("At $label:")
            different = true
            println("$diff_count differences found in $name")
            if diff_count < 40
                for i in 1:length(dd.rho)
                    !diff_mask[i] && continue
                    @printf(" - %5d: %10.5g ≢ %10.5g (%10.5g)\n", i, cpu_val[i], gpu_val[i], cpu_val[i] - gpu_val[i])
                end
            end
        end
    end

    return different
end


function comp_cpu_gpu_loop(
        params_cpu::Armon.ArmonParameters{T}, data_cpu::Armon.ArmonData{V}, 
        params_gpu::Armon.ArmonParameters{T}, data_gpu::Armon.ArmonData{W}, 
        data_gpu_tmp::Armon.ArmonData{V}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; maxtime, maxcycle, euler_projection) = params_cpu

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    while t < maxtime && cycle < maxcycle
        Armon.boundaryConditions!(params_cpu, data_cpu)
        Armon.boundaryConditions!(params_gpu, data_gpu)
        diff_data("boundaryConditions", params_cpu, data_cpu, data_gpu, data_gpu_tmp) && return cycle

        c_dt = Armon.dtCFL(params_cpu, data_cpu, dta)
        g_dt = Armon.dtCFL(params_gpu, data_gpu, dta)
        diff_data("dtCFL", params_cpu, data_cpu, data_gpu, data_gpu_tmp) && return cycle

        if !isfinite(c_dt) || c_dt <= 0.
            error("Invalid c_dt: $(c_dt)")
        end
        if !isfinite(g_dt) || g_dt <= 0.
            error("Invalid g_dt: $(g_dt)")
        end
        if !isapprox(c_dt, g_dt)
            error("dt too different: $c_dt != $g_dt")
        end

        Armon.numericalFluxes!(params_cpu, data_cpu, c_dt)
        Armon.numericalFluxes!(params_gpu, data_gpu, g_dt)
        diff_data("numericalFluxes!", params_cpu, data_cpu, data_gpu, data_gpu_tmp) && return cycle

        Armon.cellUpdate!(params_cpu, data_cpu, c_dt)
        Armon.cellUpdate!(params_gpu, data_gpu, g_dt)
        diff_data("cellUpdate!", params_cpu, data_cpu, data_gpu, data_gpu_tmp) && return cycle

        if euler_projection
            Armon.first_order_euler_remap!(params_cpu, data_cpu, c_dt)
            Armon.first_order_euler_remap!(params_gpu, data_gpu, g_dt)
            diff_data("first_order_euler_remap!", params_cpu, data_cpu, data_gpu, data_gpu_tmp) && return cycle
        end

        Armon.update_EOS!(params_cpu, data_cpu)
        Armon.update_EOS!(params_gpu, data_gpu)
        diff_data("update_EOS!", params_cpu, data_cpu, data_gpu, data_gpu_tmp) && return cycle

        dt = (c_dt + g_dt) / 2

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle
end


function diff_time_loop(params::Armon.ArmonParameters{T}) where T
    params.use_gpu = false

    data = Armon.ArmonData(T, params.nbcell + 2 * params.nghost)
    Armon.init_test(params, data)
    data.tmp_rho .= 0.
    data.tmp_urho .= 0.
    data.tmp_vrho .= 0.
    data.tmp_Erho .= 0.

    params_cpu = copy(params)
    params_gpu = copy(params)
    
    params_cpu.use_gpu = false
    params_gpu.use_gpu = true

    d_data = Armon.data_to_gpu(data)
    d_data_tmp = Armon.ArmonData(T, params.nbcell + 2 * params.nghost)

    diff_data("init", params_cpu, data, d_data, d_data_tmp) && return params_cpu, data, d_data_tmp

    cycle = comp_cpu_gpu_loop(params_cpu, data, params_gpu, d_data, d_data_tmp)
    
    println("Completed $cycle cycles.")

    all_data_from_gpu(d_data_tmp, d_data)

    return params_cpu, data, d_data_tmp
end


function disp_cpu_loop(params::Armon.ArmonParameters{T}, data::Armon.ArmonData{V}, 
        disp_cycle::Int = 0) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, euler_projection) = params

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    while t < maxtime && cycle < maxcycle
        disp_ok = cycle >= disp_cycle

        disp_ok && println("\n === Cycle $cycle ====================== \n")

        Armon.boundaryConditions!(params, data)
        NaNflag && return cycle
        check_for_NaNs("boundaryConditions!", data) && return cycle

        dt = Armon.dtCFL(params, data, dta)
        NaNflag && return cycle

        if !isfinite(dt) || dt <= 0.
            println("Invalid dt: $(dt)")
            disp_all(params, "dt", data, [:rho, :umat, :vmat, :Emat, :cmat])
            return cycle
        end

        Armon.numericalFluxes!(params, data, dt)
        NaNflag && return cycle
        check_for_NaNs("numericalFluxes!", data) && return cycle
        disp_ok && disp_all(params, "numericalFluxes!", data, [:ustar, :pstar])

        Armon.cellUpdate!(params, data, dt)
        NaNflag && return cycle
        check_for_NaNs("cellUpdate!", data) && return cycle
        disp_ok && disp_all(params, "cellUpdate!", data, [:rho, axis == Armon.X_axis ? :umat : :vmat, :Emat])

        if euler_projection
            Armon.first_order_euler_remap!(params, data, dt)
            NaNflag && return cycle
            check_for_NaNs("first_order_euler_remap!", data) && return cycle
            disp_ok && disp_all(params, "first_order_euler_remap!", data, [:tmp_rho])
        end

        Armon.update_EOS!(params, data)
        NaNflag && return cycle
        check_for_NaNs("update_EOS!", data) && return cycle

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle
end


function disp_time_loop(params::Armon.ArmonParameters{T}, disp_cycle::Int = 0) where T
    reset_nan_flag()

    data = Armon.ArmonData(T, params.nbcell + 2 * params.nghost)

    Armon.init_test(params, data)
    
    check_for_NaNs("init", data) && return params, data
    NaNflag && return params, data
    
    cycle = disp_cpu_loop(params, data, disp_cycle)
    
    println("Completed $cycle cycles.")

    return params, data
end
