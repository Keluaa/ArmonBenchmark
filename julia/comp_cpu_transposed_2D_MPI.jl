
using Printf
using PrettyTables
using MPI
using CUDA
using Polyester
using KernelAbstractions
using InteractiveUtils


const enable_NaN_tracking = false
if enable_NaN_tracking
    include("NaN_detector.jl")
else
    const NaNflag = false
    function reset_nan_flag() end
end


const use_revise = parse(Bool, get(ENV, "USE_REVISE", "false"))
if use_revise
    using Revise
end

module ArmonT
if Main.use_revise
    using Revise
    Revise.track(Main.ArmonT, "armon_2D_MPI_transposition.jl"; mode=:eval, skip_include=false)
else
    include("armon_2D_MPI_transposition.jl")
end
end

module ArmonN
include("armon_2D_MPI.jl")
end 


abstract type Params{T} end

struct ArmonParamsT{T} <: Params{T}
    p::ArmonT.Armon.ArmonParameters{T}
end

struct ArmonParamsN{T} <: Params{T}
    p::ArmonN.Armon.ArmonParameters{T}
end

abstract type Data{V} end

struct ArmonDataT{V} <: Data{V}
    d::ArmonT.Armon.ArmonData{V}
end

struct ArmonDataN{V} <: Data{V}
    d::ArmonN.Armon.ArmonData{V}
end

import Base: get

get(p::ArmonParamsT{T} where T) = p.p
get(p::ArmonParamsN{T} where T) = p.p
get(d::ArmonDataT{V} where {T, V <: AbstractArray{T}}) = d.d
get(d::ArmonDataN{V} where {T, V <: AbstractArray{T}}) = d.d


function all_data_from_ᵀ(h_paramsᵀ::ArmonParamsT{T}, 
        ᵀ_data::ArmonDataT{V}, c_data::ArmonDataT{V}) where {T, V <: AbstractArray{T}}
    paramsᵀ = get(h_paramsᵀ)
    dᵀ = get(ᵀ_data)
    dc = get(c_data)
    if paramsᵀ.use_transposition && paramsᵀ.current_axis != ArmonT.Armon.X_axis
        ArmonT.Armon.@indexing_vars(paramsᵀ)
        for name in fieldnames(typeof(dᵀ))
            name in (:tmp_comm_array,) && continue
            ᵀ_val = getfield(dᵀ, name)
            c_val = getfield(dc, name)
            for j in paramsᵀ.domain_range
                for i in paramsᵀ.row_range .+ (j - 1)
                    iᵀ = ArmonT.Armon.@iᵀ(i)
                    c_val[iᵀ] = ᵀ_val[i]
                end
            end
        end
    else
        for name in fieldnames(typeof(dᵀ))
            ᵀ_val = getfield(dᵀ, name)
            c_val = getfield(dc, name)
            c_val .= ᵀ_val
        end
    end
end


function check_for_NaNs(label::String, h_data::Data{V}) where {T, V <: AbstractArray{T}}
    data = get(h_data)
    found_NaNs = false
    for name in fieldnames(typeof(data))
        if any(isnan, getfield(data, name))
            !found_NaNs && println("Found NaNs at $label")
            println("NaN in $name")
            found_NaNs = true
        end
    end
    return found_NaNs
end


function merge_near(r::UnitRange{Int}, x::Int)
    if x ≤ first(r)
        return x:last(r)
    else
        return first(r):x
    end
end


function merge_error_boxes!(boxes::Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}, ix::Int, iy::Int)
    for (i, (box_x, box_y)) in enumerate(boxes)
        if first(box_x) - 1 ≤ ix ≤ last(box_x) + 1 && first(box_y) - 1 ≤ iy ≤ last(box_y) + 1
            boxes[i] = (merge_near(box_x, ix), merge_near(box_y, iy))
            return
        end
    end
    push!(boxes, (ix:ix, iy:iy))
    return
end


const tolerance = 1e-12
@info @sprintf("Tolerance: %e", tolerance) maxlog=1

function diff_data(label::String, h_params::ArmonParamsN{T}, 
        h_dn::ArmonDataN{V}, h_dᵀ::ArmonDataT{V}, 
        ignore_fluxes::Bool, ignore_EOS::Bool) where {T, V <: AbstractArray{T}}
    dn = get(h_dn)
    dᵀ = get(h_dᵀ)
    params = get(h_params)

    (; row_length, nghost, nbcell) = params

    different = false
    messages = []

    for name in fieldnames(typeof(dn))
        name in (:gmat, :tmp_comm_array, :tmp_rho, :tmp_urho, :tmp_vrho, :tmp_Erho) && continue
        ignore_fluxes && name in (:ustar, :pstar, :pstar_1, :ustar_1) && continue
        ignore_EOS    && name in (:pmat, :cmat, :gmat) && continue

        n_val = getfield(dn, name)
        ᵀ_val = getfield(dᵀ, name)
        
        diff_mask = .~ isapprox.(n_val, ᵀ_val; atol=tolerance)
        diff_mask .*= dn.domain_mask  # Filter out the ghost cells
        diff_count = sum(diff_mask)

        if diff_count > 0
            !different && push!(messages, "In $(params.cart_coords) at $label:")
            different = true
            push!(messages, "$diff_count differences found in $name")
            (cx, cy) = params.cart_coords
            (nx, ny) = params.global_grid
            if diff_count < 40
                for idx in 1:nbcell
                    !diff_mask[idx] && continue
                    i, j = ((idx-1) % row_length) + 1 - nghost, ((idx-1) ÷ row_length) + 1 - nghost
                    gi = cx * nx + i
                    gj = cy * ny + j
                    push!(messages, @sprintf(" - %5d (%3d,%3d) (g=%3d,%3d): %10.5g ≢ %10.5g (%10.5g)", 
                        idx, i, j, gi, gj, n_val[idx], ᵀ_val[idx], n_val[idx] - ᵀ_val[idx]))
                end
            else
                error_boxes = Vector{Tuple{UnitRange{Int}, UnitRange{Int}}}()
                for idx in 1:nbcell
                    !diff_mask[idx] && continue
                    i, j = ((idx-1) % row_length) + 1 - nghost, ((idx-1) ÷ row_length) + 1 - nghost
                    merge_error_boxes!(error_boxes, i, j)
                    length(error_boxes) > 20 && break 
                end
                for (box_x, box_y) in error_boxes
                    gbox_x = box_x .+ cx * nx
                    gbox_y = box_y .+ cy * ny
                    push!(messages, @sprintf(" - from %3d,%3d (%5d) to %3d,%3d (%5d) (g: from %3d,%3d to %3d,%3d)", 
                        first(box_x), first(box_y), first(box_x) + first(box_y) * row_length, 
                        last(box_x), last(box_y), last(box_x) + last(box_y) * row_length, 
                        first(gbox_x), first(gbox_y), 
                        last(gbox_x), last(gbox_y)))
                end
                length(error_boxes) > 20 && push!(messages, "...and more")
            end
        end
    end

    different = convert(Bool, MPI.Allreduce(convert(Int, different), MPI.BOR, ArmonT.Armon.COMM))
    if different
        for rank in 0:params.proc_size-1
            rank == params.rank && foreach(println, messages)
            MPI.Barrier(ArmonT.Armon.COMM)
        end
    end

    return different
end


function diff_data_and_notify(label::String, h_params::ArmonParamsN{T}, 
        h_dn::ArmonDataN{V}, h_dᵀ::ArmonDataT{V}, ᵀ_twin_rank::Int;
        ignore_fluxes::Bool = false, ignore_EOS::Bool = false) where {T, V <: AbstractArray{T}}
    # @warn "Difference calc disabled" maxlog=1
    different = diff_data(label, h_params, h_dn, h_dᵀ, ignore_fluxes, ignore_EOS)
    # different = false
    MPI.Send(convert(Int, different), ᵀ_twin_rank, 0, MPI.COMM_WORLD)
    return different
end


function get_diff_result(n_twin_rank::Int)
    different, _ = MPI.Recv(Int, n_twin_rank, 0, MPI.COMM_WORLD)
    different = convert(Bool, different)
    return different 
end


function diff_dt_and_notify(c_dt, is_root, ᵀ_twin_rank)
    if is_root
        messages = []
        T = typeof(c_dt)
        g_dt, _ = MPI.Recv(T, ᵀ_twin_rank, 0, MPI.COMM_WORLD)
        if !isfinite(c_dt) || c_dt <= 0.
            push!(messages, "Invalid c_dt: $(c_dt)")
        end
        if !isfinite(g_dt) || g_dt <= 0.
            push!(messages, "Invalid g_dt: $(g_dt)")
        end
        ignore_dt_diff = false
        ignore_dt_diff && @warn "dt difference calc disabled" maxlog=1
        if !isapprox(c_dt, g_dt) && !ignore_dt_diff
            push!(messages, "dt too different: $c_dt != $g_dt")
        end

        different = !isempty(messages)
        different = MPI.bcast(different, 0, MPI.COMM_WORLD)

        foreach(println, messages)
    else
        different = MPI.bcast(false, 0, MPI.COMM_WORLD)
    end

    return different
end


function get_dt_diff_result(g_dt, n_twin_rank)
    if n_twin_rank == 0
        MPI.Send(g_dt, n_twin_rank, 0, MPI.COMM_WORLD)
    end

    return MPI.bcast(false, 0, MPI.COMM_WORLD)
end


function send_data_to_n(h_paramsᵀ::ArmonParamsT{T}, 
        ᵀ_data::ArmonDataT{V}, c_data::ArmonDataT{V}, 
        n_twin_rank::Int) where {T, V <: AbstractArray{T}}
    all_data_from_ᵀ(h_paramsᵀ, ᵀ_data, c_data)
    dc = get(c_data)
    requests = MPI.Request[
        MPI.Isend(dc.x, n_twin_rank, 0, MPI.COMM_WORLD),
        MPI.Isend(dc.y, n_twin_rank, 1, MPI.COMM_WORLD),
        MPI.Isend(dc.rho, n_twin_rank, 2, MPI.COMM_WORLD),
        MPI.Isend(dc.umat, n_twin_rank, 3, MPI.COMM_WORLD),
        MPI.Isend(dc.vmat, n_twin_rank, 4, MPI.COMM_WORLD),
        MPI.Isend(dc.Emat, n_twin_rank, 5, MPI.COMM_WORLD),
        MPI.Isend(dc.pmat, n_twin_rank, 6, MPI.COMM_WORLD),
        MPI.Isend(dc.cmat, n_twin_rank, 7, MPI.COMM_WORLD),
        MPI.Isend(dc.gmat, n_twin_rank, 8, MPI.COMM_WORLD),
        MPI.Isend(dc.ustar, n_twin_rank, 9, MPI.COMM_WORLD),
        MPI.Isend(dc.pstar, n_twin_rank, 10, MPI.COMM_WORLD),
        MPI.Isend(dc.ustar_1, n_twin_rank, 11, MPI.COMM_WORLD),
        MPI.Isend(dc.pstar_1, n_twin_rank, 12, MPI.COMM_WORLD),
        MPI.Isend(dc.tmp_rho, n_twin_rank, 13, MPI.COMM_WORLD),
        MPI.Isend(dc.tmp_urho, n_twin_rank, 14, MPI.COMM_WORLD),
        MPI.Isend(dc.tmp_vrho, n_twin_rank, 15, MPI.COMM_WORLD),
        MPI.Isend(dc.tmp_Erho, n_twin_rank, 16, MPI.COMM_WORLD),
        MPI.Isend(dc.domain_mask, n_twin_rank, 17, MPI.COMM_WORLD),
        MPI.Isend(dc.tmp_comm_array, n_twin_rank, 18, MPI.COMM_WORLD)
    ]
    MPI.Waitall!(requests)
end


function recv_data_from_ᵀ(h_ᵀ_data::ArmonDataT{V}, ᵀ_twin_rank::Int) where {T, V <: AbstractArray{T}}
    ᵀ_data = get(h_ᵀ_data)
    requests = MPI.Request[
        MPI.Irecv!(ᵀ_data.x, ᵀ_twin_rank, 0, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.y, ᵀ_twin_rank, 1, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.rho, ᵀ_twin_rank, 2, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.umat, ᵀ_twin_rank, 3, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.vmat, ᵀ_twin_rank, 4, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.Emat, ᵀ_twin_rank, 5, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.pmat, ᵀ_twin_rank, 6, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.cmat, ᵀ_twin_rank, 7, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.gmat, ᵀ_twin_rank, 8, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.ustar, ᵀ_twin_rank, 9, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.pstar, ᵀ_twin_rank, 10, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.ustar_1, ᵀ_twin_rank, 11, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.pstar_1, ᵀ_twin_rank, 12, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.tmp_rho, ᵀ_twin_rank, 13, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.tmp_urho, ᵀ_twin_rank, 14, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.tmp_vrho, ᵀ_twin_rank, 15, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.tmp_Erho, ᵀ_twin_rank, 16, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.domain_mask, ᵀ_twin_rank, 17, MPI.COMM_WORLD),
        MPI.Irecv!(ᵀ_data.tmp_comm_array, ᵀ_twin_rank, 18, MPI.COMM_WORLD)
    ]
    MPI.Waitall!(requests)
end


function n_comp_loop(h_params::ArmonParamsN{T}, h_paramsᵀ::ArmonParamsT{T},
        h_n_data::ArmonDataN{V}, h_ᵀ_data::ArmonDataT{V}, ᵀ_twin_rank::Int) where {T, V <: AbstractArray{T}}
    params = get(h_params)
    n_data = get(h_n_data)
    (; maxtime, maxcycle, is_root) = params

    ignore_all_fluxes = true
    ignore_all_fluxes && @info "All fluxes differences are ignored"

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
    diff_data_and_notify("init", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank) && return cycle

    host_array = Vector{T}()

    last_i::Int, x_::V, u::V = ArmonN.Armon.update_axis_parameters(params, n_data, params.current_axis)

    while t < maxtime && cycle < maxcycle
        dt = ArmonN.Armon.dtCFL_MPI(params, n_data, dta)
        diff_dt_and_notify(dt, is_root, ᵀ_twin_rank) && return cycle

        for (axis, dt_factor) in ArmonN.Armon.split_axes(params, cycle)
            last_i, x_, u = ArmonN.Armon.update_axis_parameters(params, n_data, axis)

            is_root && println("Current axis: $(params.current_axis)")

            ArmonN.Armon.boundaryConditions!(params, n_data, host_array, axis)
            recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
            diff_data_and_notify("boundaryConditions_axis", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank; ignore_fluxes=true) && return cycle

            ArmonN.Armon.numericalFluxes!(params, n_data, dt * dt_factor, last_i, u)
            recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
            diff_data_and_notify("fluxes", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank; ignore_fluxes=ignore_all_fluxes) && return cycle

            ArmonN.Armon.cellUpdate!(params, n_data, dt * dt_factor, u, x_)
            recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
            diff_data_and_notify("cellUpdate", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank; ignore_fluxes=ignore_all_fluxes) && return cycle

            if params.euler_projection
                if !params.single_comm_per_axis_pass 
                    ArmonN.Armon.boundaryConditions!(params, n_data, host_array, axis)
                    recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
                    diff_data_and_notify("boundaryConditions_euler", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank; ignore_fluxes=ignore_all_fluxes) && return cycle
                end
                ArmonN.Armon.first_order_euler_remap!(params, n_data, dt * dt_factor)
                recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
                diff_data_and_notify("euler_remap", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank; ignore_fluxes=true, ignore_EOS=true) && return cycle
            end

            ArmonN.Armon.update_EOS!(params, n_data)
            recv_data_from_ᵀ(h_ᵀ_data, ᵀ_twin_rank)
            diff_data_and_notify("update_EOS", h_params, h_n_data, h_ᵀ_data, ᵀ_twin_rank; ignore_fluxes=true) && return cycle
        end

        is_root && println("Cycle $cycle done.")

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle
end


function ᵀ_comp_loop(
        h_paramsᵀ::ArmonParamsT{T},  ᵀ_data::ArmonDataT{V},
        h_paramsᵀᵀ::ArmonParamsT{T}, ᵀᵀ_data::ArmonDataT{V},
        c_data::ArmonDataT{V},
        n_twin_rank::Int) where {T, V <: AbstractArray{T}}
    params  = get(h_paramsᵀ)
    paramsᵀ = get(h_paramsᵀᵀ)
    data  = get(ᵀ_data)
    dataᵀ = get(ᵀᵀ_data)
    (; maxtime, maxcycle) = params

    h_params_1 = h_paramsᵀ
    h_params_2 = h_paramsᵀᵀ
    ᵀ_data_1 = ᵀ_data
    ᵀ_data_2 = ᵀᵀ_data
    params_1 = params
    params_2 = paramsᵀ
    data_1 = data
    data_2 = dataᵀ
    x::V = data_1.x
    u::V = data_1.umat

    if params.use_transposition
        y::V = data_2.y
        v::V = data_2.vmat
    else
        y = data_1.y
        v = data_1.vmat
    end

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    transposed = false

    send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
    get_diff_result(n_twin_rank) && return cycle, transposed

    # Host version of temporary array used for MPI communications
    host_array = Vector{T}(undef, length(data.tmp_comm_array))

    while t < maxtime && cycle < maxcycle
        if !params.dt_on_even_cycles || iseven(cycle)
            dt = ArmonT.Armon.dtCFL_MPI(params_1, data_1, dta)
            get_dt_diff_result(dt, n_twin_rank) && return cycle, transposed
        end

        for (axis, dt_factor, next_axis) in ArmonT.Armon.split_axes(params, cycle)
            transposition_needed = params.use_transposition && next_axis != axis
            if !params.use_transposition && params_1.current_axis != axis
                # This pass is along the other axis
                h_params_1, h_params_2 = h_params_2, h_params_1
                # ᵀ_data_1, ᵀ_data_2 = ᵀ_data_2, ᵀ_data_1  # No exchange of data arrays here
                params_1, params_2 = params_2, params_1
                # data_1, data_2 = data_2, data_1
                x, y = y, x
                u, v = v, u

                @assert params_1.current_axis == axis
            end
            
            ArmonT.Armon.boundaryConditions!(params_1, data_1, host_array, axis) |> wait
            send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
            get_diff_result(n_twin_rank) && return cycle, transposed

            ArmonT.Armon.numericalFluxes!(params_1, data_1, dt * dt_factor, u) |> wait
            send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
            get_diff_result(n_twin_rank) && return cycle, transposed

            ArmonT.Armon.cellUpdate!(params_1, data_1, dt * dt_factor, u, x) |> wait
            send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
            get_diff_result(n_twin_rank) && return cycle, transposed
 
            if params.euler_projection
                if !params.single_comm_per_axis_pass 
                    ArmonT.Armon.boundaryConditions!(params_1, data_1, host_array, axis) |> wait
                    send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
                    get_diff_result(n_twin_rank) && return cycle, transposed
                end
                ArmonT.Armon.first_order_euler_remap!(params_1, data_1, data_2, dt * dt_factor,
                    transposition_needed) |> wait

                if transposition_needed
                    # Swap the parameters and data
                    h_params_1, h_params_2 = h_params_2, h_params_1
                    ᵀ_data_1, ᵀ_data_2 = ᵀ_data_2, ᵀ_data_1
                    params_1, params_2 = params_2, params_1
                    data_1, data_2 = data_2, data_1
                    x, y = y, x
                    u, v = v, u

                    transposed = !transposed
                    
                    @assert params_1.current_axis == next_axis
                    @assert params_2.current_axis == axis
                end

                send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
                get_diff_result(n_twin_rank) && return cycle, transposed
            end

            ArmonT.Armon.update_EOS!(params_1, data_1) |> wait
            send_data_to_n(h_params_1, ᵀ_data_1, c_data, n_twin_rank)
            get_diff_result(n_twin_rank) && return cycle, transposed
        end

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle, transposed
end


function init_MPI(px, py)
    expected_size = parse(Int, Base.get(ENV, "SLURM_NTASKS", string(px * py * 2)))
    expected_armon_procs = expected_size ÷ 2
    slurm_ID = parse(Int, Base.get(ENV, "SLURM_LOCALID", "-1"))
    if expected_armon_procs ≤ slurm_ID
        # GPU process
        if slurm_ID == -1
            @warn "Running on 1 GPU only" maxlog=1
            CUDA.device!(0)  # Use only one GPU
        elseif slurm_ID - expected_armon_procs < CUDA.ndevices()
            CUDA.device!(slurm_ID - expected_armon_procs)
        else
            @warn "There will be more than one process per GPU" maxlog=1
            CUDA.device!((slurm_ID - expected_armon_procs) % CUDA.ndevices())
        end
    end
    
    MPI.Init()

    rank = MPI.Comm_rank(MPI.COMM_WORLD)
    global_size = MPI.Comm_size(MPI.COMM_WORLD)

    if isodd(global_size)
        error("There must be an even number of processes.")
    end

    armon_procs = global_size ÷ 2

    if expected_size ≠ global_size
        error("Expected $expected_size processes, got $global_size")
    end

    is_n = 0 ≤ rank < armon_procs
    color = is_n ? 1 : 2

    device_comm = MPI.Comm_split(MPI.COMM_WORLD, color, rank)
    ArmonN.Armon.set_world_comm(device_comm)
    ArmonT.Armon.set_world_comm(device_comm)

    # if is_n
    #     println("GPU process $rank is using GPU $(CUDA.device())")
    # end

    return armon_procs, is_n
end


var"@i" = ArmonT.Armon.var"@i"


function init_sides(params::ArmonT.Armon.ArmonParameters{T}, data::ArmonT.Armon.ArmonData{V}) where {T, V <: AbstractArray{T}}
    cpy_params = copy(params)
    (; test, nghost, row_range) = cpy_params
    cpy_params.row_range = (first(row_range) - nghost):step(row_range):(last(row_range) + nghost)
    if test == :Sod || test == :Sod_y || test == :Sod_circ
        ArmonT.Armon.perfectGasEOS!(cpy_params, data, 7/5)
    else
        ArmonT.Armon.BizarriumEOS!(cpy_params, data)
    end
end


function comp_cpu_ᵀ_mpi(px, py)
    armon_procs, is_n = init_MPI(px, py)

    if armon_procs ≠ px * py
        error("The number of processes must be 2 times px×py")
    end

    test = :Sod_y
    scheme = :GAD_minmod
    single_comm_per_axis_pass = true
    nghost = 3
    nx = 50
    ny = 40
    maxcycle = 500
    maxtime = 10
    # cst_dt = true
    # Dt = 0.005
    cst_dt = false
    Dt = 0.
    write_output = true
    write_ghosts = true
    axis_splitting = :Sequential
    # transpose_dims = false
    transpose_dims = true
    use_temp_vars_for_projection = false

    (MPI.Comm_rank(MPI.COMM_WORLD) == 0) && if transpose_dims
        println("Comparing 2D MPI with 2D MPI + transposition")
    else
        println("Comparing 2D MPI with 2D MPI + no transposition")
    end

    if is_n
        h_params = ArmonParamsN(ArmonN.Armon.ArmonParameters(; 
            test, scheme, nghost, nx, ny,
            Dt, cst_dt,
            euler_projection = true, axis_splitting,
            transpose_dims = false,
            maxcycle, maxtime, silent = 5,
            output_file = "n_output",
            write_output, write_ghosts,
            use_gpu = false, 
            use_MPI = true, px, py,
            single_comm_per_axis_pass,
            reorder_grid = true)
        )
        params = get(h_params)
        h_paramsᵀ = ArmonParamsT(ArmonT.Armon.ArmonParameters(;
            test, scheme, nghost, nx, ny,
            Dt, cst_dt,
            euler_projection = true, axis_splitting,
            transpose_dims = transpose_dims,
            maxcycle, maxtime, silent = 5,
            output_file = "tn_output",
            write_output, write_ghosts,
            use_gpu = false, 
            use_MPI = true, px, py,
            single_comm_per_axis_pass,
            reorder_grid = true,
            async_comms=false,
            use_temp_vars_for_projection)
        )

        global_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        coords_to_ranks = MPI.Allgather(params.cart_coords => global_rank, MPI.COMM_WORLD)
        ᵀ_coords_to_ranks = coords_to_ranks[armon_procs+1:end]
        i = findfirst(p -> p.first == params.cart_coords, ᵀ_coords_to_ranks)
        ᵀ_twin_rank = ᵀ_coords_to_ranks[i].second

        tmp_comm_size = max(params.nx, params.ny) * params.nghost * 7

        data  = ArmonN.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)
        dataᵀ = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)

        ArmonN.Armon.init_test(params, data)
        data.tmp_rho .= 0.
        data.tmp_urho .= 0.
        data.tmp_vrho .= 0.
        data.tmp_Erho .= 0.
        data.tmp_comm_array .= 0.

        cpy_params = copy(params)
        cpy_params.ideb = params.ideb - nghost
        cpy_params.ifin = params.ideb
        ArmonN.Armon.update_EOS!(cpy_params, data)
        cpy_params.ideb = params.ifin
        cpy_params.ifin = params.ifin + nghost
        ArmonN.Armon.update_EOS!(cpy_params, data)

        params.is_root && println("Setup OK, running comparison...")

        cycles = n_comp_loop(h_params, h_paramsᵀ, ArmonDataN(data), ArmonDataT(dataᵀ), ᵀ_twin_rank)

        if params.write_output
            if params.is_root
                println("Order of variables in result file: x (1), y (2), rho (3), umat (4), vmat (5), pmat (6), Emat (7), cmat (8), ustar (9)")
            end
            ArmonN.Armon.write_result(params, data, params.output_file)
        end

        params.is_root && println("Completed $cycles cycles in normal mode")
    else
        h_paramsᵀ = ArmonParamsT(ArmonT.Armon.ArmonParameters(;
            test, scheme, nghost, nx, ny,
            Dt, cst_dt,
            euler_projection = true, axis_splitting,
            transpose_dims = transpose_dims,
            maxcycle, maxtime, silent = 5,
            output_file = "t_output",
            write_output, write_ghosts,
            use_gpu = false, 
            use_MPI = true, px, py,
            single_comm_per_axis_pass,
            reorder_grid = true,
            async_comms=false,
            use_temp_vars_for_projection)
        )
        params  = get(h_paramsᵀ)
        paramsᵀ = ArmonT.Armon.transpose_params(params)

        global_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        coords_to_ranks = MPI.Allgather(params.cart_coords => global_rank, MPI.COMM_WORLD)
        n_coords_to_ranks = coords_to_ranks[1:armon_procs]
        i = findfirst(p -> p.first == params.cart_coords, n_coords_to_ranks)
        n_twin_rank = n_coords_to_ranks[i].second

        tmp_comm_size = max(params.nx, params.ny) * params.nghost * 7

        data   = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)
        dataᵀ  = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)
        c_data = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)

        data.pmat  .= NaN
        dataᵀ.pmat .= NaN

        data.tmp_rho .= 0.
        data.tmp_urho .= 0.
        data.tmp_vrho .= 0.
        data.tmp_Erho .= 0.
        data.tmp_comm_array .= 0.

        dataᵀ.tmp_rho .= 0.
        dataᵀ.tmp_urho .= 0.
        dataᵀ.tmp_vrho .= 0.
        dataᵀ.tmp_Erho .= 0.
        dataᵀ.tmp_comm_array .= 0.

        c_data.tmp_rho .= 0.
        c_data.tmp_urho .= 0.
        c_data.tmp_vrho .= 0.
        c_data.tmp_Erho .= 0.
        c_data.tmp_comm_array .= 0.

        main_range,  inner_range,  side_ghost_range_1,  side_ghost_range_2  = ArmonT.Armon.compute_ranges(params,  ArmonT.Armon.X_axis)
        main_rangeᵀ, inner_rangeᵀ, side_ghost_range_1ᵀ, side_ghost_range_2ᵀ = ArmonT.Armon.compute_ranges(paramsᵀ, ArmonT.Armon.Y_axis)

        params.domain_range = main_range
        params.row_range = inner_range
    
        paramsᵀ.domain_range = main_rangeᵀ
        paramsᵀ.row_range = inner_rangeᵀ

        ArmonT.Armon.init_test(params,  data,  side_ghost_range_1,  side_ghost_range_2,  false)
        ArmonT.Armon.init_test(paramsᵀ, dataᵀ, side_ghost_range_1ᵀ, side_ghost_range_2ᵀ, true)

        init_sides(params,  data)
        init_sides(paramsᵀ, dataᵀ)

        cycles, transposed = ᵀ_comp_loop(h_paramsᵀ, ArmonDataT(data), ArmonParamsT(paramsᵀ), ArmonDataT(dataᵀ), ArmonDataT(c_data), n_twin_rank)

        if params.write_output
            if transposed
                ArmonT.Armon.write_result(paramsᵀ, dataᵀ, params.output_file)
            else
                ArmonT.Armon.write_result(params, data, params.output_file)
            end
        end

        params.is_root && println("Completed $cycles cycles $(params.use_transposition ? "with" : "without") transposition")
    end
end


function disp(params, label, array, mask, transposed)
    array_2d = reshape(array, params.row_length, params.col_length)'
    domain_2d = reshape(mask, params.row_length, params.col_length)'

    if transposed
        array_2d  = array_2d'
        domain_2d = domain_2d'
    end

    array_2d = reverse(array_2d, dims=1)  # Put the origin on the bottom-left corner
    domain_2d = reverse(domain_2d, dims=1)

    println(label, " $(params.row_length)×$(params.col_length) $(transposed ? "(transposed)" : "")")
    pretty_table(array_2d; 
        noheader = true, 
        highlighters = (Highlighter((d,i,j) -> (domain_2d[i,j] > 0.), 
                                   foreground=:light_blue),
                        Highlighter((d,i,j) -> (abs(d[i,j]) > 1. || !isfinite(d[i,j])), 
                                   foreground=:red)))
end


function disp_ranges(params, transposed, side_ghost_range_1,  side_ghost_range_2)
    (; domain_range, row_range, nghost, row_length, col_length) = params

    all_row_range = (first(row_range) - nghost):step(row_range):(last(row_range) + nghost)
    for j in domain_range
        pos_all_row_range = all_row_range .+ (j - 1)

        f = first(pos_all_row_range)
        l = last(pos_all_row_range)
        if transposed
            fx = ((f-1) ÷ row_length) - nghost
            lx = ((l-1) ÷ row_length) - nghost
            fy = ((f-1) % row_length) - nghost
            ly = ((l-1) % row_length) - nghost
        else
            fx = ((f-1) % row_length) - nghost
            lx = ((l-1) % row_length) - nghost
            fy = ((f-1) ÷ row_length) - nghost
            ly = ((l-1) ÷ row_length) - nghost
        end

        println(" - $f ($fx,$fy) to $l ($lx,$ly)")
    end

    println("Ghost range 1:")
    for j in side_ghost_range_1
        pos_all_row_range = all_row_range .+ (j - 1)

        f = first(pos_all_row_range)
        l = last(pos_all_row_range)
        if transposed
            fx = ((f-1) ÷ row_length) - nghost
            lx = ((l-1) ÷ row_length) - nghost
            fy = ((f-1) % row_length) - nghost
            ly = ((l-1) % row_length) - nghost
        else
            fx = ((f-1) % row_length) - nghost
            lx = ((l-1) % row_length) - nghost
            fy = ((f-1) ÷ row_length) - nghost
            ly = ((l-1) ÷ row_length) - nghost
        end

        println(" - $f ($fx,$fy) to $l ($lx,$ly)")
    end

    println("Ghost range 2:")
    for j in side_ghost_range_2
        pos_all_row_range = all_row_range .+ (j - 1)

        f = first(pos_all_row_range)
        l = last(pos_all_row_range)
        if transposed
            fx = ((f-1) ÷ row_length) - nghost
            lx = ((l-1) ÷ row_length) - nghost
            fy = ((f-1) % row_length) - nghost
            ly = ((l-1) % row_length) - nghost
        else
            fx = ((f-1) % row_length) - nghost
            lx = ((l-1) % row_length) - nghost
            fy = ((f-1) ÷ row_length) - nghost
            ly = ((l-1) ÷ row_length) - nghost
        end

        println(" - $f ($fx,$fy) to $l ($lx,$ly)")
    end
end


function index_array(params, transposed)
    (; row_length, col_length, nghost, idx_row, idx_col, index_start, nx, ny) = params
    
    idx_arr = zeros(Int, row_length, col_length)
    for j in 1-nghost:ny+nghost
        for i in 1-nghost:nx+nghost
            idx = index_start + j * idx_row + i * idx_col
            idx_arr[idx] = idx
        end
    end

    return idx_arr
end

Base.adjoint(x::Tuple{Int, Int}) = x
Base.abs(x::Tuple{Int, Int}) = (abs(x[1]), abs(x[2]))
Base.isless(n::Float64, x::Tuple{Int, Int}) = any((n < x[1], n < x[2]))
Base.isfinite(x::Tuple{Int, Int}) = any((isfinite(x[1]), isfinite(x[2])))


function pos_array(params, side_ghost_range_1, side_ghost_range_2, transposed)
    (; row_length, col_length, nghost, idx_row, idx_col, index_start, nx, ny, row_range) = params
    
    idx_arr = Matrix{Tuple{Int, Int}}(undef, row_length, col_length)

    all_row_range = (first(row_range) - nghost):step(row_range):(last(row_range) + nghost)
    for j in Iterators.flatten((params.domain_range, side_ghost_range_1,  side_ghost_range_2))
        for i in all_row_range .+ (j - 1)
            if transposed
                ix = ((i-1) ÷ row_length) - nghost
                iy = ((i-1) % row_length) - nghost
            else
                ix = ((i-1) % row_length) - nghost
                iy = ((i-1) ÷ row_length) - nghost
            end

            idx_arr[i] = (ix+1, iy+1)
        end
    end

    return idx_arr
end


function small_test()
    MPI.Init()

    h_paramsᵀ = ArmonParamsT(ArmonT.Armon.ArmonParameters(;
        test=:Sod, scheme=:GAD_minmod, nghost=3, nx=10, ny=8,
        Dt=0.05, cst_dt=false,
        euler_projection = true, axis_splitting=:Sequential,
        transpose_dims = true,
        maxcycle=1, maxtime=0.2, silent = 5,
        output_file = "t_output",
        write_output=false, write_ghosts=false,
        use_gpu = false, 
        use_MPI = false, px=1, py=1,
        single_comm_per_axis_pass=true,
        reorder_grid = true,
        async_comms=false,
        use_temp_vars_for_projection=false)
    )

    params  = get(h_paramsᵀ)
    paramsᵀ = ArmonT.Armon.transpose_params(params)

    tmp_comm_size = max(params.nx, params.ny) * params.nghost * 7

    data   = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)
    dataᵀ  = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)

    data.pmat  .= NaN
    dataᵀ.pmat .= NaN

    data.tmp_rho .= 0.
    data.tmp_urho .= 0.
    data.tmp_vrho .= 0.
    data.tmp_Erho .= 0.
    data.tmp_comm_array .= 0.

    dataᵀ.tmp_rho .= 0.
    dataᵀ.tmp_urho .= 0.
    dataᵀ.tmp_vrho .= 0.
    dataᵀ.tmp_Erho .= 0.
    dataᵀ.tmp_comm_array .= 0.

    main_range,  inner_range,  side_ghost_range_1,  side_ghost_range_2  = ArmonT.Armon.compute_ranges(params,  ArmonT.Armon.X_axis)
    main_rangeᵀ, inner_rangeᵀ, side_ghost_range_1ᵀ, side_ghost_range_2ᵀ = ArmonT.Armon.compute_ranges(paramsᵀ, ArmonT.Armon.Y_axis)

    params.domain_range = main_range
    params.row_range = inner_range

    paramsᵀ.domain_range = main_rangeᵀ
    paramsᵀ.row_range = inner_rangeᵀ

    ArmonT.Armon.init_test(params,  data,  side_ghost_range_1,  side_ghost_range_2,  false)
    ArmonT.Armon.init_test(paramsᵀ, dataᵀ, side_ghost_range_1ᵀ, side_ghost_range_2ᵀ, true)

    disp(params, "rho", data.rho, data.domain_mask, false)
    disp(paramsᵀ, "rhoᵀ", dataᵀ.rho, dataᵀ.domain_mask, true)

    disp(params, "idx", index_array(params,   false), data.domain_mask, false)
    disp(paramsᵀ, "idxᵀ", index_array(paramsᵀ, true), dataᵀ.domain_mask, true)

    disp(params, "pos", pos_array(params, side_ghost_range_1, side_ghost_range_2, false), data.domain_mask, false)
    disp(paramsᵀ, "posᵀ", pos_array(paramsᵀ, side_ghost_range_1ᵀ, side_ghost_range_2ᵀ, true), dataᵀ.domain_mask, true)

    println("params:")
    disp_ranges(params,  false, side_ghost_range_1,  side_ghost_range_2)

    println("paramsᵀ:")
    disp_ranges(paramsᵀ, true,  side_ghost_range_1ᵀ, side_ghost_range_2ᵀ)

    @show params.domain_range
    @show params.row_range
    @show paramsᵀ.domain_range
    @show paramsᵀ.row_range

    @show params.idx_row
    @show params.idx_col
    @show paramsᵀ.idx_row
    @show paramsᵀ.idx_col

    @show params.row_length
    @show params.col_length
    @show paramsᵀ.row_length
    @show paramsᵀ.col_length

    return
end


function disp_code()
    h_params = ArmonParamsN(ArmonN.Armon.ArmonParameters(; 
        test=:Sod, scheme=:GAD_minmod, nghost=3, nx=8, ny=2,
        Dt=0.05, cst_dt=false,
        euler_projection = true, axis_splitting=:Sequential,
        transpose_dims = false,
        maxcycle=1, maxtime=0.2, silent = 5,
        output_file = "n_output",
        write_output=false, write_ghosts=false,
        use_gpu = false, 
        use_MPI = false, px=1, py=1,
        single_comm_per_axis_pass=true,
        reorder_grid = true)
    )

    h_paramsᵀ = ArmonParamsT(ArmonT.Armon.ArmonParameters(;
        test=:Sod, scheme=:GAD_minmod, nghost=3, nx=8, ny=2,
        Dt=0.05, cst_dt=false,
        euler_projection = true, axis_splitting=:Sequential,
        transpose_dims = true,
        maxcycle=1, maxtime=0.2, silent = 5,
        output_file = "t_output",
        write_output=false, write_ghosts=false,
        use_gpu = false, 
        use_MPI = false, px=1, py=1,
        single_comm_per_axis_pass=true,
        reorder_grid = true,
        async_comms=false,
        use_temp_vars_for_projection=false)
    )

    params   = get(h_params)
    paramsᵀ  = get(h_paramsᵀ)

    tmp_comm_size = max(params.nx, params.ny) * params.nghost * 7

    data  = ArmonN.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)
    dataᵀ = ArmonT.Armon.ArmonData(typeof(params.dx), params.nbcell, tmp_comm_size)

    println("Normal:")
    # @code_native syntax=:intel ArmonN.Armon.perfectGasEOS!(params, data, 7/5)
    @code_warntype ArmonN.Armon.perfectGasEOS!(params, data, 7/5)
    println("Transposed style:")
    # @code_native syntax=:intel ArmonT.Armon.perfectGasEOS!(paramsᵀ, dataᵀ, 7/5)
    @code_warntype ArmonT.Armon.perfectGasEOS!(paramsᵀ, dataᵀ, 7/5)
end


# small_test()
comp_cpu_ᵀ_mpi(1, 1)
# comp_cpu_ᵀ_mpi(2, 2)
