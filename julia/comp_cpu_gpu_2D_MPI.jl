
using Printf
using PrettyTables
using MPI
using CUDA


const enable_NaN_tracking = false
if enable_NaN_tracking
    include("NaN_detector.jl")
else
    const NaNflag = false
    function reset_nan_flag() end
end

const async_impl = true
if async_impl
    include("armon_2D_MPI_async.jl")
    wait_f(event) = wait(event)
else
    include("armon_2D_MPI.jl")
    wait_f(event) = nothing
end


function all_data_from_gpu(dg_tmp::Armon.ArmonData{V}, dg::Armon.ArmonData{W}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    Armon.data_from_gpu(dg_tmp, dg)
    copyto!(dg_tmp.ustar_1, dg.ustar_1)
    copyto!(dg_tmp.pstar_1, dg.pstar_1)
    copyto!(dg_tmp.domain_mask, dg.domain_mask)
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
        dd::Armon.ArmonData{V}, dg::Armon.ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; row_length, nghost, nbcell) = params

    different = false
    messages = []

    for name in fieldnames(Armon.ArmonData{V})
        name in (:gmat, :tmp_comm_array, :tmp_urho, :tmp_Erho, :tmp_vrho, :tmp_rho) && continue

        cpu_val = getfield(dd, name)
        gpu_val = getfield(dg, name)

        diff_mask = .~ isapprox.(cpu_val, gpu_val; atol=1e-10)
        diff_count = sum(diff_mask)

        if diff_count > 0
            !different && push!(messages, "In $(params.cart_coords) at $label:")
            different = true
            push!(messages, "$diff_count differences found in $name")
            if diff_count < 40
                (cx, cy) = params.cart_coords
                (nx, ny) = params.global_grid
                for idx in 1:nbcell
                    !diff_mask[idx] && continue
                    i, j = ((idx-1) % row_length) + 1 - nghost, ((idx-1) ÷ row_length) + 1 - nghost
                    gi = cx * nx + i
                    gj = cy * ny + j
                    push!(messages, @sprintf(" - %5d (%3d,%3d) (g=%3d,%3d): %10.5g ≢ %10.5g (%10.5g)", idx, i, j, gi, gj, cpu_val[idx], gpu_val[idx], cpu_val[idx] - gpu_val[idx]))
                end
            end
        end
    end

    different = convert(Bool, MPI.Allreduce(convert(Int, different), MPI.BOR, Armon.COMM))
    if different
        for rank in 0:params.proc_size-1
            rank == params.rank && foreach(println, messages)
            MPI.Barrier(Armon.COMM)
        end
    end

    return different
end


function diff_data_and_notify(label::String, params::Armon.ArmonParameters{T}, 
        dd::Armon.ArmonData{V}, dg::Armon.ArmonData{V}, gpu_twin_rank::Int) where {T, V <: AbstractArray{T}}
    different = diff_data(label, params, dd, dg)
    MPI.Send(convert(Int, different), gpu_twin_rank, 0, MPI.COMM_WORLD)
    return different
end


function get_diff_result(params::Armon.ArmonParameters{T}, cpu_twin_rank::Int) where {T}
    different, _ = MPI.Recv(Int, cpu_twin_rank, 0, MPI.COMM_WORLD)
    different = convert(Bool, different)
    return different 
end


function diff_dt_and_notify(c_dt, is_root, gpu_twin_rank)
    if is_root
        messages = []
        T = typeof(c_dt)
        g_dt, _ = MPI.Recv(T, gpu_twin_rank, 0, MPI.COMM_WORLD)
        if !isfinite(c_dt) || c_dt <= 0.
            push!(messages, "Invalid c_dt: $(c_dt)")
        end
        if !isfinite(g_dt) || g_dt <= 0.
            push!(messages, "Invalid g_dt: $(g_dt)")
        end
        if !isapprox(c_dt, g_dt)
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


function get_dt_diff_result(g_dt, cpu_twin_rank)
    if cpu_twin_rank == 0
        MPI.Send(g_dt, cpu_twin_rank, 0, MPI.COMM_WORLD)
    end

    return MPI.bcast(false, 0, MPI.COMM_WORLD)
end


function send_data_to_cpu(g_data::Armon.ArmonData{V}, c_data::Armon.ArmonData{W}, 
        cpu_twin_rank::Int) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    all_data_from_gpu(c_data, g_data)
    requests = MPI.Request[
        MPI.Isend(c_data.x, cpu_twin_rank, 0, MPI.COMM_WORLD),
        MPI.Isend(c_data.y, cpu_twin_rank, 1, MPI.COMM_WORLD),
        MPI.Isend(c_data.rho, cpu_twin_rank, 2, MPI.COMM_WORLD),
        MPI.Isend(c_data.umat, cpu_twin_rank, 3, MPI.COMM_WORLD),
        MPI.Isend(c_data.vmat, cpu_twin_rank, 4, MPI.COMM_WORLD),
        MPI.Isend(c_data.Emat, cpu_twin_rank, 5, MPI.COMM_WORLD),
        MPI.Isend(c_data.pmat, cpu_twin_rank, 6, MPI.COMM_WORLD),
        MPI.Isend(c_data.cmat, cpu_twin_rank, 7, MPI.COMM_WORLD),
        MPI.Isend(c_data.gmat, cpu_twin_rank, 8, MPI.COMM_WORLD),
        MPI.Isend(c_data.ustar, cpu_twin_rank, 9, MPI.COMM_WORLD),
        MPI.Isend(c_data.pstar, cpu_twin_rank, 10, MPI.COMM_WORLD),
        MPI.Isend(c_data.ustar_1, cpu_twin_rank, 11, MPI.COMM_WORLD),
        MPI.Isend(c_data.pstar_1, cpu_twin_rank, 12, MPI.COMM_WORLD),
        MPI.Isend(c_data.tmp_rho, cpu_twin_rank, 13, MPI.COMM_WORLD),
        MPI.Isend(c_data.tmp_urho, cpu_twin_rank, 14, MPI.COMM_WORLD),
        MPI.Isend(c_data.tmp_vrho, cpu_twin_rank, 15, MPI.COMM_WORLD),
        MPI.Isend(c_data.tmp_Erho, cpu_twin_rank, 16, MPI.COMM_WORLD),
        MPI.Isend(c_data.domain_mask, cpu_twin_rank, 17, MPI.COMM_WORLD),
        MPI.Isend(c_data.tmp_comm_array, cpu_twin_rank, 18, MPI.COMM_WORLD)
    ]
    MPI.Waitall!(requests)
end


function recv_data_from_gpu(g_data::Armon.ArmonData{V}, gpu_twin_rank::Int) where {T, V <: AbstractArray{T}}
    requests = MPI.Request[
        MPI.Irecv!(g_data.x, gpu_twin_rank, 0, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.y, gpu_twin_rank, 1, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.rho, gpu_twin_rank, 2, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.umat, gpu_twin_rank, 3, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.vmat, gpu_twin_rank, 4, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.Emat, gpu_twin_rank, 5, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.pmat, gpu_twin_rank, 6, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.cmat, gpu_twin_rank, 7, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.gmat, gpu_twin_rank, 8, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.ustar, gpu_twin_rank, 9, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.pstar, gpu_twin_rank, 10, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.ustar_1, gpu_twin_rank, 11, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.pstar_1, gpu_twin_rank, 12, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.tmp_rho, gpu_twin_rank, 13, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.tmp_urho, gpu_twin_rank, 14, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.tmp_vrho, gpu_twin_rank, 15, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.tmp_Erho, gpu_twin_rank, 16, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.domain_mask, gpu_twin_rank, 17, MPI.COMM_WORLD),
        MPI.Irecv!(g_data.tmp_comm_array, gpu_twin_rank, 18, MPI.COMM_WORLD)
    ]
    MPI.Waitall!(requests)
end


function cpu_comp_loop(params::Armon.ArmonParameters{T}, data::Armon.ArmonData{V}, 
        g_data::Armon.ArmonData{V}, gpu_twin_rank::Int) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, is_root) = params
    
    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    recv_data_from_gpu(g_data, gpu_twin_rank)
    diff_data_and_notify("init", params, data, g_data, gpu_twin_rank) && return cycle

    host_array = Vector{T}()

    last_i::Int, x_::V, u::V = Armon.update_axis_parameters(params, data, params.current_axis)

    while t < maxtime && cycle < maxcycle
        dt = Armon.dtCFL_MPI(params, data, dta)
        diff_dt_and_notify(dt, is_root, gpu_twin_rank) && return cycle

        for (axis, dt_factor) in Armon.split_axes(params, cycle)
            last_i, x_, u = Armon.update_axis_parameters(params, data, axis)

            is_root && println("Current axis: $(params.current_axis)")

            @static if async_impl
                Armon.boundaryConditions!(params, data, host_array, axis) |> wait_f
            else
                Armon.boundaryConditions_MPI!(params, data, host_array, axis)
            end
            recv_data_from_gpu(g_data, gpu_twin_rank)
            diff_data_and_notify("boundaryConditions_axis", params, data, g_data, gpu_twin_rank) && return cycle

            @static if async_impl
                Armon.numericalFluxes_inner!(params, data, dt * dt_factor, u) |> wait_f
                recv_data_from_gpu(g_data, gpu_twin_rank)
                diff_data_and_notify("fluxes_inner", params, data, g_data, gpu_twin_rank) && return cycle

                Armon.numericalFluxes_outer!(params, data, dt * dt_factor, u) |> wait_f
                recv_data_from_gpu(g_data, gpu_twin_rank)
                diff_data_and_notify("fluxes_outer", params, data, g_data, gpu_twin_rank) && return cycle
            else
                Armon.numericalFluxes!(params, data, dt * dt_factor, last_i, u)
                recv_data_from_gpu(g_data, gpu_twin_rank)
                diff_data_and_notify("fluxes", params, data, g_data, gpu_twin_rank) && return cycle
            end

            Armon.cellUpdate!(params, data, dt * dt_factor, u, x_) |> wait_f
            recv_data_from_gpu(g_data, gpu_twin_rank)
            diff_data_and_notify("cellUpdate", params, data, g_data, gpu_twin_rank) && return cycle

            if params.euler_projection
                if !params.single_comm_per_axis_pass
                    @static if async_impl
                        Armon.boundaryConditions!(params, data, host_array, axis) |> wait_f
                    else
                        Armon.boundaryConditions_MPI!(params, data, host_array, axis)
                    end
                    recv_data_from_gpu(g_data, gpu_twin_rank)
                    diff_data_and_notify("boundaryConditions_euler", params, data, g_data, gpu_twin_rank) && return cycle
                end

                Armon.first_order_euler_remap!(params, data, dt * dt_factor) |> wait_f
                recv_data_from_gpu(g_data, gpu_twin_rank)
                diff_data_and_notify("euler_remap", params, data, g_data, gpu_twin_rank) && return cycle
            end

            Armon.update_EOS!(params, data) |> wait_f
            recv_data_from_gpu(g_data, gpu_twin_rank)
            diff_data_and_notify("update_EOS", params, data, g_data, gpu_twin_rank) && return cycle
        end

        is_root && println("Cycle $cycle done.")

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle
end


function gpu_comp_loop(params::Armon.ArmonParameters{T}, data::Armon.ArmonData{V},
        data_tmp::Armon.ArmonData{W},
        cpu_twin_rank::Int) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; maxtime, maxcycle) = params

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    send_data_to_cpu(data, data_tmp, cpu_twin_rank)
    get_diff_result(params, cpu_twin_rank) && return cycle

    # Host version of temporary array used for MPI communications
    host_array = Vector{T}(undef, length(data.tmp_comm_array))

    last_i::Int, x_::V, u::V = Armon.update_axis_parameters(params, data, params.current_axis)

    while t < maxtime && cycle < maxcycle
        dt = Armon.dtCFL_MPI(params, data, dta)
        get_dt_diff_result(dt, cpu_twin_rank) && return cycle

        for (axis, dt_factor) in Armon.split_axes(params, cycle)
            last_i, x_, u = Armon.update_axis_parameters(params, data, axis)

            @static if async_impl
                Armon.boundaryConditions!(params, data, host_array, axis) |> wait_f
            else
                Armon.boundaryConditions_MPI!(params, data, host_array, axis)
            end
            send_data_to_cpu(data, data_tmp, cpu_twin_rank)
            get_diff_result(params, cpu_twin_rank) && return cycle

            @static if async_impl
                Armon.numericalFluxes_inner!(params, data, dt * dt_factor, u) |> wait
                send_data_to_cpu(data, data_tmp, cpu_twin_rank)
                get_diff_result(params, cpu_twin_rank) && return cycle

                Armon.numericalFluxes_outer!(params, data, dt * dt_factor, u) |> wait
                send_data_to_cpu(data, data_tmp, cpu_twin_rank)
                get_diff_result(params, cpu_twin_rank) && return cycle
            else
                Armon.numericalFluxes!(params, data, dt * dt_factor, last_i, u)
                send_data_to_cpu(data, data_tmp, cpu_twin_rank)
                get_diff_result(params, cpu_twin_rank) && return cycle
            end
            
            Armon.cellUpdate!(params, data, dt * dt_factor, u, x_) |> wait_f
            send_data_to_cpu(data, data_tmp, cpu_twin_rank)
            get_diff_result(params, cpu_twin_rank) && return cycle
 
            if params.euler_projection
                if !params.single_comm_per_axis_pass 
                    @static if async_impl
                        Armon.boundaryConditions!(params, data, host_array, axis) |> wait
                    else
                        Armon.boundaryConditions_MPI!(params, data, host_array, axis)
                    end
                    send_data_to_cpu(data, data_tmp, cpu_twin_rank)
                    get_diff_result(params, cpu_twin_rank) && return cycle
                end

                Armon.first_order_euler_remap!(params, data, dt * dt_factor) |> wait_f
                send_data_to_cpu(data, data_tmp, cpu_twin_rank)
                get_diff_result(params, cpu_twin_rank) && return cycle
            end

            Armon.update_EOS!(params, data) |> wait_f
            send_data_to_cpu(data, data_tmp, cpu_twin_rank)
            get_diff_result(params, cpu_twin_rank) && return cycle
        end

        dta = dt
        cycle += 1
        t += dt
    end

    return cycle
end


function init_MPI(px, py)
    expected_size = parse(Int, get(ENV, "SLURM_NTASKS", string(px * py * 2)))
    expected_armon_procs = expected_size ÷ 2
    slurm_ID = parse(Int, get(ENV, "SLURM_LOCALID", "-1"))
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

    is_cpu = 0 ≤ rank < armon_procs
    color = is_cpu ? 1 : 2

    device_comm = MPI.Comm_split(MPI.COMM_WORLD, color, rank)
    Armon.set_world_comm(device_comm)

    if !is_cpu
        println("GPU process $rank is using GPU $(CUDA.device())")
    end

    return armon_procs, is_cpu
end


function comp_cpu_gpu_mpi(px, py)
    armon_procs, is_cpu = init_MPI(px, py)

    if armon_procs ≠ px * py
        error("The number of processes must be 2 times px×py")
    end

    test = :Sod
    scheme = :GAD_minmod
    single_comm_per_axis_pass = false
    nghost = 2
    nx = 40
    ny = 40
    maxcycle = 20
    cst_dt = false
    Dt = 0.

    if is_cpu
        params = Armon.ArmonParameters(; 
            test, scheme, nghost, nx, ny,
            Dt, cst_dt,
            euler_projection = true, axis_splitting = :Sequential,
            maxcycle, silent = 5,
            output_file = "cpu_output",
            write_output = true, write_ghosts = false,
            use_gpu = false, 
            use_MPI = true, px, py,
            single_comm_per_axis_pass,
            reorder_grid = true)

        async_impl && (params.async_comms = false)

        global_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        coords_to_ranks = MPI.Allgather(params.cart_coords => global_rank, MPI.COMM_WORLD)
        gpu_coords_to_ranks = coords_to_ranks[armon_procs+1:end]
        i = findfirst(p -> p.first == params.cart_coords, gpu_coords_to_ranks)
        gpu_twin_rank = gpu_coords_to_ranks[i].second

        data = Armon.ArmonData(typeof(params.dx), params.nbcell, max(params.nx, params.ny) * params.nghost * 7)
        Armon.init_test(params, data)
        data.tmp_rho .= 0.
        data.tmp_urho .= 0.
        data.tmp_vrho .= 0.
        data.tmp_Erho .= 0.
        data.tmp_comm_array .= 0.

        g_data = Armon.ArmonData(typeof(params.dx), params.nbcell, max(params.nx, params.ny) * params.nghost * 7)

        params.is_root && println("Setup OK, running comparison...")

        cycles = cpu_comp_loop(params, data, g_data, gpu_twin_rank)

        if params.write_output
            Armon.write_result(params, data, params.output_file)
        end

        params.is_root && println("Completed $cycles cycles on CPU")
    else
        params = Armon.ArmonParameters(;
            test, scheme, nghost, nx, ny,
            Dt, cst_dt,
            euler_projection = true, axis_splitting = :Sequential,
            maxcycle, silent = 5,
            output_file = "gpu_output",
            write_output = true, write_ghosts = false,
            use_gpu = true, 
            use_MPI = true, px, py,
            single_comm_per_axis_pass,
            reorder_grid = true)

        async_impl && (params.async_comms = false)

        global_rank = MPI.Comm_rank(MPI.COMM_WORLD)
        coords_to_ranks = MPI.Allgather(params.cart_coords => global_rank, MPI.COMM_WORLD)
        cpu_coords_to_ranks = coords_to_ranks[1:armon_procs]
        i = findfirst(p -> p.first == params.cart_coords, cpu_coords_to_ranks)
        cpu_twin_rank = cpu_coords_to_ranks[i].second

        data = Armon.ArmonData(typeof(params.dx), params.nbcell, max(params.nx, params.ny) * params.nghost * 7)
        Armon.init_test(params, data)
        data.tmp_rho .= 0.
        data.tmp_urho .= 0.
        data.tmp_vrho .= 0.
        data.tmp_Erho .= 0.
        data.tmp_comm_array .= 0.

        d_data = Armon.data_to_gpu(data)
        
        cycles = gpu_comp_loop(params, d_data, data, cpu_twin_rank)

        if params.write_output
            Armon.data_from_gpu(data, d_data)
            Armon.write_result(params, data, params.output_file)
        end

        params.is_root && println("Completed $cycles cycles on GPU")
    end
end


comp_cpu_gpu_mpi(2, 2)
