
import .Armon: ArmonData, read_data_from_file, write_sub_domain_file, inflate

using MPI
MPI.Init()
MPI.Barrier(MPI.COMM_WORLD)


function read_sub_domain_from_global_domain_file!(params::ArmonParameters, data::ArmonData, file::IO)
    (g_nx, g_ny) = params.global_grid
    (cx, cy) = params.cart_coords
    (; nx, ny, nghost) = params

    # Ranges of the global domain
    global_cols = 1:g_nx
    global_rows = 1:g_ny
    if params.write_ghosts
        global_cols = inflate(global_cols, nghost)
        global_rows = inflate(global_rows, nghost)
    end

    # Ranges of the sub-domain
    col_range = 1:ny
    row_range = 1:nx
    if params.write_ghosts
        offset = nghost
    else
        offset = 0
    end

    # Position of the origin and end of this sub-domain
    pos_x = cx * length(row_range) + 1 - offset
    pos_y = cy * length(col_range) + 1 - offset
    end_x = pos_x + length(row_range) - 1 + offset * 2
    end_y = pos_y + length(col_range) - 1 + offset * 2
    col_offset = cy * length(col_range)

    # Ranges of the sub-domain in the global domain
    sub_domain_rows = pos_y:end_y
    cols_before = first(global_cols):(pos_x-1)
    cols_after = (end_x+1):last(global_cols)

    if params.write_ghosts
        col_range = inflate(col_range, nghost)
        row_range = inflate(row_range, nghost)
        offset = nghost
    else
        offset = 0
    end

    skip_cells(range) = for _ in range
        skipchars(!=('\n'), file)
        skip(file, 1)  # Skip the '\n'
    end

    for iy in global_rows
        if iy in sub_domain_rows
            skip_cells(cols_before)

            # `col_range = iy:iy` since we can only read one row at a time
            # The offset then transforms `iy` to the local index of the row
            col_range = (iy:iy) .- col_offset
            read_data_from_file(params, data, col_range, row_range, file)

            skip_cells(cols_after)
        else
            # Skip the entire row, made of g_nx cells (+ ghosts if any)
            skip_cells(global_cols)
        end

        skip(file, 1)  # Skip the additional '\n' at the end of each row of cells
    end
end


function ref_data_for_sub_domain(params::ArmonParameters{T}) where T
    file_path = get_reference_data_file_name(params.test, T)
    ref_data = ArmonData(T, params.nbcell, params.comm_array_size)
    ref_dt::T = 0
    ref_cycles = 0

    open(file_path, "r") do ref_file
        ref_dt = parse(T, readuntil(ref_file, ','))
        ref_cycles = parse(Int, readuntil(ref_file, '\n'))
        read_sub_domain_from_global_domain_file!(params, ref_data, ref_file)
    end

    return ref_dt, ref_cycles, ref_data
end


function ref_params_for_sub_domain(test::Symbol, type::Type, px, py; overriden_options...)
    ref_options = Dict(
        :use_MPI => true, :px => px, :py => py,
        :single_comm_per_axis_pass => false, :reorder_grid => true,
        :async_comms => false
    )
    merge!(ref_options, overriden_options)
    return get_reference_params(test, type; ref_options...)
end


function set_comm_for_grid(px, py)
    new_grid_size = px * py
    global_rank = MPI.Comm_rank(MPI.COMM_WORLD)
    # Only the first `new_grid_size` ranks will be part of the new communicator
    in_grid = global_rank < new_grid_size
    color = in_grid ? 0 : MPI.API.MPI_UNDEFINED[]
    sub_comm = MPI.Comm_split(MPI.COMM_WORLD, color, global_rank)
    Armon.set_world_comm(sub_comm)
    return sub_comm, in_grid
end


macro MPI_test(comm, expr, kws...)
    # Similar to @test in Test.jl
    skip = [kw.args[2] for kw in kws if kw.args[1] === :skip]
    kws = filter(kw -> kw.args[1] ∉ (:skip, :broken), kws)
    length(skip) > 1 && error("'skip' only allowed once")
    length(kws) > 1 && error("Cannot handle keywords other than 'skip'")
    skip = length(skip) > 0 ? first(skip) : false

    # Run `expr` only if !skip, reduce the test result, then only the root prints and calls @test
    return esc(quote
        let comm = $comm, skip::Bool = $skip, test_rank_ok::Int = skip ? false : $expr;
            test_result = skip ? 0 : MPI.Allreduce(test_rank_ok, MPI.PROD, comm)
            test_result = test_result > 0
            if !test_result && !skip
                # Print which ranks failed the test
                test_results = MPI.Gather(test_rank_ok, 0, comm)
                ranks = MPI.Gather(MPI.Comm_rank(comm), 0, comm)
                if is_root
                    test_results = Bool.(test_results)
                    if !any(test_results)
                        println("All ranks failed this test:")
                    else
                        failed_ranks = ranks[.!test_results]
                        println("$(length(failed_ranks)) ranks failed this test (ranks: $failed_ranks):")
                    end
                end
            end
            is_root && @test test_result skip=skip
        end
    end)
end


macro root_test(expr, kws...)
    return esc(quote
        if is_root
            @test($expr, $(kws...))
        end
    end)
end


# All grid should be able to perfectly divide the number of cells in each direction in the reference
# case (100×100)
domain_combinations = [
    (1, 1),
    (1, 4),
    (4, 1),
    (2, 2),
    (4, 4),
    (5, 2),
    (2, 5),
    (5, 5)
]

total_proc_count = MPI.Comm_size(MPI.COMM_WORLD)

@testset "MPI" begin
    @testset "$(px)×$(py)" for (px, py) in domain_combinations
        enough_processes = px * py ≤ total_proc_count
        if enough_processes
            is_root && @info "Testing with a $(px)×$(py) domain"
            comm, proc_in_grid = set_comm_for_grid(px, py)
        else
            is_root && @info "Not enough processes to test a $(px)×$(py) domain"
            comm, proc_in_grid = MPI.COMM_NULL, false
        end

        # Reference tests
        @testset "$test with $type" for type in (Float64,),
                                        test in (:Sod, :Sod_y, :Sod_circ, :Sedov, :Bizarrium)
            @MPI_test comm begin
                ref_params = ref_params_for_sub_domain(test, type, px, py)
                dt, cycles, data = run_armon_reference(ref_params)
                ref_dt, ref_cycles, ref_data = ref_data_for_sub_domain(ref_params)

                @root_test dt ≈ ref_dt atol=abs_tol(type) rtol=rel_tol(type)
                @root_test cycles == ref_cycles

                diff_count = count_differences(ref_params, data, ref_data)
                if WRITE_FAILED
                    global_diff_count = MPI.Allreduce(diff_count, MPI.SUM, comm)
                    if global_diff_count > 0
                        write_sub_domain_file(ref_params, data, "test_$(test)_$(type)_$(px)x$(py)"; no_msg=true)
                        write_sub_domain_file(ref_params, ref_data, "ref_$(test)_$(type)_$(px)x$(py)"; no_msg=true)
                    end
                    println("[$(MPI.Comm_rank(comm))]: found $diff_count")
                end

                diff_count == 0
            end skip=!enough_processes || !proc_in_grid

            # TODO: mass + energy conservation tests
        end

        # TODO: thread pinning tests (no overlaps, no gaps)
        # TODO: GPU assignment tests (overlaps only if there is more processes than GPUs in a node)
        # TODO: add @debug statements for those tests to get a view of the structure of cores&gpus assigned to each rank

        MPI.free(comm)
    end
end
