module Armon

using Printf
using Polyester
using ThreadPinning
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using MPI

export ArmonParameters, armon

# TODO LIST
# better test implementation (common sturcture, one test = f(x, y) -> rho, pmat, umat, vmat, Emat + boundary conditions + EOS)
# use types and function overloads to define limiters and tests (in the hope that everything gets inlined)
# center the positions of the cells in the output file

# GPU init

const use_ROCM = parse(Bool, get(ENV, "USE_ROCM_GPU", "false"))
const block_size = parse(Int, get(ENV, "GPU_BLOCK_SIZE", "32"))
const use_std_lib_threads = parse(Bool, get(ENV, "USE_STD_LIB_THREADS", "false"))

if use_ROCM
    using AMDGPU
    using ROCKernels
    AMDGPU.allowscalar(false)
else
    using CUDA
    using CUDAKernels
    CUDA.allowscalar(false)
end

const device = use_ROCM ? ROCDevice() : CUDADevice()

# MPI Init

COMM = MPI.COMM_WORLD

function set_world_comm(comm::MPI.Comm)
    # Allows to customize which processes will be part of the grid
    global COMM = comm
end

# VTune performance analysis

include("vtune_lib.jl")
using .VTune

#
# Axis enum
#

@enum Axis X_axis Y_axis

#
# Parameters
# 

mutable struct ArmonParameters{Flt_T}
    # Test problem type, riemann solver and solver scheme
    test::Symbol
    riemann::Symbol
    scheme::Symbol
    
    # Domain parameters
    nghost::Int
    nx::Int
    ny::Int
    dx::Flt_T
    cfl::Flt_T
    Dt::Flt_T
    cst_dt::Bool
    dt_on_even_cycles::Bool 
    euler_projection::Bool
    axis_splitting::Symbol

    # Indexing
    row_length::Int
    col_length::Int
    nbcell::Int
    ideb::Int
    ifin::Int
    index_start::Int
    idx_row::Int
    idx_col::Int
    current_axis::Axis
    s::Int  # Stride

    # Bounds
    maxtime::Flt_T
    maxcycle::Int
    
    # Output
    silent::Int
    output_dir::String
    output_file::String
    write_output::Bool
    write_ghosts::Bool
    merge_files::Bool
    animation_step::Int
    measure_time::Bool

    # Performance
    use_ccall::Bool
    use_threading::Bool
    use_simd::Bool
    use_gpu::Bool

    # MPI
    use_MPI::Bool
    is_root::Bool
    rank::Int
    root_rank::Int
    proc_size::Int
    proc_dims::NTuple{2, Int}
    cart_comm::MPI.Comm
    cart_coords::NTuple{2, Int}  # Coordinates of this process in the cartesian grid
    neighbours::NamedTuple{(:top, :bottom, :left, :right), NTuple{4, Int}}  # Ranks of the neighbours of this process
    global_grid::NTuple{2, Int}  # Dimensions (nx, ny) of the global grid
    single_comm_per_axis_pass::Bool
    reorder_grid::Bool

    # Asynchronicity
    async_comms::Bool
end


# Constructor for ArmonParameters
function ArmonParameters(;
        ieee_bits = 64,
        test = :Sod, riemann = :acoustic, scheme = :GAD_minmod,
        nghost = 2, nx = 10, ny = 10, 
        cfl = 0.6, Dt = 0., cst_dt = false, dt_on_even_cycles = false,
        euler_projection = false, transpose_dims = false, axis_splitting = :Sequential,
        maxtime = 0, maxcycle = 500_000,
        silent = 0, output_dir = ".", output_file = "output",
        write_output = true, write_ghosts = false, merge_files = false, animation_step = 0, 
        measure_time = true,
        use_ccall = false, use_threading = true, 
        use_simd = true, interleaving = false,
        use_gpu = false,
        use_MPI = true, px = 1, py = 1,
        single_comm_per_axis_pass = false, reorder_grid = true, 
        async_comms = true
    )

    flt_type = (ieee_bits == 64) ? Float64 : Float32

    # Make sure that all floating point types are the same
    cfl = flt_type(cfl)
    Dt = flt_type(Dt)
    maxtime = flt_type(maxtime)
    
    if cst_dt && Dt == zero(flt_type)
        error("Dt == 0 with constant step enabled")
    end
    
    if use_ccall
        error("The C librairy only supports 1D")
    end

    if interleaving
        error("No support for interleaving in 2D")
    end

    if transpose_dims
        error("No support for axis transposition in 2D")
    end

    if write_output && write_ghosts && merge_files
        error("Writing the ghost cells to a single output file is not possible")
    end

    if (single_comm_per_axis_pass && (scheme == :Godunov && nghost < 2 || nghost < 3)
            || (scheme == :Godunov && nghost < 1 || nghost < 2))
        error("Not enough ghost cells for the scheme.")
    end

    if (nx % px != 0) || (ny % py != 0)
        error("The dimensions of the global domain ($nx x $ny) are not divisible by the number of processors ($px x $py)")
    end

    # MPI
    if use_MPI
        rank = MPI.Comm_rank(COMM)
        proc_size = MPI.Comm_size(COMM)

        # Create a cartesian grid communicator of px × py processes. reorder=true can be very
        # important for performance since it will optimize the layout of the processes.
        C_COMM = MPI.Cart_create(COMM, [Int32(px), Int32(py)], [Int32(0), Int32(0)], reorder_grid)
        (cx, cy) = MPI.Cart_coords(C_COMM)

        neighbours = (
            top    = MPI.Cart_shift(C_COMM, 1,  1)[2],
            bottom = MPI.Cart_shift(C_COMM, 1, -1)[2],
            left   = MPI.Cart_shift(C_COMM, 0, -1)[2],
            right  = MPI.Cart_shift(C_COMM, 0,  1)[2]
        )
    else
        rank = 0
        proc_size = 1
        C_COMM = COMM
        (cx, cy) = (0, 0)
        neighbours = (
            top    = MPI.MPI_PROC_NULL, 
            bottom = MPI.MPI_PROC_NULL,
            left   = MPI.MPI_PROC_NULL,
            right  = MPI.MPI_PROC_NULL
        )
    end

    root_rank = 0
    is_root = rank == root_rank

    # Dimensions of the global domain
    g_nx = nx
    g_ny = ny

    dx = flt_type(1. / g_nx)

    # Dimensions of an array of the sub-domain
    nx ÷= px
    ny ÷= py
    row_length = nghost * 2 + nx
    col_length = nghost * 2 + ny
    nbcell = row_length * col_length

    # First and last index of the real domain of an array
    ideb = row_length * nghost + nghost + 1
    ifin = row_length * (ny - 1 + nghost) + nghost + nx
    index_start = ideb - row_length - 1  # Used only by the `@i` macro

    # Used only for indexing with the `@i` macro
    idx_row = row_length
    idx_col = 1
    
    return ArmonParameters{flt_type}(
        test, riemann, scheme,
        nghost, nx, ny, dx,
        cfl, Dt, cst_dt, dt_on_even_cycles,
        euler_projection, axis_splitting,
        row_length, col_length, nbcell,
        ideb, ifin, index_start,
        idx_row, idx_col,
        X_axis, 1,
        maxtime, maxcycle,
        silent, output_dir, output_file,
        write_output, write_ghosts, merge_files, animation_step,
        measure_time,
        use_ccall, use_threading, use_simd, use_gpu,
        use_MPI, is_root, rank, root_rank, 
        proc_size, (px, py), C_COMM, (cx, cy), neighbours, (g_nx, g_ny),
        single_comm_per_axis_pass, reorder_grid, 
        async_comms 
    )
end


function print_parameters(p::ArmonParameters{T}) where T
    println("Parameters:")
    print(" - multithreading: ", p.use_threading)
    if p.use_threading
        if p.use_ccall
            println(" (OMP threads: ", ENV["OMP_NUM_THREADS"], ")")
        elseif use_std_lib_threads
            println(" (Julia standard threads: ", Threads.nthreads(), ")")
        else
            println(" (Julia threads: ", Threads.nthreads(), ")")
        end
    else
        println("")
    end
    println(" - use_simd:   ", p.use_simd)
    println(" - use_ccall:  ", p.use_ccall)
    println(" - use_gpu:    ", p.use_gpu)
    if p.use_gpu
        println(" - block size: ", block_size)
    end
    println(" - use_MPI:    ", p.use_MPI)
    println(" - ieee_bits:  ", sizeof(T) * 8)
    println("")
    println(" - test:       ", p.test)
    println(" - riemann:    ", p.riemann)
    println(" - scheme:     ", p.scheme)
    println(" - splitting:  ", p.axis_splitting)
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt, p.dt_on_even_cycles ? ", updated only for even cycles" : "")
    println(" - euler proj: ", p.euler_projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
    println("")
    println(" - domain:     ", p.nx, "x", p.ny, " (", p.nghost, " ghosts)")
    println(" - nbcell:     ", @sprintf("%g", p.nx * p.ny), " (", p.nbcell, " total)")
    println(" - global:     ", p.global_grid[1], "x", p.global_grid[2])
    println(" - proc grid:  ", p.proc_dims[1], "x", p.proc_dims[2], " ($(p.reorder_grid ? "" : "not ")reordered)")
    println(" - coords:     ", p.cart_coords[1], "x", p.cart_coords[2], " (rank: ", p.rank, "/", p.proc_size-1, ")")
    println(" - comms per axis: ", p.single_comm_per_axis_pass ? 1 : 2)
    println(" - asynchronous communications: ", p.async_comms)
    println(" - measure step times: ", p.measure_time)
end


# Default copy method
function Base.copy(p::ArmonParameters{T}) where T
    return ArmonParameters([getfield(p, k) for k in fieldnames(ArmonParameters{T})]...)
end

#
# Data
#

"""
Generic array holder for all variables and temporary variables used throughout the solver.
`V` can be a `Vector` of floats (`Float32` or `Float64`) on CPU, `CuArray` or `ROCArray` on GPU.
`Vector`, `CuArray` and `ROCArray` are all subtypes of `AbstractArray`.
"""
struct ArmonData{V}
    x::V
    y::V
    rho::V
    umat::V
    vmat::V
    Emat::V
    pmat::V
    cmat::V
    gmat::V
    ustar::V
    pstar::V
    ustar_1::V
    pstar_1::V
    tmp_rho::V
    tmp_urho::V
    tmp_vrho::V
    tmp_Erho::V
    domain_mask::V
    tmp_comm_array::V
end


function ArmonData(type::Type, size::Int64, tmp_comm_size::Int64)
    return ArmonData{Vector{type}}(
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, size),
        Vector{type}(undef, tmp_comm_size)
    )
end


function data_to_gpu(data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    device_type = use_ROCM ? ROCArray : CuArray
    return ArmonData{device_type{T}}(
        device_type(data.x),
        device_type(data.y),
        device_type(data.rho),
        device_type(data.umat),
        device_type(data.vmat),
        device_type(data.Emat),
        device_type(data.pmat),
        device_type(data.cmat),
        device_type(data.gmat),
        device_type(data.ustar),
        device_type(data.pstar),
        device_type(data.ustar_1),
        device_type(data.pstar_1),
        device_type(data.tmp_rho),
        device_type(data.tmp_urho),
        device_type(data.tmp_vrho),
        device_type(data.tmp_Erho),
        device_type(data.domain_mask),
        device_type(data.tmp_comm_array)
    )
end


function data_from_gpu(host_data::ArmonData{V}, device_data::ArmonData{W}) where 
        {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    # We only need to copy the non-temporary arrays 
    copyto!(host_data.x, device_data.x)
    copyto!(host_data.y, device_data.y)
    copyto!(host_data.rho, device_data.rho)
    copyto!(host_data.umat, device_data.umat)
    copyto!(host_data.vmat, device_data.vmat)
    copyto!(host_data.Emat, device_data.Emat)
    copyto!(host_data.pmat, device_data.pmat)
    copyto!(host_data.cmat, device_data.cmat)
    copyto!(host_data.gmat, device_data.gmat)
    copyto!(host_data.ustar, device_data.ustar)
    copyto!(host_data.pstar, device_data.pstar)
end

#
# Threading and SIMD control macros
#

"""
Controls which multi-threading library to use.
"""
macro threads(expr)
    if use_std_lib_threads
        return esc(quote
            Threads.@threads $(expr)
        end)
    else
        return esc(quote
            @batch $(expr)
        end)
    end
end


"""
    @threaded(expr)

Allows to enable/disable multithreading of the loop depending on the parameters.

```julia
    @threaded for i = 1:n
        y[i] = log10(x[i]) + x[i]
    end
```
"""
macro threaded(expr)
    return esc(quote
        if params.use_threading
            @inbounds @threads $(expr)
        else
            $(expr)
        end
    end)
end


"""
    @simd_loop(expr)

Allows to enable/disable SIMD optimisations for a loop.
When SIMD is enabled, it is assumed that there is no dependencies between each iterations of the loop.

```julia
    @simd_loop for i = 1:n
        y[i] = x[i] * (x[i-1])
    end
```
"""
macro simd_loop(expr)
    return esc(quote 
        if params.use_simd
            @fastmath @inbounds @simd ivdep $(expr)
        else
            @inbounds $(expr)
        end
    end)
end


"""
    @simd_threaded_loop(expr)

Allows to enable/disable multithreading and/or SIMD of the loop depending on the parameters.
When using SIMD, `@fastmath` and `@inbounds` are used.

In order to use SIMD and multithreading at the same time, the range of the loop is split in even 
batches.
Each batch has a size of `params.simd_batch` iterations, meaning that the inner `@simd` loop has a
fixed number of iterations, while the outer threaded loop will have `N ÷ params.simd_batch`
iterations.

The loop range is assumed to be increasing, i.e. this is correct: 1:2:100, this is not: 100:-2:1
The inner `@simd` loop assumes there is no dependencies between each iteration.

```julia
    @simd_threaded_loop for i = 1:n
        y[i] = log10(x[i]) + x[i]
    end
```
"""
macro simd_threaded_loop(expr)
    if !Meta.isexpr(expr, :for, 2)
        throw(ArgumentError("Expected a valid for loop"))
    end

    # Only in for the case of a threaded loop with SIMD:
    # Extract the range of the loop and replace it with the new variables
    modified_loop_expr = copy(expr)
    range_expr = modified_loop_expr.args[1]

    if range_expr.head == :block
        # Compound range expression: "j in 1:3, i in 4:6"
        # Use the first range as the threaded loop range
        loop_range = range_expr.args[1].args[2]
        range_expr.args[1].args[2] = :(__ideb:__step:__ifin)
    elseif range_expr.head == :(=)
        # Single range expression: "j in 1:3"
        loop_range = range_expr.args[2]
        range_expr.args[2] = :(__ideb:__step:__ifin)
    else
        error("Expected range expression")
    end

    return esc(quote
        if params.use_threading
            if params.use_simd
                __loop_range = $(loop_range)
                __total_iter = length(__loop_range)
                __num_threads = Threads.nthreads()
                # Equivalent to __total_iter ÷ __num_threads
                __batch = convert(Int, cld(__total_iter, __num_threads))::Int
                __first_i = first(__loop_range)
                __last_i = last(__loop_range)
                __step = step(__loop_range)
                @threads for __i_thread = 1:__num_threads
                    __ideb = __first_i + (__i_thread - 1) * __batch * __step
                    __ifin = min(__ideb + (__batch - 1) * __step, __last_i)
                    @fastmath @inbounds @simd ivdep $(modified_loop_expr)
                end
            else
                @inbounds @threads $(expr)
            end
        else
            @simd_loop $(expr)
        end
    end
    )
end


"""
    @simd_threaded_iter(range, expr)

Same as `@simd_threaded_loop(expr)`, but instead of slicing the range of the for loop in `expr`,
we slice the `range` given as the first parameter and distribute the slices evenly to the threads.

The inner `@simd` loop assumes there is no dependencies between each iteration.

```julia
    @simd_threaded_iter 4:2:100 for i in 1:100
        y[i] = log10(x[i]) + x[i]
    end
    # is equivalent to (without threading and SIMD)
    for j in 4:2:100
        for i in (1:100) .+ (j - 1)
            y[i] = log10(x[i]) + x[i]
        end
    end
```
"""
macro simd_threaded_iter(range, expr)
    if !Meta.isexpr(expr, :for, 2)
        throw(ArgumentError("Expected a valid for loop"))
    end

    # Only in for the case of a threaded loop with SIMD:
    # Extract the range of the loop and replace it with the new expression
    modified_loop_expr = copy(expr)
    range_expr = modified_loop_expr.args[1]

    if range_expr.head == :(=)
        loop_range = range_expr.args[2]
        range_expr.args[2] = :($loop_range .+ (__j - 1))
    else
        error("Expected range vector")
    end

    return esc(quote
        if params.use_threading
            if params.use_simd
                @threads for __j in $(range)
                    @fastmath @inbounds @simd ivdep $(modified_loop_expr)
                end
            else
                @threads for __j in $(range)
                    @inbounds $(modified_loop_expr)
                end
            end
        else
            if params.use_simd
                for __j in $(range)
                    @fastmath @inbounds @simd ivdep $(modified_loop_expr)
                end
            else
                for __j in $(range)
                    $(modified_loop_expr)
                end
            end
        end
    end
    )
end

#
# Execution Time Measurement
#

in_warmup_cycle = false
is_warming_up() = in_warmup_cycle


axis_time_contrib = Dict{Axis, Dict{String, Float64}}()
total_time_contrib = Dict{String, Float64}()
const time_contrib_lock = ReentrantLock()


function add_common_time(label, time)
    lock(time_contrib_lock) do
        if !haskey(total_time_contrib, label)
            global total_time_contrib[label] = time
        else
            global total_time_contrib[label] += time
        end
    end
end


function add_axis_time(axis, label, time)
    lock(time_contrib_lock) do
        if !haskey(axis_time_contrib, axis)
            global axis_time_contrib[axis] = Dict{String, Float64}()
        end

        if !haskey(axis_time_contrib[axis], label)
            global axis_time_contrib[axis][label] = time
        else
            global axis_time_contrib[axis][label] += time
        end
    end
end


function build_time_expr(label, common_time_only, expr; use_wait=true, exclude_from_total=false)
    return esc(quote
        if params.measure_time && !is_warming_up()
            @static if $(use_wait)
                var"_$(label)_res"   = $(expr)
                var"_$(label)_time"  = @elapsed wait(var"_$(label)_res")
                var"_$(label)_time" *= 1e9
            else
                var"_$(label)_start" = time_ns()
                var"_$(label)_res"   = $(expr)
                var"_$(label)_end"   = time_ns()
                var"_$(label)_time"  = var"_$(label)_end" - var"_$(label)_start"
            end

            @static if $(!common_time_only)
                add_axis_time(params.current_axis, $(label), var"_$(label)_time")
            end
            @static if $(!exclude_from_total)
                add_common_time($(label), var"_$(label)_time")
            end
            
            var"_$(label)_res"
        else
            $(expr)
        end
    end)
end


function extract_function_name(expr)
    if expr.head == :call
        function_name = expr.args[1]
    elseif isa(expr.args[2], Expr) && expr.args[2].head == :call
        function_name = expr.args[2].args[1]
    else
        error("Could not find the function name of the provided expression")
    end
    return string(function_name)
end


macro time_event(label, expr)   return build_time_expr(label, false, expr) end
macro time_event_c(label, expr) return build_time_expr(label, true,  expr) end
macro time_expr(label, expr)    return build_time_expr(label, false, expr; use_wait=false) end
macro time_expr_c(label, expr)  return build_time_expr(label, true,  expr; use_wait=false) end
macro time_event_a(label, expr) return build_time_expr(label, false,  expr; exclude_from_total=true) end
macro time_expr_a(label, expr)  return build_time_expr(label, false,  expr; exclude_from_total=true, use_wait=false) end
macro time_event(expr)          return build_time_expr(extract_function_name(expr), false, expr) end
macro time_event_c(label, expr) return build_time_expr(extract_function_name(expr), true,  expr) end
macro time_expr(expr)           return build_time_expr(extract_function_name(expr), false, expr; use_wait=false) end
macro time_expr_c(expr)         return build_time_expr(extract_function_name(expr), true,  expr; use_wait=false) end
macro time_event_a(expr)        return build_time_expr(extract_function_name(expr), false, expr; exclude_from_total=true) end
macro time_expr_a(expr)         return build_time_expr(extract_function_name(expr), false, expr; exclude_from_total=true, use_wait=false) end


# Equivalent to `@time` but with a better output
macro pretty_time(expr)
    function_name = extract_function_name(expr)
    return esc(quote
        if params.is_root && params.silent <= 3
            # Same structure as `@time` (see `@macroexpand @time`), using some undocumented functions.
            gc_info_before = Base.gc_num()
            time_before = Base.time_ns()
            compile_time_before = Base.cumulative_compile_time_ns_before()

            $(expr)

            compile_time_after = Base.cumulative_compile_time_ns_after()
            time_after = Base.time_ns()
            gc_info_after = Base.gc_num()
            
            elapsed_time = time_after - time_before
            compile_time = compile_time_after - compile_time_before
            gc_diff = Base.GC_Diff(gc_info_after, gc_info_before)

            allocations_size = gc_diff.allocd / 1e3
            allocations_count = Base.gc_alloc_count(gc_diff)
            gc_time = gc_diff.total_time
    
            println("\nTime info for $($(function_name)):")
            if allocations_count > 0
                @printf(" - %d allocations for %g kB\n", 
                    allocations_count, convert(Float64, allocations_size))
            end
            if gc_time > 0
                @printf(" - GC:      %10.5f ms (%5.2f%%)\n", 
                    gc_time / 1e6, gc_time / elapsed_time * 100)
            end
            if compile_time > 0
                @printf(" - Compile: %10.5f ms (%5.2f%%)\n", 
                    compile_time / 1e6, compile_time / elapsed_time * 100)
            end
            @printf(" - Total:   %10.5f ms\n", elapsed_time / 1e6)
        else
            $(expr)
        end
    end)
end

# 
# Indexing macros
#

"""
    @indexing_vars(params)

Brings the parameters needed for the `@i` macro into the current scope.
"""
macro indexing_vars(params)
    return esc(quote
        (; index_start, row_length, col_length, idx_row, idx_col) = $(params)
    end)
end

"""
    @i(i, j)

Converts the two-dimensional indexes `i` and `j` to a mono-dimensional index.
Since the variables of `@indexing_vars` are updated whenever the arrays are transposed, this macro
handles the transposition of the arrays.

```julia
    idx = @i(i, j)
```
"""
macro i(i, j)
    return esc(quote
        index_start + $(j) * idx_row + $(i) * idx_col
    end)
end

#
# GPU Kernels
#

@kernel function gpu_acoustic_kernel!(
        main_range_start, main_range_step, inner_range_start, inner_range_length, s,
        ustar, pstar, @Const(rho), @Const(u), @Const(pmat), @Const(cmat))
    idx = @index(Global)
    ix, iy = divrem(idx - 1, inner_range_length)
    j = main_range_start  + ix * main_range_step - 1
    i = inner_range_start + iy + j

    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    ustar[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
    pstar[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
end


@kernel function gpu_acoustic_GAD_minmod_kernel!(
        main_range_start, main_range_step, inner_range_start, inner_range_length, s,
        ustar, pstar, @Const(rho), @Const(u), @Const(pmat), @Const(cmat), 
        @Const(ustar_1), @Const(pstar_1), dt, dx)
    idx = @index(Global)
    ix, iy = divrem(idx - 1, inner_range_length)
    j = main_range_start  + ix * main_range_step - 1
    i = inner_range_start + iy + j

    r_u_m = (ustar_1[i+s] -      u[i]) / (ustar_1[i] -    u[i-s] + 1e-6)
    r_p_m = (pstar_1[i+s] -   pmat[i]) / (pstar_1[i] - pmat[i-s] + 1e-6)
    r_u_p = (   u[i-s] - ustar_1[i-s]) / (   u[i] -   ustar_1[i] + 1e-6)
    r_p_p = (pmat[i-s] - pstar_1[i-s]) / (pmat[i] -   pstar_1[i] + 1e-6)

    r_u_m = max(0., min(1., r_u_m))
    r_p_m = max(0., min(1., r_p_m))
    r_u_p = max(0., min(1., r_u_p))
    r_p_p = max(0., min(1., r_p_p))

    dm_l = rho[i-s] * dx
    dm_r = rho[i]   * dx
    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    Dm   = (dm_l + dm_r) / 2
    θ    = 1/2 * (1 - (rc_l + rc_r) / 2 * (dt / Dm))
    
    ustar[i] = ustar_1[i] + θ * (r_u_p * (   u[i] - ustar_1[i]) - r_u_m * (ustar_1[i] -    u[i-s]))
    pstar[i] = pstar_1[i] + θ * (r_p_p * (pmat[i] - pstar_1[i]) - r_p_m * (pstar_1[i] - pmat[i-s]))
end


@kernel function gpu_update_perfect_gas_EOS_kernel!(i_0, gamma,
        @Const(rho), @Const(Emat), @Const(umat), @Const(vmat), pmat, cmat, gmat)
    i = @index(Global) + i_0

    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = (gamma - 1.) * rho[i] * e
    cmat[i] = sqrt(gamma * pmat[i] / rho[i])
    gmat[i] = (1. + gamma) / 2
end


@kernel function gpu_update_bizarrium_EOS_kernel!(i_0, 
        @Const(rho), @Const(Emat), @Const(umat), @Const(vmat), pmat, cmat, gmat)
    i = @index(Global) + i_0

    type = eltype(rho)

    # O. Heuzé, S. Jaouen, H. Jourdren, 
    # "Dissipative issue of high-order shock capturing schemes wtih non-convex equations of state"
    # JCP 2009

    rho0::type = 10000.
    K0::type   = 1e+11
    Cv0::type  = 1000.
    T0::type   = 300.
    eps0::type = 0.
    G0::type   = 1.5
    s::type    = 1.5
    q::type    = -42080895/14941154
    r::type    = 727668333/149411540

    x::type = rho[i] / rho0 - 1
    g::type = G0 * (1-rho0 / rho[i])

    f0::type = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)
    f1::type = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)
    f2::type = (2*q+6*r*x+2*s*f1)/(1-s*x)
    f3::type = (6*r+3*s*f2)/(1-s*x)

    epsk0::type     = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0
    pk0::type       = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)
    pk0prime::type  = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)
    pk0second::type = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 + 
                                                    6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = pk0 + G0 * rho0 * (e - epsk0)
    cmat[i] = sqrt(G0 * rho0 * (pmat[i] - pk0) - pk0prime) / rho[i]
    gmat[i] = 0.5 / (rho[i]^3 * cmat[i]^2) * (pk0second + (G0 * rho0)^2 * (pmat[i] - pk0))
end


@kernel function gpu_boundary_conditions_left_kernel!(index_start, idx_row, idx_col, u_factor_left, 
        rho, umat, vmat, pmat, cmat, gmat)
    thread_i = @index(Global)

    idx = @i(1,thread_i)
    idxm1 = @i(0,thread_i)
    
    rho[idxm1]  = rho[idx]
    umat[idxm1] = umat[idx] * u_factor_left
    vmat[idxm1] = vmat[idx]
    pmat[idxm1] = pmat[idx]
    cmat[idxm1] = cmat[idx]
    gmat[idxm1] = gmat[idx]
end


@kernel function gpu_boundary_conditions_right_kernel!(index_start, idx_row, idx_col, nx, u_factor_right, 
        rho, umat, vmat, pmat, cmat, gmat)
    thread_i = @index(Global)
    
    idx = @i(nx,thread_i)
    idxp1 = @i(nx+1,thread_i)

    rho[idxp1] = rho[idx]
    umat[idxp1] = umat[idx] * u_factor_right
    vmat[idxp1] = vmat[idx]
    pmat[idxp1] = pmat[idx]
    cmat[idxp1] = cmat[idx]
    gmat[idxp1] = gmat[idx]
end


@kernel function gpu_boundary_conditions_top_kernel!(index_start, idx_row, idx_col, ny, v_factor_top, 
        rho, umat, vmat, pmat, cmat, gmat)
    thread_i = @index(Global)

    idx = @i(thread_i,ny)
    idxp1 = @i(thread_i,ny+1)

    rho[idxp1]  = rho[idx]
    umat[idxp1] = umat[idx]
    vmat[idxp1] = vmat[idx] * v_factor_top
    pmat[idxp1] = pmat[idx]
    cmat[idxp1] = cmat[idx]
    gmat[idxp1] = gmat[idx]
end


@kernel function gpu_boundary_conditions_bottom_kernel!(index_start, idx_row, idx_col, v_factor_bottom, 
        rho, umat, vmat, pmat, cmat, gmat)
    thread_i = @index(Global)

    idx = @i(thread_i,1)
    idxm1 = @i(thread_i,0)

    rho[idxm1]  = rho[idx]
    umat[idxm1] = umat[idx]
    vmat[idxm1] = vmat[idx] * v_factor_bottom
    pmat[idxm1] = pmat[idx]
    cmat[idxm1] = cmat[idx]
    gmat[idxm1] = gmat[idx]
end


@kernel function gpu_read_border_array_X_kernel!(pos, nghost, nx, row_length,
        value_array, rho, umat, vmat, pmat, cmat, gmat, Emat)
    thread_i = @index(Global)

    (i, i_g) = divrem(thread_i - 1, nghost)
    i_arr = (i_g * nx + i) * 7
    idx = i_g * row_length + pos + i

    value_array[i_arr+1] =  rho[idx]
    value_array[i_arr+2] = umat[idx]
    value_array[i_arr+3] = vmat[idx]
    value_array[i_arr+4] = pmat[idx]
    value_array[i_arr+5] = cmat[idx]
    value_array[i_arr+6] = gmat[idx]
    value_array[i_arr+7] = Emat[idx]
end


@kernel function gpu_read_border_array_Y_kernel!(pos, nghost, ny, row_length,
        value_array, rho, umat, vmat, pmat, cmat, gmat, Emat)
    thread_i = @index(Global)

    (i, i_g) = divrem(thread_i - 1, nghost)
    i_arr = (i_g * ny + i) * 7
    idx = i * row_length + pos + i_g

    value_array[i_arr+1] =  rho[idx]
    value_array[i_arr+2] = umat[idx]
    value_array[i_arr+3] = vmat[idx]
    value_array[i_arr+4] = pmat[idx]
    value_array[i_arr+5] = cmat[idx]
    value_array[i_arr+6] = gmat[idx]
    value_array[i_arr+7] = Emat[idx]
end


@kernel function gpu_write_border_array_X_kernel!(pos, nghost, nx, row_length,
        value_array, rho, umat, vmat, pmat, cmat, gmat, Emat)
    thread_i = @index(Global)

    (i, i_g) = divrem(thread_i - 1, nghost)
    i_arr = (i_g * nx + i) * 7
    idx = i_g * row_length + pos + i

     rho[idx] = value_array[i_arr+1]
    umat[idx] = value_array[i_arr+2]
    vmat[idx] = value_array[i_arr+3]
    pmat[idx] = value_array[i_arr+4]
    cmat[idx] = value_array[i_arr+5]
    gmat[idx] = value_array[i_arr+6]
    Emat[idx] = value_array[i_arr+7]
end


@kernel function gpu_write_border_array_Y_kernel!(pos, nghost, ny, row_length,
        value_array, rho, umat, vmat, pmat, cmat, gmat, Emat)
    thread_i = @index(Global)

    (i, i_g) = divrem(thread_i - 1, nghost)
    i_arr = (i_g * ny + i) * 7
    idx = i * row_length + pos + i_g

     rho[idx] = value_array[i_arr+1]
    umat[idx] = value_array[i_arr+2]
    vmat[idx] = value_array[i_arr+3]
    pmat[idx] = value_array[i_arr+4]
    cmat[idx] = value_array[i_arr+5]
    gmat[idx] = value_array[i_arr+6]
    Emat[idx] = value_array[i_arr+7]
end


@kernel function gpu_dtCFL_reduction_euler_kernel!(dx, dy, out,
        @Const(umat), @Const(vmat), @Const(cmat), @Const(domain_mask))
    i = @index(Global)

    c = cmat[i]
    u = umat[i]
    v = vmat[i]
    mask = domain_mask[i]

    out[i] = mask * min(
        dx / abs(max(abs(u + c), abs(u - c))),
        dy / abs(max(abs(v + c), abs(v - c)))
    )
end


@kernel function gpu_dtCFL_reduction_lagrange_kernel!(out, @Const(cmat), @Const(domain_mask))
    i = @index(Global)
    out[i] = 1. / (cmat[i] * domain_mask[i])
end


@kernel function gpu_cell_update_kernel!(i_0, dx, dt, s,
        @Const(ustar), @Const(pstar), rho, u, Emat, domain_mask)
    i = @index(Global) + i_0

    mask = domain_mask[i]
    dm = rho[i] * dx
    rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]) * mask)
    u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) * mask
    Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) * mask
end


@kernel function gpu_cell_update_lagrange_kernel!(i_0, ifin, dt, s, x, @Const(ustar))
    i = @index(Global) + i_0

    x[i] += dt * ustar[i]

    if i == ifin
        x[i+s] += dt * ustar[i+s]
    end
end


@kernel function gpu_first_order_euler_remap_kernel!(i_0, dx, dt, s,
        @Const(ustar), rho, umat, vmat, Emat, 
        tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, @Const(domain_mask))
    i = @index(Global) + i_0

    mask = domain_mask[i]
    dX = dx + dt * (ustar[i+s] - ustar[i])
    L₁ =  max(0, ustar[i])   * dt * mask
    L₃ = -min(0, ustar[i+s]) * dt * mask
    L₂ = dX - L₁ - L₃
    
    tmp_rho[i]  = (L₁ * rho[i-s] 
                 + L₂ * rho[i] 
                 + L₃ * rho[i+s]) / dX
    tmp_urho[i] = (L₁ * rho[i-s] * umat[i-s] 
                 + L₂ * rho[i]   * umat[i] 
                 + L₃ * rho[i+s] * umat[i+s]) / dX
    tmp_vrho[i] = (L₁ * rho[i-s] * vmat[i-s] 
                 + L₂ * rho[i]   * vmat[i] 
                 + L₃ * rho[i+s] * vmat[i+s]) / dX
    tmp_Erho[i] = (L₁ * rho[i-s] * Emat[i-s] 
                 + L₂ * rho[i]   * Emat[i] 
                 + L₃ * rho[i+s] * Emat[i+s]) / dX
end


@kernel function gpu_first_order_euler_remap_2_kernel!(i_0, 
        rho, umat, vmat, Emat, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho)
    i = @index(Global) + i_0

    # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E)
    rho[i] = tmp_rho[i]
    umat[i] = tmp_urho[i] / tmp_rho[i]
    vmat[i] = tmp_vrho[i] / tmp_rho[i]
    Emat[i] = tmp_Erho[i] / tmp_rho[i]
end


# Construction of the kernels for a common device and block size
gpu_acoustic! = gpu_acoustic_kernel!(device, block_size)
gpu_acoustic_GAD_minmod! = gpu_acoustic_GAD_minmod_kernel!(device, block_size)
gpu_update_perfect_gas_EOS! = gpu_update_perfect_gas_EOS_kernel!(device, block_size)
gpu_update_bizarrium_EOS! = gpu_update_bizarrium_EOS_kernel!(device, block_size)
gpu_boundary_conditions_left! = gpu_boundary_conditions_left_kernel!(device, block_size)
gpu_boundary_conditions_right! = gpu_boundary_conditions_right_kernel!(device, block_size)
gpu_boundary_conditions_top! = gpu_boundary_conditions_top_kernel!(device, block_size)
gpu_boundary_conditions_bottom! = gpu_boundary_conditions_bottom_kernel!(device, block_size)
gpu_read_border_array_X! = gpu_read_border_array_X_kernel!(device, block_size)
gpu_read_border_array_Y! = gpu_read_border_array_Y_kernel!(device, block_size)
gpu_write_border_array_X! = gpu_write_border_array_X_kernel!(device, block_size)
gpu_write_border_array_Y! = gpu_write_border_array_Y_kernel!(device, block_size)
gpu_dtCFL_reduction_euler! = gpu_dtCFL_reduction_euler_kernel!(device, block_size)
gpu_dtCFL_reduction_lagrange! = gpu_dtCFL_reduction_lagrange_kernel!(device, block_size)
gpu_cell_update! = gpu_cell_update_kernel!(device, block_size)
gpu_cell_update_lagrange! = gpu_cell_update_lagrange_kernel!(device, block_size)
gpu_first_order_euler_remap_1! = gpu_first_order_euler_remap_kernel!(device, block_size)
gpu_first_order_euler_remap_2! = gpu_first_order_euler_remap_2_kernel!(device, block_size)

#
# Acoustic Riemann problem solvers
# 

function acoustic!(params::ArmonParameters{T}, data::ArmonData{V}, 
        u::V, main_range::StepRange{Int, Int}, inner_range::StepRange{Int, Int},
        is_outer::Bool;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, pmat, cmat) = data
    (; s) = params

    step_label = is_outer ? "acoustic_outer!" : "acoustic_inner!"

    if params.use_gpu
        event = gpu_acoustic!(first(main_range), step(main_range), first(inner_range), length(inner_range), s,
                              ustar, pstar, rho, u, pmat, cmat;
                              ndrange=length(main_range) * length(inner_range), dependencies)
        return @time_event step_label event
    end

    @time_expr_a step_label @simd_threaded_iter main_range for i in inner_range
        rc_l = rho[i-s] * cmat[i-s]
        rc_r = rho[i]   * cmat[i]
        ustar[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
        pstar[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
    end

    return NoneEvent()
end


function acoustic_GAD!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, u::V, main_range::StepRange{Int, Int}, inner_range::StepRange{Int, Int},
        is_outer::Bool; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, pmat, cmat, ustar_1, pstar_1) = data
    (; scheme, dx, s) = params

    step_label_1st = is_outer ? "acoustic_outer!"     : "acoustic_inner!"
    step_label_2nd = is_outer ? "acoustic_GAD_outer!" : "acoustic_GAD_inner!"

    if params.current_axis == X_axis
        main_range_1st_order = main_range
        inner_range_1st_order = first(inner_range)-s:step(inner_range):last(inner_range)+s
    else
        main_range_1st_order = first(main_range)-s:step(main_range):last(main_range)+s
        inner_range_1st_order = inner_range
    end
    
    if params.use_gpu
        if params.scheme != :GAD_minmod
            error("Only the minmod limiter is implemented for GPU")
        end

        first_kernel = @time_event_a step_label_1st gpu_acoustic!(
            first(main_range_1st_order), step(main_range_1st_order), 
            first(inner_range_1st_order), length(inner_range_1st_order), s,
            ustar_1, pstar_1, rho, u, pmat, cmat;
            ndrange=length(main_range_1st_order) * length(inner_range_1st_order), dependencies)

        second_kernel = @time_event_a step_label_2nd gpu_acoustic_GAD_minmod!(
            first(main_range), step(main_range), 
            first(inner_range), length(inner_range), s,
            ustar, pstar, rho, u, pmat, cmat, ustar_1, pstar_1, dt, dx;
            ndrange=length(main_range) * length(inner_range), dependencies=first_kernel)

        return second_kernel
    end

    # First order
    @time_expr_a step_label_1st @simd_threaded_iter main_range_1st_order for i in inner_range_1st_order
        rc_l = rho[i-s] * cmat[i-s]
        rc_r = rho[i]   * cmat[i]
        ustar_1[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
        pstar_1[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
    end

    # Second order, for each flux limiter
    @time_expr_a step_label_2nd if scheme == :GAD_minmod
        @simd_threaded_iter main_range for i in inner_range
            r_u_m = (ustar_1[i+s] -      u[i]) / (ustar_1[i] -    u[i-s] + 1e-6)
            r_p_m = (pstar_1[i+s] -   pmat[i]) / (pstar_1[i] - pmat[i-s] + 1e-6)
            r_u_p = (   u[i-s] - ustar_1[i-s]) / (   u[i] -   ustar_1[i] + 1e-6)
            r_p_p = (pmat[i-s] - pstar_1[i-s]) / (pmat[i] -   pstar_1[i] + 1e-6)

            r_u_m = max(0., min(1., r_u_m))
            r_p_m = max(0., min(1., r_p_m))
            r_u_p = max(0., min(1., r_u_p))
            r_p_p = max(0., min(1., r_p_p))

            dm_l = rho[i-s] * dx
            dm_r = rho[i]   * dx
            rc_l = rho[i-s] * cmat[i-s]
            rc_r = rho[i]   * cmat[i]
            Dm   = (dm_l + dm_r) / 2
            θ    = 1/2 * (1 - (rc_l + rc_r) / 2 * (dt / Dm))

            ustar[i] = ustar_1[i] + θ * (r_u_p * (   u[i] - ustar_1[i]) - r_u_m * (ustar_1[i] -    u[i-s]))
            pstar[i] = pstar_1[i] + θ * (r_p_p * (pmat[i] - pstar_1[i]) - r_p_m * (pstar_1[i] - pmat[i-s]))
        end
    elseif scheme == :GAD_superbee
        @simd_threaded_iter main_range for i in inner_range
            r_u_m = (ustar_1[i+s] -      u[i]) / (ustar_1[i] -    u[i-s] + 1e-6)
            r_p_m = (pstar_1[i+s] -   pmat[i]) / (pstar_1[i] - pmat[i-s] + 1e-6)
            r_u_p = (   u[i-s] - ustar_1[i-s]) / (   u[i] -   ustar_1[i] + 1e-6)
            r_p_p = (pmat[i-s] - pstar_1[i-s]) / (pmat[i] -   pstar_1[i] + 1e-6)

            r_u_m = max(0., min(1., 2. * r_u_m), min(2., r_u_m))
            r_p_m = max(0., min(1., 2. * r_p_m), min(2., r_p_m))
            r_u_p = max(0., min(1., 2. * r_u_p), min(2., r_u_p))
            r_p_p = max(0., min(1., 2. * r_p_p), min(2., r_p_p))

            dm_l = rho[i-s] * dx
            dm_r = rho[i]   * dx
            rc_l = rho[i-s] * cmat[i-s]
            rc_r = rho[i]   * cmat[i]
            Dm = (dm_l + dm_r) / 2
            θ  = 1/2 * (1 - (rc_l + rc_r) / 2 * (dt / Dm))
            
            ustar[i] = ustar_1[i] + θ * (r_u_p * (   u[i] - ustar_1[i]) - r_u_m * (ustar_1[i] -    u[i-s]))
            pstar[i] = pstar_1[i] + θ * (r_p_p * (pmat[i] - pstar_1[i]) - r_p_m * (pstar_1[i] - pmat[i-s]))
        end
    elseif scheme == :GAD_no_limiter
        @simd_threaded_iter main_range for i in inner_range
            dm_l = rho[i-s] * dx
            dm_r = rho[i]   * dx
            rc_l = rho[i-s] * cmat[i-s]
            rc_r = rho[i]   * cmat[i]
            Dm = (dm_l + dm_r) / 2
            θ  = 1/2 * (1 - (rc_l + rc_r) / 2 * (dt / Dm))

            ustar[i] = ustar_1[i] + θ * ((   u[i] - ustar_1[i]) - (ustar_1[i] -    u[i-s]))
            pstar[i] = pstar_1[i] + θ * ((pmat[i] - pstar_1[i]) - (pstar_1[i] - pmat[i-s]))
        end
    else
        error("The choice of the scheme for the acoustic solver is not recognized: ", scheme)
    end

    return NoneEvent()
end


function numericalFluxes!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, u::V, main_range::StepRange{Int, Int}, inner_range::StepRange{Int, Int},
        is_outer::Bool; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            return acoustic!(params, data, u, main_range, inner_range, is_outer; dependencies)
        else
            return acoustic_GAD!(params, data, dt, u, main_range, inner_range, is_outer; dependencies)
        end
    else
        error("The choice of Riemann solver is not recognized: ", params.riemann)
    end
end


function numericalFluxes_inner!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T, u::V;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; nx, ny, nghost, row_length, s) = params
    @indexing_vars(params)

    # Add one more ring of cells to compute on the sides if needed
    o = convert(Int, params.single_comm_per_axis_pass)

    # In both cases, we also include one more cell/row on the left/top of the inner domain to have 
    # the fluxes at the left/bottom and right/top of every cell in the inner domain. 

    if params.current_axis == X_axis
        # Compute the fluxes row by row, excluding 'nghost' columns at the left and right
        main_range  = @i(1,1-o):row_length:@i(1,ny+o)
        inner_range = nghost+1:nx+1-nghost
    else
        # Compute the fluxes row by row, excluding 'nghost' rows at the top and bottom
        main_range  = @i(1,1+nghost):row_length:@i(1,ny+1-nghost)
        inner_range = 1-o:nx+o
    end

    main_range = StepRange{Int, Int}(main_range)
    inner_range = StepRange{Int, Int}(inner_range)

    @perf_task "loop" "fluxes_inner" event = numericalFluxes!(params, data, dt, u, main_range, inner_range, false; dependencies)
    return event
end


function numericalFluxes_outer!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, u::V; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; nx, ny, nghost, row_length, s) = params
    @indexing_vars(params)

    # Add one more ring of cells to compute on the sides if needed
    o = convert(Int, params.single_comm_per_axis_pass)

    if params.current_axis == X_axis
        # Compute the fluxes row by row, for the first 'nghost+o' columns on the left
        main_range  = @i(1-o,1-o):row_length:@i(1-o,ny+o)
        inner_range = 1:nghost+o
    else
        # Compute the fluxes row by row, for the first 'nghost+o' rows at the bottom
        main_range  = @i(1,1-o):row_length:@i(1,nghost)
        inner_range = 1-o:nx+o
    end
    
    main_range = StepRange{Int, Int}(main_range)
    inner_range = StepRange{Int, Int}(inner_range)

    @perf_task "comms" "fluxes_outer_1" event = numericalFluxes!(params, data, dt, u, main_range, inner_range, true; dependencies)

    if params.current_axis == X_axis
        # Compute the fluxes row by row, for the last 'nghost+o' columns on the right
        main_range  = @i(1,1-o):row_length:@i(1+o,ny+o)
        inner_range = nx-nghost+1:nx+o
    else
        # Compute the fluxes row by row, for the last 'nghost+o' rows at the top
        main_range  = @i(1-o,ny-nghost+1+o):row_length:@i(1-o,ny+2*o)
        inner_range = 1-o:nx+o
    end

    main_range = StepRange{Int, Int}(main_range)
    inner_range = StepRange{Int, Int}(inner_range)

    # Shift the computation to the right/top by one column/row since the flux at index 'i' is the 
    # flux between the cells 'i-s' and 'i', but we also need the fluxes on the other side, between
    # the cells 'i' and 'i+s'. Therefore we need this shift to compute all fluxes needed later.
    inner_range = inner_range .+ 1

    @perf_task "comms" "fluxes_outer_2" event = numericalFluxes!(params, data, dt, u, main_range, inner_range, true; dependencies=event)
    return event
end

#
# Equations of State
#

function perfectGasEOS!(params::ArmonParameters{T}, data::ArmonData{V}, gamma::T) where {T, V <: AbstractArray{T}}
    (; umat, vmat, pmat, cmat, gmat, rho, Emat) = data
    (; ideb, ifin) = params

    @simd_threaded_loop for i in ideb:ifin
        e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
        pmat[i] = (gamma - 1.) * rho[i] * e
        cmat[i] = sqrt(gamma * pmat[i] / rho[i])
        gmat[i] = (1. + gamma) / 2
    end
end


function BizarriumEOS!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; umat, vmat, pmat, cmat, gmat, rho, Emat) = data
    (; ideb, ifin) = params

    # O. Heuzé, S. Jaouen, H. Jourdren, 
    # "Dissipative issue of high-order shock capturing schemes with non-convex equations of state",
    # JCP 2009

    rho0 = 10000; K0 = 1e+11; Cv0 = 1000; T0 = 300; eps0 = 0; G0 = 1.5; s = 1.5
    q = -42080895/14941154; r = 727668333/149411540

    @simd_threaded_loop for i in ideb:ifin
        x = rho[i]/rho0 - 1; g = G0*(1-rho0/rho[i]) # Formula (4b)

        f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)  # Formula (15b)
        f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x) # Formula (16a)
        f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)         # Formula (16b)
        f3 = (6*r+3*s*f2)/(1-s*x)               # Formula (16c)

        epsk0 = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0                             # Formula (15a)
        pk0 = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)                           # Formula (17a)
        pk0prime = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2) # Formula (17b)
        pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 +        # Formula (17c)
                                            6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

        e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
        pmat[i] = pk0 + G0*rho0*(e - epsk0)                                      # Formula (5b)
        cmat[i] = sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i]              # Formula (8)
        gmat[i] = 0.5/(rho[i]^3*cmat[i]^2)*(pk0second+(G0*rho0)^2*(pmat[i]-pk0)) # Formula (8) + (11)
    end
end


function update_EOS!(params::ArmonParameters{T}, data::ArmonData{V};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat, Emat) = data
    (; ideb, ifin, test) = params

    if test == :Sod || test == :Sod_y || test == :Sod_circ
        gamma::T = 7/5
        if params.use_gpu
            event = gpu_update_perfect_gas_EOS!(ideb - 1, gamma, rho, Emat,
                umat, vmat, pmat, cmat, gmat;
                ndrange=length(ideb:ifin), dependencies)
            return @time_event "update_EOS!" event
        else
            @time_expr "update_EOS!" perfectGasEOS!(params, data, gamma)
            return NoneEvent()
        end
    elseif test == :Bizarrium
        if params.use_gpu
            event = gpu_update_bizarrium_EOS!(ideb - 1, rho, Emat,
                umat, vmat, pmat, cmat, gmat;
                ndrange=length(ideb:ifin), dependencies)
            return @time_event "update_EOS!" event
        else
            @time_expr "update_EOS!" BizarriumEOS!(params, data)
            return NoneEvent()
        end
    end
end

#
# Test initialisation
#

function init_test(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; x, y, rho, umat, vmat, pmat, cmat, Emat, 
       ustar, pstar, ustar_1, pstar_1, domain_mask) = data
    (; test, nghost, nbcell, nx, ny, row_length, cart_coords, global_grid) = params

    if test == :Sod || test == :Sod_y || test == :Sod_circ
        if params.maxtime == 0
            params.maxtime = 0.20
        end
    
        if params.cfl == 0
            params.cfl = 0.95
        end

        gamma::T   = 7/5
        left_ρ::T  = 1.
        right_ρ::T = 0.125
        left_p::T  = 1.0
        right_p::T = 0.1
        right_u::T = 0.

        if test == :Sod
            cond = (x_, y_) -> x_ ≤ 0.5
        elseif test == :Sod_y
            cond = (x_, y_) -> y_ ≤ 0.5
        else
            cond = (x_, y_) -> (x_ - 0.5)^2 + (y_ - 0.5)^2 ≤ 0.125
        end
    elseif test == :Bizarrium
        if params.maxtime == 0
            params.maxtime = 80e-6
        end
    
        if params.cfl == 0
            params.cfl = 0.6
        end

        gamma   = 2
        left_ρ  = 1.42857142857e+4
        right_ρ = 10000.
        left_p  = 6.40939744478e+10
        right_p = 312.5e6
        right_u = 250.

        cond = (x_, y_) -> x_ ≤ 0.5
    else
        error("Unknown test case: $test")
    end

    (cx, cy) = cart_coords
    (g_nx, g_ny) = global_grid

    # Position of the origin of this sub-domain
    pos_x = cx * nx
    pos_y = cy * ny

    one_more_ring = params.single_comm_per_axis_pass

    @simd_threaded_loop for i in 1:nbcell
        ix = ((i-1) % row_length) - nghost
        iy = ((i-1) ÷ row_length) - nghost

        # Global indexes, used only to know to compute the position of the cell
        g_ix = ix + pos_x
        g_iy = iy + pos_y

        x[i] = g_ix / g_nx
        y[i] = g_iy / g_ny

        if cond(x[i] + 1. / (2*g_nx), y[i] + 1. / (2*g_ny))
            rho[i]  = left_ρ
            Emat[i] = left_p / ((gamma - 1.) * rho[i])
            umat[i] = 0.
        else
            rho[i]  = right_ρ
            Emat[i] = right_p / ((gamma - 1.) * rho[i])
            umat[i] = right_u
        end

        vmat[i] = 0.

        # Set to 1 if the cell is in the real domain or 0 in the ghost domain
        if one_more_ring
            domain_mask[i] = (
                (-1 ≤   ix < nx+1 && -1 ≤   iy < ny+1)  # Include as well one ring of ghost cells...
             && ( 0 ≤   ix < nx   ||  0 ≤   iy < ny  )  # ...while excluding the corners of the sub-domain...
             && ( 0 ≤ g_ix < g_nx &&  0 ≤ g_iy < g_ny)  # ...and only if it is in the global domain
            ) ? 1. : 0
        else
            domain_mask[i] = (0 ≤ ix < nx && 0 ≤ iy < ny) ? 1. : 0
        end

        # Set to zero to make sure no non-initialized values changes the result
        pmat[i] = 0.
        cmat[i] = 1.  # Set to 1 as a max speed of 0 will create NaNs
        ustar[i] = 0.
        pstar[i] = 0.
        ustar_1[i] = 0.
        pstar_1[i] = 0.
    end

    if test == :Sod || test == :Sod_y || test == :Sod_circ
        perfectGasEOS!(params, data, gamma)
    else
        BizarriumEOS!(params, data)
    end
    
    return
end

#
# Boundary conditions
#

function boundaryConditions_left!(params::ArmonParameters{T}, data::ArmonData{V};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, ny) = params
    @indexing_vars(params)

    u_factor_left::T = 1.
    if test == :Sod || test == :Sod_circ
        u_factor_left = -1.
    end

    if params.use_gpu
        bc_event = gpu_boundary_conditions_left!(index_start, idx_row, idx_col, u_factor_left,
            rho, umat, vmat, pmat, cmat, gmat;
            ndrange=ny, dependencies)
        return @time_event_a "boundary_conditions!" bc_event
    end

    @time_expr_a "boundary_conditions!" @simd_loop for j in 1:ny
        idx = @i(1,j)
        idxm1 = @i(0,j)
        rho[idxm1]  = rho[idx]
        umat[idxm1] = umat[idx] * u_factor_left
        vmat[idxm1] = vmat[idx]
        pmat[idxm1] = pmat[idx]
        cmat[idxm1] = cmat[idx]
        gmat[idxm1] = gmat[idx]
    end

    return NoneEvent()
end


function boundaryConditions_right!(params::ArmonParameters{T}, data::ArmonData{V};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, nx, ny) = params
    @indexing_vars(params)

    u_factor_right::T = 1.
    if test == :Sod || test == :Sod_circ
        u_factor_right = -1.
    end

    if params.use_gpu
        bc_event = gpu_boundary_conditions_right!(index_start, idx_row, idx_col, nx, u_factor_right,
            rho, umat, vmat, pmat, cmat, gmat;
            ndrange=ny, dependencies)
        return @time_event_a "boundary_conditions!" bc_event
    end

   @time_expr_a "boundary_conditions!" @simd_loop for j in 1:ny
        idx = @i(nx,j)
        idxp1 = @i(nx+1,j)
        rho[idxp1] = rho[idx]
        umat[idxp1] = umat[idx] * u_factor_right
        vmat[idxp1] = vmat[idx]
        pmat[idxp1] = pmat[idx]
        cmat[idxp1] = cmat[idx]
        gmat[idxp1] = gmat[idx]
    end

    return NoneEvent()
end


function boundaryConditions_top!(params::ArmonParameters{T}, data::ArmonData{V};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, nx, ny) = params
    @indexing_vars(params)

    v_factor_top::T = 1.
    if test == :Sod_y || test == :Sod_circ
        v_factor_top = -1.
    end

    if params.use_gpu
        bc_event = gpu_boundary_conditions_top!(index_start, idx_row, idx_col, ny, v_factor_top,
            rho, umat, vmat, pmat, cmat, gmat;
            ndrange=nx, dependencies)
        return @time_event_a "boundary_conditions!" bc_event
    end
    
    @time_expr_a "boundary_conditions!" @simd_loop for i in 1:nx
        idx = @i(i,ny)
        idxp1 = @i(i,ny+1)
        rho[idxp1]  = rho[idx]
        umat[idxp1] = umat[idx]
        vmat[idxp1] = vmat[idx] * v_factor_top
        pmat[idxp1] = pmat[idx]
        cmat[idxp1] = cmat[idx]
        gmat[idxp1] = gmat[idx]
    end

    return NoneEvent()
end


function boundaryConditions_bottom!(params::ArmonParameters{T}, data::ArmonData{V};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, nx) = params
    @indexing_vars(params)

    v_factor_bottom::T = 1.
    if test == :Sod_y || test == :Sod_circ
        v_factor_bottom = -1.
    end

    if params.use_gpu
        bc_event = gpu_boundary_conditions_bottom!(index_start, idx_row, idx_col, v_factor_bottom,
            rho, umat, vmat, pmat, cmat, gmat;
            ndrange=nx, dependencies)
        return @time_event_a "boundary_conditions!" bc_event
    end

    @time_expr_a "boundary_conditions!" @simd_loop for i in 1:nx
        idx = @i(i,1)
        idxm1 = @i(i,0)
        rho[idxm1]  = rho[idx]
        umat[idxm1] = umat[idx]
        vmat[idxm1] = vmat[idx] * v_factor_bottom
        pmat[idxm1] = pmat[idx]
        cmat[idxm1] = cmat[idx]
        gmat[idxm1] = gmat[idx]
    end

    return NoneEvent()
end


function read_border_array_X!(params::ArmonParameters{T}, data::ArmonData{V}, 
        value_array::W, pos::Int; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, row_length, nx) = params
    (; rho, umat, vmat, pmat, cmat, gmat, Emat, tmp_comm_array) = data

    if params.use_gpu
        read_event = gpu_read_border_array_X!(pos, nghost, nx, row_length,
            tmp_comm_array, rho, umat, vmat, pmat, cmat, gmat, Emat;
            ndrange=nx*nghost, dependencies)
        copy_event = async_copy!(device, value_array, tmp_comm_array; dependencies=read_event)
        return @time_event_a "border_array" copy_event
    end

    @perf_task "comms" "read_X" @time_expr_a "border_array" for i_g in 0:nghost-1
        ghost_row = i_g * nx * 7
        row_pos = i_g * row_length + pos
        @simd_loop for i in 0:nx-1
            i_arr = ghost_row + i * 7
            idx = row_pos + i
            value_array[i_arr+1] =  rho[idx]
            value_array[i_arr+2] = umat[idx]
            value_array[i_arr+3] = vmat[idx]
            value_array[i_arr+4] = pmat[idx]
            value_array[i_arr+5] = cmat[idx]
            value_array[i_arr+6] = gmat[idx]
            value_array[i_arr+7] = Emat[idx]
        end
    end

    return NoneEvent()
end


function read_border_array_Y!(params::ArmonParameters{T}, data::ArmonData{V}, 
        value_array::W, pos::Int; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, row_length, ny) = params
    (; rho, umat, vmat, pmat, cmat, gmat, Emat, tmp_comm_array) = data
    
    if params.use_gpu
        read_event = gpu_read_border_array_Y!(pos, nghost, ny, row_length,
            tmp_comm_array, rho, umat, vmat, pmat, cmat, gmat, Emat;
            ndrange=ny*nghost, dependencies)
        copy_event = async_copy!(device, value_array, tmp_comm_array; dependencies=read_event)
        return @time_event_a "border_array" copy_event
    end

    @perf_task "comms" "read_Y" @time_expr_a "border_array" for i_g in 0:nghost-1
        ghost_col = i_g * ny * 7
        @simd_loop for i in 0:ny-1
            i_arr = ghost_col + i * 7
            idx = pos + row_length * i + i_g
            value_array[i_arr+1] =  rho[idx]
            value_array[i_arr+2] = umat[idx]
            value_array[i_arr+3] = vmat[idx]
            value_array[i_arr+4] = pmat[idx]
            value_array[i_arr+5] = cmat[idx]
            value_array[i_arr+6] = gmat[idx]
            value_array[i_arr+7] = Emat[idx]
        end
    end

    return NoneEvent()
end


function write_border_array_X!(params::ArmonParameters{T}, data::ArmonData{V}, 
        value_array::W, pos::Int; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, row_length, nx) = params
    (; rho, umat, vmat, pmat, cmat, gmat, Emat, tmp_comm_array) = data

    if params.use_gpu
        copy_event = async_copy!(device, tmp_comm_array, value_array; dependencies)
        write_event = gpu_write_border_array_X!(pos, nghost, nx, row_length,
            tmp_comm_array, rho, umat, vmat, pmat, cmat, gmat, Emat;
            ndrange=nx*nghost, dependencies=copy_event)
        return @time_event_a "border_array" write_event
    end

    @perf_task "comms" "write_X" @time_expr_a "border_array" for i_g in 0:nghost-1
        ghost_row = i_g * nx * 7
        row_pos = i_g * row_length + pos
        @simd_loop for i in 0:nx-1
            i_arr = ghost_row + i * 7
            idx = row_pos + i
             rho[idx] = value_array[i_arr+1]
            umat[idx] = value_array[i_arr+2]
            vmat[idx] = value_array[i_arr+3]
            pmat[idx] = value_array[i_arr+4]
            cmat[idx] = value_array[i_arr+5]
            gmat[idx] = value_array[i_arr+6]
            Emat[idx] = value_array[i_arr+7]
        end
    end

    return NoneEvent()
end


function write_border_array_Y!(params::ArmonParameters{T}, data::ArmonData{V}, 
        value_array::W, pos::Int; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, row_length, ny) = params
    (; rho, umat, vmat, pmat, cmat, gmat, Emat, tmp_comm_array) = data

    if params.use_gpu
        copy_event = async_copy!(device, tmp_comm_array, value_array; dependencies)
        write_event = gpu_write_border_array_Y!(pos, nghost, ny, row_length,
            tmp_comm_array, rho, umat, vmat, pmat, cmat, gmat, Emat;
            ndrange=ny*nghost, dependencies=copy_event)
        return @time_event_a "border_array" write_event
    end

    @perf_task "comms" "write_Y" @time_expr_a "border_array" for i_g in 0:nghost-1
        ghost_col = i_g * ny * 7
        @simd_loop for i in 0:ny-1
            i_arr = ghost_col + i * 7
            idx = pos + row_length * i + i_g
             rho[idx] = value_array[i_arr+1]
            umat[idx] = value_array[i_arr+2]
            vmat[idx] = value_array[i_arr+3]
            pmat[idx] = value_array[i_arr+4]
            cmat[idx] = value_array[i_arr+5]
            gmat[idx] = value_array[i_arr+6]
            Emat[idx] = value_array[i_arr+7]
        end
    end

    return NoneEvent()
end


function exchange_with_neighbour(params::ArmonParameters{T}, array::V, neighbour_rank::Int,
        cart_comm::MPI.Comm) where {T, V <: AbstractArray{T}}
    @perf_task "comms" "MPI_sendrecv" @time_expr_a "boundaryConditions!_MPI" MPI.Sendrecv!(array, neighbour_rank, 0, array, neighbour_rank, 0, cart_comm)
end


function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}, host_array::W, axis::Axis; 
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nx, ny, nghost, neighbours, cart_comm, cart_coords) = params
    (; tmp_comm_array) = data
    @indexing_vars(params)
    # TODO : use active RMA instead? => maybe but it will (maybe) not work with GPUs: https://www.open-mpi.org/faq/?category=runcuda
    # TODO : use CUDA/ROCM-aware MPI
    # TODO : use 4 views for each side for each variable ? (2 will be contigous, 2 won't) <- pre-calculate them!
    # TODO : try to mix the comms: send to left and receive from right, then vice-versa. Maybe it can speed things up?

    if params.use_gpu
        comm_array = host_array
    else
        comm_array = tmp_comm_array
    end

    # We only exchange the ghost domains along the current axis.
    # even x/y coordinate in the cartesian process grid:
    #   - send+receive left  or top
    #   - send+receive right or bottom
    # odd  x/y coordinate in the cartesian process grid:
    #   - send+receive right or bottom
    #   - send+receive left  or top
    (cx, cy) = cart_coords
    if axis == X_axis
        if cx % 2 == 0
            order = [:left, :right]
        else
            order = [:right, :left]
        end
    else
        if cy % 2 == 0
            order = [:top, :bottom]
        else
            order = [:bottom, :top]
        end
    end

    prev_event = dependencies

    for side in order
        if side == :left
            if neighbours.left == MPI.MPI_PROC_NULL
                @perf_task "loop" "BC_left" prev_event = boundaryConditions_left!(params, data; dependencies=prev_event)
            else
                read_event = read_border_array_Y!(params, data, comm_array, @i(1, 1);
                    dependencies=prev_event)
                Event(exchange_with_neighbour, params, comm_array, neighbours.left, cart_comm;
                    dependencies=read_event) |> wait
                prev_event = write_border_array_Y!(params, data, comm_array, @i(1-nghost, 1))
            end
        elseif side == :right
            if neighbours.right == MPI.MPI_PROC_NULL
                @perf_task "loop" "BC_right" prev_event = boundaryConditions_right!(params, data; dependencies=prev_event)
            else
                read_event = read_border_array_Y!(params, data, comm_array, @i(nx-nghost+1, 1);
                    dependencies=prev_event)
                Event(exchange_with_neighbour, params, comm_array, neighbours.right, cart_comm;
                    dependencies=read_event) |> wait
                prev_event = write_border_array_Y!(params, data, comm_array, @i(nx+1, 1))
            end
        elseif side == :top
            if neighbours.top == MPI.MPI_PROC_NULL
                @perf_task "loop" "BC_top" prev_event = boundaryConditions_top!(params, data; dependencies=prev_event)
            else
                read_event = read_border_array_X!(params, data, comm_array, @i(1, ny-nghost+1);
                    dependencies=prev_event)
                Event(exchange_with_neighbour, params, comm_array, neighbours.top, cart_comm;
                    dependencies=read_event) |> wait
                prev_event = write_border_array_X!(params, data, comm_array, @i(1, ny+1))
            end
        else # side == :bottom
            if neighbours.bottom == MPI.MPI_PROC_NULL
                @perf_task "loop" "BC_bottom" prev_event = boundaryConditions_bottom!(params, data; dependencies=prev_event)
            else
                read_event = read_border_array_X!(params, data, comm_array, @i(1, 1);
                    dependencies=prev_event)
                Event(exchange_with_neighbour, params, comm_array, neighbours.bottom, cart_comm;
                    dependencies=read_event) |> wait
                prev_event = write_border_array_X!(params, data, comm_array, @i(1, 1-nghost))
            end
        end
    end

    return prev_event
end

#
# Time step computation
#

function dtCFL(params::ArmonParameters{T}, data::ArmonData{V}, dta::T;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; cmat, umat, vmat, domain_mask, tmp_rho) = data
    (; cfl, Dt, ideb, ifin, global_grid) = params
    @indexing_vars(params)

    (g_nx, g_ny) = global_grid

    dt::T = Inf
    dx::T = 1. / g_nx
    dy::T = 1. / g_ny

    if params.cst_dt
        # Constant time step
        dt = Dt
    elseif params.use_gpu && use_ROCM
        # AMDGPU doesn't support ArrayProgramming, however its implementation of `reduce` is quite
        # fast. Therefore first we compute dt for all cells and store the result in a temporary
        # array, then we reduce this array.
        # TODO : fix this
        if params.euler_projection
            gpu_dtCFL_reduction_euler!(dx, dy, tmp_rho, umat, vmat, cmat, domain_mask;
                ndrange=length(cmat), dependencies) |> wait
            dt = reduce(min, tmp_rho)
        else
            gpu_dtCFL_reduction_lagrange!(tmp_rho, cmat, domain_mask;
                ndrange=length(cmat), dependencies) |> wait
            dt = reduce(min, tmp_rho) * min(dx, dy)
        end
    elseif params.euler_projection
        if params.use_gpu
            wait(dependencies)
            # We need the absolute value of the divisor since the result of the max can be negative,
            # because of some IEEE 754 non-compliance since fast math is enabled when compiling this
            # code for GPU, e.g.: `@fastmath max(-0., 0.) == -0.`, while `max(-0., 0.) == 0.`
            # If the mask is 0, then: `dx / -0.0 == -Inf`, which will then make the result incorrect.
            dt_x = @inbounds reduce(min, @views (dx ./ abs.(
                max.(
                    abs.(umat[ideb:ifin] .+ cmat[ideb:ifin]), 
                    abs.(umat[ideb:ifin] .- cmat[ideb:ifin])
                ) .* domain_mask[ideb:ifin])))
            dt_y = @inbounds reduce(min, @views (dy ./ abs.(
                max.(
                    abs.(vmat[ideb:ifin] .+ cmat[ideb:ifin]), 
                    abs.(vmat[ideb:ifin] .- cmat[ideb:ifin])
                ) .* domain_mask[ideb:ifin])))
            dt = min(dt_x, dt_y)
        else
            @batch threadlocal=typemax(T) for i in ideb:ifin
                dt_x = dx / (max(abs(umat[i] + cmat[i]), abs(umat[i] - cmat[i])) * domain_mask[i])
                dt_y = dy / (max(abs(vmat[i] + cmat[i]), abs(vmat[i] - cmat[i])) * domain_mask[i])
                threadlocal = min(threadlocal, dt_x, dt_y)
            end
            dt = minimum(threadlocal)
        end
    else
        if params.use_gpu
            dt = reduce(min, @views (1. ./ (cmat .* domain_mask)))
        else
            @batch threadlocal=typemax(T) for i in ideb:ifin
                threadlocal = min(threadlocal, 1. / (cmat[i] * domain_mask[i]))
            end
            dt = minimum(threadlocal)
        end
        dt *= min(dx, dy)
    end

    if !isfinite(dt) || dt ≤ 0
        return dt  # Let it crash
    elseif dta == 0  # First cycle
        if Dt != 0
            return Dt
        else
            return cfl * dt
        end
    else
        # CFL condition and maximum increase per cycle of the time step
        return convert(T, min(cfl * dt, 1.05 * dta))
    end
end


function dtCFL_MPI(params::ArmonParameters{T}, data::ArmonData{V}, dta::T;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    @perf_task "loop" "dtCFL" local_dt::T = @time_expr_c dtCFL(params, data, dta; dependencies)

    if params.cst_dt || !params.use_MPI
        return local_dt
    end

    # Reduce all local_dts and broadcast the result to all processes
    @perf_task "comms" "MPI_dt" @time_expr_c "dt_Allreduce_MPI" dt = MPI.Allreduce(local_dt, MPI.Op(min, T), params.cart_comm)
    return dt
end

# 
# Cell update and euler projection
# 

function cellUpdate!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T,
        u::V, x::V; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, Emat, domain_mask) = data
    (; dx, ideb, ifin, s) = params

    if params.single_comm_per_axis_pass
        (; nx, ny) = params
        @indexing_vars(params)
        first_i = @i(0, 0)
        last_i = @i(nx+1, ny+1)
    else
        first_i = ideb
        last_i = ifin
    end

    if params.use_gpu
        event = gpu_cell_update!(first_i - 1, dx, dt, s, ustar, pstar, rho, u, Emat, domain_mask;
            ndrange=length(first_i:last_i), dependencies)
        if !params.euler_projection
            event = gpu_cell_update_lagrange!(first_i - 1, last_i, dt, s, x, ustar;
                ndrange=length(first_i:last_i), dependencies=event)
        end
        return @time_event "cellUpdate!" event
    end

    @time_expr "cellUpdate!" @simd_threaded_loop for i in first_i:last_i
        mask = domain_mask[i]
        dm = rho[i] * dx
        rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]) * mask)
        u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) * mask
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) * mask
    end

    if !params.euler_projection
        @time_expr "cellUpdate!" @simd_threaded_loop for i in first_i:last_i
            x[i] += dt * ustar[i]
        end
    end

    return NoneEvent()
end


function first_order_euler_remap!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, Emat, ustar, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, domain_mask) = data
    (; dx, ideb, ifin, s) = params
    @indexing_vars(params)

    if params.use_gpu
        event = gpu_first_order_euler_remap_1!(ideb - 1, dx, dt, s,
            ustar, rho, umat, vmat, Emat, 
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, domain_mask;
            ndrange=length(ideb:ifin), dependencies)
        event = gpu_first_order_euler_remap_2!(ideb - 1, rho, umat, vmat, Emat,
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho;
            ndrange=length(ideb:ifin), dependencies=event)
        return @time_event "euler_remap" event
    end

    # Projection of the conservative variables
    @time_expr "euler_remap" @simd_threaded_loop for i in ideb:ifin
        dX = dx + dt * (ustar[i+s] - ustar[i])
        L₁ =  max(0, ustar[i])   * dt * domain_mask[i]
        L₃ = -min(0, ustar[i+s]) * dt * domain_mask[i]
        L₂ = dX - L₁ - L₃
        
        tmp_rho_    = (L₁ * rho[i-s]             + L₂ * rho[i]           + L₃ * rho[i+s]            ) / dX
        tmp_urho[i] = (L₁ * rho[i-s] * umat[i-s] + L₂ * rho[i] * umat[i] + L₃ * rho[i+s] * umat[i+s]) / dX / tmp_rho_
        tmp_vrho[i] = (L₁ * rho[i-s] * vmat[i-s] + L₂ * rho[i] * vmat[i] + L₃ * rho[i+s] * vmat[i+s]) / dX / tmp_rho_
        tmp_Erho[i] = (L₁ * rho[i-s] * Emat[i-s] + L₂ * rho[i] * Emat[i] + L₃ * rho[i+s] * Emat[i+s]) / dX / tmp_rho_
        tmp_rho[i]  = tmp_rho_
    end

    # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E)
    @time_expr "euler_remap" @simd_threaded_loop for i in ideb:ifin
        rho[i]  = tmp_rho[i]
        umat[i] = tmp_urho[i]
        vmat[i] = tmp_vrho[i]
        Emat[i] = tmp_Erho[i]
    end

    return NoneEvent()
end

#
# Transposition parameters
#

function update_axis_parameters(params::ArmonParameters{T}, data::ArmonData{V}, 
        axis::Axis) where {T, V <: AbstractArray{T}}
    (; row_length, global_grid) = params
    (g_nx, g_ny) = global_grid

    params.current_axis = axis

    last_i::Int = params.ifin + 2
    
    if axis == X_axis
        params.s = 1
        params.dx = 1. / g_nx

        axis_positions::V = data.x
        axis_velocities::V = data.umat
    else  # axis == Y_axis
        params.s = row_length
        params.dx = 1. / g_ny
        
        axis_positions = data.y
        axis_velocities = data.vmat

        last_i += row_length  # include one more row to compute the fluxes at the top
    end

    return last_i, axis_positions, axis_velocities
end

#
# Axis splitting
#

function split_axes(params::ArmonParameters{T}, cycle::Int) where T
    axis_1, axis_2 = X_axis, Y_axis
    if iseven(cycle)
        axis_1, axis_2 = axis_2, axis_1
    end

    if params.axis_splitting == :Sequential
        return [
            (X_axis, T(1.0)),
            (Y_axis, T(1.0)),
        ]
    elseif params.axis_splitting == :SequentialSym
        return [
            (axis_1, T(1.0)),
            (axis_2, T(1.0)),
        ]
    elseif params.axis_splitting == :Strang
        return [
            (axis_1, T(0.5)),
            (axis_2, T(1.0)),
            (axis_1, T(0.5)),
        ]
    else
        error("Unknown axes splitting method: $(params.axis_splitting)")
    end
end

#
# Output 
#

function write_result_single_file(params::ArmonParameters{T}, data::ArmonData{V}, 
        file_name::String) where {T, V <: AbstractArray{T}}
    (; silent, nx, ny, nghost, row_length, col_length, output_dir, merge_files,
       is_root, cart_comm, cart_coords, global_grid) = params

    if is_root && merge_files && !params.euler_projection
        @warn "The output is written in an uniform matrix format, which is incompatible with an non uniform grid." maxlog=1
    end

    MPI.Barrier(cart_comm)  # To make sure that no process opens the file before the output dir is created

    # Type of the view to the real array of this sub-domain (so by excluding the ghost cells)
    local_subarray_type = MPI.Types.create_subarray((row_length, col_length), (nx, ny), (nghost, nghost), 
        MPI.Datatype(T); rowmajor=true)
    MPI.Types.commit!(local_subarray_type)

    # A buffer for one view to the array we want to write
    data_buffer = MPI.Buffer(data.rho, 1, local_subarray_type)

    # Type of the view of this process to the global domain array 
    (cx, cy) = cart_coords
    subarray_type = MPI.Types.create_subarray(global_grid, (nx, ny), (cx * nx, cy * ny), 
        MPI.Datatype(T); rowmajor=true)
    MPI.Types.commit!(subarray_type)

    # Open/Create the global file, then write our view to it
    output_file_path = joinpath(output_dir, file_name)
    f = MPI.File.open(cart_comm, output_file_path; write=true, create=true)

    MPI.File.set_view!(f, 0, MPI.Datatype(T), subarray_type)
    MPI.File.write_all(f, data_buffer)

    if is_root
        # Write the dimensions of the file domain to some utility files for easy gnuplot-ing
        (g_nx, g_ny) = global_grid
        
        open(output_file_path * "_DIM_X", "w") do f
            println(f, g_nx)
        end

        open(output_file_path * "_DIM_Y", "w") do f
            println(f, g_ny)
        end
    end

    if is_root && silent < 2
        println("\nWrote to file " * output_file_path)
    end
end


function write_sub_domain_file(params::ArmonParameters{T}, data::ArmonData{V}, 
        file_name::String) where {T, V <: AbstractArray{T}}
    (; silent, output_dir, nx, ny, cart_coords, cart_comm, is_root, write_ghosts, nghost) = params
    @indexing_vars(params)

    output_file_path = joinpath(output_dir, file_name)

    if is_root
        remove_all_outputs = "rm -f $(output_file_path)_*"
        run(`bash -c $remove_all_outputs`)
    end

    # Wait for the root command to complete
    params.use_MPI && MPI.Barrier(cart_comm)

    (cx, cy) = cart_coords

    f = open("$(output_file_path)_$(cx)x$(cy)", "w")
   
    vars_to_write = [data.x, data.y, data.rho, data.umat, data.vmat, data.pmat]

    if write_ghosts
        for j in 1-nghost:ny+nghost
            for i in 1-nghost:nx+nghost
                @printf(f, "%9.6f", vars_to_write[1][@i(i, j)])
                for var in vars_to_write[2:end]
                    @printf(f, ", %9.6f", var[@i(i,j)])
                end
                print(f, "\n")
            end
        end
    else
        for j in 1:ny
            for i in 1:nx
                @printf(f, "%9.6f", vars_to_write[1][@i(i, j)])
                for var in vars_to_write[2:end]
                    @printf(f, ", %9.6f", var[@i(i,j)])
                end
                print(f, "\n")
            end
        end
    end

    close(f)

    if is_root && silent < 2
        println("\nWrote to files $(output_file_path)_*x*")
    end
end


function write_result(params::ArmonParameters{T}, data::ArmonData{V}, 
        file_name::String) where {T, V <: AbstractArray{T}}
    (; output_dir, merge_files, is_root) = params

    if is_root && !isdir(output_dir)
        mkpath(output_dir)
    end

    if merge_files
        write_result_single_file(params, data, file_name)
    else
        write_sub_domain_file(params, data, file_name)
    end
end

# 
# Main time loop
# 

function time_loop(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root, dt_on_even_cycles) = params
    
    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.
    total_cycles_time::T = 0.

    t1 = time_ns()
    t_warmup = t1

    if params.use_gpu
        # Host version of temporary array used for MPI communications
        host_array = Vector{T}(undef, length(data.tmp_comm_array))
    else
        host_array = Vector{T}()
    end

    last_i::Int, x_::V, u::V = update_axis_parameters(params, data, params.current_axis)

    prev_event = NoneEvent()

    while t < maxtime && cycle < maxcycle
        cycle_start = time_ns()
        
        if !dt_on_even_cycles || iseven(cycle)
            dt = dtCFL_MPI(params, data, dta; dependencies=prev_event)
            prev_event = NoneEvent()

            if is_root && (!isfinite(dt) || dt <= 0.)
                error("Invalid dt at cycle $cycle: $dt")
            end
        end

        for (axis, dt_factor) in split_axes(params, cycle)
            last_i, x_, u = update_axis_parameters(params, data, axis)
           
            @perf_task "loop" "comms+fluxes" @time_expr_c "comms+fluxes" if params.async_comms 
                @sync begin
                    @async begin
                        event_2 = numericalFluxes_inner!(params, data, dt * dt_factor, u; dependencies=prev_event)
                        wait(event_2)
                    end

                    bc_params = copy(params)
                    bc_params.use_threading = false
                    @async begin
                        event_1 = boundaryConditions!(bc_params, data, host_array, axis; dependencies=prev_event)
                        event_1 = numericalFluxes_outer!(bc_params, data, dt * dt_factor, u; dependencies=event_1)
                        wait(event_1)
                    end
                end

                event = NoneEvent()
            else
                event = boundaryConditions!(params, data, host_array, axis; dependencies=prev_event)
                event = numericalFluxes_outer!(params, data, dt * dt_factor, u; dependencies=event)
                event = numericalFluxes_inner!(params, data, dt * dt_factor, u; dependencies=event)
                params.measure_time && wait(event)
            end

            @perf_task "loop" "cellUpdate" event = cellUpdate!(params, data, dt * dt_factor, u, x_; dependencies=event)
    
            if params.euler_projection
                if !params.single_comm_per_axis_pass 
                    event = boundaryConditions!(params, data, host_array, axis; dependencies=event)
                end
                @perf_task "loop" "euler_proj" event = first_order_euler_remap!(params, data, dt * dt_factor; dependencies=event)
            end

            @perf_task "loop" "EOS" prev_event = update_EOS!(params, data; dependencies=event)
        end

        if !is_warming_up()
            total_cycles_time += time_ns() - cycle_start
        end

        dta = dt
        cycle += 1
        t += dt

        if is_root
            if silent <= 1
                @printf("Cycle %4d: dt = %.18f, t = %.18f\n", cycle, dt, t)
            end
        end

        if cycle == 5
            t_warmup = time_ns()
            global in_warmup_cycle = false
        end
        
        if animation_step != 0 && (cycle - 1) % animation_step == 0
            write_result(params, data, joinpath("anim", params.output_file) * "_" *
                @sprintf("%03d", (cycle - 1) ÷ animation_step))
        end
    end

    t2 = time_ns()

    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((cycle - 5) * nb_cells)

    if is_root
        if cycle <= 5 && maxcycle > 5
            error("More than 5 cycles are needed to compute the grind time, got: $cycle")
        elseif t2 < t_warmup
            error("Clock error: $t2 < $t_warmup")
        end

        if silent < 3
            println(" ")
            println("Total time:  ", round((t2 - t1) / 1e9, digits=5),         " sec")
            println("Cycles time: ", round(total_cycles_time / 1e9, digits=5), " sec")
            println("Warmup:      ", round((t_warmup - t1) / 1e9, digits=5),   " sec")
            println("Grind time:  ", round(grind_time / 1e3, digits=5),        " µs/cell/cycle")
            println("Cells/sec:   ", round(1 / grind_time * 1e3, digits=5),    " Mega cells/sec")
            println("Cycles:      ", cycle)
        end
    end

    return dt, cycle, convert(T, 1 / grind_time), total_cycles_time
end

# 
# Main function
# 

function armon(params::ArmonParameters{T}) where T
    (; silent, is_root) = params

    if params.measure_time
        empty!(axis_time_contrib)
        empty!(total_time_contrib)
        global in_warmup_cycle = true
    end

    if is_root && silent < 3
        print_parameters(params)
    end

    if params.use_MPI && silent < 3
        (; rank, proc_size, cart_coords) = params
    
        # Local info
        node_local_comm = MPI.Comm_split_type(COMM, MPI.MPI_COMM_TYPE_SHARED, rank)
        local_rank = MPI.Comm_rank(node_local_comm)
        local_size = MPI.Comm_size(node_local_comm)

        is_root && println("\nProcesses info:")
        rank > 0 && MPI.Recv(Bool, rank-1, 1, COMM)
        @printf(" - %2d/%-2d, local: %2d/%-2d, coords: (%2d,%-2d), cores: %3d to %3d\n", 
            rank, proc_size, local_rank, local_size, cart_coords[1], cart_coords[2], minimum(getcpuids()), maximum(getcpuids()))
        rank < proc_size-1 && MPI.Send(true, rank+1, 1, COMM)
    end

    if is_root && params.animation_step != 0
        if isdir("anim")
            rm.("anim/" .* readdir("anim"))
        else
            mkdir("anim")
        end
    end
    
    # Allocate without initialisation in order to correctly map the NUMA space using the first-touch
    # policy when working on CPU only
    @perf_task "init" "alloc" data = ArmonData(T, params.nbcell, max(params.nx, params.ny) * params.nghost * 7)

    @perf_task "init" "init_test" @pretty_time init_test(params, data)

    if params.use_gpu
        copy_time = @elapsed d_data = data_to_gpu(data)
        (is_root && silent <= 2) && @printf("Time for copy to device: %.3g sec\n", copy_time)

        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, d_data)

        data_from_gpu(data, d_data)
    else
        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, data)
    end

    if params.write_output
        write_result(params, data, params.output_file)
    end

    sorted_time_contrib = sort(collect(total_time_contrib))

    if params.measure_time
        sync_total_time = mapreduce(x->x[2], +, sorted_time_contrib)
        async_efficiency = (sync_total_time - total_time) / total_time
        async_efficiency = max(async_efficiency, 0.)
    else
        sync_total_time = 1.
        async_efficiency = 0.
    end

    if is_root && params.measure_time && silent < 3 && !isempty(axis_time_contrib)
        axis_time = Dict{Axis, Float64}()

        # Print the time of each step for each axis
        for (axis, time_contrib_axis) in sort(collect(axis_time_contrib); lt=(a, b)->(a[1] < b[1]))
            isempty(time_contrib_axis) && continue
            
            axis_total_time = mapreduce(x->x[2], +, collect(time_contrib_axis))
            axis_time[axis] = axis_total_time

            println("\nTime for each step of the $axis:          ( axis%,  total%)")
            for (step_label, step_time) in sort(collect(time_contrib_axis))
                @printf(" - %-25s %10.5f ms (%5.2f%%) (%5.2f%%)\n", 
                    step_label, step_time / 1e6, step_time / axis_total_time * 100, step_time / total_time * 100)
            end
            @printf(" => %-24s %10.5f ms          (%5.2f%%)\n", "Axis total time:", 
                axis_total_time / 1e6, axis_total_time / total_time * 100)
        end

        # Print the total distribution of time
        println("\nTotal time repartition: ")
        for (step_label, step_time) in sorted_time_contrib
            @printf(" - %-25s %10.5f ms (%5.2f%%)\n",
                    step_label, step_time / 1e6, step_time / total_time * 100)
        end

        @printf("\nAsynchronicity efficiency: %.2f sec / %.2f sec = %.2f%% (effective time / total steps time)\n",
            total_time / 1e9, sync_total_time / 1e9, total_time / sync_total_time * 100)
    end

    return dt, cycles, cells_per_sec, sorted_time_contrib, async_efficiency
end

end
