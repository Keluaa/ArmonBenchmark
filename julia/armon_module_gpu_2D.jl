module Armon

using Printf
using Polyester
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

const use_ROCM = parse(Bool, get(ENV, "USE_ROCM_GPU", "false"))
const block_size = parse(Int, get(ENV, "GPU_BLOCK_SIZE", "32"))
const use_std_lib_threads = parse(Bool, get(ENV, "USE_STD_LIB_THREADS", "false"))

if use_ROCM
    using AMDGPU
    using ROCKernels
    println("Using ROCM GPU")
else
    using CUDA
    using CUDAKernels
    CUDA.allowscalar(false)
    println("Using CUDA GPU")
end


export ArmonParameters, armon


# TODO LIST
# fix ROCM dtCFL (+ take into account the ghost cells)
# make sure that the first touch is preserved in 2D
# GPU dtCFL time
# perf scripts
# better test implementation (common sturcture, one test = f(x, y) -> rho, pmat, umat, vmat, Emat + boundary conditions + EOS)
# use types and function overloads to define limiters and tests (in the hope that everything gets inlined)
# center the positions of the cells in the output file

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
    euler_projection::Bool
    transpose_dims::Bool
    axis_splitting::Symbol

    # Indexing
    row_length::Int
    col_length::Int
    nbcell::Int
    ideb::Int
    ifin::Int
    index_start::Int
    idebᵀ::Int
    ifinᵀ::Int
    index_startᵀ::Int
    idx_row::Int
    idx_col::Int
    current_axis::Axis
    s::Int  # Stride

    # Bounds
    maxtime::Flt_T
    maxcycle::Int
    
    # Output
    silent::Int
    output_dir::String
    output_file::String
    write_output::Bool
    write_ghosts::Bool
    animation_step::Int
    measure_time::Bool

    # Performance
    use_ccall::Bool
    use_threading::Bool
    use_simd::Bool
    use_gpu::Bool
end


# Constructor for ArmonParameters
function ArmonParameters(;
        ieee_bits = 64,
        test = :Sod, riemann = :acoustic, scheme = :GAD_minmod,
        nghost = 2, nx = 10, ny = 10, 
        cfl = 0.6, Dt = 0., cst_dt = false,
        euler_projection = false, transpose_dims = false, axis_splitting = :Sequential,
        maxtime = 0, maxcycle = 500_000,
        silent = 0, output_dir = ".", output_file = "output",
        write_output = true, write_ghosts = false, animation_step = 0, 
        measure_time = true,
        use_ccall = false, use_threading = true, 
        use_simd = true, interleaving = false,
        use_gpu = false
    )

    flt_type = (ieee_bits == 64) ? Float64 : Float32

    # Make sure that all floating point types are the same
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

    dx = flt_type(1. / nx)

    # Dimensions of an array
    row_length = nghost * 2 + nx
    col_length = nghost * 2 + ny
    nbcell = row_length * col_length

    # First and last index of the real domain of an array
    ideb = row_length * nghost + nghost + 1
    ifin = row_length * (ny - 1 + nghost) + nghost + nx
    index_start = ideb - row_length - 1  # Used only by the `@i` macro

    # Same as the 3 values above, but for a transposed array
    idebᵀ = col_length * nghost + nghost + 1
    ifinᵀ = col_length * (nx - 1 + nghost) + nghost + ny
    index_startᵀ = idebᵀ - col_length - 1

    # Used only for indexing with the `@i` macro
    idx_row = row_length
    idx_col = 1
    
    return ArmonParameters{flt_type}(
        test, riemann, scheme,
        nghost, nx, ny, dx,
        cfl, Dt, cst_dt,
        euler_projection, transpose_dims, axis_splitting,
        row_length, col_length, nbcell,
        ideb, ifin, index_start, 
        idebᵀ, ifinᵀ, index_startᵀ,
        idx_row, idx_col,
        X_axis, 1,
        maxtime, maxcycle,
        silent, output_dir, output_file,
        write_output, write_ghosts, animation_step,
        measure_time,
        use_ccall, use_threading, use_simd, use_gpu
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
    println(" - ieee_bits:  ", sizeof(T) * 8)
    println("")
    println(" - test:       ", p.test)
    println(" - riemann:    ", p.riemann)
    println(" - scheme:     ", p.scheme)
    println(" - splitting:  ", p.axis_splitting)
    println(" - domain:     ", p.nx, "x", p.ny, " (", p.nghost, " ghosts)")
    println(" - nbcell:     ", p.nx * p.ny, " (", p.nbcell, " total)")
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt)
    println(" - euler proj: ", p.euler_projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
end


# Default copy method
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
    domain_maskᵀ::V
end


function ArmonData(type::Type, size::Int64)
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
        Vector{type}(undef, size)
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
        device_type(data.domain_maskᵀ)
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
Controls which multi-threading librairy to use.
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
    @simd_threaded_loop(expr)

Allows to enable/disable multithreading and/or SIMD of the loop depending on the parameters.
When using SIMD, `@fastmath` and `@inbounds` are used.

In order to use SIMD and multithreading at the same time, the range of the loop is split in even 
batches.
Each batch has a size of `params.simd_batch` iterations, meaning that the inner `@simd` loop has a
fixed number of iterations, while the outer threaded loop will have `N ÷ params.simd_batch`
iterations.

The loop range must have a step of 1, i.e. this is forbidden: 1:4:450
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

    # Only in for the case of a threaded loop with SIMD:
    # Extract the range of the loop and replace it with the new variables
    modified_loop_expr = copy(expr)
    range_expr = modified_loop_expr.args[1]

    if range_expr.head == :block
        # Compound range expression: "j in 1:3, i in 4:6"
        # Use the first range as the threaded loop range
        loop_range = range_expr.args[1].args[2]
        range_expr.args[1].args[2] = :(__ideb:__ifin)
    elseif range_expr.head == Symbol("=")
        # Single range expression: "j in 1:3"
        loop_range = range_expr.args[2]
        range_expr.args[2] = :(__ideb:__ifin)
    else
        error("Expected range expression")
    end

    return esc(quote
        if params.use_threading
            if params.use_simd
                __loop_range = $(loop_range)
                __total_iter = length(__loop_range)
                __num_threads = Threads.nthreads()
                # Equivalent to __total_iter ÷ __num_threads
                __batch = convert(Int, cld(__total_iter, __num_threads))::Int
                __first_i = first(__loop_range)
                __last_i = last(__loop_range)
                @threads for __i_thread = 1:__num_threads
                    __ideb = __first_i + (__i_thread - 1) * __batch
                    __ifin = min(__ideb + __batch - 1, __last_i)
                    @fastmath @inbounds @simd ivdep $(modified_loop_expr)
                end
            else
                @inbounds @threads $(expr)
            end
        else
            if params.use_simd
                @fastmath @inbounds @simd ivdep $(expr)
            else
                $(expr)
            end
        end
    end)
end

#
# Execution Time Measurement
#

time_contrib = Dict{Axis, Dict{String, Float64}}()
macro time_pos(expr)
    if expr.head == :call
        function_name = expr.args[1]
    elseif isa(expr.args[2], Expr) && expr.args[2].head == :call
        function_name = expr.args[2].args[1]
    else
        error("Could not find the function name of the provided expression")
    end
    function_name = string(function_name)

    return esc(quote
        if params.measure_time
            _t_start = time_ns()
            $(expr)
            _t_end = time_ns()

            if !haskey(time_contrib, params.current_axis)
                global time_contrib[params.current_axis] = Dict{String, Float64}()
            end

            if !haskey(time_contrib, $(function_name))
                global time_contrib[params.current_axis][$(function_name)] = 0.
            end

            global time_contrib[params.current_axis][$(function_name)] += _t_end - _t_start
        else
            $(expr)
        end
    end)
end


# Equivalent to `@time` but with a better output
macro time_expr(expr)
    if expr.head == :call
        function_name = expr.args[1]
    elseif isa(expr.args[2], Expr) && expr.args[2].head == :call
        function_name = expr.args[2].args[1]
    else
        error("Could not find the function name of the provided expression")
    end
    function_name = string(function_name)

    return esc(quote
        if params.silent <= 3
            # Same structure as `@time` (see `@macroexpand @time`), using some undocumented functions.
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

Brings the parameters needed for the `@i` and `@iᵀ` macros into the current scope.
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

"""
    @iᵀ(idx)

Returns the index of the same element pointed by the mono-dimensional index `idx` but for a 
transposed array.

Note that the method used incorrectly transposes the last element of the array (`idx = rows*cols`) 
with the first element (`idx = 1`).

For square domains, `@iᵀ(@iᵀ(idx)) == idx`, this is not true for rectangular domains.
"""
macro iᵀ(i)
    return esc(quote
        ((col_length * ($(i) - 1)) % (row_length * col_length - 1)) + 1 
    end)
end

#
# GPU Kernels
#

const device = use_ROCM ? ROCDevice() : CUDADevice()
const reduction_block_size = 1024;
const reduction_block_size_log2 = convert(Int, log2(reduction_block_size))


@kernel function gpu_acoustic_kernel!(i_0, s, ustar, pstar, 
        @Const(rho), @Const(u), @Const(pmat), @Const(cmat))
    i = @index(Global) + i_0
    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    ustar[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
    pstar[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
end


@kernel function gpu_acoustic_GAD_minmod_kernel!(i_0, s, ustar, pstar, 
        @Const(rho), @Const(u), @Const(pmat), @Const(cmat), @Const(ustar_1), @Const(pstar_1), 
        dt, dx)
    i = @index(Global) + i_0

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
    θ    = (rc_l + rc_r) / 2 * (dt / Dm)
    
    ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (      u[i] - ustar_1[i]) -
                                             r_u_m * (ustar_1[i] -     u[i-s]))
    pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (   pmat[i] - pstar_1[i]) -
                                             r_p_m * (pstar_1[i] -  pmat[i-s]))
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


@kernel function gpu_boundary_conditions_kernel!(index_start, idx_row, idx_col, nx, ny,
        u_factor_left, u_factor_right, v_factor_bottom, v_factor_top,
        rho, umat, vmat, pmat, cmat, gmat)
    thread_i = @index(Global)

    if thread_i ≤ ny
        # Condition for the left border of the domain
        idx = @i(1,thread_i)
        idxm1 = @i(0,thread_i)
        rho[idxm1]  = rho[idx]
        umat[idxm1] = umat[idx] * u_factor_left
        vmat[idxm1] = vmat[idx]
        pmat[idxm1] = pmat[idx]
        cmat[idxm1] = cmat[idx]
        gmat[idxm1] = gmat[idx]

        # Condition for the right border of the domain
        idx = @i(nx,thread_i)
        idxp1 = @i(nx+1,thread_i)
        rho[idxp1]  = rho[idx]
        umat[idxp1] = umat[idx] * u_factor_right
        vmat[idxp1] = vmat[idx]
        pmat[idxp1] = pmat[idx]
        cmat[idxp1] = cmat[idx]
        gmat[idxp1] = gmat[idx]
    end

    if thread_i ≤ nx
        # Condition for the bottom border of the domain
        idx = @i(thread_i,1)
        idxm1 = @i(thread_i,0)
        rho[idxm1]  = rho[idx]
        umat[idxm1] = umat[idx]
        vmat[idxm1] = vmat[idx] * v_factor_bottom
        pmat[idxm1] = pmat[idx]
        cmat[idxm1] = cmat[idx]
        gmat[idxm1] = gmat[idx]

        # Condition for the top border of the domain
        idx = @i(thread_i,ny)
        idxp1 = @i(thread_i,ny+1)
        rho[idxp1]  = rho[idx]
        umat[idxp1] = umat[idx]
        vmat[idxp1] = vmat[idx] * v_factor_top
        pmat[idxp1] = pmat[idx]
        cmat[idxp1] = cmat[idx]
        gmat[idxp1] = gmat[idx]
    end
end


@kernel function gpu_dtCFL_reduction_kernel!(euler, ideb, ifin, x, cmat, umat, result, tmp_values, tmp_err_i)
    tid = @index(Local)

    values = @localmem eltype(x) reduction_block_size

    min_val_thread::eltype(x) = Inf
    if euler
        for i in ideb+tid-1:reduction_block_size:ifin
            dt_i = (x[i+1] - x[i]) / max(abs(umat[i] + cmat[i]), abs(umat[i] - cmat[i]))
            if isnan(dt_i) && tmp_err_i[tid] == -1
                tmp_err_i[tid] = i
                tmp_values[tid] = x[i+1] - x[i]
            end
            min_val_thread = min(min_val_thread, dt_i)
        end
    else
        for i in ideb+tid-1:reduction_block_size:ifin
            dt_i = (x[i+1] - x[i]) / cmat[i]
            min_val_thread = min(min_val_thread, dt_i)
        end
    end
    values[tid] = min_val_thread
    #tmp_values[tid] = min_val_thread

    @synchronize
    
    step_size = reduction_block_size >> 1
    
    @unroll for _ in 1:reduction_block_size_log2
        if tid <= step_size
            values[tid] = min(values[tid], values[tid + step_size])
        end

        step_size >>= 1

        @synchronize
    end

    if tid == 1
        result[1] = values[1]
    end
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


@kernel function gpu_first_order_euler_remap_2_transpose_kernel!(i_0, row_length, col_length, 
        rho, umat, vmat, Emat, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho)
    i = @index(Global) + i_0

    # Naive matrix transposition
    iᵀ = @iᵀ(i)

    # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E)
    rho[iᵀ] = tmp_rho[i]
    umat[iᵀ] = tmp_urho[i] / tmp_rho[i]
    vmat[iᵀ] = tmp_vrho[i] / tmp_rho[i]
    Emat[iᵀ] = tmp_Erho[i] / tmp_rho[i]
end


# Construction of the kernels for a common device and block size
gpu_acoustic! = gpu_acoustic_kernel!(device, block_size)
gpu_acoustic_GAD_minmod! = gpu_acoustic_GAD_minmod_kernel!(device, block_size)
gpu_update_perfect_gas_EOS! = gpu_update_perfect_gas_EOS_kernel!(device, block_size)
gpu_update_bizarrium_EOS! = gpu_update_bizarrium_EOS_kernel!(device, block_size)
gpu_boundary_conditions! = gpu_boundary_conditions_kernel!(device, block_size)
gpu_dtCFL_reduction! = gpu_dtCFL_reduction_kernel!(device, reduction_block_size, reduction_block_size)
gpu_cell_update! = gpu_cell_update_kernel!(device, block_size)
gpu_cell_update_lagrange! = gpu_cell_update_lagrange_kernel!(device, block_size)
gpu_first_order_euler_remap_1! = gpu_first_order_euler_remap_kernel!(device, block_size)
gpu_first_order_euler_remap_2! = gpu_first_order_euler_remap_2_kernel!(device, block_size)
gpu_first_order_euler_remap_2ᵀ! = gpu_first_order_euler_remap_2_transpose_kernel!(device, block_size)

#
# Acoustic Riemann problem solvers
# 

function acoustic!(params::ArmonParameters{T}, data::ArmonData{V}, 
        last_i::Int, u::V) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, pmat, cmat) = data
    (; ideb, s) = params

    if params.use_gpu
        gpu_acoustic!(ideb - 1, s, ustar, pstar, rho, u, pmat, cmat, 
            ndrange=length(ideb:last_i)) |> wait
        return
    end

    @simd_threaded_loop for i in ideb:last_i
        rc_l = rho[i-s] * cmat[i-s]
        rc_r = rho[i]   * cmat[i]
        ustar[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
        pstar[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
    end
end


function acoustic_GAD!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, last_i::Int, u::V) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, pmat, cmat, ustar_1, pstar_1) = data
    (; scheme, dx, ideb, s) = params

    if params.use_gpu
        if params.scheme != :GAD_minmod
            error("Only the minmod limiter is implemented for GPU")
        end

        gpu_acoustic!(ideb - 1, s, ustar_1, pstar_1, 
            rho, u, pmat, cmat, ndrange=length(ideb:last_i)) |> wait
        gpu_acoustic_GAD_minmod!(ideb - 1, s, ustar, pstar, 
            rho, u, pmat, cmat, ustar_1, pstar_1, dt, dx, ndrange=length(ideb:last_i)) |> wait
        return
    end

    # First order
    @simd_threaded_loop for i in ideb:last_i
        rc_l = rho[i-s] * cmat[i-s]
        rc_r = rho[i]   * cmat[i]
        ustar_1[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
        pstar_1[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
    end

    # Second order, for each flux limiter
    if scheme == :GAD_minmod
        @simd_threaded_loop for i in ideb:last_i
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
            θ    = (rc_l + rc_r) / 2 * (dt / Dm)

            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (      u[i] - ustar_1[i]) -
                                                     r_u_m * (ustar_1[i] -     u[i-s]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (   pmat[i] - pstar_1[i]) - 
                                                     r_p_m * (pstar_1[i] -  pmat[i-s]))
        end
    elseif scheme == :GAD_superbee
        @simd_threaded_loop for i in ideb:last_i
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
            θ  = (rc_l + rc_r) / 2 * (dt / Dm)
            
            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (      u[i] - ustar_1[i]) - 
                                                     r_u_m * (ustar_1[i] -     u[i-s]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (   pmat[i] - pstar_1[i]) -
                                                     r_p_m * (pstar_1[i] -  pmat[i-s]))
        end
    elseif scheme == :GAD_no_limiter
        @simd_threaded_loop for i in ideb:last_i
            dm_l = rho[i-s] * dx
            dm_r = rho[i]   * dx
            rc_l = rho[i-s] * cmat[i-s]
            rc_r = rho[i]   * cmat[i]
            Dm = (dm_l + dm_r) / 2
            θ  = (rc_l + rc_r) / 2 * (dt / Dm)

            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (      u[i] - ustar_1[i]) - 
                                                     r_u_m * (ustar_1[i] -     u[i-s]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (   pmat[i] - pstar_1[i]) - 
                                                     r_p_m * (pstar_1[i] -  pmat[i-s]))
        end
    else
        error("The choice of the scheme for the acoustic solver is not recognized: ", scheme)
    end

    return
end


function numericalFluxes!(params::ArmonParameters{T}, data::ArmonData{V}, 
    dt::T, last_i::Int, u::V) where {T, V <: AbstractArray{T}}
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            acoustic!(params, data, last_i, u)
        else
            acoustic_GAD!(params, data, dt, last_i, u)
        end
    else
        error("The choice of Riemann solver is not recognized: ", params.riemann)
    end
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
    # "Dissipative issue of high-order shock capturing schemes wtih non-convex equations of state",
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


function update_EOS!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat, Emat) = data
    (; ideb, ifin, test) = params

    if test == :Sod || test == :Sod_y || test == :Sod_circ
        gamma::T = 7/5
        if params.use_gpu
            gpu_update_perfect_gas_EOS!(ideb - 1, gamma, rho, Emat, 
                umat, vmat, pmat, cmat, gmat, ndrange=length(ideb:ifin)) |> wait
        else
            perfectGasEOS!(params, data, gamma)
        end
    elseif test == :Bizarrium
        if params.use_gpu
            gpu_update_bizarrium_EOS!(ideb - 1, rho, Emat, 
                umat, vmat, pmat, cmat, gmat, ndrange=length(ideb:ifin)) |> wait
        else
            BizarriumEOS!(params, data)
        end
    end
end

#
# Test initialisation
# 

function init_test(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; x, y, rho, umat, vmat, pmat, cmat, Emat, 
       ustar, pstar, ustar_1, pstar_1, domain_mask, domain_maskᵀ) = data
    (; test, nghost, nbcell, nx, ny, row_length, col_length, transpose_dims) = params

    if test == :Sod || test == :Sod_y || test == :Sod_circ
        if params.maxtime == 0
            params.maxtime = 0.20
        end
    
        if params.cfl == 0
            params.cfl = 0.95
        end

        gamma::T   = 7/5
        left_p::T  = 1.0
        right_p::T = 0.1

        if test == :Sod
            cond = (x_, y_) -> x_ ≤ 0.5
        elseif test == :Sod_y
            cond = (x_, y_) -> y_ ≤ 0.5
        else
            cond = (x_, y_) -> (x_ - 0.5)^2 + (y_ - 0.5)^2 ≤ 0.125
        end
    
        @simd_threaded_loop for i in 1:nbcell
            ix = ((i-1) % row_length) - nghost
            iy = ((i-1) ÷ row_length) - nghost

            if transpose_dims
                iᵀ = @iᵀ(i)
            else
                iᵀ = i
            end

            x[i]  = ix / nx
            y[iᵀ] = iy / ny
    
            if cond(x[i] + 1. / (2*nx), y[iᵀ] + 1. / (2*ny))
                rho[i] = 1.
                Emat[i] = left_p / ((gamma - 1.) * rho[i])
                umat[i] = 0.
                vmat[i] = 0.
            else
                rho[i] = 0.125
                Emat[i] = right_p / ((gamma - 1.) * rho[i])
                umat[i] = 0.
                vmat[i] = 0.
            end

            # Set to 1 if the cell is in the real domain or 0 in the ghost domain
            domain_mask[i] = domain_maskᵀ[iᵀ] = (0 ≤ ix < nx && 0 ≤ iy < ny) ? 1. : 0.

            # Set to zero to make sure no non-initialized values changes the result
            pmat[i] = 0.
            cmat[i] = 1.  # Set to 1 as a max speed of 0 will create NaNs
            ustar[i] = 0.
            pstar[i] = 0.
            ustar_1[i] = 0.
            pstar_1[i] = 0.
        end

        if transpose_dims
            # Handle the edge case of the first and last elements
            y[1] = y[col_length+1]
            y[nbcell] = y[nbcell - col_length]
            domain_maskᵀ[1] = domain_mask[1]
            domain_maskᵀ[nbcell] = domain_mask[nbcell]
        end

        perfectGasEOS!(params, data, gamma)

    elseif test == :Bizarrium
        if params.maxtime == 0
            params.maxtime = 80e-6
        end
    
        if params.cfl == 0
            params.cfl = 0.6
        end

        # TODO
        error("NYI")
    
        @simd_threaded_loop for i in 1:nbcell
            ix = ((i-1) % row_length) - nghost
            iy = ((i-1) ÷ row_length) - nghost

            if transpose_dims
                iᵀ = @iᵀ(i)
            else
                iᵀ = i
            end

            x[i]  = ix / nx
            y[iᵀ] = iy / ny
    
            if x[i] < 0.5
                rho[i] = 1.42857142857e+4
                umat[i] = 0.
                vmat[i] = 0.
                Emat[i] = 4.48657821135e+6
            else
                rho[i] =  10000.
                umat[i] = 250.
                vmat[i] = 0.
                Emat[i] = 0.5 * umat[i]^2
            end
            
            domain_mask[i] = domain_maskᵀ[iᵀ] = (0 ≤ ix < nx && 0 ≤ iy < ny) ? 1. : 0.

            pmat[i] = 0.
            cmat[i] = 1.
            ustar[i] = 0.
            pstar[i] = 0.
            ustar_1[i] = 0.
            pstar_1[i] = 0.
        end

        if transpose_dims
            y[1] = y[col_length+1]
            y[nbcell] = y[nbcell - col_length]
            domain_maskᵀ[1] = domain_mask[1]
            domain_maskᵀ[nbcell] = domain_mask[nbcell]
        end

        BizarriumEOS!(params, data)
    else
        error("Unknown test case: " * string(test))
    end
    
    return
end

#
# Boundary conditions
#

function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, nx, ny) = params
    @indexing_vars(params)

    u_factor_left::T   = 1.
    u_factor_right::T  = 1.
    v_factor_bottom::T = 1.
    v_factor_top::T    = 1.

    if test == :Sod
        u_factor_left   = -1.
        u_factor_right  = -1.
    elseif test == :Sod_y
        v_factor_top    = -1.
        v_factor_bottom = -1.
    elseif test == :Sod_circ
        u_factor_left   = -1.
        u_factor_right  = -1.
        v_factor_bottom = -1.
        v_factor_top    = -1.
    end

    if params.use_gpu
        gpu_boundary_conditions!(index_start, idx_row, idx_col, nx, ny, 
            u_factor_left, u_factor_right, v_factor_bottom, v_factor_top,
            rho, umat, vmat, pmat, cmat, gmat, ndrange=max(nx, ny)) |> wait
        return
    end

    @threaded for j in 1:ny
        # Condition for the left border of the domain
        idx = @i(1,j)
        idxm1 = @i(0,j)
        rho[idxm1]  = rho[idx]
        umat[idxm1] = umat[idx] * u_factor_left
        vmat[idxm1] = vmat[idx]
        pmat[idxm1] = pmat[idx]
        cmat[idxm1] = cmat[idx]
        gmat[idxm1] = gmat[idx]

        # Condition for the right border of the domain
        idx = @i(nx,j)
        idxp1 = @i(nx+1,j)
        rho[idxp1] = rho[idx]
        umat[idxp1] = umat[idx] * u_factor_right
        vmat[idxp1] = vmat[idx]
        pmat[idxp1] = pmat[idx]
        cmat[idxp1] = cmat[idx]
        gmat[idxp1] = gmat[idx]
    end

    @threaded for i in 1:nx
        # Condition for the bottom border of the domain
        idx = @i(i,1)
        idxm1 = @i(i,0)
        rho[idxm1]  = rho[idx]
        umat[idxm1] = umat[idx]
        vmat[idxm1] = vmat[idx] * v_factor_bottom
        pmat[idxm1] = pmat[idx]
        cmat[idxm1] = cmat[idx]
        gmat[idxm1] = gmat[idx]

        # Condition for the top border of the domain
        idx = @i(i,ny)
        idxp1 = @i(i,ny+1)
        rho[idxp1]  = rho[idx]
        umat[idxp1] = umat[idx]
        vmat[idxp1] = vmat[idx] * v_factor_top
        pmat[idxp1] = pmat[idx]
        cmat[idxp1] = cmat[idx]
        gmat[idxp1] = gmat[idx]
    end
end

#
# Time step computation
#

function dtCFL(params::ArmonParameters{T}, data::ArmonData{V}, 
        dta::T, domain_mask::V) where {T, V <: AbstractArray{T}}
    (; x, cmat, umat, vmat) = data
    (; cfl, Dt, ideb, ifin, nx, ny) = params
    @indexing_vars(params)

    dt::T = Inf
    dx::T = 1. / nx
    dy::T = 1. / ny

    if params.cst_dt
        # Constant time step
        dt = Dt
    elseif params.use_gpu && use_ROCM
        error("dtCFL for ROCM NYI")  # TODO : fix the reduction + min(dt_x, dt_y)
        # ROCM doesn't support Array Programming, so an explicit reduction kernel is needed
        result = zeros(T, 1)  # temporary array of a single value, holding the result of the reduction
        tmp_values = ones(T, 1024)
        d_tmp_values = ROCArray(tmp_values)
        tmp_err_i = -ones(Int, 1024)
        d_tmp_err_i = ROCArray(tmp_err_i)
        d_result = ROCArray(result)
        gpu_dtCFL_reduction!(params.euler_projection, ideb, ifin, x, cmat, umat, d_result, d_tmp_values, d_tmp_err_i) |> wait
        copyto!(result, d_result)
        copyto!(tmp_values, d_tmp_values)
        copyto!(tmp_err_i, d_tmp_err_i)
        dt = result[1]
        println("ROCM reduction result: ", dt)
        for (tid, (value, err)) in enumerate(zip(tmp_values, tmp_err_i)ù)
            #@printf("TID %2d: %f\n", tid, value)
            #if err >= 0
                @printf("TID %3d: err pos=%d, err=%g\n", tid, err, value)
            #end
        end
    elseif params.euler_projection
        if params.use_gpu
            # We need the absolute value of the divisor since the result of the max can be negative, 
            # because of some IEEE 754 non-compliance since fast math is enabled when compiling this 
            # code for GPU, e.g.: `@fastmath max(-0., 0.) == -0.`, while `max(-0., 0.) == 0.`
            # If the mask is 0, then: `dx / -0.0 == -Inf`, which will then make the result incorrect.
            dt_x = @inbounds reduce(min, @views (dx ./ abs.(
                max.(
                    abs.(umat .+ cmat), 
                    abs.(umat .- cmat)
                ) .* domain_mask)))
            dt_y = @inbounds reduce(min, @views (dy ./ abs.(
                max.(
                    abs.(vmat .+ cmat), 
                    abs.(vmat .- cmat)
                ) .* domain_mask)))
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
        return dt  # Let it crash
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

# 
# Cell update and euler projection
# 

function cellUpdate!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T,
        u::V, x::V, domain_mask::V) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, Emat) = data
    (; dx, ideb, ifin, s) = params

    if params.use_gpu
        gpu_cell_update!(ideb - 1, dx, dt, s, ustar, pstar, rho, u, Emat, domain_mask,
            ndrange=length(ideb:ifin)) |> wait
        if !params.euler_projection
            gpu_cell_update_lagrange!(ideb - 1, ifin, dt, s, x, ustar, 
                ndrange=length(ideb:ifin)) |> wait
        end
        return
    end

    @simd_threaded_loop for i in ideb:ifin
        mask = domain_mask[i]
        dm = rho[i] * dx
        rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]) * mask)
        u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) * mask
        Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) * mask
    end
 
    if !params.euler_projection
        @simd_threaded_loop for i in ideb:ifin
            x[i] += dt * ustar[i]
        end
    end
end


function first_order_euler_remap!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, domain_mask::V) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, Emat, ustar, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho) = data
    (; dx, ideb, ifin, nbcell, row_length, transpose_dims, s) = params
    @indexing_vars(params)

    if transpose_dims
        # When transposing the temporary arrays `tmp_rho` (and others) some uninitialized values 
        # might leak into the computing domain of other functions. Therefore this range is made very
        # large to ensure all values are set before transposing.
        projection_range = 2:nbcell-1
    else
        # We can't use the same range in both cases since the stride is not the same and we might go
        # out of bounds.
        projection_range = ideb:ifin
    end

    if params.use_gpu
        gpu_first_order_euler_remap_1!(first(projection_range) - 1, dx, dt, s, 
            ustar, rho, umat, vmat, Emat, 
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, domain_mask, 
            ndrange=length(projection_range)) |> wait

        if transpose_dims
            gpu_first_order_euler_remap_2ᵀ!(first(projection_range) - 1, row_length, col_length,
                rho, umat, vmat, Emat, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, 
                ndrange=length(projection_range)) |> wait
        else
            gpu_first_order_euler_remap_2!(first(projection_range) - 1, rho, umat, vmat, Emat, 
                tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, 
                ndrange=length(projection_range)) |> wait
        end
        
        return
    end

    # Projection of the conservative variables
    @simd_threaded_loop for i in projection_range
        dX = dx + dt * (ustar[i+s] - ustar[i])
        L₁ =  max(0, ustar[i])   * dt * domain_mask[i]
        L₃ = -min(0, ustar[i+s]) * dt * domain_mask[i]
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

    if transpose_dims        
        # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E) + transposition (including the ghost cells)
        @simd_threaded_loop for i in projection_range
            # Thanks to the temporary arrays, we can do out-of-place transposition, which is much 
            # easier and faster than in-place transposition
            iᵀ = @iᵀ(i)
            rho[iᵀ]  = tmp_rho[i]
            umat[iᵀ] = tmp_urho[i] / tmp_rho[i]
            vmat[iᵀ] = tmp_vrho[i] / tmp_rho[i]
            Emat[iᵀ] = tmp_Erho[i] / tmp_rho[i]
        end
    else
        # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E)
        @simd_threaded_loop for i in projection_range
            rho[i]  = tmp_rho[i]
            umat[i] = tmp_urho[i] / tmp_rho[i]
            vmat[i] = tmp_vrho[i] / tmp_rho[i]
            Emat[i] = tmp_Erho[i] / tmp_rho[i]
        end
    end
end

# 
# Transposition parameters
#

function update_axis_parameters(params::ArmonParameters{T}, data::ArmonData{V}, 
        axis::Axis) where {T, V <: AbstractArray{T}}
    (; nx, ny, row_length, transpose_dims) = params

    params.current_axis = axis

    last_i::Int = params.ifin + 1
    
    if axis == X_axis
        params.s = 1
        params.idx_row = row_length
        params.idx_col = 1

        params.dx = 1. / nx
        axis_positions::V = data.x
        axis_velocities::V = data.umat
        axis_domain_mask::V = data.domain_mask
    else  # axis == Y_axis
        if transpose_dims
            params.s = 1
            params.idx_row = 1
            params.idx_col = row_length
        else
            params.s = row_length
            params.idx_row = row_length
            params.idx_col = 1
        end
        
        params.dx = 1. / ny
        
        axis_positions = data.y
        axis_velocities = data.vmat
        axis_domain_mask = data.domain_maskᵀ

        last_i += row_length  # include one more row to compute the fluxes at the top
    end

    return last_i, axis_positions, axis_velocities, axis_domain_mask
end


function transpose_parameters(params::ArmonParameters{T}) where T
    # Swap the rows with the columns
    params.row_length, params.col_length = params.col_length, params.row_length
    
    # Swap the pre-calculated indexes
    params.ideb, params.idebᵀ = params.idebᵀ, params.ideb
    params.ifin, params.ifinᵀ = params.ifinᵀ, params.ifin
    params.index_start, params.index_startᵀ = params.index_startᵀ, params.index_start
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

function write_result(params::ArmonParameters{T}, data::ArmonData{V}, 
        file_name::String) where {T, V <: AbstractArray{T}}
    (; x, y, rho) = data
    (; silent, write_ghosts, nx, ny, nghost, output_dir, transpose_dims) = params
    @indexing_vars(params)

    if !isdir(output_dir)
        mkpath(output_dir)
    end

    output_file_path = joinpath(output_dir, file_name)
    f = open(output_file_path, "w")

    if write_ghosts
        for j in 1-nghost:ny+nghost
            for i in 1-nghost:nx+nghost
                print(f, x[@i(i, j)], ", ", 
                         y[transpose_dims ? @i(j, i) : @i(i, j)], ", ", 
                         rho[@i(i, j)], "\n")
            end
        end
    else
        for j in 1:ny
            for i in 1:nx
                print(f, x[@i(i, j)], ", ", 
                         y[transpose_dims ? @i(j, i) : @i(i, j)], ", ", 
                         rho[@i(i, j)], "\n")
            end
            print(f, "\n")
        end
    end

    close(f)

    if silent < 2
        println("\nWrote to file " * output_file_path)
    end
end

# 
# Main time loop
# 

function time_loop(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent, animation_step, transpose_dims) = params
    
    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    t1 = time_ns()
    t_warmup = t1

    last_i::Int, x_::V, u::V, mask::V = update_axis_parameters(params, data, params.current_axis)

    while t < maxtime && cycle < maxcycle
        @time_pos dt = dtCFL(params, data, dta, mask)

        if !isfinite(dt) || dt <= 0.
            error("Invalid dt at cycle $(cycle): $(dt)")
        end

        for (axis, dt_factor) in split_axes(params, cycle)
            last_i, x_, u, mask = update_axis_parameters(params, data, axis)

            @time_pos boundaryConditions!(params, data)
            @time_pos numericalFluxes!(params, data, dt * dt_factor, last_i, u)
            @time_pos cellUpdate!(params, data, dt * dt_factor, u, x_, mask)
    
            if params.euler_projection
                @time_pos first_order_euler_remap!(params, data, dt * dt_factor, mask)
                transpose_dims && transpose_parameters(params)
            end

            @time_pos update_EOS!(params, data)
        end

        dta = dt
        cycle += 1
        t += dt

        if silent <= 1
            @printf("Cycle %4d: dt = %.18f, t = %.18f\n", cycle, dt, t)
        end

        if cycle == 5
            t_warmup = time_ns()
        end

        if animation_step != 0 && (cycle - 1) % animation_step == 0
            write_result(params, data, joinpath("anim", params.output_file) * "_" *
                @sprintf("%03d", (cycle - 1) ÷ animation_step))
        end
    end

    t2 = time_ns()

    if cycle <= 5 && maxcycle > 5
        error("More than 5 cycles are needed to compute the grind time, got: $(cycle)")
    elseif t2 < t_warmup
        error("Clock error: $(t2) < $(t_warmup)")
    end
    
    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((cycle - 5) * nb_cells)

    if silent < 3
        println(" ")
        println("Time:       ", round((t2 - t1) / 1e9, digits=5),       " sec")
        println("Warmup:     ", round((t_warmup - t1) / 1e9, digits=5), " sec")
        println("Grind time: ", round(grind_time / 1e3, digits=5),      " µs/cell/cycle")
        println("Cells/sec:  ", round(1 / grind_time * 1e3, digits=5),  " Mega cells/sec")
        println("Cycles:     ", cycle)
    end

    return dt, convert(T, 1 / grind_time)
end

# 
# Main function
# 

function armon(params::ArmonParameters{T}) where T
    (; silent) = params

    if params.measure_time
        empty!(time_contrib)
    end

    if silent < 3
        print_parameters(params)
    end

    if params.animation_step != 0
        if isdir("anim")
            rm.("anim/" .* readdir("anim"))
        else
            mkdir("anim")
        end
    end
    
    # Allocate without initialisation in order to correctly map the NUMA space using the first-touch
    # policy when working on CPU only
    data = ArmonData(T, params.nbcell)

    @time_expr init_test(params, data)

    if params.use_gpu
        copy_time = @elapsed d_data = data_to_gpu(data)
        silent <= 2 && @printf("Time for copy to device: %.3g sec\n", copy_time)

        @time_expr dt, cells_per_sec = time_loop(params, d_data)

        data_from_gpu(data, d_data)
    else
        @time_expr dt, cells_per_sec = time_loop(params, data)
    end

    if params.write_output
        write_result(params, data, params.output_file)
    end

    if params.measure_time && silent < 3 && !isempty(time_contrib)
        axis_time = Dict{Axis, Float64}()

        # Print the time of each step for each axis
        for (axis, time_contrib_axis) in sort(collect(time_contrib); lt=(a, b)->(a[1] < b[1]))
            isempty(time_contrib_axis) && continue
            
            total_time = mapreduce(x->x[2], +, collect(time_contrib_axis))
            axis_time[axis] = total_time

            println("\nTime for each step of the $(axis): ")
            for (step_label, step_time) in sort(collect(time_contrib_axis))
                @printf(" - %-25s %10.5f ms (%5.2f%%)\n", 
                    step_label, step_time / 1e6, step_time / total_time * 100)
            end
            @printf(" => %-24s %10.5f ms\n", "Total time:", total_time / 1e6)
        end

        # Print the distribution of time between axis
        if length(axis_time) > 1
            total_time = mapreduce(x->x[2], +, collect(axis_time))

            println("\nAxis time repartition: ")
            for (axis, time_) in sort(collect(axis_time))
                @printf(" - %-5s %10.5f ms (%5.2f%%)\n", 
                    SubString(string(axis), 1:1), time_ / 1e6, time_ / total_time * 100)
            end
        end
    end

    return dt, cells_per_sec, sort(collect(time_contrib); lt=(a, b)->(a[1] < b[1]))
end

end
