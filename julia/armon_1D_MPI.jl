module Armon

using Printf
using Polyester
using KernelAbstractions
using KernelAbstractions.Extras: @unroll
using MPI

export ArmonParameters, armon

# GPU init

const use_ROCM = parse(Bool, get(ENV, "USE_ROCM_GPU", "false"))

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
const block_size = haskey(ENV, "GPU_BLOCK_SIZE") ? parse(Int, ENV["GPU_BLOCK_SIZE"]) : 32
const reduction_block_size = 1024;
const reduction_block_size_log2 = convert(Int, log2(reduction_block_size))

# MPI init

const COMM = MPI.COMM_WORLD

#
# Parameters
# 

mutable struct ArmonParameters{Flt_T}
    # Test problem type, riemann solver and solver scheme
    test::Symbol
    riemann::Symbol
    scheme::Symbol
    
    # Solver parameters
    iterations::Int
    
    # Domain parameters
    nghost::Int
    nbcell::Int
    cfl::Flt_T
    Dt::Flt_T
    ideb::Int
    ifin::Int
    euler_projection::Bool
    cst_dt::Bool

    # Bounds
    maxtime::Flt_T
    maxcycle::Int
    
    # Output
    silent::Int
    write_output::Bool
    measure_time::Bool

    # Performance
    use_ccall::Bool
    use_threading::Bool
    use_simd::Bool
    interleaving::Bool
    use_gpu::Bool

    # MPI
    use_MPI::Bool
    is_root::Bool
    rank::Int
    root_rank::Int
    proc_size::Int
    total_nbcell::Int
    global_cells_range::UnitRange{Int}
end


# Constructor for ArmonParameters
function ArmonParameters(; ieee_bits = 64,
                           test = :Sod, riemann = :acoustic, scheme = :GAD_minmod,
                           iterations = 4, 
                           nghost = 2, nbcell = 100, cfl = 0.6, Dt = 0., 
                           euler_projection = false, cst_dt = false,
                           maxtime = 0, maxcycle = 500_000,
                           silent = 0, write_output = true, measure_time = true,
                           use_ccall = false, use_threading = true, 
                           use_simd = true, interleaving = false,
                           use_gpu = false,
                           use_MPI = true)

    flt_type = (ieee_bits == 64) ? Float64 : Float32

    # Make sure that all floating point types are the same
    cfl = flt_type(cfl)
    Dt = flt_type(Dt)
    maxtime = flt_type(maxtime)
    
    if riemann == :one_iteration_acoustic iterations = 1 end
    if riemann == :two_iteration_acoustic iterations = 2 end

    if cst_dt && Dt == zero(flt_type)
        error("Dt == 0 with constant step enabled")
    end

    if use_MPI
        rank = MPI.Comm_rank(COMM)
        proc_size = MPI.Comm_size(COMM)
    else
        rank = 0
        proc_size = 1
    end

    root_rank = 0
    is_root = rank == root_rank

    total_nbcell = nbcell
    cells_per_proc = nbcell ÷ proc_size

    if rank != proc_size - 1
        cells_range = (cells_per_proc * rank + 1):(cells_per_proc * (rank + 1))
    else
        cells_range = (cells_per_proc * rank + 1):nbcell
    end

    nbcell = length(cells_range)

    ideb = nghost + 1
    ifin = nghost + nbcell
    
    return ArmonParameters{flt_type}(test, riemann, scheme, 
                                     iterations, 
                                     nghost, nbcell, cfl, Dt, ideb, ifin, 
                                     euler_projection, cst_dt,
                                     maxtime, maxcycle,
                                     silent, write_output, measure_time,
                                     use_ccall, use_threading,
                                     use_simd, interleaving,
                                     use_gpu,
                                     use_MPI, is_root, rank, root_rank, proc_size, 
                                     total_nbcell, cells_range)
end


function print_parameters(p::ArmonParameters{T}) where T
    println("Parameters:")
    print(" - multithreading: ", p.use_threading)
    if p.use_threading
        if p.use_ccall
            println(" (OMP threads: ", ENV["OMP_NUM_THREADS"], ")")
        else
            println(" (Julia threads: ", Threads.nthreads(), ")")
        end
    else
        println("")
    end
    println(" - interleaving: ", p.interleaving)
    println(" - use_simd:   ", p.use_simd)
    println(" - use_ccall:  ", p.use_ccall)
    println(" - use_gpu:    ", p.use_gpu)
    println(" - use_MPI:    ", p.use_MPI)
    println(" - ieee_bits:  ", sizeof(T) * 8)
    println(" - block size: ", block_size)
    println("")
    println(" - test:       ", p.test)
    println(" - riemann:    ", p.riemann)
    println(" - scheme:     ", p.scheme)
    println(" - iterations: ", p.iterations)
    println(" - nbcell:     ", p.nbcell)
    println(" - nghost:     ", p.nghost)
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt)
    println(" - euler proj: ", p.euler_projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
    println(" - rank:       ", p.rank, "/", p.proc_size - 1)
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
'V' can be a Vector of floats (Float32 or Float64) on CPU, CuArray or ROCArray on GPU.
Vector, CuArray and ROCArray are all subtypes of AbstractArray.
"""
struct ArmonData{V}
    x::V
    X::V
    rho::V
    umat::V
    vmat::V
    emat::V
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
        Vector{type}(undef, size)
    )
end


function data_to_gpu(data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    device_type = use_ROCM ? ROCArray : CuArray
    return ArmonData{device_type{T}}(
        device_type(data.x),
        device_type(data.X),
        device_type(data.rho),
        device_type(data.umat),
        device_type(data.vmat),
        device_type(data.emat),
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
        device_type(data.tmp_Erho)
    )
end


function data_from_gpu(host_data::ArmonData{V}, device_data::ArmonData{W}) where 
        {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    # We only need to copy the non-temporary arrays 
    copyto!(host_data.x, device_data.x)
    copyto!(host_data.X, device_data.X)
    copyto!(host_data.rho, device_data.rho)
    copyto!(host_data.umat, device_data.umat)
    copyto!(host_data.vmat, device_data.vmat)
    copyto!(host_data.emat, device_data.emat)
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


const USE_STD_LIB_THREADS = haskey(ENV, "USE_STD_LIB_THREADS") && ENV["USE_STD_LIB_THREADS"] == "true"


"""
Controls which multi-threading librairy to use.
"""
macro threads(expr)
    if USE_STD_LIB_THREADS
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

    @threaded for i = 1:n
        y[i] = log10(x[i]) + x[i]
    end
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

    # Only in for the case of a threaded loop with SIMD:
    # Extract the range of the loop and replace it with the new variables
    modified_loop_expr = copy(expr)
    range_expr = modified_loop_expr.args[1]
    loop_range = copy(range_expr.args[2])  # The original loop range
    range_expr.args[2] = :(__ideb:__ifin)  # The new loop range

    # Same but with interleaving
    interleaved_loop_expr = copy(expr)
    range_expr = interleaved_loop_expr.args[1]
    range_expr.args[2] = :((__first_i + __i_thread):__num_threads:__last_i)  # The interleaved loop range

    return esc(quote
        if params.use_threading
            if params.use_simd
                if params.interleaving
                    __loop_range = $(loop_range)
                    __total_iter = length(__loop_range)
                    __num_threads = Threads.nthreads()
                    __first_i = first(__loop_range)
                    __last_i = last(__loop_range)
                    @threads for __i_thread = 1:__num_threads
                        @fastmath @inbounds @simd ivdep $(interleaved_loop_expr)
                    end
                else
                    __loop_range = $(loop_range)
                    __total_iter = length(__loop_range)
                    __num_threads = Threads.nthreads()
                    # Equivalent to __total_iter ÷ __num_threads
                    __batch = convert(Int, cld(__total_iter, __num_threads))::Int
                    __first_i = first(__loop_range)
                    __last_i = last(__loop_range)
                    @threads for __i_thread = 1:__num_threads
                        __ideb = __first_i + (__i_thread - 1) * __batch
                        __ifin = min(__ideb + __batch - 1, __last_i)
                        @fastmath @inbounds @simd ivdep $(modified_loop_expr)
                    end
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

time_contrib = Dict{String, Float64}()
macro time_pos(params, label, expr) 
    return esc(quote
        if params.measure_time
            _t_start = time_ns()
            $(expr)
            _t_end = time_ns()
            if haskey(time_contrib, $(label))
                global time_contrib[$(label)] += _t_end - _t_start
            else
                global time_contrib[$(label)] = _t_end - _t_start
            end
        else
            $(expr)
        end
    end)
end

#
# GPU Kernels
#

@kernel function gpu_acoustic_kernel!(i_0, ustar, pstar, 
        @Const(rho), @Const(umat), @Const(pmat), @Const(cmat))
    i = @index(Global) + i_0
    rc_l = rho[i-1] * cmat[i-1]
    rc_r = rho[i]   * cmat[i]
    ustar[i] = (rc_l*umat[i-1] + rc_r*umat[i] +           (pmat[i-1] - pmat[i])) / (rc_l + rc_r)
    pstar[i] = (rc_r*pmat[i-1] + rc_l*pmat[i] + rc_l*rc_r*(umat[i-1] - umat[i])) / (rc_l + rc_r)
end


@kernel function gpu_acoustic_GAD_minmod_kernel!(i_0, ustar, pstar, 
        @Const(rho), @Const(umat), @Const(pmat), @Const(cmat), @Const(ustar_1), @Const(pstar_1), 
        dt, @Const(x))
    i = @index(Global) + i_0

    r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6)
    r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6)
    r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6)
    r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6)

    r_u_m = max(0., min(1., r_u_m))
    r_p_m = max(0., min(1., r_p_m))
    r_u_p = max(0., min(1., r_u_p))
    r_p_p = max(0., min(1., r_p_p))

    dm_l = rho[i-1] * (x[i] - x[i-1])
    dm_r = rho[i]   * (x[i+1] - x[i])
    Dm = (dm_l + dm_r) / 2
    rc_l = rho[i-1] * cmat[i-1]
    rc_r = rho[i]   * cmat[i]
    θ = (rc_l + rc_r) / 2 * (dt / Dm)
    
    ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * 
        (r_u_p * (umat[i] - ustar_1[i]) - r_u_m * (ustar_1[i] - umat[i-1]))
    pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * 
        (r_p_p * (pmat[i] - pstar_1[i]) - r_p_m * (pstar_1[i] - pmat[i-1]))
end


@kernel function gpu_cell_update_lagrange_kernel!(i_0, ifin, dt, 
        x_, X, @Const(ustar), @Const(pstar), rho, umat, vmat, emat, Emat)
    i = @index(Global) + i_0

    X[i] = x_[i] + dt * ustar[i]

    if i == ifin
        X[i+1] = x_[i+1] + dt * ustar[i+1]
    end

    @synchronize

    dm = rho[i] * (x_[i+1] - x_[i])
    # We must use this instead of X[i+1]-X[i] since X can be overwritten by other workgroups
    rho[i]  = dm / (x_[i+1] + dt * ustar[i+1] - (x_[i] + dt * ustar[i]))
    umat[i] = umat[i] + dt / dm * (pstar[i] - pstar[i+1])
    Emat[i] = Emat[i] + dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1])
    emat[i] = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)

    @synchronize

    x_[i] = X[i]

    if i == ifin
        x_[i+1] = X[i+1]
    end
end


@kernel function gpu_cell_update_euler_kernel!(i_0, ifin, dt, 
        x_, X, @Const(ustar), @Const(pstar), rho, umat, vmat, emat, Emat)
    i = @index(Global) + i_0

    X[i] = x_[i] + dt * ustar[i]

    if i == ifin
        X[i+1] = x_[i+1] + dt * ustar[i+1]
    end

    @synchronize

    dm = rho[i] * (x_[i+1] - x_[i])
    # We must use this instead of X[i+1]-X[i] since X can be overwritten by other workgroups
    dx = x_[i+1] + dt * ustar[i+1] - (x_[i] + dt * ustar[i])
    rho[i]  = dm / dx
    umat[i] = umat[i] + dt / dm * (pstar[i] - pstar[i+1])
    Emat[i] = Emat[i] + dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1])
    emat[i] = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
end


@kernel function gpu_first_order_euler_remap_kernel!(i_0, dt, 
        X, @Const(ustar), rho, umat, vmat, Emat, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho)
    i = @index(Global) + i_0

    dx = X[i+1] - X[i]
    L₁ = max(0, ustar[i]) * dt
    L₃ = -min(0, ustar[i+1]) * dt
    L₂ = dx - L₁ - L₃
    
    tmp_rho[i]  = (L₁ * rho[i-1] 
                 + L₂ * rho[i] 
                 + L₃ * rho[i+1]) / dx
    tmp_urho[i] = (L₁ * rho[i-1] * umat[i-1] 
                 + L₂ * rho[i]   * umat[i] 
                 + L₃ * rho[i+1] * umat[i+1]) / dx
    tmp_vrho[i] = (L₁ * rho[i-1] * vmat[i-1] 
                 + L₂ * rho[i]   * vmat[i] 
                 + L₃ * rho[i+1] * vmat[i+1]) / dx
    tmp_Erho[i] = (L₁ * rho[i-1] * Emat[i-1] 
                 + L₂ * rho[i]   * Emat[i] 
                 + L₃ * rho[i+1] * Emat[i+1]) / dx
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


@kernel function gpu_update_perfect_gas_EOS_kernel!(i_0, gamma,
        @Const(rho), @Const(emat), pmat, cmat, gmat)
    i = @index(Global) + i_0
    
    pmat[i] = (gamma - 1.) * rho[i] * emat[i]
    cmat[i] = sqrt(gamma * pmat[i] / rho[i])
    gmat[i] = (1. + gamma) / 2
end


@kernel function gpu_update_bizarrium_EOS_kernel!(i_0, 
        @Const(rho), @Const(emat), pmat, cmat, gmat)
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

    pmat[i] = pk0 + G0 * rho0 * (emat[i] - epsk0)
    cmat[i] = sqrt(G0 * rho0 * (pmat[i] - pk0) - pk0prime) / rho[i]
    gmat[i] = 0.5 / (rho[i]^3 * cmat[i]^2) * (pk0second + (G0 * rho0)^2 * (pmat[i] - pk0))
end


@kernel function gpu_boundary_conditions_left_kernel!(ideb, rho, umat, vmat, pmat, cmat, gmat)
    rho[ideb-1]  = rho[ideb]
    umat[ideb-1] = -umat[ideb]
    vmat[ideb-1] = vmat[ideb]
    pmat[ideb-1] = pmat[ideb]
    cmat[ideb-1] = cmat[ideb]
    gmat[ideb-1] = gmat[ideb]
end


@kernel function gpu_boundary_conditions_right_kernel!(test_bizarrium, ifin, rho, umat, vmat, pmat,
        cmat, gmat)
    rho[ifin+1]  = rho[ifin]
    vmat[ifin-1] = vmat[ifin]
    pmat[ifin+1] = pmat[ifin]
    cmat[ifin+1] = cmat[ifin]
    gmat[ifin+1] = gmat[ifin]

    if test_bizarrium
        umat[ifin+1] = umat[ifin]
    else
        umat[ifin+1] = -umat[ifin]
    end
end


@kernel function gpu_dtCFL_reduction_euler_kernel!(i_0, out, @Const(x), @Const(umat), @Const(cmat))
    i = @index(Global) + i_0

    c = cmat[i]
    u = umat[i]
    out[i] = (x[i+1] - x[i]) / max(abs(u + c), abs(u - c))
end


@kernel function gpu_dtCFL_reduction_lagrange_kernel!(i_0, out, @Const(x), @Const(cmat))
    i = @index(Global) + i_0
    out[i] = (x[i+1] - x[i]) / cmat[i]
end


gpu_acoustic! = gpu_acoustic_kernel!(device, block_size)
gpu_acoustic_GAD_minmod! = gpu_acoustic_GAD_minmod_kernel!(device, block_size)
gpu_cell_update_lagrange! = gpu_cell_update_lagrange_kernel!(device, block_size)
gpu_cell_update_euler! = gpu_cell_update_euler_kernel!(device, block_size)
gpu_first_order_euler_remap_1! = gpu_first_order_euler_remap_kernel!(device, block_size)
gpu_first_order_euler_remap_2! = gpu_first_order_euler_remap_2_kernel!(device, block_size)
gpu_update_perfect_gas_EOS! = gpu_update_perfect_gas_EOS_kernel!(device, block_size)
gpu_update_bizarrium_EOS! = gpu_update_bizarrium_EOS_kernel!(device, block_size)
gpu_boundary_conditions_left! = gpu_boundary_conditions_left_kernel!(device, 1, 1)
gpu_boundary_conditions_right! = gpu_boundary_conditions_right_kernel!(device, 1, 1)
gpu_dtCFL_reduction_euler! = gpu_dtCFL_reduction_euler_kernel!(device, block_size)
gpu_dtCFL_reduction_lagrange! = gpu_dtCFL_reduction_lagrange_kernel!(device, block_size)

#
# Equations of State
#

function perfectGasEOS!(params::ArmonParameters{T}, data::ArmonData{V}, gamma::T) where {T, V <: AbstractArray{T}}
    (; pmat, cmat, gmat, rho, emat) = data
    (; ideb, ifin) = params

    @simd_threaded_loop for i in ideb:ifin
        pmat[i] = (gamma - 1.) * rho[i] * emat[i]
        cmat[i] = sqrt(gamma * pmat[i] / rho[i])
        gmat[i] = (1. + gamma) / 2
    end
end


function BizarriumEOS!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; pmat, cmat, gmat, rho, emat) = data
    (; ideb, ifin) = params

    # O. Heuzé, S. Jaouen, H. Jourdren, 
    # "Dissipative issue of high-order shock capturing schemes wtih non-convex equations of state",
    # JCP 2009

    rho0 = 10000; K0 = 1e+11; Cv0 = 1000; T0 = 300; eps0 = 0; G0 = 1.5; s = 1.5
    q = -42080895/14941154; r = 727668333/149411540

    @simd_threaded_loop for i in ideb:ifin
        x = rho[i]/rho0 - 1; g = G0*(1-rho0/rho[i]); # formula (4b)

        f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)   # Formula (15b)
        f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)  # Formula (16a)
        f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)          # Formula (16b)
        f3 = (6*r+3*s*f2)/(1-s*x)                # Formula (16c)

        epsk0 = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0                                # Formula (15a)
        pk0 = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)                              # Formula (17a)
        pk0prime = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)    # Formula (17b)
        pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 +           # Formula (17c)
                                             6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

        pmat[i] = pk0 + G0*rho0*(emat[i] - epsk0)                                 # Formula (5b)
        cmat[i] = sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i]               # Formula (8)
        gmat[i] = 0.5/(rho[i]^3*cmat[i]^2)*(pk0second+(G0*rho0)^2*(pmat[i]-pk0))  # Formula (8) + (11)
    end
end

#
# Acoustic Riemann problem solvers
# 

function acoustic!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, umat, pmat, cmat) = data
    (; ideb, ifin) = params

    if params.use_ccall
        # void acoustic(double* restrict ustar, double* restrict pstar, 
        #       const double* restrict rho, const double* restrict cmat,
        #       const double* restrict umat, const double* restrict pmat, 
        #       int ideb, int ifin)
        ccall((:acoustic, "./libacoustic.so"), Cvoid, (
                Ref{T}, Ref{T}, 
                Ref{T}, Ref{T}, 
                Ref{T}, Ref{T}, 
                Int32, Int32),
            ustar, pstar, rho, cmat, umat, pmat, ideb, ifin)
    elseif params.use_gpu
        gpu_acoustic!(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ndrange=length(ideb:ifin+1)) |> wait
    else
        @simd_threaded_loop for i in ideb:ifin+1
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            ustar[i] = (rc_l * umat[i-1] + rc_r * umat[i] +               (pmat[i-1] - pmat[i])) /
                (rc_l + rc_r)
            pstar[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + rc_l * rc_r * (umat[i-1] - umat[i])) /
                (rc_l + rc_r)
        end
    end
end


function acoustic_GAD!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1, x) = data
    (; scheme, ideb, ifin) = params

    if params.use_ccall
        scheme_int::Int32 = (scheme == :GAD_minmod) ? 1 : ((scheme == :GAD_superbee) ? 2 : 0)
        # void acoustic_GAD(double* restrict ustar, double* restrict pstar, 
        #           double* restrict ustar_1, double* restrict pstar_1,
        #           const double* restrict rho, const double* restrict cmat,
        #           const double* restrict umat, const double* restrict pmat, 
        #           const double* restrict x,
        #           double dt, int ideb, int ifin,
        #           int scheme)
        ccall((:acoustic_GAD, "./libacoustic.so"), Cvoid, (
                Ref{T}, Ref{T}, 
                Ref{T}, Ref{T}, 
                Ref{T}, Ref{T}, 
                Ref{T}, Ref{T}, 
                Ref{T},
                Int32, Int32,
                Int32),
            ustar, pstar, ustar_1, pstar_1, rho, cmat, umat, pmat, x, ideb, ifin, scheme_int)

        return
    elseif params.use_gpu
        if params.scheme != :GAD_minmod
            error("Only the minmod limiter is implemented for GPU")
        end
        gpu_acoustic!(ideb - 1, ustar_1, pstar_1, rho, umat, pmat, cmat, ndrange=length(ideb:ifin+1)) |> wait
        gpu_acoustic_GAD_minmod!(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1,
            dt, x, ndrange=length(ideb:ifin+1)) |> wait
        return
    end

    # First order
    @simd_threaded_loop for i in ideb-1:ifin+2
        rc_l = rho[i-1] * cmat[i-1]
        rc_r = rho[i]   * cmat[i]
        ustar_1[i] = (rc_l * umat[i-1] + rc_r * umat[i] +               (pmat[i-1] - pmat[i])) /
            (rc_l + rc_r)
        pstar_1[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + rc_l * rc_r * (umat[i-1] - umat[i])) /
            (rc_l + rc_r)
    end
    
    # Second order, for each flow limiter
    if scheme == :GAD_minmod
        @simd_threaded_loop for i in ideb:ifin+1
            r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6)
            r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6)
            r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6)
            r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6)

            r_u_m = max(0., min(1., r_u_m))
            r_p_m = max(0., min(1., r_p_m))
            r_u_p = max(0., min(1., r_u_p))
            r_p_p = max(0., min(1., r_p_p))

            dm_l = rho[i-1] * (x[i] - x[i-1])
            dm_r = rho[i]   * (x[i+1] - x[i])
            Dm = (dm_l + dm_r) / 2
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            θ = (rc_l + rc_r) / 2 * (dt / Dm)
            
            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (umat[i] - ustar_1[i]) -
                                                     r_u_m * (ustar_1[i] - umat[i-1]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (pmat[i] - pstar_1[i]) - 
                                                     r_p_m * (pstar_1[i] - pmat[i-1]))
        end
    elseif scheme == :GAD_superbee
        @simd_threaded_loop for i in ideb:ifin+1
            r_u_m = (ustar_1[i+1] - umat[i]) / (ustar_1[i] - umat[i-1] + 1e-6)
            r_p_m = (pstar_1[i+1] - pmat[i]) / (pstar_1[i] - pmat[i-1] + 1e-6)
            r_u_p = (umat[i-1] - ustar_1[i-1]) / (umat[i] - ustar_1[i] + 1e-6)
            r_p_p = (pmat[i-1] - pstar_1[i-1]) / (pmat[i] - pstar_1[i] + 1e-6)

            r_u_m = max(0., min(1., 2. * r_u_m), min(2., r_u_m))
            r_p_m = max(0., min(1., 2. * r_p_m), min(2., r_p_m))
            r_u_p = max(0., min(1., 2. * r_u_p), min(2., r_u_p))
            r_p_p = max(0., min(1., 2. * r_p_p), min(2., r_p_p))
    
            dm_l = rho[i-1] * (x[i] - x[i-1])
            dm_r = rho[i]   * (x[i+1] - x[i])
            Dm = (dm_l + dm_r) / 2
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            θ = (rc_l + rc_r) / 2 * (dt / Dm)
            
            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (umat[i] - ustar_1[i]) - 
                                                     r_u_m * (ustar_1[i] - umat[i-1]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (pmat[i] - pstar_1[i]) -
                                                     r_p_m * (pstar_1[i] - pmat[i-1]))
        end
    elseif scheme == :GAD_no_limiter
        @simd_threaded_loop for i in ideb:ifin+1
            dm_l = rho[i-1] * (x[i] - x[i-1])
            dm_r = rho[i]   * (x[i+1] - x[i])
            Dm = (dm_l + dm_r) / 2
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            θ = (rc_l + rc_r) / 2 * (dt / Dm)

            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * ((umat[i] - ustar_1[i]) - 
                                                     (ustar_1[i] - umat[i-1]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * ((pmat[i] - pstar_1[i]) - 
                                                     (pstar_1[i] - pmat[i-1]))
        end
    else
        error("The choice of the scheme for the acoustic solver is not recognized: ", scheme)
    end

    return
end

#
# Test initialisation
#

function init_test(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; x, rho, pmat, umat, vmat, emat, Emat, cmat, gmat) = data
    (; X, ustar, pstar, ustar_1, pstar_1, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho) = data
    (; test, nghost, nbcell, total_nbcell, global_cells_range) = params

    if test == :Sod
        if params.maxtime == 0
            params.maxtime = 0.20
        end
    
        if params.cfl == 0
            params.cfl = 0.95
        end

        gamma::T = 1.4

        # -1-nghost                      -> local range to 0 to N_local-1
        # +first(global_cells_range) - 1 -> local range to global range
        offset = - 1 - nghost + first(global_cells_range) - 1

        @simd_threaded_loop for i in 1:nbcell+2*nghost
            x[i] = (i + offset) / total_nbcell

            if x[i] < 0.5
                rho[i] = 1.
                pmat[i] = 1.
                umat[i] = 0.
                vmat[i] = 0.
            else
                rho[i] = 0.125
                pmat[i] = 0.1
                umat[i] = 0.
                vmat[i] = 0.
            end

            emat[i] = Emat[i] = pmat[i] / ((gamma - 1.) * rho[i])
            cmat[i] = sqrt(gamma * pmat[i] / rho[i])
            gmat[i] = 0.5 * (1. + gamma)

            X[i] = 0.
            ustar[i] = 0.
            pstar[i] = 0.
            ustar_1[i] = 0.
            pstar_1[i] = 0.
            tmp_rho[i] = 0.
            tmp_urho[i] = 0.
            tmp_vrho[i] = 0.
            tmp_Erho[i] = 0.
        end
    elseif test == :Bizarrium
        if params.maxtime == 0
            params.maxtime = 80e-6
        end
    
        if params.cfl == 0
            params.cfl = 0.6
        end

        offset = - 1 - nghost + first(global_cells_range) - 1
    
        @simd_threaded_loop for i in 1:nbcell+2*nghost
            x[i] = (i + offset) / total_nbcell
    
            if x[i] < 0.5
                rho[i] = 1.42857142857e+4
                umat[i] = 0.
                vmat[i] = 0.
                emat[i] = Emat[i] = 4.48657821135e+6
            else
                rho[i] =  10000.
                umat[i] = 250.
                vmat[i] = 0.
                emat[i] = 0.
                Emat[i] = 0.5 * umat[i]^2
            end

            X[i] = 0.
            ustar[i] = 0.
            pstar[i] = 0.
            ustar_1[i] = 0.
            pstar_1[i] = 0.
            tmp_rho[i] = 0.
            tmp_urho[i] = 0.
            tmp_vrho[i] = 0.
            tmp_Erho[i] = 0.
        end
    
        BizarriumEOS!(params, data)
    end
    
    return
end

#
# Boundary conditions
#

function boundaryConditions_left!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; ideb) = params

    if params.use_gpu
        gpu_boundary_conditions_left!(ideb, rho, umat, vmat, pmat, cmat, gmat) |> wait
        return
    end

    rho[ideb-1]  = rho[ideb]
    umat[ideb-1] = -umat[ideb]
    vmat[ideb-1] = vmat[ideb]
    pmat[ideb-1] = pmat[ideb]
    cmat[ideb-1] = cmat[ideb]
    gmat[ideb-1] = gmat[ideb]
end


function boundaryConditions_right!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, ifin) = params

    if params.use_gpu
        gpu_boundary_conditions_right!(test == :Bizarrium, ifin, rho, umat, vmat, pmat, cmat, gmat) |> wait
        return
    end

    rho[ifin+1]  = rho[ifin]
    vmat[ifin+1] = vmat[ifin]
    pmat[ifin+1] = pmat[ifin]
    cmat[ifin+1] = cmat[ifin]
    gmat[ifin+1] = gmat[ifin]

    if test == :Bizarrium
        umat[ifin+1] = umat[ifin]
    else
        umat[ifin+1] = -umat[ifin]
    end
end


function read_border_array!(value_array::V, pos::Int, nghost::Int, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat, Emat) = data
    i_arr = 0
    for i in 0:nghost-1
        value_array[i_arr+1] = rho[pos+i]
        value_array[i_arr+2] = umat[pos+i]
        value_array[i_arr+3] = vmat[pos+i]
        value_array[i_arr+4] = pmat[pos+i]
        value_array[i_arr+5] = cmat[pos+i]
        value_array[i_arr+6] = gmat[pos+i]
        value_array[i_arr+7] = Emat[pos+i]
        i_arr += 7
    end
end


function write_border_array(value_array::V, pos::Int, nghost::Int, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat, Emat) = data
    i_arr = 0
    for i in 0:nghost-1
        rho[pos+i]  = value_array[i_arr+1]
        umat[pos+i] = value_array[i_arr+2]
        vmat[pos+i] = value_array[i_arr+3]
        pmat[pos+i] = value_array[i_arr+4]
        cmat[pos+i] = value_array[i_arr+5]
        gmat[pos+i] = value_array[i_arr+6]
        Emat[pos+i] = value_array[i_arr+7]
        i_arr += 7
    end
end


function boundaryConditions_MPI!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; nghost, ideb, ifin, rank, proc_size) = params
    # TODO : use active RMA instead?
    # TODO : use CUDA/ROCM-aware MPI

    tmp_array = Vector{T}(undef, nghost * 7)

    # rank even:
    #   - send+receive left
    #   - send+receive right
    # rank odd:
    #   - send+receive right
    #   - send+receive left
    if rank % 2 == 0
        order = [:left, :right]
    else
        order = [:right, :left]
    end

    for side in order
        if side == :left
            if rank == 0
                boundaryConditions_left!(params, data)
            else
                read_border_array!(tmp_array, ideb, nghost, data)
                @time_pos params "boundaryConditions_MPI!" MPI.Sendrecv!(tmp_array, rank - 1, 0, tmp_array, rank - 1, 0, COMM)
                write_border_array(tmp_array, ideb-nghost, nghost, data)
            end
        else
            if rank == proc_size - 1
                boundaryConditions_right!(params, data)
            else
                read_border_array!(tmp_array, ifin-nghost+1, nghost, data)
                @time_pos params "boundaryConditions_MPI!" MPI.Sendrecv!(tmp_array, rank + 1, 0, tmp_array, rank + 1, 0, COMM)
                write_border_array(tmp_array, ifin+1, nghost, data)
            end
        end
    end
end

#
# Time step computation
#

function dtCFL(params::ArmonParameters{T}, data::ArmonData{V}, dta::T) where {T, V <: AbstractArray{T}}
    (; x, cmat, umat, tmp_rho) = data
    (; cfl, Dt, ideb, ifin) = params

    dt::T = Inf

    if params.cst_dt
        # Constant time step
        dt = Dt
    elseif params.use_gpu && use_ROCM
        # ROCM doesn't support Array Programming, so first we compute `dt` for all cells in the 
        # domain, then we reduce those values.
        if params.euler_projection
            gpu_dtCFL_reduction_euler!(ideb - 1, tmp_rho, x, umat, cmat, ndrange=length(ideb:ifin)) |> wait
        else
            gpu_dtCFL_reduction_lagrange!(ideb - 1, tmp_rho, x, cmat, ndrange=length(ideb:ifin)) |> wait
        end
        dt = reduce(min, tmp_rho[ideb:ifin])
    elseif params.euler_projection
        if params.use_gpu
            dt = reduce(min, @views (
                (x[ideb+1:ifin+1] .- x[ideb:ifin]) ./ max.(
                    abs.(umat[ideb:ifin] .+ cmat[ideb:ifin]), 
                    abs.(umat[ideb:ifin] .- cmat[ideb:ifin]))))
        else
            @batch threadlocal=typemax(T) for i in ideb:ifin
                dt_i = (x[i+1] - x[i]) / max(abs(umat[i] + cmat[i]), abs(umat[i] - cmat[i]))
                threadlocal = min(threadlocal, dt_i)
            end
            dt = minimum(threadlocal)
        end
    else
        if params.use_gpu
            dt = reduce(min, @views ((x[ideb+1:ifin+1] .- x[ideb:ifin]) ./ cmat[ideb:ifin]))
        else
            @batch threadlocal=typemax(T) for i in ideb:ifin
                threadlocal = min(threadlocal, (x[i+1] - x[i]) / cmat[i])
            end
            dt = minimum(threadlocal)
        end
    end

    if dta == 0  # First cycle
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

function dtCFL_MPI(params::ArmonParameters{T}, data::ArmonData{V}, dta::T) where {T, V <: AbstractArray{T}}
    local_dt::T = dtCFL(params, data, dta)

    # Reduce all local_dts and broadcast the result to all processes
    @time_pos params "dt_Allreduce_MPI" dt = MPI.Allreduce(local_dt, MPI.Op(min, T), COMM)
    return dt
end

#
# Numerical fluxes computation
#

function numericalFluxes!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T) where {T, V <: AbstractArray{T}}
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            acoustic!(params, data)
        else
            acoustic_GAD!(params, data, dt)
        end
    else
        error("The choice of Riemann solver is not recognized: ", params.riemann)
    end
end

# 
# Cell update
# 

function first_order_euler_remap!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T) where {T, V <: AbstractArray{T}}
    (; X, rho, umat, vmat, Emat, ustar, tmp_rho, tmp_urho, tmp_vrho, tmp_Erho) = data
    (; ideb, ifin) = params

    if params.use_gpu
        gpu_first_order_euler_remap_1!(ideb - 1, dt, X, ustar, rho, umat, vmat, Emat, 
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, ndrange=length(ideb:ifin)) |> wait
        gpu_first_order_euler_remap_2!(ideb - 1, rho, umat, vmat, Emat, 
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, ndrange=length(ideb:ifin)) |> wait
        return
    end

    # Projection of the conservative variables
    @simd_threaded_loop for i in ideb:ifin
        dx = X[i+1] - X[i]
        L₁ =  max(0, ustar[i]) * dt
        L₃ = -min(0, ustar[i+1]) * dt
        L₂ = dx - L₁ - L₃
        
        tmp_rho[i]  = (L₁ * rho[i-1] 
                     + L₂ * rho[i] 
                     + L₃ * rho[i+1]) / dx
        tmp_urho[i] = (L₁ * rho[i-1] * umat[i-1] 
                     + L₂ * rho[i]   * umat[i] 
                     + L₃ * rho[i+1] * umat[i+1]) / dx
        tmp_vrho[i] = (L₁ * rho[i-1] * vmat[i-1] 
                     + L₂ * rho[i]   * vmat[i] 
                     + L₃ * rho[i+1] * vmat[i+1]) / dx
        tmp_Erho[i] = (L₁ * rho[i-1] * Emat[i-1] 
                     + L₂ * rho[i]   * Emat[i] 
                     + L₃ * rho[i+1] * Emat[i+1]) / dx
    end

    # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E)
    @simd_threaded_loop for i in ideb:ifin
        rho[i]  = tmp_rho[i]
        umat[i] = tmp_urho[i] / tmp_rho[i]
        vmat[i] = tmp_vrho[i] / tmp_rho[i]
        Emat[i] = tmp_Erho[i] / tmp_rho[i]
    end
end


function cellUpdate!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T) where {T, V <: AbstractArray{T}}
    (; x, X, ustar, pstar, rho, umat, vmat, emat, Emat) = data
    (; ideb, ifin) = params

    if params.use_gpu
        if params.euler_projection
            gpu_cell_update_euler!(ideb - 1, ifin, dt, x, X, ustar, pstar, 
                rho, umat, vmat, emat, Emat, ndrange=length(ideb:ifin)) |> wait
        else
            gpu_cell_update_lagrange!(ideb - 1, ifin, dt, x, X, ustar, pstar, 
                rho, umat, vmat, emat, Emat, ndrange=length(ideb:ifin)) |> wait
        end
        return
    end

    @simd_threaded_loop for i in ideb:ifin+1
        X[i] = x[i] + dt * ustar[i]
    end
 
    @simd_threaded_loop for i in ideb:ifin
        dm = rho[i] * (x[i+1] - x[i])
        rho[i] = dm / (X[i+1] - X[i])
        umat[i] = umat[i] + dt / dm * (pstar[i] - pstar[i+1])
        vmat[i] = vmat[i]
        Emat[i] = Emat[i] + dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1])
        emat[i] = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    end

    if !params.euler_projection
        @simd_threaded_loop for i in ideb:ifin+1
            x[i] = X[i]
        end
    end
end


function update_EOS!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, emat, pmat, cmat, gmat) = data
    (; ideb, ifin, test) = params

    if test == :Sod || test == :Leblanc || test == :Woodward
        gamma::T = 0.0

        if test == :Sod || test == :Woodward
            gamma = 1.4
        elseif test == :Leblanc
            gamma = 5/3
        end

        if params.use_gpu
            gpu_update_perfect_gas_EOS!(ideb - 1, gamma, rho, emat, 
                pmat, cmat, gmat, ndrange=length(ideb:ifin)) |> wait
        else
            perfectGasEOS!(params, data, gamma)
        end
    elseif test == :Bizarrium
        if params.use_gpu
            gpu_update_bizarrium_EOS!(ideb - 1, rho, emat, 
                pmat, cmat, gmat, ndrange=length(ideb:ifin)) |> wait
        else
            BizarriumEOS!(params, data)
        end
    end
end

# 
# Main time loop
# 

function time_loop(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, nbcell, silent, is_root) = params

    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    t1 = time_ns()
    t_warmup = t1

    while t < maxtime && cycle < maxcycle
        @time_pos params "boundaryConditions" boundaryConditions_MPI!(params, data)
        @time_pos params "dtCFL"              dt = dtCFL_MPI(params, data, dta)
        @time_pos params "numericalFluxes!"   numericalFluxes!(params, data, dt)
        @time_pos params "cellUpdate!"        cellUpdate!(params, data, dt)

        if params.euler_projection
            @time_pos params "boundaryConditions" boundaryConditions_MPI!(params, data)
            @time_pos params "first_order_euler_remap!" first_order_euler_remap!(params, data, dt)
        end

        @time_pos params "update_EOS!"        update_EOS!(params, data)

        dta = dt
        cycle += 1
        t += dt

        if is_root
            if silent <= 1
                println("Cycle = ", cycle, ", dt = ", dt, ", t = ", t)
            end
    
            if !isfinite(dt) || dt <= 0.
                error("Invalid dt at cycle $(cycle): $(dt)")
            end
    
            if cycle == 5
                t_warmup = time_ns()
            end
        end
    end

    t2 = time_ns()

    grind_time = (t2 - t_warmup) / ((cycle - 5)*nbcell)

    if is_root
        if cycle <= 5 && maxcycle > 5
            error("More than 5 cycles are needed to compute the grind time, got: $(cycle)")
        elseif t2 < t_warmup
            error("Clock error: $(t2) < $(t_warmup)")
        end
    
        if silent < 3
            println(" ")
            println("Time:       ", round((t2 - t1) / 1e9, digits=5),       " sec")
            println("Warmup:     ", round((t_warmup - t1) / 1e9, digits=5), " sec")
            println("Grind time: ", round(grind_time / 1e3, digits=5),      " µs/cell/cycle")
            println("Cells/sec:  ", round(1 / grind_time * 1e3, digits=5),  " Mega cells/sec")
            println("Cycles: ", cycle)
            println(" ")
        end
    end
    
    return dt, cycle, convert(T, 1 / grind_time)
end

#
# Output 
#

function write_result(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; x, rho, umat, vmat, pmat, emat, cmat, gmat, ustar, pstar) = data
    (; ideb, ifin) = params

    f = open("output_proc=$(params.rank)", "w")

    for i in ideb:ifin
        print(f, 0.5*(x[i]+x[i+1]), ", ", 
            rho[i],  ", ", umat[i], ", ", vmat[i],  ", ", pmat[i],  ", ", emat[i],  ", ", 
            cmat[i], ", ", gmat[i], ", ", ustar[i], ", ", pstar[i], "\n")
    end
    
    close(f)

    MPI.Barrier(COMM)

    if params.is_root
        concat_all_outputs = "cat output_proc={0..$(params.proc_size-1)} > output"
        run(`bash -c $concat_all_outputs`)
        remove_all_outputs = "rm -f output_proc={0..$(params.proc_size-1)}"
        run(`bash -c $remove_all_outputs`)
        println("Output file closed")
    end
end

# 
# Main function
# 

function armon(params::ArmonParameters{T}) where T
    (; nghost, nbcell, silent, is_root) = params

    if params.measure_time
        empty!(time_contrib)
    end

    if is_root && silent < 3
        print_parameters(params)
    end
    
    # Allocate without initialisation in order to correctly map the NUMA space using the first-touch
    # policy when working on CPU only
    data_size = nbcell + 2 * nghost
    data = ArmonData(T, data_size)

    init_time = @elapsed init_test(params, data)
    (is_root && silent <= 2) && @printf("Init time: %.3g sec\n", init_time)

    if params.use_gpu
        copy_time = @elapsed d_data = data_to_gpu(data)
        (is_root && silent <= 2) && @printf("Time for copy to device: %.3g sec\n", copy_time)

        if (is_root && silent <= 3)
            @time dt, cycles, cells_per_sec = time_loop(params, d_data)
        else
            dt, cycles, cells_per_sec = time_loop(params, d_data)
        end

        data_from_gpu(data, d_data)
    else
        if (is_root && silent <= 3)
            @time dt, cycles, cells_per_sec = time_loop(params, data)
        else
            dt, cycles, cells_per_sec = time_loop(params, data)
        end
    end

    if params.write_output
        write_result(params, data)
    end

    if params.measure_time && is_root && silent < 3 && !isempty(time_contrib)
        total_time = mapreduce(x->x[2], +, collect(time_contrib))
        println("\nTotal time of each step:")
        for (label, time_) in sort(collect(time_contrib))
            @printf(" - %-25s %10.5f ms (%5.2f%%)\n", label, time_ / 1e6, time_ / total_time * 100)
        end
    end

    return dt, cycles, cells_per_sec, sort(collect(time_contrib))
end

end
