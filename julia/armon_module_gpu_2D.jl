module Armon

using Printf
using Polyester
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

use_ROCM = haskey(ENV, "USE_ROCM_GPU") && ENV["USE_ROCM_GPU"] == "true"

if use_ROCM
    using AMDGPU
    using ROCKernels
    println("Using ROCM GPU")
    macro cuda(args...) end
else
    using CUDA
    using CUDAKernels
    CUDA.allowscalar(false)
    println("Using CUDA GPU")
end


export ArmonParameters, armon


# TODO : dtCFL ?
# TODO : boundaryConditions (y) ?
# p, c, g? -> scalaire
# Passer en for j in ny, for i in nx
# Transposition rho, Emat (emat à mettre dans update EOS)

# EOS à l'init
# NAN aléatoires

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
    nx::Int
    ny::Int
    row_length::Int
    col_length::Int
    nbcell::Int
    ideb::Int
    ifin::Int
    index_start::Int
    cfl::Flt_T
    Dt::Flt_T
    euler_projection::Bool
    cst_dt::Bool
    transpose_dims::Bool

    # Bounds
    maxtime::Flt_T
    maxcycle::Int
    
    # Output
    silent::Int
    write_output::Bool
    write_ghosts::Bool
    measure_time::Bool

    # Performance
    use_ccall::Bool
    use_threading::Bool
    use_simd::Bool
    use_gpu::Bool
end


# Constructor for ArmonParameters
function ArmonParameters(; ieee_bits = 64,
                           test = :Sod, riemann = :acoustic, scheme = :GAD_minmod,
                           iterations = 4, 
                           nghost = 2, nx = 10, ny = 10, cfl = 0.6, Dt = 0., 
                           euler_projection = false, cst_dt = false, transpose_dims = false,
                           maxtime = 0, maxcycle = 500_000,
                           silent = 0, write_output = true, write_ghosts = false, measure_time = true,
                           use_ccall = false, use_threading = true, 
                           use_simd = true, interleaving = false,
                           use_gpu = false)

    flt_type = (ieee_bits == 64) ? Float64 : Float32

    # Make sure that all floating point types are the same
    cfl = flt_type(cfl)
    Dt = flt_type(Dt)
    maxtime = flt_type(maxtime)
    
    if riemann == :one_iteration_acoustic iterations = 1 end
    if riemann == :two_iteration_acoustic iterations = 2 end

    if cst_dt && Dt == zero(flt_type)
        error("Dt == 0 with constant step enabled")
    end
    
    if use_ccall
        error("The C librairy only supports 1D")
    end

    if interleaving
        error("No support for interleaving in 2D")
    end

    row_length = nghost * 2 + nx
    col_length = nghost * 2 + ny
    ideb = row_length * nghost + nghost + 1
    ifin = row_length * (ny - 1 + nghost) + nghost + nx
    nbcell = row_length * col_length
    index_start = ideb - row_length - 1
    
    return ArmonParameters{flt_type}(test, riemann, scheme, 
                                     iterations, 
                                     nghost, nx, ny, row_length, col_length, nbcell,
                                     ideb, ifin, index_start,
                                     cfl, Dt, euler_projection, cst_dt, transpose_dims,
                                     maxtime, maxcycle,
                                     silent, write_output, write_ghosts, measure_time,
                                     use_ccall, use_threading, use_simd, use_gpu)
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
    println(" - use_simd:   ", p.use_simd)
    println(" - use_ccall:  ", p.use_ccall)
    println(" - use_gpu:    ", p.use_gpu)
    println(" - ieee_bits:  ", sizeof(T) * 8)
    println("")
    println(" - test:       ", p.test)
    println(" - riemann:    ", p.riemann)
    println(" - scheme:     ", p.scheme)
    println(" - iterations: ", p.iterations)
    println(" - domain:     ", p.nx, "x", p.ny, " (", p.nghost, " ghosts)")
    println(" - nbcell:     ", length(p.ideb:p.ifin), " (", p.nbcell, " total)")
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt)
    println(" - euler proj: ", p.euler_projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
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
    y::V
    Y::V
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
        device_type(data.y),
        device_type(data.Y),
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
    copyto!(host_data.y, device_data.y)
    copyto!(host_data.Y, device_data.Y)
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

    # Only in for the case of a threaded loop with SIMD:
    # Extract the range of the loop and replace it with the new variables
    modified_loop_expr = copy(expr)
    range_expr = modified_loop_expr.args[1]

    if range_expr.head == :block
        # Compound range expression: "j in 1:3, i in 4:6"
        # Use the first range as the threaded loop range
        loop_range = copy(range_expr.args[1].args[2])
        range_expr.args[1].args[2] = :(__ideb:__ifin)
    elseif range_expr.head == Symbol("=")
        # Single range expression: "j in 1:3"
        loop_range = copy(range_expr.args[2])
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
# Indexing macros
#

macro indexing_vars(params)
    return esc(quote
        (; index_start, row_length, nghost) = $(params)
    end)
end

macro i(i, j)
    return esc(quote
        index_start + $(j) * row_length + $(i)
    end)
end

macro I(i)
    return esc(quote
        (($(i)-1) % row_length) + 1 - nghost, (($(i)-1) ÷ row_length) + 1 - nghost
    end)
end

#
# GPU Kernels
#

const use_native_CUDA = haskey(ENV, "USE_NATIVE_CUDA") && ENV["USE_NATIVE_CUDA"] == "true"

if use_native_CUDA
    # Replace KernelAbstractions.jl's macros with their CUDA.jl equivalent.
    println("Using native CUDA")

    macro identity(expr) return esc(expr) end

    macro make_return_nothing(func)
        push!(func.args[2].args, Expr(:return, :nothing))
        return esc(quote
            $(func)
        end)
    end

    macro cuda_index(scope)
        if scope == :Local
            return esc(quote
                threadIdx().x
            end)
        elseif scope == :Global
            return esc(quote
                (blockIdx().x - 1) * blockDim().x + threadIdx().x
            end)
        else
            error("Unknown index scope: " * string(scope))
        end
    end

    macro cuda_synchronize()
        return esc(quote 
            sync_threads() 
        end)
    end

    macro cuda_localmem(type, dims)
        return esc(quote
            CuStaticSharedArray(type, dims)
        end)
    end

    var"@Const" = var"@identity"
    var"@kernel" = var"@make_return_nothing"
    var"@index" = var"@cuda_index"
    var"@synchronize" = var"@cuda_synchronize"
    var"@localmem" = var"@cuda_localmem"
end


const device = use_ROCM ? ROCDevice() : CUDADevice()
const block_size = haskey(ENV, "GPU_BLOCK_SIZE") ? parse(Int, ENV["GPU_BLOCK_SIZE"]) : 32
const reduction_block_size = 1024;
const reduction_block_size_log2 = convert(Int, log2(reduction_block_size))


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
    # We must use this instead of X[i+1]-X[i] since X can be overwritten by other workgroups
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
    # We must use this instead of X[i+1]-X[i] since X can be overwritten by other workgroups
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

    data_type = eltype(rho)

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


@kernel function gpu_boundary_conditions_kernel!(test_bizarrium, ideb, ifin, 
        rho, umat, vmat, pmat, cmat, gmat)
    rho[ideb-1]  = rho[ideb]
    umat[ideb-1] = -umat[ideb]
    vmat[ideb-1] = vmat[ideb]
    pmat[ideb-1] = pmat[ideb]
    cmat[ideb-1] = cmat[ideb]
    gmat[ideb-1] = gmat[ideb]

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


# Construction of the kernels for a common device and block size
if use_native_CUDA
    global compiled_kernels = Dict{Function, CUDA.HostKernel}()

    function convert_KA_to_CUDA_call(kernel, args...; ndrange=1, block_size=block_size)
        if !haskey(compiled_kernels, kernel)
            compiled_kernel = @cuda launch=false kernel(args...)
            compiled_kernels[kernel] = compiled_kernel
        end
        
        threads = block_size
        blocks = cld(ndrange, threads)
        return compiled_kernels[kernel](args...; threads, blocks)

        # threads = block_size
        # blocks = cld(ndrange, threads)
        # return @cuda threads=threads blocks=blocks kernel(args...)
    end

    gpu_acoustic!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_acoustic_kernel!, args...; ndrange, block_size)
    gpu_acoustic_GAD_minmod!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_acoustic_GAD_minmod_kernel!, args...; ndrange, block_size)
    gpu_cell_update_lagrange!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_cell_update_lagrange_kernel!, args...; ndrange, block_size)
    gpu_cell_update_euler!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_cell_update_euler_kernel!, args...; ndrange, block_size)
    gpu_first_order_euler_remap_1!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_first_order_euler_remap_kernel!, args...; ndrange, block_size)
    gpu_first_order_euler_remap_2!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_first_order_euler_remap_2_kernel!, args...; ndrange, block_size)
    gpu_update_perfect_gas_EOS!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_update_perfect_gas_EOS_kernel!, args...; ndrange, block_size)
    gpu_update_bizarrium_EOS!(args...; ndrange) = convert_KA_to_CUDA_call(gpu_update_bizarrium_EOS_kernel!, args...; ndrange, block_size)
    gpu_boundary_conditions!(args...) = convert_KA_to_CUDA_call(gpu_boundary_conditions_kernel!, args...; ndrange=1, block_size=1)
    gpu_dtCFL_reduction!(args...) = convert_KA_to_CUDA_call(gpu_dtCFL_reduction_kernel!, args...; ndrange=reduction_block_size, block_size=reduction_block_size)
else
    gpu_acoustic! = gpu_acoustic_kernel!(device, block_size)
    gpu_acoustic_GAD_minmod! = gpu_acoustic_GAD_minmod_kernel!(device, block_size)
    gpu_cell_update_lagrange! = gpu_cell_update_lagrange_kernel!(device, block_size)
    gpu_cell_update_euler! = gpu_cell_update_euler_kernel!(device, block_size)
    gpu_first_order_euler_remap_1! = gpu_first_order_euler_remap_kernel!(device, block_size)
    gpu_first_order_euler_remap_2! = gpu_first_order_euler_remap_2_kernel!(device, block_size)
    gpu_update_perfect_gas_EOS! = gpu_update_perfect_gas_EOS_kernel!(device, block_size)
    gpu_update_bizarrium_EOS! = gpu_update_bizarrium_EOS_kernel!(device, block_size)
    gpu_boundary_conditions! = gpu_boundary_conditions_kernel!(device, 1, 1)
    gpu_dtCFL_reduction! = gpu_dtCFL_reduction_kernel!(device, reduction_block_size, reduction_block_size)
end

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
        x = rho[i]/rho0 - 1; g = G0*(1-rho0/rho[i]) # formula (4b)

        f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)   # Formula (15b)
        f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)  # Formula (16a)
        f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)          # Formula (16b)
        f3 = (6*r+3*s*f2)/(1-s*x)                # Formula (16c)

        epsk0 = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0                                # Formula (15a)
        pk0 = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)                              # Formula (17a)
        pk0prime = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)    # Formula (17b)
        pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 +           # Formula (17c)
                                            6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

        pmat[i] = pk0 + G0*rho0*(emat[i] - epsk0)                                     # Formula (5b)
        cmat[i] = sqrt(G0*rho0*(pmat[i] - pk0) - pk0prime) / rho[i]                 # Formula (8)
        gmat[i] = 0.5/(rho[i]^3*cmat[i]^2)*(pk0second+(G0*rho0)^2*(pmat[i]-pk0))  # Formula (8) + (11)
    end
end

#
# Acoustic Riemann problem solvers
# 

function acoustic!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; ustar, pstar, rho, umat, pmat, cmat) = data
    (; ideb, ifin) = params

    if params.use_gpu
        gpu_acoustic!(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ndrange=length(ideb:ifin+1))
    else
        @simd_threaded_loop for i in ideb:ifin+1
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            ustar[i] = (rc_l * umat[i-1] + rc_r * umat[i] +
                              (pmat[i-1] - pmat[i])) / (rc_l + rc_r)
            pstar[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] + 
                rc_l * rc_r * (umat[i-1] - umat[i])) / (rc_l + rc_r)
        end
    end
end


function acoustic_GAD!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T) where {T, V <: AbstractArray{T}}
    (; x, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1) = data
    (; scheme, ideb, ifin) = params

    if params.use_gpu
        if params.scheme != :GAD_minmod
            error("Only the minmod limiter is implemented for GPU")
        end
        gpu_acoustic!(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ndrange=length(ideb:ifin+1))
        gpu_acoustic_GAD_minmod!(ideb - 1, ustar, pstar, rho, umat, pmat, cmat, ustar_1, pstar_1,
            dt, umat, ndrange=length(ideb:ifin+1))
        return
    end

    # First order
    @simd_threaded_loop for i in ideb:ifin
        rc_l = rho[i-1] * cmat[i-1]
        rc_r = rho[i]   * cmat[i]
        ustar_1[i] = (rc_l * umat[i-1] + rc_r * umat[i] +
                          (pmat[i-1] - pmat[i])) / (rc_l + rc_r)
        pstar_1[i] = (rc_r * pmat[i-1] + rc_l * pmat[i] +
            rc_l * rc_r * (umat[i-1] - umat[i])) / (rc_l + rc_r)
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
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            Dm = (dm_l + dm_r) / 2
            θ  = (rc_l + rc_r) / 2 * (dt / Dm)
            
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
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            Dm = (dm_l + dm_r) / 2
            θ  = (rc_l + rc_r) / 2 * (dt / Dm)
            
            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (umat[i] - ustar_1[i]) - 
                                                     r_u_m * (ustar_1[i] - umat[i-1]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (pmat[i] - pstar_1[i]) -
                                                     r_p_m * (pstar_1[i] - pmat[i-1]))
        end
    elseif scheme == :GAD_no_limiter
        @simd_threaded_loop for i in ideb:ifin+1
            dm_l = rho[i-1] * (x[i] - x[i-1])
            dm_r = rho[i]   * (x[i+1] - x[i])
            rc_l = rho[i-1] * cmat[i-1]
            rc_r = rho[i]   * cmat[i]
            Dm = (dm_l + dm_r) / 2
            θ  = (rc_l + rc_r) / 2 * (dt / Dm)

            ustar[i] = ustar_1[i] + 1/2 * (1 - θ) * (r_u_p * (umat[i] - ustar_1[i]) - 
                                                     r_u_m * (ustar_1[i] - umat[i-1]))
            pstar[i] = pstar_1[i] + 1/2 * (1 - θ) * (r_p_p * (pmat[i] - pstar_1[i]) - 
                                                     r_p_m * (pstar_1[i] - pmat[i-1]))
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
    (; x, y, rho, pmat, umat, vmat, emat, Emat, cmat, gmat) = data
    (; test, nghost, nbcell, nx, ny, row_length) = params

    if test == :Sod
        if params.maxtime == 0
            params.maxtime = 0.20
        end
    
        if params.cfl == 0
            params.cfl = 0.95
        end

        gamma::T = 1.4
    
        @simd_threaded_loop for i in 1:nbcell
            ix = (i-1) % row_length
            iy = (i-1) ÷ row_length

            x[i] = (ix-nghost) / nx
            y[i] = (iy-nghost) / ny
    
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
        end
    elseif test == :Bizarrium
        if params.maxtime == 0
            params.maxtime = 80e-6
        end
    
        if params.cfl == 0
            params.cfl = 0.6
        end
    
        @simd_threaded_loop for i in 1:nbcell
            ix = (i-1) % row_length
            iy = (i-1) ÷ row_length

            x[i] = (ix-nghost) / nx
            y[i] = (iy-nghost) / ny
    
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
        end
    
        BizarriumEOS!(params, data)
    end
    
    return
end

#
# Boundary conditions
#

function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, umat, vmat, pmat, cmat, gmat) = data
    (; test, ideb, ifin, nx, ny) = params
    @indexing_vars(params)

    if params.use_gpu
        gpu_boundary_conditions!(test == :Bizarrium, ideb, ifin, rho, umat, vmat, pmat, cmat, gmat)
        return
    end

    # Mirror the u component on the right of the x axis for the Sod test case, but not for Bizarrium
    u_factor_right = test == :Bizarrium ? 1 : -1

    @threaded for j in 1:ny
        # Condition for the left border of the domain
        rho[@i(0,j)]  = rho[@i(1,j)]
        umat[@i(0,j)] =-umat[@i(1,j)]
        vmat[@i(0,j)] = vmat[@i(1,j)]
        pmat[@i(0,j)] = pmat[@i(1,j)]
        cmat[@i(0,j)] = cmat[@i(1,j)]
        gmat[@i(0,j)] = gmat[@i(1,j)]

        # Condition for the right border of the domain
        rho[@i(nx+1, j)] = rho[@i(nx,j)]
        umat[@i(nx+1,j)] = umat[@i(nx,j)] * u_factor_right
        vmat[@i(nx+1,j)] = vmat[@i(nx,j)]
        pmat[@i(nx+1,j)] = pmat[@i(nx,j)]
        cmat[@i(nx+1,j)] = cmat[@i(nx,j)]
        gmat[@i(nx+1,j)] = gmat[@i(nx,j)]
    end

    #=
    @threaded for i in 1:nx
        # Condition for the bottom border of the domain
        rho[@i(i,0)]  = rho[@i(i,1)]
        umat[@i(i,0)] = umat[@i(i,1)]
        vmat[@i(i,0)] = vmat[@i(i,1)]
        pmat[@i(i,0)] = pmat[@i(i,1)]
        cmat[@i(i,0)] = cmat[@i(i,1)]
        gmat[@i(i,0)] = gmat[@i(i,1)]

        # Condition for the top border of the domain
        rho[@i(i,ny+1)]  = rho[@i(i,ny)]
        umat[@i(i,ny+1)] = umat[@i(i,ny)]
        vmat[@i(i,ny+1)] = vmat[@i(i,ny)]
        pmat[@i(i,ny+1)] = pmat[@i(i,ny)]
        cmat[@i(i,ny+1)] = cmat[@i(i,ny)]
        gmat[@i(i,ny+1)] = gmat[@i(i,ny)]
    end
    =#
end

#
# Time step computation
#

function dtCFL(params::ArmonParameters{T}, data::ArmonData{V}, dta::T) where {T, V <: AbstractArray{T}}
    (; x, cmat, umat) = data
    (; cfl, Dt, ideb, ifin) = params
    @indexing_vars(params)

    dt::T = Inf

    # TODO : think about treating ghost cells differenlty: their values must not change the result (set them to Inf)

    if params.cst_dt
        # Constant time step
        dt = Dt
    elseif params.use_gpu && use_ROCM
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
        for (tid, (value, err)) in enumerate(zip(tmp_values, tmp_err_i))
            #@printf("TID %2d: %f\n", tid, value)
            #if err >= 0
                @printf("TID %3d: err pos=%d, err=%g\n", tid, err, value)
            #end
        end
    elseif params.euler_projection
        if params.use_gpu
            dt = reduce(min, @views (
                (x[ideb+1:ifin+1] .- x[ideb:ifin]) ./ max.(
                    abs.(umat[ideb:ifin] .+ cmat[ideb:ifin]), 
                    abs.(umat[ideb:ifin] .- cmat[ideb:ifin]))))
        else
            (; nx, ny) = params
            @indexing_vars(params)
            @batch threadlocal=typemax(T) for i in ideb:ifin
                ix, iy = @I(i)
                if 1 ≤ ix ≤ nx && 1 ≤ iy ≤ ny
                    dt_i = (x[i+1] - x[i]) / max(abs(umat[i] + cmat[i]), 
                                                 abs(umat[i] - cmat[i]))
                    threadlocal = min(threadlocal, dt_i)
                end
                # dt_i = (x[i+1] - x[i]) / max(abs(umat[i] + cmat[i]), 
                #                              abs(umat[i] - cmat[i]))
                # threadlocal = min(threadlocal, dt_i)
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
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, ndrange=length(ideb:ifin))
        gpu_first_order_euler_remap_2!(ideb - 1, rho, umat, vmat, Emat, 
            tmp_rho, tmp_urho, tmp_vrho, tmp_Erho, ndrange=length(ideb:ifin))
        return
    end

    # Projection of the conservative variables
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

    # (ρ, ρu, ρv, ρE) -> (ρ, u, v, E)
    @simd_threaded_loop for i in ideb:ifin
        rho[i]  = tmp_rho[i]
        umat[i] = tmp_urho[i] / tmp_rho[i]
        vmat[i] = tmp_vrho[i] / tmp_rho[i]
        Emat[i] = tmp_Erho[i] / tmp_rho[i]
    end
end


function cellUpdate!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T) where {T, V <: AbstractArray{T}}
    (; x, X, ustar, pstar, rho, umat, vmat, emat, Emat) = data
    (; ideb, ifin, nx) = params

    if params.use_gpu
        if params.euler_projection
            gpu_cell_update_euler!(ideb - 1, ifin, dt, x, X, ustar, pstar, 
                rho, umat, vmat, emat, Emat, ndrange=length(ideb:ifin))
        else
            gpu_cell_update_lagrange!(ideb - 1, ifin, dt, x, X, ustar, pstar, 
                rho, umat, vmat, emat, Emat, ndrange=length(ideb:ifin))
        end
        return
    end

    @simd_threaded_loop for i in ideb:ifin
        X[i] = x[i] + dt * ustar[i]
    end

    @simd_threaded_loop for i in ideb:ifin
        dx  = 1. / nx
        #dm  = rho[i] * (x[i+1] - x[i])
        dm  = rho[i] * dx
        #rho[i]  = dm / (X[i+1] - X[i])
        rho[i]  = dm / (dx + dt * (ustar[i+1] - ustar[i]))

        umat[i] = umat[i] + dt / dm * (pstar[i] - pstar[i+1])
        vmat[i] = vmat[i]
        Emat[i] = Emat[i] + dt / dm * (pstar[i] * ustar[i] - pstar[i+1] * ustar[i+1])
        emat[i] = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    end
 
    if !params.euler_projection
        @simd_threaded_loop for i in ideb:ifin
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
                pmat, cmat, gmat, ndrange=length(ideb:ifin))
        else
            perfectGasEOS!(params, data, gamma)
        end
    elseif test == :Bizarrium
        if params.use_gpu
            gpu_update_bizarrium_EOS!(ideb - 1, rho, emat, 
                pmat, cmat, gmat, ndrange=length(ideb:ifin))
        else
            BizarriumEOS!(params, data)
        end
    end
end

# 
# Main time loop
# 

function time_loop(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent) = params
    
    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.

    t1 = time_ns()
    t_warmup = t1

    while t < maxtime && cycle < maxcycle
        @time_pos params "boundaryConditions" boundaryConditions!(params, data)
        @time_pos params "dtCFL"              dt = dtCFL(params, data, dta)

        @time_pos params "numericalFluxes!"   numericalFluxes!(params, data, dt)
        @time_pos params "cellUpdate!"        cellUpdate!(params, data, dt)

        if params.euler_projection
            @time_pos params "first_order_euler_remap!" first_order_euler_remap!(params, data, dt)
        end

        @time_pos params "update_EOS!"        update_EOS!(params, data)

        dta = dt
        cycle += 1
        t += dt

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

    t2 = time_ns()

    if cycle <= 5 && maxcycle > 5
        error("More than 5 cycles are needed to compute the grind time, got: $(cycle)")
    elseif t2 < t_warmup
        error("Clock error: $(t2) < $(t_warmup)")
    end
    
    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((cycle - 5)*nb_cells)

    if silent < 3
        println(" ")
        println("Time:       ", round((t2 - t1) / 1e9, digits=5),       " sec")
        println("Warmup:     ", round((t_warmup - t1) / 1e9, digits=5), " sec")
        println("Grind time: ", round(grind_time / 1e3, digits=5),      " µs/cell/cycle")
        println("Cells/sec:  ", round(1 / grind_time * 1e3, digits=5),  " Mega cells/sec")
        println("Cycles: ", cycle)
        println(" ")
    end

    return convert(T, 1 / grind_time)
end

#
# Output 
#

function write_result(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; x, y, rho) = data
    (; silent, write_ghosts, nx, ny, nghost) = params
    @indexing_vars(params)

    f = open("output", "w")

    if write_ghosts
        for j in 1-nghost:ny+nghost
            for i in 1-nghost:nx+nghost
                print(f, x[@i(i, j)], ", ", y[@i(i, j)], ", ", rho[@i(i, j)], "\n")
            end
        end
    else
        for j in 1:ny
            for i in 1:nx
                print(f, x[@i(i, j)], ", ", y[@i(i, j)], ", ", rho[@i(i, j)], "\n")
            end
        end
    end
    
    close(f)

    if silent < 2
        println("Output file closed")
    end
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
        if params.use_gpu
            println(" - gpu block size: ", block_size)
        end
    end
    
    # Allocate without initialisation in order to correctly map the NUMA space using the first-touch
    # policy when working on CPU only
    data = ArmonData(T, params.nbcell)

    x .= NaN
    X .= NaN
    y .= NaN
    Y .= NaN
    rho .= NaN
    umat .= NaN
    vmat .= NaN
    emat .= NaN
    Emat .= NaN
    pmat .= NaN
    cmat .= NaN
    gmat .= NaN
    ustar .= NaN
    pstar .= NaN
    ustar_1 .= NaN
    pstar_1 .= NaN
    tmp_rho .= NaN
    tmp_urho .= NaN
    tmp_vrho .= NaN
    tmp_Erho .= NaN

    init_time = @elapsed init_test(params, data)
    silent <= 2 && @printf("Init time: %.3g sec\n", init_time)

    println("x:        ", x)
    println("X:        ", X)
    println("y:        ", y)
    println("Y:        ", Y)
    println("rho:      ", rho)
    println("umat:     ", umat)
    println("vmat:     ", vmat)
    println("emat:     ", emat)
    println("Emat:     ", Emat)
    println("pmat:     ", pmat)
    println("cmat:     ", cmat)
    println("gmat:     ", gmat)
    println("ustar:    ", ustar)
    println("pstar:    ", pstar)
    println("ustar_1:  ", ustar_1)
    println("pstar_1:  ", pstar_1)
    println("tmp_rho:  ", tmp_rho)
    println("tmp_urho: ", tmp_urho)
    println("tmp_vrho: ", tmp_vrho)
    println("tmp_Erho: ", tmp_Erho)

    if params.use_gpu
        copy_time = @elapsed d_data = data_to_gpu(data)
        silent <= 2 && @printf("Time for copy to device: %.3g sec\n", copy_time)

        if silent <= 3
            @time cells_per_sec = time_loop(params, d_data)
        else
            cells_per_sec = time_loop(params, d_data)
        end

        data_from_gpu(data, d_data)
    else
        if silent <= 3
            @time cells_per_sec = time_loop(params, data)
        else
            cells_per_sec = time_loop(params, data)
        end
    end

    if params.write_output
        write_result(params, data)
    end

    if params.measure_time && silent < 3 && !isempty(time_contrib)
        total_time = mapreduce(x->x[2], +, collect(time_contrib))
        println("\nTotal time of each step:")
        for (label, time_) in sort(collect(time_contrib))
            @printf(" - %-25s %10.5f ms (%5.2f%%)\n", label, time_ / 1e6, time_ / total_time * 100)
        end
    end

    return cells_per_sec, sort(collect(time_contrib))
end

end
