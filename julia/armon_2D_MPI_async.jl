module Armon

using Printf
using Polyester
using ThreadPinning
using KernelAbstractions
using MPI
using MacroTools
using AMDGPU
using ROCKernels
using CUDA
using CUDAKernels

export ArmonParameters, armon

# TODO LIST
# better test implementation (common sturcture, one test = f(x, y) -> rho, pmat, umat, vmat, Emat + boundary conditions + EOS)
# use types and function overloads to define limiters and tests (in the hope that everything gets inlined)
# center the positions of the cells in the output file
# Remove all generics : 'where {T, V <: AbstractVector{T}}' etc... when T and V are not used in the method. Omitting the 'where' will not change anything.
# Merge GAD and euler 2nd kernels into a single one + delete the unneeded arrays
# Bug: `conservation_vars` doesn't give correct values with MPI, even though the solution is correct
# Bug: fix dtCFL on AMDGPU

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
    axis_splitting::Symbol
    projection::Symbol

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
    output_precision::Int
    merge_files::Bool
    animation_step::Int
    measure_time::Bool

    # Performance
    use_ccall::Bool
    use_threading::Bool
    use_simd::Bool
    use_gpu::Bool
    device::Device
    block_size::Int

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
    extra_ring_width::Int  # Number of cells to compute additionally when 'single_comm_per_axis_pass' is true
    reorder_grid::Bool
    comm_array_size::Int

    # Asynchronicity
    async_comms::Bool

    # Tests & Comparaison
    compare::Bool
    is_ref::Bool
    comparison_tolerance::Float64
end


# Constructor for ArmonParameters
function ArmonParameters(;
        ieee_bits = 64,
        test = :Sod, riemann = :acoustic, scheme = :GAD_minmod, projection = :euler,
        nghost = 2, nx = 10, ny = 10, 
        cfl = 0.6, Dt = 0., cst_dt = false, dt_on_even_cycles = false,
        transpose_dims = false, axis_splitting = :Sequential,
        maxtime = 0, maxcycle = 500_000,
        silent = 0, output_dir = ".", output_file = "output",
        write_output = true, write_ghosts = false, output_precision = 6, merge_files = false,
        animation_step = 0, 
        measure_time = true,
        use_ccall = false, use_threading = true, 
        use_simd = true, interleaving = false,
        use_gpu = false, device = :CUDA, block_size = 1024,
        use_MPI = true, px = 1, py = 1,
        single_comm_per_axis_pass = false, reorder_grid = true, 
        async_comms = true,
        compare = false, is_ref = false, comparison_tolerance = 1e-10
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

    min_nghost = 1
    min_nghost += (scheme != :Godunov)
    min_nghost += single_comm_per_axis_pass
    min_nghost += (projection == :euler_2nd)

    if nghost < min_nghost
        error("Not enough ghost cells for the scheme and/or projection, at least $min_nghost are needed.")
    end

    if (nx % px != 0) || (ny % py != 0)
        error("The dimensions of the global domain ($nx x $ny) are not divisible by the number of processors ($px x $py)")
    end

    # GPU
    if use_gpu
        if device == :CUDA
            CUDA.allowscalar(false)
            device = CUDADevice()
        elseif device == :ROCM
            AMDGPU.allowscalar(false)
            device = ROCDevice()
        elseif device == :CPU
            device = CPU()  # Useful in some cases for debugging
        else
            error("Unknown GPU device: $device")
        end
    else
        device = CPU()
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
            top    = MPI.PROC_NULL, 
            bottom = MPI.PROC_NULL,
            left   = MPI.PROC_NULL,
            right  = MPI.PROC_NULL
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

    # Ring width
    if single_comm_per_axis_pass
        extra_ring_width = 1
        extra_ring_width += projection == :euler_2nd
    else
        extra_ring_width = 0
    end

    if use_MPI
        comm_array_size = max(nx, ny) * nghost * 7
    else
        comm_array_size = 0
    end
    
    return ArmonParameters{flt_type}(
        test, riemann, scheme,
        nghost, nx, ny, dx,
        cfl, Dt, cst_dt, dt_on_even_cycles,
        axis_splitting, projection,
        row_length, col_length, nbcell,
        ideb, ifin, index_start,
        idx_row, idx_col,
        X_axis, 1,
        maxtime, maxcycle,
        silent, output_dir, output_file,
        write_output, write_ghosts, output_precision, merge_files, animation_step,
        measure_time,
        use_ccall, use_threading, use_simd, use_gpu, device, block_size,
        use_MPI, is_root, rank, root_rank, 
        proc_size, (px, py), C_COMM, (cx, cy), neighbours, (g_nx, g_ny),
        single_comm_per_axis_pass, extra_ring_width, reorder_grid, comm_array_size,
        async_comms,
        compare, is_ref, comparison_tolerance
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
        println()
    end
    println(" - use_simd:   ", p.use_simd)
    println(" - use_ccall:  ", p.use_ccall)
    print(" - use_gpu:    ", p.use_gpu)
    if p.use_gpu
        print(", ")
        if p.device == CPU()
            println("CPU")
        elseif p.device == CUDADevice()
            println("CUDA")
        elseif p.device == ROCDevice()
            println("ROCm")
        else
            println("<unknown device>")
        end
        println(" - block size: ", p.block_size)
    else
        println()
    end
    println(" - use_MPI:    ", p.use_MPI)
    println(" - ieee_bits:  ", sizeof(T) * 8)
    println()
    println(" - test:       ", p.test)
    println(" - riemann:    ", p.riemann)
    println(" - scheme:     ", p.scheme)
    println(" - splitting:  ", p.axis_splitting)
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt, p.dt_on_even_cycles ? ", updated only for even cycles" : "")
    println(" - euler proj: ", p.projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
    println()
    println(" - domain:     ", p.nx, "x", p.ny, " (", p.nghost, " ghosts)")
    println(" - nbcell:     ", @sprintf("%g", p.nx * p.ny), " (", p.nbcell, " total)")
    println(" - global:     ", p.global_grid[1], "x", p.global_grid[2])
    println(" - proc grid:  ", p.proc_dims[1], "x", p.proc_dims[2], " ($(p.reorder_grid ? "" : "not ")reordered)")
    println(" - coords:     ", p.cart_coords[1], "x", p.cart_coords[2], " (rank: ", p.rank, "/", p.proc_size-1, ")")
    println(" - comms per axis: ", p.single_comm_per_axis_pass ? 1 : 2)
    println(" - asynchronous communications: ", p.async_comms)
    println(" - measure step times: ", p.measure_time)
    println()
    if p.write_output
        println(" - write output: ", p.write_output, " (precision: ", p.output_precision, " digits)")
        println(" - write ghosts: ", p.write_ghosts)
        println(" - output file: ", p.output_file)
        if p.compare
            println(" - compare: ", p.compare, p.is_ref ? ", as reference" : "")
            println(" - tolerance: ", p.comparison_tolerance)
        end
        println()
    end
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


function data_to_gpu(data::ArmonData{V}, device_array) where {T, V <: AbstractArray{T}}
    return ArmonData{device_array{T}}(
        device_array(data.x),
        device_array(data.y),
        device_array(data.rho),
        device_array(data.umat),
        device_array(data.vmat),
        device_array(data.Emat),
        device_array(data.pmat),
        device_array(data.cmat),
        device_array(data.gmat),
        device_array(data.ustar),
        device_array(data.pstar),
        device_array(data.ustar_1),
        device_array(data.pstar_1),
        device_array(data.tmp_rho),
        device_array(data.tmp_urho),
        device_array(data.tmp_vrho),
        device_array(data.tmp_Erho),
        device_array(data.domain_mask),
        device_array(data.tmp_comm_array)
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
# Domain ranges
#

include("domain_ranges.jl")

#
# Threading and SIMD control macros
#

include("generic_kernel.jl")

#
# Execution Time Measurement
#

include("timing_macros.jl")

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
# Limiters
#

abstract type Limiter end

struct NoLimiter       <: Limiter end
struct MinmodLimiter   <: Limiter end
struct SuperbeeLimiter <: Limiter end

limiter(_::T, ::NoLimiter)       where T = one(T)
limiter(r::T, ::MinmodLimiter)   where T = max(zero(T), min(one(T), r))
limiter(r::T, ::SuperbeeLimiter) where T = max(zero(T), min(2r, one(T)), min(r, 2*one(T)))

#
# Kernels
# 

@generic_kernel function acoustic!_kernel(s::Int, ustar_::V, pstar_::V, 
        rho::V, u::V, pmat::V, cmat::V) where V
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()
    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    ustar_[i] = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
    pstar_[i] = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)
end


@generic_kernel function acoustic_GAD_minmod!_kernel(s::Int, dt::T, dx::T, ustar::V, pstar::V,
        rho::V, u::V, pmat::V, cmat::V, ustar_1::V, pstar_1::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()

    r_u_m = (ustar_1[i+s] -      u[i]) / (ustar_1[i] -    u[i-s] + 1e-6)
    r_p_m = (pstar_1[i+s] -   pmat[i]) / (pstar_1[i] - pmat[i-s] + 1e-6)
    r_u_p = (   u[i-s] - ustar_1[i-s]) / (   u[i] -   ustar_1[i] + 1e-6)
    r_p_p = (pmat[i-s] - pstar_1[i-s]) / (pmat[i] -   pstar_1[i] + 1e-6)

    # TODO : make the limiter a parameter
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


@generic_kernel function acoustic_GAD_minmod_single_kernel!_kernel(
        s::Int, dt::T, dx::T, ustar::V, pstar::V,
        rho::V, u::V, pmat::V, cmat::V, ustar_1::V, pstar_1::V)
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()

    i = i - s
    rc_l₋ = rho[i-s] * cmat[i-s]
    rc_r₋ = rho[i]   * cmat[i]
    ustar_i₋ = (rc_l₋*   u[i-s] + rc_r₋*   u[i] +             (pmat[i-s] - pmat[i])) / (rc_l₋ + rc_r₋)
    pstar_i₋ = (rc_r₋*pmat[i-s] + rc_l₋*pmat[i] + rc_l₋*rc_r₋*(   u[i-s] -    u[i])) / (rc_l₋ + rc_r₋)

    i = i + s
    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    ustar_i = (rc_l*   u[i-s] + rc_r*   u[i] +           (pmat[i-s] - pmat[i])) / (rc_l + rc_r)
    pstar_i = (rc_r*pmat[i-s] + rc_l*pmat[i] + rc_l*rc_r*(   u[i-s] -    u[i])) / (rc_l + rc_r)

    i = i + s
    rc_l₊ = rho[i-s] * cmat[i-s]
    rc_r₊ = rho[i]   * cmat[i]
    ustar_i₊ = (rc_l₊*   u[i-s] + rc_r₊*   u[i] +             (pmat[i-s] - pmat[i])) / (rc_l₊ + rc_r₊)
    pstar_i₊ = (rc_r₊*pmat[i-s] + rc_l₊*pmat[i] + rc_l₊*rc_r₊*(   u[i-s] -    u[i])) / (rc_l₊ + rc_r₊)

    i = i - s
    
    r_u_m = (ustar_i₊ -      u[i]) / (ustar_i -    u[i-s] + 1e-6)
    r_p_m = (pstar_i₊ -   pmat[i]) / (pstar_i - pmat[i-s] + 1e-6)
    r_u_p = (   u[i-s] - ustar_i₋) / (   u[i] -   ustar_i + 1e-6)
    r_p_p = (pmat[i-s] - pstar_i₋) / (pmat[i] -   pstar_i + 1e-6)

    r_u_m = max(0., min(1., r_u_m))
    r_p_m = max(0., min(1., r_p_m))
    r_u_p = max(0., min(1., r_u_p))
    r_p_p = max(0., min(1., r_p_p))

    dm_l = rho[i-s] * dx
    dm_r = rho[i]   * dx
    Dm   = (dm_l + dm_r) / 2
    θ    = 1/2 * (1 - (rc_l + rc_r) / 2 * (dt / Dm))
    
    ustar[i] = ustar_i + θ * (r_u_p * (   u[i] - ustar_i) - r_u_m * (ustar_i -    u[i-s]))
    pstar[i] = pstar_i + θ * (r_p_p * (pmat[i] - pstar_i) - r_p_m * (pstar_i - pmat[i-s]))
end


@generic_kernel function update_perfect_gas_EOS!_kernel(gamma::T, 
        rho::V, Emat::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()
    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = (gamma - 1.) * rho[i] * e
    cmat[i] = sqrt(gamma * pmat[i] / rho[i])
    gmat[i] = (1. + gamma) / 2
end


@generic_kernel function update_bizarrium_EOS!_kernel(
        rho::V, umat::V, vmat::V, Emat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, dynamic_label, debug)

    i = @index_2D_lin()

    # O. Heuzé, S. Jaouen, H. Jourdren, 
    # "Dissipative issue of high-order shock capturing schemes wtih non-convex equations of state"
    # JCP 2009

    @kernel_init begin
        rho0::T = 10000.
        K0::T   = 1e+11
        Cv0::T  = 1000.
        T0::T   = 300.
        eps0::T = 0.
        G0::T   = 1.5
        s::T    = 1.5
        q::T    = -42080895/14941154
        r::T    = 727668333/149411540
    end

    x = rho[i] / rho0 - 1
    g = G0 * (1-rho0 / rho[i])

    f0 = (1+(s/3-2)*x+q*x^2+r*x^3)/(1-s*x)
    f1 = (s/3-2+2*q*x+3*r*x^2+s*f0)/(1-s*x)
    f2 = (2*q+6*r*x+2*s*f1)/(1-s*x)
    f3 = (6*r+3*s*f2)/(1-s*x)

    epsk0     = eps0 - Cv0*T0*(1+g) + 0.5*(K0/rho0)*x^2*f0
    pk0       = -Cv0*T0*G0*rho0 + 0.5*K0*x*(1+x)^2*(2*f0+x*f1)
    pk0prime  = -0.5*K0*(1+x)^3*rho0 * (2*(1+3x)*f0 + 2*x*(2+3x)*f1 + x^2*(1+x)*f2)
    pk0second = 0.5*K0*(1+x)^4*rho0^2 * (12*(1+2x)*f0 + 6*(1+6x+6*x^2)*f1 + 
                                                    6*x*(1+x)*(1+2x)*f2 + x^2*(1+x)^2*f3)

    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = pk0 + G0 * rho0 * (e - epsk0)
    cmat[i] = sqrt(G0 * rho0 * (pmat[i] - pk0) - pk0prime) / rho[i]
    gmat[i] = 0.5 / (rho[i]^3 * cmat[i]^2) * (pk0second + (G0 * rho0)^2 * (pmat[i] - pk0))
end


@generic_kernel function cell_update!_kernel(s::Int, dx::T, dt::T, 
        ustar::V, pstar::V, rho::V, u::V, Emat::V, domain_mask::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=cellUpdate!)

    i = @index_1D_lin()
    mask = domain_mask[i]
    dm = rho[i] * dx
    rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]) * mask)
    u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) * mask
    Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) * mask
end


@generic_kernel function cell_update_lagrange!_kernel(ifin_::Int, s::Int, dt::T, 
        x_::V, ustar::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=cell_update!)

    i = @index_1D_lin()

    x_[i] += dt * ustar[i]

    if i == ifin_
        x_[i+s] += dt * ustar[i+s]
    end
end


@generic_kernel function euler_projection!_kernel(s::Int, dx::T, dt::T,
        ustar::V, rho::V, umat::V, vmat::V, Emat::V,
        advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=euler_remap)

    i = @index_2D_lin()

    dX = dx + dt * (ustar[i+s] - ustar[i])

    tmp_ρ  = (dX * rho[i]           - (advection_ρ[i+s]  - advection_ρ[i] )) / dx
    tmp_uρ = (dX * rho[i] * umat[i] - (advection_uρ[i+s] - advection_uρ[i])) / dx
    tmp_vρ = (dX * rho[i] * vmat[i] - (advection_vρ[i+s] - advection_vρ[i])) / dx
    tmp_Eρ = (dX * rho[i] * Emat[i] - (advection_Eρ[i+s] - advection_Eρ[i])) / dx

    rho[i]  = tmp_ρ
    umat[i] = tmp_uρ / tmp_ρ
    vmat[i] = tmp_vρ / tmp_ρ
    Emat[i] = tmp_Eρ / tmp_ρ
end


@generic_kernel function first_order_euler_remap!_kernel(s::Int, dt::T,
        ustar::V, rho::V, umat::V, vmat::V, Emat::V,
        advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=euler_remap_1st)

    i = @index_2D_lin()

    is = i
    disp = dt * ustar[i]
    if disp > 0
        i = i - s
    end

    advection_ρ[is]  = disp * (rho[i]          )
    advection_uρ[is] = disp * (rho[i] * umat[i])
    advection_vρ[is] = disp * (rho[i] * vmat[i])
    advection_Eρ[is] = disp * (rho[i] * Emat[i])
end


@generic_kernel function second_order_euler_remap!_kernel(s::Int, dx::T, dt::T,
        ustar::V, rho::V, umat::V, vmat::V, Emat::V,
        advection_ρ::V, advection_uρ::V, advection_vρ::V, advection_Eρ::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=euler_remap_2nd)

    i = @index_2D_lin()

    is = i
    disp = dt * ustar[i]
    if disp > 0
        Δxₑ = -(dx - dt * ustar[i-s])
        i = i - s
    else
        Δxₑ = dx + dt * ustar[i+s]
    end

    Δxₗ₋  = dx + dt * (ustar[i]    - ustar[i-s])
    Δxₗ   = dx + dt * (ustar[i+s]  - ustar[i]  )
    Δxₗ₊  = dx + dt * (ustar[i+2s] - ustar[i+s])

    r₋  = (2 * Δxₗ) / (Δxₗ + Δxₗ₋)
    r₊  = (2 * Δxₗ) / (Δxₗ + Δxₗ₊)

    slopes_ρ  = slope_minmod(rho[i-s]            , rho[i]          , rho[i+s]            , r₋, r₊)
    slopes_uρ = slope_minmod(rho[i-s] * umat[i-s], rho[i] * umat[i], rho[i+s] * umat[i+s], r₋, r₊)
    slopes_vρ = slope_minmod(rho[i-s] * vmat[i-s], rho[i] * vmat[i], rho[i+s] * vmat[i+s], r₋, r₊)
    slopes_Eρ = slope_minmod(rho[i-s] * Emat[i-s], rho[i] * Emat[i], rho[i+s] * Emat[i+s], r₋, r₊)

    length_factor = Δxₑ / (2 * Δxₗ)
    advection_ρ[is]  = disp * (rho[i]           - slopes_ρ  * length_factor)
    advection_uρ[is] = disp * (rho[i] * umat[i] - slopes_uρ * length_factor)
    advection_vρ[is] = disp * (rho[i] * vmat[i] - slopes_vρ * length_factor)
    advection_Eρ[is] = disp * (rho[i] * Emat[i] - slopes_Eρ * length_factor)
end


@generic_kernel function boundaryConditions!_kernel(stride::Int, i_start::Int, d::Int,
        u_factor::T, v_factor::T, rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, label=boundaryConditions!, no_threading)

    idx = @index_1D_lin()
    i = idx * stride + i_start
    i₊ = i + d

    rho[i]  = rho[i₊]
    umat[i] = umat[i₊] * u_factor
    vmat[i] = vmat[i₊] * v_factor
    pmat[i] = pmat[i₊]
    cmat[i] = cmat[i₊]
    gmat[i] = gmat[i₊]
end


@generic_kernel function read_border_array!_kernel(side_length::Int, nghost::Int,
        rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V) where V
    @kernel_options(add_time, async, label=border_array, no_threading)

    idx = @index_2D_lin()
    itr = @iter_idx()

    (i, i_g) = divrem(itr - 1, nghost)
    i_arr = (i_g * side_length + i) * 7

    value_array[i_arr+1] =  rho[idx]
    value_array[i_arr+2] = umat[idx]
    value_array[i_arr+3] = vmat[idx]
    value_array[i_arr+4] = pmat[idx]
    value_array[i_arr+5] = cmat[idx]
    value_array[i_arr+6] = gmat[idx]
    value_array[i_arr+7] = Emat[idx]
end


@generic_kernel function write_border_array!_kernel(side_length::Int, nghost::Int,
        rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V) where V
    @kernel_options(add_time, async, label=border_array, no_threading)

    idx = @index_2D_lin()
    itr = @iter_idx()

    (i, i_g) = divrem(itr - 1, nghost)
    i_arr = (i_g * side_length + i) * 7

     rho[idx] = value_array[i_arr+1]
    umat[idx] = value_array[i_arr+2]
    vmat[idx] = value_array[i_arr+3]
    pmat[idx] = value_array[i_arr+4]
    cmat[idx] = value_array[i_arr+5]
    gmat[idx] = value_array[i_arr+6]
    Emat[idx] = value_array[i_arr+7]
end

#
# GPU-only Kernels
#

@kernel function gpu_dtCFL_reduction_euler_kernel!(dx, dy, out, umat, vmat, cmat, domain_mask)
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


@kernel function gpu_dtCFL_reduction_lagrange_kernel!(out, cmat, domain_mask)
    i = @index(Global)
    out[i] = 1. / (cmat[i] * domain_mask[i])
end

#
# Acoustic Riemann problem solvers
# 

function numericalFluxes!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, u::V, range::DomainRange, label::Symbol;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            step_label = "acoustic_$(label)!"
            return acoustic!(params, data, step_label, range, data.ustar, data.pstar, u; dependencies)
        elseif params.scheme == :GAD_minmod_single
            step_label = "acoustic_GAD_$(label)!"
            return acoustic_GAD_minmod_single!(params, data, step_label, range, dt, u; dependencies)
        else
            # 1st order
            # Add one column/row on both sides since the second order solver relies on the first order fluxes
            # of the neighbouring cells.
            range_1st_order = inflate_dir(range, params.current_axis)

            step_label = "acoustic_$(label)!"
            dependencies = acoustic!(params, data, step_label, range_1st_order, data.ustar_1, data.pstar_1, u; dependencies)

            params.scheme != :GAD_minmod && error("Only the GAD scheme with minmod limiter is supported right now") # TODO : fix this

            # 2nd order
            step_label = "acoustic_GAD_$(label)!"
            return acoustic_GAD_minmod!(params, data, step_label, range, dt, u; dependencies)
        end
    else
        error("The choice of Riemann solver is not recognized: ", params.riemann)
    end
end

#
# Equations of State
#

function update_EOS!(params::ArmonParameters{T}, data::ArmonData{V}, range::DomainRange, 
        label::Symbol; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    step_label = "update_EOS_$(label)!"
    if params.test in (:Sod, :Sod_y, :Sod_circ)
        gamma::T = 7/5
        return update_perfect_gas_EOS!(params, data, step_label, range, gamma; dependencies)
    elseif params.test == :Bizarrium
        return update_bizarrium_EOS!(params, data, step_label, range; dependencies)
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
    r = params.extra_ring_width

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

        # Set the domain mask to 1 if the cell should be computed or 0 otherwise
        if one_more_ring
            domain_mask[i] = (
                (-r ≤   ix < nx+r && -r ≤   iy < ny+r)  # Include as well a ring of ghost cells...
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

    return
end

#
# Boundary conditions
#

function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}, side::Symbol;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; test, row_length, nx, ny) = params
    @indexing_vars(params)

    u_factor::T = 1.
    v_factor::T = 1.
    stride::Int = 1
    d::Int = 1

    if side == :left
        if test == :Sod || test == :Sod_circ
            u_factor = -1.
        end

        stride = row_length
        i_start = @i(0,1)
        loop_range = 1:ny
        d = 1
    
    elseif side == :right
        if test == :Sod || test == :Sod_circ
            u_factor = -1.
        end

        stride = row_length
        i_start = @i(nx+1,1)
        loop_range = 1:ny
        d = -1

    elseif side == :top
        if test == :Sod_y || test == :Sod_circ
            v_factor = -1.
        end

        stride = 1
        i_start = @i(1,ny+1)
        loop_range = 1:nx
        d = -row_length

    elseif side == :bottom
        if test == :Sod_y || test == :Sod_circ
            v_factor = -1.
        end
        
        stride = 1
        i_start = @i(1,0)
        loop_range = 1:nx
        d = row_length
    else
        error("Unknown side: $side")
    end

    i_start -= stride  # Adjust for the fact that `@index_1D_lin()` is 1-indexed

    return boundaryConditions!(params, data, loop_range, stride, i_start, d, u_factor, v_factor; dependencies)
end

#
# Halo exchange
#

function read_border_array!(params::ArmonParameters{T}, data::ArmonData{V}, value_array::W,
        side::Symbol; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, nx, ny, row_length) = params
    (; tmp_comm_array) = data
    @indexing_vars(params)

    if side == :left
        main_range = @i(1, 1):row_length:@i(1, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :right
        main_range = @i(nx-nghost+1, 1):row_length:@i(nx-nghost+1, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :top
        main_range = @i(1, ny-nghost+1):row_length:@i(1, ny)
        inner_range = 1:nx
        side_length = nx
    elseif side == :bottom
        main_range = @i(1, 1):row_length:@i(1, nghost)
        inner_range = 1:nx
        side_length = nx
    else
        error("Unknown side: $side")
    end

    range = DomainRange((main_range, inner_range))
    event = read_border_array!(params, data, range, side_length, tmp_comm_array; dependencies)

    if params.use_gpu
        # Copy `tmp_comm_array` from the GPU to the CPU in `value_array`
        event = async_copy!(params.device, value_array, tmp_comm_array; dependencies=event)
        event = @time_event_a "border_array" event
    end

    return event
end


function write_border_array!(params::ArmonParameters{T}, data::ArmonData{V}, value_array::W,
        side::Symbol; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; nghost, nx, ny, row_length) = params
    (; tmp_comm_array) = data
    @indexing_vars(params)

    # Write the border array to the ghost cells of the data arrays
    if side == :left
        main_range = @i(1-nghost, 1):row_length:@i(1-nghost, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :right
        main_range = @i(nx+1, 1):row_length:@i(nx+1, ny)
        inner_range = 1:nghost
        side_length = ny
    elseif side == :top
        main_range = @i(1, ny+1):row_length:@i(1, ny+nghost)
        inner_range = 1:nx
        side_length = nx
    elseif side == :bottom
        main_range = @i(1, 1-nghost):row_length:@i(1, 0)
        inner_range = 1:nx
        side_length = nx
    else
        error("Unknown side: $side")
    end

    if params.use_gpu
        # Copy `value_array` from the CPU to the GPU in `tmp_comm_array`
        event = async_copy!(params.device, tmp_comm_array, value_array; dependencies)
        event = @time_event_a "border_array" event
    else
        event = dependencies
    end

    range = DomainRange((main_range, inner_range))
    event = write_border_array!(params, data, range, side_length, tmp_comm_array; dependencies=event)

    return event
end


function exchange_with_neighbour(params::ArmonParameters{T}, array::V, neighbour_rank::Int,
        cart_comm::MPI.Comm) where {T, V <: AbstractArray{T}}
    @perf_task "comms" "MPI_sendrecv" @time_expr_a "boundaryConditions!_MPI" MPI.Sendrecv!(array, neighbour_rank, 0, array, neighbour_rank, 0, cart_comm)
end


function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}, host_array::W, axis::Axis; 
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; neighbours, cart_comm, cart_coords) = params
    # TODO : use active RMA instead? => maybe but it will (maybe) not work with GPUs: https://www.open-mpi.org/faq/?category=runcuda
    # TODO : use CUDA/ROCM-aware MPI
    # TODO : use 4 views for each side for each variable ? (2 will be contigous, 2 won't) <- pre-calculate them!
    # TODO : try to mix the comms: send to left and receive from right, then vice-versa. Maybe it can speed things up?    

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

    comm_array = params.use_gpu ? host_array : data.tmp_comm_array

    prev_event = dependencies

    for side in order
        neighbour = neighbours[side]
        if neighbour == MPI.PROC_NULL
            prev_event = boundaryConditions!(params, data, side; dependencies=prev_event)
        else
            read_event = read_border_array!(params, data, comm_array, side; dependencies=prev_event)
            Event(exchange_with_neighbour, params, comm_array, neighbour, cart_comm;
                dependencies=read_event) |> wait
            prev_event = write_border_array!(params, data, comm_array, side)
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
    elseif params.use_gpu && params.device == ROCDevice()
        # AMDGPU doesn't support ArrayProgramming, however its implementation of `reduce` is quite
        # fast. Therefore first we compute dt for all cells and store the result in a temporary
        # array, then we reduce this array.
        # TODO : fix this
        if params.projection != :none
            gpu_dtCFL_reduction_euler! = gpu_dtCFL_reduction_euler_kernel!(params.device, params.block_size)
            gpu_dtCFL_reduction_euler!(dx, dy, tmp_rho, umat, vmat, cmat, domain_mask;
                ndrange=length(cmat), dependencies) |> wait
            dt = reduce(min, tmp_rho)
        else
            gpu_dtCFL_reduction_lagrange! = gpu_dtCFL_reduction_lagrange_kernel!(params.device, params.block_size)
            gpu_dtCFL_reduction_lagrange!(tmp_rho, cmat, domain_mask;
                ndrange=length(cmat), dependencies) |> wait
            dt = reduce(min, tmp_rho) * min(dx, dy)
        end
    elseif params.projection != :none
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
            wait(dependencies)
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
# Lagrangian cell update
# 

function cellUpdate!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T,
        u::V, x::V; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; ideb, ifin) = params

    if params.single_comm_per_axis_pass
        (; nx, ny) = params
        @indexing_vars(params)
        r = params.extra_ring_width
        first_i = @i(1-r, 1-r)
        last_i = @i(nx+r, ny+r)
    else
        first_i = ideb
        last_i = ifin
    end

    cell_update!(params, data, first_i:last_i, dt, u; dependencies)
    if params.projection == :none
        cell_update_lagrange!(params, data, first_i:last_i, last_i, dt, x; dependencies)
    end

    return NoneEvent()
end

#
# Euler projection
#

function slope_minmod(uᵢ₋::T, uᵢ::T, uᵢ₊::T, r₋::T, r₊::T) where T
    Δu₊ = r₊ * (uᵢ₊ - uᵢ )
    Δu₋ = r₋ * (uᵢ  - uᵢ₋)
    s = sign(Δu₊)
    return s * max(0, min(s * Δu₊, s * Δu₋))
end


function projection_remap!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    params.projection == :none && return
    
    if !params.single_comm_per_axis_pass
        # Additionnal communications phase needed to get the new values of the lagrangian cells
        dependencies = boundaryConditions!(params, data, host_array, axis; dependencies)
    end

    (; tmp_rho, tmp_urho, tmp_vrho, tmp_Erho) = data
    domain_ranges = compute_domain_ranges(params)
    advection_range = full_domain_projection_advection(domain_ranges)

    advection_ρ  = tmp_rho
    advection_uρ = tmp_urho
    advection_vρ = tmp_vrho
    advection_Eρ = tmp_Erho

    if params.projection == :euler
        event = first_order_euler_remap!(params, data, advection_range, dt,
            advection_ρ, advection_uρ, advection_vρ, advection_Eρ; dependencies)
    elseif params.projection == :euler_2nd
        event = second_order_euler_remap!(params, data, advection_range, dt,
            advection_ρ, advection_uρ, advection_vρ, advection_Eρ; dependencies)
    else
        error("Unknown projection scheme: $(params.projection)")
    end

    return euler_projection!(params, data, full_domain(domain_ranges), dt,
        advection_ρ, advection_uρ, advection_vρ, advection_Eρ; dependencies=event)
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
# Inner / Outer domains definition
#

function compute_domain_ranges(params::ArmonParameters)
    (; nx, ny, nghost, row_length, current_axis) = params
    @indexing_vars(params)

    # Full range
    col_range = @i(1,1):row_length:@i(1,ny)
    row_range = 1:nx
    full_range = DomainRange((col_range, row_range))
    
    # Inner range

    if current_axis == X_axis
        # Parse the cells row by row, excluding 'nghost' columns at the left and right
        col_range = @i(1,1):row_length:@i(1,ny)
        row_range = nghost+1:nx-nghost
    else
        # Parse the cells row by row, excluding 'nghost' rows at the top and bottom
        col_range = @i(1,nghost+1):row_length:@i(1,ny-nghost)
        row_range = 1:nx
    end

    inner_range = DomainRange((col_range, row_range))

    # Outer range: left/bottom

    if current_axis == X_axis
        # Parse the cells row by row, for the first 'nghost' columns on the left
        col_range = @i(1,1):row_length:@i(1,ny)
        row_range = 1:nghost
    else
        # Parse the cells row by row, for the first 'nghost' rows at the bottom
        col_range = @i(1,1):row_length:@i(1,nghost)
        row_range = 1:nx
    end
    
    outer_lb_range = DomainRange((col_range, row_range))

    # Outer range: right/top
    if current_axis == X_axis
        # Parse the cells row by row, for the last 'nghost' columns on the right
        col_range = @i(1,1):row_length:@i(1,ny)
        row_range = nx-nghost+1:nx
    else
        # Parse the cells row by row, for the last 'nghost' rows at the top
        col_range = @i(1,ny-nghost+1):row_length:@i(1,ny)
        row_range = 1:nx
    end
    outer_rt_range = DomainRange((col_range, row_range))

    if params.single_comm_per_axis_pass
        r = params.extra_ring_width

        # Add 'r' columns/rows on each of the 4 sides
        full_range  = DomainRange((inflate(full_range.col,  r), inflate(full_range.row,  r)))
        inner_range = DomainRange((inflate(inner_range.col, r), inflate(inner_range.row, r)))
        
        if current_axis == X_axis
            # Shift the outer domain to the left and right by 'r' cells, and add 'r' rows at the top and bottom
            outer_lb_range = DomainRange((inflate(outer_lb_range.col, r), shift(outer_lb_range.row, -r)))
            outer_rt_range = DomainRange((inflate(outer_rt_range.col, r), shift(outer_rt_range.row,  r)))
        else
            # Shift the outer domain to the top and bottom by 'r' cells, and add 'r' columns at the left and right
            outer_lb_range = DomainRange((shift(outer_lb_range.col, -r), inflate(outer_lb_range.row, r)))
            outer_rt_range = DomainRange((shift(outer_rt_range.col,  r), inflate(outer_rt_range.row, r)))
        end
    end

    return DomainRanges(full_range, inner_range, outer_lb_range, outer_rt_range, current_axis)
end

#
# Output 
#

function write_result_single_file(params::ArmonParameters{T}, data::ArmonData{V}, 
        file_name::String) where {T, V <: AbstractArray{T}}
    (; silent, nx, ny, nghost, row_length, col_length, output_dir, merge_files,
       is_root, cart_comm, cart_coords, global_grid) = params

    if is_root && merge_files && params.projection == :none
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
        file_name::String; no_msg=false) where {T, V <: AbstractArray{T}}
    (; silent, output_dir, nx, ny, cart_coords, cart_comm, is_root, nghost) = params
    @indexing_vars(params)

    output_file_path = joinpath(output_dir, file_name)

    if is_root
        # Advanced globing to remove all files sharing the same file name
        remove_all_outputs = "rm -f $(output_file_path)_*([0-9])x*([0-9])"
        run(`bash -O extglob -c $remove_all_outputs`)
    end

    # Wait for the root command to complete
    params.use_MPI && MPI.Barrier(cart_comm)

    (cx, cy) = cart_coords

    f = open("$(output_file_path)_$(cx)x$(cy)", "w")
   
    vars_to_write = [data.x, data.y, data.rho, data.umat, data.vmat, data.pmat]

    if params.write_ghosts
        col_range = 1-nghost:ny+nghost
        row_range = 1-nghost:nx+nghost
    else
        col_range = 1:ny
        row_range = 1:nx
    end

    p = params.output_precision
    first_format = Printf.Format("%$(p+3).$(p)f")
    format = Printf.Format(", %$(p+3).$(p)f")

    for j in col_range
        for i in row_range
            idx = @i(i, j)
            Printf.format(f, first_format, vars_to_write[1][idx])
            for var in vars_to_write[2:end]
                Printf.format(f, format, var[idx])
            end
            print(f, "\n")
        end
    end

    close(f)

    if !no_msg && is_root && silent < 2
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


function read_sub_domain_file!(params::ArmonParameters{T}, data::ArmonData{V}, 
        file_name::String) where {T, V <: AbstractArray{T}}
    (; output_dir, cart_coords, nx, ny, nghost) = params
    @indexing_vars(params)

    comp_file_path = joinpath(output_dir, file_name)

    (cx, cy) = cart_coords

    f = open("$(comp_file_path)_$(cx)x$(cy)", "r")
   
    vars_to_read = [data.x, data.y, data.rho, data.umat, data.vmat, data.pmat]

    if params.write_ghosts
        col_range = 1-nghost:ny+nghost
        row_range = 1-nghost:nx+nghost
    else
        col_range = 1:ny
        row_range = 1:nx
    end

    for j in col_range
        for i in row_range
            idx = @i(i, j)
            for var in vars_to_read[1:end-1]
                var[idx] = parse(T, readuntil(f, ','))
            end
            vars_to_read[end][idx] = parse(T, readuntil(f, '\n'))
        end
    end

    close(f)
end


function compare_data(label::String, params::ArmonParameters{T}, 
        ref_data::ArmonData{V}, our_data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; row_length, nghost, nbcell, comparison_tolerance) = params
    different = false
    fields_to_compare = (:x, :y, :rho, :umat, :vmat, :pmat)
    for name in fields_to_compare
        ref_val = getfield(ref_data, name)
        our_val = getfield(our_data, name)

        diff_mask = .~ isapprox.(ref_val, our_val; atol=comparison_tolerance)
        !params.write_ghosts && diff_mask .*= our_data.domain_mask
        diff_count = sum(diff_mask)

        if diff_count > 0
            !different && println("At $label:")
            different = true
            print("$diff_count differences found in $name")
            if diff_count < 201
                println(" (ref ≢ current)")
                for idx in 1:nbcell
                    !diff_mask[idx] && continue
                    i, j = ((idx-1) % row_length) + 1 - nghost, ((idx-1) ÷ row_length) + 1 - nghost
                    @printf(" - %5d (%3d,%3d): %10.5g ≢ %10.5g (%10.5g)\n", idx, i, j, 
                        ref_val[idx], our_val[idx], ref_val[idx] - our_val[idx])
                end
            else
                println()
            end
        end
    end
    return different
end


function compare_with_file(params::ArmonParameters{T}, 
        data::ArmonData{V}, file_name::String, label::String) where {T, V <: AbstractArray{T}}
    ref_data = ArmonData(T, params.nbcell, params.comm_array_size)
    read_sub_domain_file!(params, ref_data, file_name)
    different = compare_data(label, params, ref_data, data)
    if params.use_MPI
        different = MPI.Allreduce(different, |, params.cart_comm)
    end
    return different
end


function step_checkpoint(params::ArmonParameters{T}, 
        data::ArmonData{V}, cpu_data::Union{Nothing, ArmonData{W}},
        step_label::String, cycle::Int, axis::Union{Axis, Nothing};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    if params.compare
        wait(dependencies)

        if !isnothing(cpu_data) && W != V
            data_from_gpu(cpu_data, data)
        else
            cpu_data = data
        end

        step_file_name = params.output_file * @sprintf("_%03d_%s", cycle, step_label)
        step_file_name *= isnothing(axis) ? "" : "_" * string(axis)[1:1]

        if params.is_ref
            write_sub_domain_file(params, cpu_data, step_file_name; no_msg=true)
        else
            different = compare_with_file(params, cpu_data, step_file_name, step_label)
            if different
                write_sub_domain_file(params, cpu_data, step_file_name * "_diff"; no_msg=true)
                println("Difference file written to $(step_file_name)_diff")
            end
            return different
        end
    end

    return false
end

#
# Conservation test
# 

function conservation_vars(params::ArmonParameters{T}, data::ArmonData{V}) where {T, V <: AbstractArray{T}}
    (; rho, Emat, domain_mask, x, y) = data
    (; ideb, ifin, dx, row_length) = params
    
    if params.use_gpu
        if params.projection == :none
            total_mass = @inbounds reduce(+, @views (
                rho[ideb:ifin] .* domain_mask[ideb:ifin]
                .* (x[ideb+1:ifin+1] .- x[ideb:ifin])
                .* (y[ideb+row_length:ifin+row_length] .- y[ideb:ifin])))
            total_energy = @inbounds reduce(+, @views (
                rho[ideb:ifin] .* Emat[ideb:ifin] .* domain_mask[ideb:ifin]
                .* (x[ideb+1:ifin+1] .- x[ideb:ifin])
                .* (y[ideb+row_length:ifin+row_length] .- y[ideb:ifin])))
        else
            total_mass = @inbounds reduce(+, @views (
                rho[ideb:ifin] .* domain_mask[ideb:ifin] .* (dx * dx)))
            total_energy = @inbounds reduce(+, @views (
                rho[ideb:ifin] .* Emat[ideb:ifin] .* domain_mask[ideb:ifin] .* (dx * dx)))
        end
    else
        if params.projection == :none
            @batch threadlocal=zeros(T, 2) for i in ideb:ifin
                ds = (x[i+1] - x[i]) * (y[i+row_length] - y[i])
                threadlocal[1] += rho[i] * ds           * domain_mask[i]  # mass
                threadlocal[2] += rho[i] * ds * Emat[i] * domain_mask[i]  # energy
            end
        else
            ds = dx * dx
            @batch threadlocal=zeros(T, 2) for i in ideb:ifin
                threadlocal[1] += rho[i] * ds           * domain_mask[i]  # mass
                threadlocal[2] += rho[i] * ds * Emat[i] * domain_mask[i]  # energy
            end
        end

        threadlocal  = sum(threadlocal)  # Reduce the result of each thread
        total_mass   = threadlocal[1]
        total_energy = threadlocal[2]
    end

    if params.use_MPI
        total_mass   = MPI.Reduce(total_mass,   MPI.Op(+, T), params.cart_comm)
        total_energy = MPI.Reduce(total_energy, MPI.Op(+, T), params.cart_comm)
    end

    return total_mass, total_energy
end

# 
# Main time loop
# 

function time_loop(params::ArmonParameters{T}, data::ArmonData{V},
        cpu_data::Union{ArmonData{W}, Nothing}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root, dt_on_even_cycles) = params
    
    cycle  = 0
    t::T   = 0.
    dta::T = 0.
    dt::T  = 0.
    total_cycles_time::T = 0.

    t1 = time_ns()
    t_warmup = t1

    if params.use_MPI && params.use_gpu
        # Host version of temporary array used for MPI communications
        host_array = Vector{T}(undef, params.comm_array_size)
    else
        host_array = Vector{T}()
    end

    if silent <= 1
        initial_mass, initial_energy = conservation_vars(params, data)
    end

    last_i::Int, x_::V, u::V = update_axis_parameters(params, data, params.current_axis)
    domain_ranges = compute_domain_ranges(params)

    EOS_up_to_date = false  # p, c and g are not initialized by `init_test`
    prev_event = NoneEvent()

    while t < maxtime && cycle < maxcycle
        cycle_start = time_ns()
        
        if !dt_on_even_cycles || iseven(cycle)
            if !EOS_up_to_date
                update_EOS!(params, data, full_domain(domain_ranges), :full; dependencies=prev_event) |> wait
                step_checkpoint(params, data, cpu_data, "update_EOS!", cycle, params.current_axis) && @goto stop
                prev_event = NoneEvent()
                EOS_up_to_date = true
            end

            dt = dtCFL_MPI(params, data, dta; dependencies=prev_event)
            prev_event = NoneEvent()

            if is_root && (!isfinite(dt) || dt <= 0.)
                error("Invalid dt at cycle $cycle: $dt")
            end
        end

        for (axis, dt_factor) in split_axes(params, cycle)
            last_i, x_, u = update_axis_parameters(params, data, axis)
            domain_ranges = compute_domain_ranges(params)
            
            @perf_task "loop" "comms+fluxes" @time_expr_c "comms+fluxes" if params.async_comms
                error("oops")
                @sync begin
                    @async begin
                        if !EOS_up_to_date 
                            event_2 = update_EOS!(params, data, inner_domain(domain_ranges), :inner; dependencies=prev_event)
                        else
                            event_2 = prev_event
                        end

                        event_2 = numericalFluxes!(params, data, dt * dt_factor, u, inner_fluxes_domain(domain_ranges), :inner; dependencies=event_2)
                        wait(event_2)
                    end

                    bc_params = copy(params)
                    bc_params.use_threading = false  # TODO : check if this is still necessary
                    @async begin
                        if !EOS_up_to_date
                            event_1 = update_EOS!(params, data, outer_lb_domain(domain_ranges), :outer; dependencies=prev_event)
                            event_1 = update_EOS!(params, data, outer_rt_domain(domain_ranges), :outer; dependencies=event_1)
                        else
                            event_1 = prev_event
                        end

                        event_1 = boundaryConditions!(bc_params, data, host_array, axis; dependencies=event_1)

                        event_1 = numericalFluxes!(bc_params, data, dt * dt_factor, u, outer_fluxes_lb_domain(domain_ranges), :outer; dependencies=event_1)
                        event_1 = numericalFluxes!(bc_params, data, dt * dt_factor, u, outer_fluxes_rt_domain(domain_ranges), :outer; dependencies=event_1)
                        wait(event_1)
                    end
                end

                event = NoneEvent()
            else
                if !EOS_up_to_date
                    event = update_EOS!(params, data, full_domain(domain_ranges), :full; dependencies=prev_event)
                    step_checkpoint(params, data, cpu_data, "update_EOS!", cycle, axis; dependencies=event) && @goto stop
                else
                    event = prev_event
                end
                
                event = boundaryConditions!(params, data, host_array, axis; dependencies=event)
                step_checkpoint(params, data, cpu_data, "boundaryConditions!", cycle, axis; dependencies=event) && @goto stop
                
                event = numericalFluxes!(params, data, dt * dt_factor, u, full_fluxes_domain(domain_ranges), :full; dependencies=event)
                step_checkpoint(params, data, cpu_data, "numericalFluxes!", cycle, axis; dependencies=event) && @goto stop
                
                params.measure_time && wait(event)
            end

            @perf_task "loop" "cellUpdate" event = cellUpdate!(params, data, dt * dt_factor, u, x_; dependencies=event)
            step_checkpoint(params, data, cpu_data, "cellUpdate!", cycle, axis; dependencies=event) && @goto stop

            @perf_task "loop" "euler_proj" event = projection_remap!(params, data, dt * dt_factor; dependencies=event)
            step_checkpoint(params, data, cpu_data, "projection_remap!", cycle, axis; dependencies=event) && @goto stop
            
            prev_event = event
            EOS_up_to_date = false
        end

        if !is_warming_up()
            total_cycles_time += time_ns() - cycle_start
        end

        dta = dt
        cycle += 1
        t += dt

        if is_root
            if silent <= 1
                current_mass, current_energy = conservation_vars(params, data)
                ΔM = abs(initial_mass - current_mass)
                ΔE = abs(initial_energy - current_energy)
                @printf("Cycle %4d: dt = %.18f, t = %.18f, |ΔM| = %.6f, |ΔE| = %.6f\n", cycle, dt, t, ΔM, ΔE)
            end
        elseif silent <= 1
            conservation_vars(params, data)
        end

        if cycle == 5
            t_warmup = time_ns()
            set_warmup(false)
        end
        
        if animation_step != 0 && (cycle - 1) % animation_step == 0
            write_result(params, data, joinpath("anim", params.output_file) * "_" *
                @sprintf("%03d", (cycle - 1) ÷ animation_step))
        end
    end

    @label stop

    t2 = time_ns()

    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((cycle - 5) * nb_cells)

    if is_root
        if params.compare
            # ignore timing errors
        elseif cycle <= 5 && maxcycle > 5
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
            println("Last Δt:     ", @sprintf("%.18f", dt),                    " sec")
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
        set_warmup(true)
    end

    if is_root && silent < 3
        print_parameters(params)
    end

    if params.use_MPI && silent < 3
        (; rank, proc_size, cart_coords) = params
    
        # Local info
        node_local_comm = MPI.Comm_split_type(COMM, MPI.COMM_TYPE_SHARED, rank)
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
        device_array = params.device == CUDADevice() ? CuArray : params.device == ROCDevice() ? ROCArray : Array
        copy_time = @elapsed d_data = data_to_gpu(data, device_array)
        (is_root && silent <= 2) && @printf("Time for copy to device: %.3g sec\n", copy_time)

        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, d_data, data)

        data_from_gpu(data, d_data)
    else
        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, data, nothing)
    end

    if params.write_output
        write_result(params, data, params.output_file)
    end

    sorted_time_contrib = sort(collect(total_time_contrib))

    if params.measure_time && length(sorted_time_contrib) > 0
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

            println("\nTime for each step of the $axis:          ( axis%) (total%)")
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
