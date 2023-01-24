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
# center the positions of the cells in the output file
# Remove all generics : 'where {T, V <: AbstractVector{T}}' etc... when T and V are not used in the method. Omitting the 'where' will not change anything.
# Bug: `conservation_vars` doesn't give correct values with MPI, even though the solution is correct
# Bug: fix dtCFL on AMDGPU
# Bug: steps are not properly categorized and filtered at the output, giving wrong asynchronicity efficiency
# Bug: some time measurements are incorrect on GPU

# MPI Init

COMM = MPI.COMM_WORLD

function set_world_comm(comm::MPI.Comm)
    # Allows to customize which processes will be part of the grid
    global COMM = comm
end

# VTune performance analysis

include("vtune_lib.jl")
using .VTune

# Hardware counters measurements

include("perf_utils.jl")

#
# Axis enum
#

@enum Axis X_axis Y_axis

#
# GPU device
#

GPUDevice = Union{Device, ROCDevice}  # ROCKernels uses AMDGPU's ROCDevice, unlike CUDAKernels and KernelsAbstractions...

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

limiter_from_name(::Val{:no_limiter}) = NoLimiter()
limiter_from_name(::Val{:minmod})     = MinmodLimiter()
limiter_from_name(::Val{:superbee})   = SuperbeeLimiter()

limiter_from_name(::Val{s}) where s = error("Unknown limiter name: '$s'")
limiter_from_name(s::Symbol) = limiter_from_name(Val(s))

Base.show(io::IO, ::NoLimiter)       = print(io, "No limiter")
Base.show(io::IO, ::MinmodLimiter)   = print(io, "Minmod limiter")
Base.show(io::IO, ::SuperbeeLimiter) = print(io, "Superbee limiter")

#
# Test cases
#

abstract type TestCase end
abstract type TwoStateTestCase <: TestCase end

struct Sod       <: TwoStateTestCase end
struct Sod_y     <: TwoStateTestCase end
struct Sod_circ  <: TwoStateTestCase end
struct Bizarrium <: TwoStateTestCase end
struct Sedov{T}  <: TwoStateTestCase
    r::T
end

create_test(::T, ::T, ::Type{Test}) where {T, Test <: TestCase} = Test()

function create_test(Δx::T, Δy::T, ::Type{Sedov}) where T
    r_Sedov::T = sqrt(Δx^2 + Δy^2) / sqrt(2)
    return Sedov{T}(r_Sedov)
end

test_from_name(::Val{:Sod})       = Sod
test_from_name(::Val{:Sod_y})     = Sod_y
test_from_name(::Val{:Sod_circ})  = Sod_circ
test_from_name(::Val{:Bizarrium}) = Bizarrium
test_from_name(::Val{:Sedov})     = Sedov

test_from_name(::Val{s}) where s = error("Unknown test case: '$s'")
test_from_name(s::Symbol) = test_from_name(Val(s))

test_name(::Test) where {Test <: TestCase} = Test.name.name

default_domain_size(::Type{<:TestCase}) = (1, 1)
default_domain_size(::Type{Sedov}) = (2, 2)

default_domain_origin(::Type{<:TestCase}) = (0, 0)
default_domain_origin(::Type{Sedov}) = (-1, -1)

default_CFL(::Union{Sod, Sod_y, Sod_circ}) = 0.95
default_CFL(::Bizarrium) = 0.6
default_CFL(::Sedov) = 0.7

default_max_time(::Union{Sod, Sod_y, Sod_circ}) = 0.20
default_max_time(::Bizarrium) = 80e-6
default_max_time(::Sedov) = 1.0

Base.show(io::IO, ::Sod)       = print(io, "Sod shock tube")
Base.show(io::IO, ::Sod_y)     = print(io, "Sod shock tube (along the Y axis)")
Base.show(io::IO, ::Sod_circ)  = print(io, "Sod shock tube (cylindrical symmetry around the Z axis)")
Base.show(io::IO, ::Bizarrium) = print(io, "Bizarrium")
Base.show(io::IO, ::Sedov)     = print(io, "Sedov")

# TODO : use 0.0625 for Sod_circ since 1/8 makes no sense and is quite arbitrary
test_region_high(x::T, _::T, ::Sod)       where T = x ≤ 0.5
test_region_high(_::T, y::T, ::Sod_y)     where T = y ≤ 0.5
test_region_high(x::T, y::T, ::Sod_circ)  where T = (x - T(0.5))^2 + (y - T(0.5))^2 ≤ T(0.125)
test_region_high(x::T, _::T, ::Bizarrium) where T = x ≤ 0.5
test_region_high(x::T, y::T, s::Sedov{T}) where T = x^2 + y^2 ≤ s.r^2

function init_test_params(::Union{Sod, Sod_y, Sod_circ})
    return (
        7/5,   # gamma
        1.,    # high_ρ
        0.125, # low_ρ
        1.0,   # high_p
        0.1,   # low_p
        0.,    # high_u
        0.,    # low_u
        0.,    # high_v
        0.,    # low_v
    )
end

function init_test_params(::Bizarrium)
    return (
        2,                 # gamma
        1.42857142857e+4,  # high_ρ
        10000.,            # low_ρ
        6.40939744478e+10, # high_p
        312.5e6,           # low_p
        0.,                # high_u
        250.,              # low_u
        0.,                # high_v
        0.,                # low_v
    )
end

function init_test_params(p::Sedov{T}) where T
    return (
        7/5,   # gamma
        1.,    # high_ρ
        1.,    # low_ρ
        (1.4 - 1) * 0.851072 / (π * p.r^2), # high_p
        1e-14, # low_p
        0.,    # high_u
        0.,    # low_u
        0.,    # high_v
        0.,    # low_v
    )
end

function boundaryCondition(side::Symbol, ::Union{Sod, Bizarrium})::NTuple{2, Int}
    return (side == :left || side == :right) ? (-1, 1) : (1, 1)
end

function boundaryCondition(side::Symbol, ::Sod_y)::NTuple{2, Int}
    return (side == :left || side == :right) ? (1, 1) : (1, -1)
end

function boundaryCondition(side::Symbol, ::Sod_circ)::NTuple{2, Int}
    return (side == :left || side == :right) ? (-1, 1) : (1, -1)
end

function boundaryCondition(::Symbol, ::Sedov)::NTuple{2, Int}
    return (1, 1)
end

#
# Parameters
#

mutable struct ArmonParameters{Flt_T}
    # Test problem type, riemann solver and solver scheme
    test::TestCase
    riemann::Symbol
    scheme::Symbol
    riemann_limiter::Limiter
    
    # Domain parameters
    nghost::Int
    nx::Int
    ny::Int
    dx::Flt_T
    domain_size::NTuple{2, Flt_T}
    origin::NTuple{2, Flt_T}
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
    stencil_width::Int

    # Bounds
    maxtime::Flt_T
    maxcycle::Int
    
    # Output
    silent::Int
    output_dir::String
    output_file::String
    write_output::Bool
    write_ghosts::Bool
    write_slices::Bool
    output_precision::Int
    animation_step::Int
    measure_time::Bool
    measure_hw_counters::Bool
    hw_counters_options::String
    hw_counters_output::String
    return_data::Bool

    # Performance
    use_threading::Bool
    use_simd::Bool
    use_gpu::Bool
    device::GPUDevice
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

    # Tests & Comparison
    compare::Bool
    is_ref::Bool
    comparison_tolerance::Float64
end


# Constructor for ArmonParameters
function ArmonParameters(;
        ieee_bits = 64,
        test = :Sod, riemann = :acoustic, scheme = :GAD_minmod, projection = :euler,
        riemann_limiter = :minmod,
        nghost = 2, nx = 10, ny = 10, stencil_width = nothing,
        domain_size = nothing, origin = nothing,
        cfl = 0., Dt = 0., cst_dt = false, dt_on_even_cycles = false,
        axis_splitting = :Sequential,
        maxtime = 0, maxcycle = 500_000,
        silent = 0, output_dir = ".", output_file = "output",
        write_output = false, write_ghosts = false, write_slices = false, output_precision = nothing,
        animation_step = 0, 
        measure_time = true, measure_hw_counters = false,
        hw_counters_options = nothing, hw_counters_output = nothing,
        use_threading = true, use_simd = true,
        use_gpu = false, device = :CUDA, block_size = 1024,
        use_MPI = true, px = 1, py = 1,
        single_comm_per_axis_pass = false, reorder_grid = true, 
        async_comms = false,
        compare = false, is_ref = false, comparison_tolerance = 1e-10,
        return_data = false
    )

    flt_type = (ieee_bits == 64) ? Float64 : Float32

    if isnothing(output_precision)
        output_precision = flt_type == Float64 ? 17 : 9  # Exact output by default
    end

    # Make sure that all floating point types are the same
    cfl = flt_type(cfl)
    Dt = flt_type(Dt)
    maxtime = flt_type(maxtime)

    domain_size = isnothing(domain_size) ? nothing : Tuple(flt_type.(domain_size))
    origin = isnothing(origin) ? nothing : Tuple(flt_type.(origin))
    
    if cst_dt && Dt == zero(flt_type)
        error("Dt == 0 with constant step enabled")
    end

    if measure_hw_counters
        use_gpu && error("Hardware counters are not supported on GPU")
        async_comms && error("Hardware counters in an asynchronous context are NYI")
        !measure_time && error("Hardware counters are only done when timings are measured as well")

        hw_counters_options = @something hw_counters_options default_perf_options()
        hw_counters_output = @something hw_counters_output ""
    else
        hw_counters_options = ""
        hw_counters_output = ""
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

    if isnothing(stencil_width)
        stencil_width = min_nghost
    elseif stencil_width < min_nghost
        @warn "The detected minimum stencil width is $min_nghost, but $stencil_width was given. \
               The Boundary conditions might be false." maxlog=1
    elseif stencil_width > nghost
        error("The stencil width given ($stencil_width) cannot be bigger than the number of ghost cells ($nghost)")
    end

    if riemann_limiter isa Symbol
        riemann_limiter = limiter_from_name(riemann_limiter)
    elseif !(riemann_limiter isa Limiter)
        error("Expected a Limiter type or a symbol, got: $riemann_limiter")
    end

    if test isa Symbol
        test_type = test_from_name(test)
        test = nothing
    elseif test isa TestCase
        test_type = typeof(test)
    else
        error("Expected a TestCase type or a symbol, got: $test")
    end

    if single_comm_per_axis_pass
        error("single_comm_per_axis_pass=true is broken")
    end

    # MPI
    if use_MPI
        !MPI.Initialized() && error("'use_MPI=true' but MPI has not yet been initialized")

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

    # GPU
    if use_gpu
        if device == :CUDA
            CUDA.allowscalar(false)
            device = CUDADevice()
        elseif device == :ROCM
            AMDGPU.allowscalar(false)
            device = ROCDevice()
        elseif device == :CPU
            is_root && @warn "`use_gpu=true` but the device is set to the CPU. Therefore no kernel will run on a GPU." maxlog=1
            device = CPU()  # Useful in some cases for debugging
        else
            error("Unknown GPU device: $device")
        end
    else
        device = CPU()
    end

    # Initialize the test
    if isnothing(domain_size)
        domain_size = default_domain_size(test_type)
        domain_size = Tuple(flt_type.(domain_size))
    end

    if isnothing(origin)
        origin = default_domain_origin(test_type)
        origin = Tuple(flt_type.(origin))
    end

    if isnothing(test)
        (sx, sy) = domain_size
        Δx::flt_type = sx / nx
        Δy::flt_type = sy / ny
        test = create_test(Δx, Δy, test_type)
    end

    if cfl == 0
        cfl = default_CFL(test)
    end

    if maxtime == 0
        maxtime = default_max_time(test)
    end

    # Dimensions of the global domain
    g_nx = nx
    g_ny = ny

    dx = flt_type(domain_size[1] / g_nx)

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
        test, riemann, scheme, riemann_limiter,
        nghost, nx, ny, dx, domain_size, origin,
        cfl, Dt, cst_dt, dt_on_even_cycles,
        axis_splitting, projection,
        row_length, col_length, nbcell,
        ideb, ifin, index_start,
        idx_row, idx_col,
        X_axis, 1, stencil_width,
        maxtime, maxcycle,
        silent, output_dir, output_file,
        write_output, write_ghosts, write_slices, output_precision, animation_step,
        measure_time,
        measure_hw_counters, hw_counters_options, hw_counters_output,
        return_data,
        use_threading, use_simd, use_gpu, device, block_size,
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
        if use_std_lib_threads
            println(" (Julia standard threads: ", Threads.nthreads(), ")")
        else
            println(" (Julia threads: ", Threads.nthreads(), ")")
        end
    else
        println()
    end
    println(" - use_simd:   ", p.use_simd)
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
    print(" - riemann:    ", p.riemann)
    if p.scheme != :Godunov
        println(", ", p.riemann_limiter)
    else
        println()
    end
    println(" - scheme:     ", p.scheme)
    println(" - splitting:  ", p.axis_splitting)
    println(" - cfl:        ", p.cfl)
    println(" - Dt:         ", p.Dt, p.dt_on_even_cycles ? ", updated only for even cycles" : "")
    println(" - euler proj: ", p.projection)
    println(" - cst dt:     ", p.cst_dt)
    println(" - stencil width: ", p.stencil_width)
    println(" - maxtime:    ", p.maxtime)
    println(" - maxcycle:   ", p.maxcycle)
    println()
    println(" - domain:     ", p.nx, "×", p.ny, " (", p.nghost, " ghosts)")
    println(" - domain size: ", join(p.domain_size, " × "), ", origin: (", join(p.origin, ", "), ")")
    println(" - nbcell:     ", @sprintf("%g", p.nx * p.ny), " (", p.nbcell, " total)")
    println(" - global:     ", p.global_grid[1], "×", p.global_grid[2])
    println(" - proc grid:  ", p.proc_dims[1], "×", p.proc_dims[2], " ($(p.reorder_grid ? "" : "not ")reordered)")
    println(" - coords:     ", p.cart_coords[1], "×", p.cart_coords[2], " (rank: ", p.rank, "/", p.proc_size-1, ")")
    println(" - comms per axis: ", p.single_comm_per_axis_pass ? 1 : 2)
    println(" - asynchronous communications: ", p.async_comms)
    println(" - measure step times: ", p.measure_time)
    if p.measure_hw_counters
        println(" - hardware counters measured: ", p.hw_counters_options)
    end
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


function get_device_array(params::ArmonParameters)
    if params.device == CUDADevice()
        return CuArray
    elseif params.device == ROCDevice()
        return ROCArray
    else  # params.device == CPU()
        return Array
    end
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
    work_array_1::V
    work_array_2::V
    work_array_3::V
    work_array_4::V
    domain_mask::V
    tmp_comm_array::V
end


function ArmonData(params::ArmonParameters{T}) where T
    return ArmonData(T, params.nbcell, params.comm_array_size)
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
        device_array(data.work_array_1),
        device_array(data.work_array_2),
        device_array(data.work_array_3),
        device_array(data.work_array_4),
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


function memory_required_for(params::ArmonParameters{T}) where T
    return memory_required_for(params.nbcell, params.comm_array_size, T)
end


function memory_required_for(N, communication_array_size, float_type)
    field_count = fieldcount(ArmonData{AbstractArray{float_type}})
    floats = (field_count - 1) * N + communication_array_size
    return floats * sizeof(float_type)
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
# Kernels
#

function acoustic_Godunov(ρᵢ::T, ρᵢ₋₁::T, cᵢ::T, cᵢ₋₁::T, uᵢ::T, uᵢ₋₁::T, pᵢ::T, pᵢ₋₁::T) where T
    rc_l = ρᵢ₋₁ * cᵢ₋₁
    rc_r = ρᵢ   * cᵢ
    ustarᵢ = (rc_l * uᵢ₋₁ + rc_r * uᵢ +               (pᵢ₋₁ - pᵢ)) / (rc_l + rc_r)
    pstarᵢ = (rc_r * pᵢ₋₁ + rc_l * pᵢ + rc_l * rc_r * (uᵢ₋₁ - uᵢ)) / (rc_l + rc_r)
    return ustarᵢ, pstarᵢ
end


@generic_kernel function acoustic!(s::Int, ustar_::V, pstar_::V, 
        rho::V, u::V, pmat::V, cmat::V) where V
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()
    ustar_[i], pstar_[i] = acoustic_Godunov(
        rho[i], rho[i-s], cmat[i], cmat[i-s],
          u[i],   u[i-s], pmat[i], pmat[i-s]
    )
end


@generic_kernel function acoustic_GAD!(s::Int, dt::T, dx::T, 
        ustar::V, pstar::V, rho::V, u::V, pmat::V, cmat::V,
        ::LimiterType) where {T, V <: AbstractArray{T}, LimiterType <: Limiter}
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()

    # First order acoustic solver on the left cell
    ustar_i₋, pstar_i₋ = acoustic_Godunov(
        rho[i-s], rho[i-2s], cmat[i-s], cmat[i-2s],
          u[i-s],   u[i-2s], pmat[i-s], pmat[i-2s]
    )

    # First order acoustic solver on the current cell
    ustar_i, pstar_i = acoustic_Godunov(
        rho[i], rho[i-s], cmat[i], cmat[i-s],
          u[i],   u[i-s], pmat[i], pmat[i-s]
    )

    # First order acoustic solver on the right cell
    ustar_i₊, pstar_i₊ = acoustic_Godunov(
        rho[i+s], rho[i], cmat[i+s], cmat[i],
          u[i+s],   u[i], pmat[i+s], pmat[i]
    )

    # Second order GAD acoustic solver on the current cell

    r_u₋ = (ustar_i₊ -      u[i]) / (ustar_i -    u[i-s] + 1e-6)
    r_p₋ = (pstar_i₊ -   pmat[i]) / (pstar_i - pmat[i-s] + 1e-6)
    r_u₊ = (   u[i-s] - ustar_i₋) / (   u[i] -   ustar_i + 1e-6)
    r_p₊ = (pmat[i-s] - pstar_i₋) / (pmat[i] -   pstar_i + 1e-6)

    r_u₋ = limiter(r_u₋, LimiterType())
    r_p₋ = limiter(r_p₋, LimiterType())
    r_u₊ = limiter(r_u₊, LimiterType())
    r_p₊ = limiter(r_p₊, LimiterType())

    dm_l = rho[i-s] * dx
    dm_r = rho[i]   * dx
    Dm   = (dm_l + dm_r) / 2

    rc_l = rho[i-s] * cmat[i-s]
    rc_r = rho[i]   * cmat[i]
    θ    = 1/2 * (1 - (rc_l + rc_r) / 2 * (dt / Dm))
    
    ustar[i] = ustar_i + θ * (r_u₊ * (   u[i] - ustar_i) - r_u₋ * (ustar_i -    u[i-s]))
    pstar[i] = pstar_i + θ * (r_p₊ * (pmat[i] - pstar_i) - r_p₋ * (pstar_i - pmat[i-s]))
end


@generic_kernel function update_perfect_gas_EOS!(gamma::T, 
        rho::V, Emat::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()
    e = Emat[i] - 0.5 * (umat[i]^2 + vmat[i]^2)
    pmat[i] = (gamma - 1.) * rho[i] * e
    cmat[i] = sqrt(gamma * pmat[i] / rho[i])
    gmat[i] = (1. + gamma) / 2
end


@generic_kernel function update_bizarrium_EOS!(
        rho::V, umat::V, vmat::V, Emat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, dynamic_label)

    i = @index_2D_lin()

    # O. Heuzé, S. Jaouen, H. Jourdren, 
    # "Dissipative issue of high-order shock capturing schemes with non-convex equations of state"
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


@generic_kernel function cell_update!(s::Int, dx::T, dt::T, 
        ustar::V, pstar::V, rho::V, u::V, Emat::V, domain_mask::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=cellUpdate!)

    i = @index_1D_lin()
    mask = domain_mask[i]
    dm = rho[i] * dx
    rho[i]   = dm / (dx + dt * (ustar[i+s] - ustar[i]) * mask)
    u[i]    += dt / dm * (pstar[i]            - pstar[i+s]             ) * mask
    Emat[i] += dt / dm * (pstar[i] * ustar[i] - pstar[i+s] * ustar[i+s]) * mask
end


@generic_kernel function cell_update_lagrange!(ifin_::Int, s::Int, dt::T, 
        x_::V, ustar::V, domain_mask::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, label=cell_update!)

    i = @index_1D_lin()

    x_[i] += dt * ustar[i] * domain_mask[i]

    if i == ifin_
        x_[i+s] += dt * ustar[i+s] * domain_mask[i+s]
    end
end


@generic_kernel function euler_projection!(s::Int, dx::T, dt::T,
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


@generic_kernel function first_order_euler_remap!(s::Int, dt::T,
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


@generic_kernel function second_order_euler_remap!(s::Int, dx::T, dt::T,
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


@generic_kernel function boundaryConditions!(stencil_width::Int, stride::Int, i_start::Int, d::Int,
        u_factor::T, v_factor::T, rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V) where {T, V <: AbstractArray{T}}
    @kernel_options(add_time, async, label=boundaryConditions!)

    idx = @index_1D_lin()
    i  = idx * stride + i_start
    i₊ = i + d

    for _ in 1:stencil_width
        rho[i]  = rho[i₊]
        umat[i] = umat[i₊] * u_factor
        vmat[i] = vmat[i₊] * v_factor
        pmat[i] = pmat[i₊]
        cmat[i] = cmat[i₊]
        gmat[i] = gmat[i₊]

        i  -= d
        i₊ += d
    end
end


@generic_kernel function read_border_array!(side_length::Int, nghost::Int,
        rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V) where V
    @kernel_options(add_time, async, label=border_array)

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


@generic_kernel function write_border_array!(side_length::Int, nghost::Int,
        rho::V, umat::V, vmat::V, pmat::V, cmat::V, gmat::V, Emat::V, value_array::V) where V
    @kernel_options(add_time, async, label=border_array)

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


@generic_kernel function init_test(
        row_length::Int, nghost::Int, nx::Int, ny::Int, 
        domain_size::NTuple{2, T}, origin::NTuple{2, T},
        cart_coords::NTuple{2, Int}, global_grid::NTuple{2, Int},
        single_comm_per_axis_pass::Bool, extra_ring_width::Int,
        x::V, y::V, rho::V, Emat::V, umat::V, vmat::V, 
        domain_mask::V, pmat::V, cmat::V, ustar::V, pstar::V, 
        test_case::Test) where {T, V <: AbstractArray{T}, Test <: TwoStateTestCase}
    @kernel_options(add_time, label=init_test, no_gpu)

    i = @index_1D_lin()

    @kernel_init begin
        (cx, cy) = cart_coords
        (g_nx, g_ny) = global_grid
        (sx, sy) = domain_size
        (ox, oy) = origin

        # Position of the origin of this sub-domain
        pos_x = cx * nx
        pos_y = cy * ny

        r = extra_ring_width

        (gamma::T,
            high_ρ::T, low_ρ::T,
            high_p::T, low_p::T,
            high_u::T, low_u::T, 
            high_v::T, low_v::T) = init_test_params(test_case)
    end
    
    ix = ((i-1) % row_length) - nghost
    iy = ((i-1) ÷ row_length) - nghost

    # Global indexes, used only to know to compute the position of the cell
    g_ix = ix + pos_x
    g_iy = iy + pos_y

    x[i] = g_ix / g_nx * sx + ox
    y[i] = g_iy / g_ny * sy + oy

    x_mid = x[i] + sx / (2*g_nx)
    y_mid = y[i] + sy / (2*g_ny)

    if test_region_high(x_mid, y_mid, test_case)
        rho[i]  = high_ρ
        Emat[i] = high_p / ((gamma - one(T)) * rho[i])
        umat[i] = high_u
        vmat[i] = high_v
    else
        rho[i]  = low_ρ
        Emat[i] = low_p / ((gamma - one(T)) * rho[i])
        umat[i] = low_u
        vmat[i] = low_v
    end

    # Set the domain mask to 1 if the cell should be computed or 0 otherwise
    if single_comm_per_axis_pass
        domain_mask[i] = (
               (-r ≤   ix < nx+r && -r ≤   iy < ny+r)  # Include as well a ring of ghost cells...
            && ( 0 ≤   ix < nx   ||  0 ≤   iy < ny  )  # ...while excluding the corners of the sub-domain...
            && ( 0 ≤ g_ix < g_nx &&  0 ≤ g_iy < g_ny)  # ...and only if it is in the global domain
        ) ? 1 : 0
    else
        domain_mask[i] = (0 ≤ ix < nx && 0 ≤ iy < ny) ? 1 : 0
    end

    # Set to zero to make sure no non-initialized values changes the result
    pmat[i] = 0
    cmat[i] = 1  # Set to 1 as a max speed of 0 will create NaNs
    ustar[i] = 0
    pstar[i] = 0
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
        dt::T, range::DomainRange, label::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}}
    u = params.current_axis == X_axis ? data.umat : data.vmat
    if params.riemann == :acoustic  # 2-state acoustic solver (Godunov)
        if params.scheme == :Godunov
            step_label = "acoustic_$(label)!"
            return acoustic!(params, data, step_label, range, data.ustar, data.pstar, u; 
                dependencies, no_threading)
        elseif params.scheme == :GAD
            step_label = "acoustic_GAD_$(label)!"
            return acoustic_GAD!(params, data, step_label, range, dt, u, params.riemann_limiter; 
                dependencies, no_threading)
        else
            error("Unknown acoustic scheme: ", params.scheme)
        end
    else
        error("Unknown Riemann solver: ", params.riemann)
    end
end


function numericalFluxes!(params::ArmonParameters{T}, data::ArmonData{V}, 
        dt::T, domain_ranges::DomainRanges, label::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}}

    if label == :inner
        range = inner_fluxes_domain(domain_ranges)
    elseif label == :outer_lb
        range = outer_fluxes_lb_domain(domain_ranges)
        label = :outer
        sides = (:left, :bottom)
    elseif label == :outer_rt
        range = outer_fluxes_rt_domain(domain_ranges)
        label = :outer
        sides = (:right, :top)
    elseif label == :full
        range = full_fluxes_domain(domain_ranges)
        sides = (:left, :bottom, :right, :top)
    else
        error("Wrong region label: $label")
    end

    if params.use_MPI && label != :inner
        if params.current_axis == X_axis
            sides = filter(in((:left, :right)), sides)
        else  # Y_axis
            sides = filter(in((:top, :bottom)), sides)
        end

        # Extend the range if there is a neighbour in the current direction
        for side in sides
            params.neighbours[side] == MPI.PROC_NULL && continue
            # TODO: compute the '1' dynamically depending on the stencil of each kernel
            if side in (:left, :bottom)
                range = prepend_dir(range, params.current_axis, 1)
            else
                range = expand_dir(range, params.current_axis, 1)
            end
        end
    end

    return numericalFluxes!(params, data, dt, range, label; dependencies, no_threading)
end

#
# Equations of State
#

function update_EOS!(params::ArmonParameters{T}, data::ArmonData, ::TestCase,
        range::DomainRange, label::Symbol; dependencies, no_threading) where T
    step_label = "update_EOS_$(label)!"
    gamma::T = 7/5
    return update_perfect_gas_EOS!(params, data, step_label, range, gamma; dependencies, no_threading)
end


function update_EOS!(params::ArmonParameters, data::ArmonData, ::Bizarrium,
        range::DomainRange, label::Symbol; dependencies, no_threading)
    step_label = "update_EOS_$(label)!"
    return update_bizarrium_EOS!(params, data, step_label, range; dependencies, no_threading)
end


function update_EOS!(params::ArmonParameters, data::ArmonData,
        range::DomainRange, label::Symbol; dependencies=NoneEvent(), no_threading=false)
    return update_EOS!(params, data, params.test, range, label; dependencies, no_threading)
end

#
# Test initialisation
#

function init_test(params::ArmonParameters, data::ArmonData)
    return init_test(params, data, 1:params.nbcell, params.test)
end

#
# Boundary conditions
#

function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}, side::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}}
    (; row_length, nx, ny) = params
    @indexing_vars(params)

    (u_factor::T, v_factor::T) = boundaryCondition(side, params.test)

    stride::Int = 1
    d::Int = 1

    if side == :left
        stride = row_length
        i_start = @i(0,1)
        loop_range = 1:ny
        d = 1
    elseif side == :right
        stride = row_length
        i_start = @i(nx+1,1)
        loop_range = 1:ny
        d = -1
    elseif side == :top
        stride = 1
        i_start = @i(1,ny+1)
        loop_range = 1:nx
        d = -row_length
    elseif side == :bottom
        stride = 1
        i_start = @i(1,0)
        loop_range = 1:nx
        d = row_length
    else
        error("Unknown side: $side")
    end

    i_start -= stride  # Adjust for the fact that `@index_1D_lin()` is 1-indexed

    return boundaryConditions!(params, data, loop_range, stride, i_start, d, u_factor, v_factor; 
        dependencies, no_threading)
end

#
# Halo exchange
#

function read_border_array!(params::ArmonParameters{T}, data::ArmonData{V}, value_array::W, side::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
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

    range = DomainRange(main_range, inner_range)
    event = read_border_array!(params, data, range, side_length, tmp_comm_array;
        dependencies, no_threading)

    if params.use_gpu
        # Copy `tmp_comm_array` from the GPU to the CPU in `value_array`
        event = async_copy!(params.device, value_array, tmp_comm_array; dependencies=event)
        event = @time_event_a "border_array" event
    end

    return event
end


function write_border_array!(params::ArmonParameters{T}, data::ArmonData{V}, value_array::W, side::Symbol;
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
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

    range = DomainRange(main_range, inner_range)
    event = write_border_array!(params, data, range, side_length, tmp_comm_array; 
        dependencies=event, no_threading)

    return event
end


function exchange_with_neighbour(params::ArmonParameters{T}, array::V, neighbour_rank::Int) where {T, V <: AbstractArray{T}}
    @perf_task "comms" "MPI_sendrecv" @time_expr_a "boundaryConditions!_MPI" MPI.Sendrecv!(array, 
        neighbour_rank, 0, array, neighbour_rank, 0, params.cart_comm)
end


function boundaryConditions!(params::ArmonParameters{T}, data::ArmonData{V}, host_array::W, axis::Axis; 
        dependencies=NoneEvent(), no_threading=false) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; neighbours, cart_coords) = params
    # TODO : use active RMA instead? => maybe but it will (maybe) not work with GPUs: 
    #   https://www.open-mpi.org/faq/?category=runcuda
    # TODO : use CUDA/ROCM-aware MPI
    # TODO : use 4 views for each side for each variable ? (2 will be contiguous, 2 won't)
    #   <- pre-calculate them!
    # TODO : try to mix the comms: send to left and receive from right, then vice-versa. 
    #  Maybe it can speed things up?    

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
            prev_event = boundaryConditions!(params, data, side;
                dependencies=prev_event, no_threading)
        else
            read_event = read_border_array!(params, data, comm_array, side; 
                dependencies=prev_event, no_threading)
            Event(exchange_with_neighbour, params, comm_array, neighbour; 
                dependencies=read_event) |> wait
            prev_event = write_border_array!(params, data, comm_array, side; 
                no_threading)
        end
    end

    return prev_event
end

#
# Time step computation
#

function dtCFL(params::ArmonParameters{T}, data::ArmonData{V}, prev_dt::T;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; cmat, umat, vmat, domain_mask, work_array_1) = data
    (; cfl, Dt, ideb, ifin, global_grid, domain_size) = params
    @indexing_vars(params)

    (g_nx, g_ny) = global_grid
    (sx, sy) = domain_size

    dt::T = Inf
    dx::T = sx / g_nx
    dy::T = sy / g_ny

    if params.cst_dt
        # Constant time step
        return Dt
    elseif params.use_gpu && params.device isa ROCDevice
        # AMDGPU doesn't support ArrayProgramming, however its implementation of `reduce` is quite
        # fast. Therefore first we compute dt for all cells and store the result in a temporary
        # array, then we reduce this array.
        # TODO : fix this
        if params.projection != :none
            gpu_dtCFL_reduction_euler! = gpu_dtCFL_reduction_euler_kernel!(params.device, params.block_size)
            gpu_dtCFL_reduction_euler!(dx, dy, work_array_1, umat, vmat, cmat, domain_mask;
                ndrange=length(cmat), dependencies) |> wait
            dt = reduce(min, work_array_1)
        else
            gpu_dtCFL_reduction_lagrange! = gpu_dtCFL_reduction_lagrange_kernel!(params.device, params.block_size)
            gpu_dtCFL_reduction_lagrange!(work_array_1, cmat, domain_mask;
                ndrange=length(cmat), dependencies) |> wait
            dt = reduce(min, work_array_1) * min(dx, dy)
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
    elseif prev_dt == 0
        # First cycle: use the initial time step if defined
        return Dt != 0 ? Dt : cfl * dt
    else
        # CFL condition and maximum increase per cycle of the time step
        return convert(T, min(cfl * dt, 1.05 * prev_dt))
    end
end


function dtCFL_MPI(params::ArmonParameters{T}, data::ArmonData{V}, prev_dt::T;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    @perf_task "loop" "dtCFL" local_dt::T = @time_expr_c dtCFL(params, data, prev_dt; dependencies)

    if params.cst_dt || !params.use_MPI
        return local_dt
    end

    # Reduce all local_dts and broadcast the result to all processes
    @perf_task "comms" "MPI_dt" @time_expr_c "dt_Allreduce_MPI" dt = MPI.Allreduce(
        local_dt, MPI.Op(min, T), params.cart_comm)
    return dt
end

#
# Lagrangian cell update
#

function cellUpdate!(params::ArmonParameters{T}, data::ArmonData{V}, dt::T;
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}}
    (; ideb, ifin) = params

    # TODO : use the new ranges functions to achieve this
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

    u = params.current_axis == X_axis ? data.umat : data.vmat
    event = cell_update!(params, data, first_i:last_i, dt, u; dependencies)
    if params.projection == :none
        x = params.current_axis == X_axis ? data.x : data.y
        event = cell_update_lagrange!(params, data, first_i:last_i, last_i, dt, x; dependencies=event)
    end

    return event
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


function projection_remap!(params::ArmonParameters{T}, data::ArmonData{V}, host_array::W, 
        dt::T; dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    params.projection == :none && return dependencies

    if params.use_MPI && !params.single_comm_per_axis_pass
        # Additional communications phase needed to get the new values of the lagrangian cells
        # TODO: put this outside of the function
        # TODO: this should be done also in the non-MPI case, because of cellUpdate! changing more
        #  cells that it should.
        dependencies = boundaryConditions!(params, data, host_array, params.current_axis; dependencies)
    end

    (; work_array_1, work_array_2, work_array_3, work_array_4) = data
    domain_ranges = compute_domain_ranges(params)
    advection_range = full_domain_projection_advection(domain_ranges)

    advection_ρ  = work_array_1
    advection_uρ = work_array_2
    advection_vρ = work_array_3
    advection_Eρ = work_array_4

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
# Axis indexing parameters
#

function update_axis_parameters(params::ArmonParameters{T}, axis::Axis) where T
    (; row_length, global_grid, domain_size) = params
    (g_nx, g_ny) = global_grid
    (sx, sy) = domain_size

    params.current_axis = axis

    if axis == X_axis
        params.s = 1
        params.dx = sx / g_nx
    else  # axis == Y_axis
        params.s = row_length
        params.dx = sy / g_ny
    end
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
    elseif params.axis_splitting == :X_only
        return [(X_axis, T(1.0))]
    elseif params.axis_splitting == :Y_only
        return [(Y_axis, T(1.0))]
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
    full_range = DomainRange(col_range, row_range)

    # Inner range

    if current_axis == X_axis
        # Parse the cells row by row, excluding 'nghost' columns at the left and right
        col_range = @i(1,1):row_length:@i(1,ny)
        row_range = nghost+1:nx-nghost
    else
        # Parse the cells row by row, excluding 'nghost' rows at the top and bottom
        col_range = @i(1,nghost+1):row_length:@i(1,ny-nghost)
        row_range = 1:nx
    end

    inner_range = DomainRange(col_range, row_range)

    # Outer range: left/bottom

    if current_axis == X_axis
        # Parse the cells row by row, for the first 'nghost' columns on the left
        col_range = @i(1,1):row_length:@i(1,ny)
        row_range = 1:nghost
    else
        # Parse the cells row by row, for the first 'nghost' rows at the bottom
        col_range = @i(1,1):row_length:@i(1,nghost)
        row_range = 1:nx
    end
    
    outer_lb_range = DomainRange(col_range, row_range)

    # Outer range: right/top
    if current_axis == X_axis
        # Parse the cells row by row, for the last 'nghost' columns on the right
        col_range = @i(1,1):row_length:@i(1,ny)
        row_range = nx-nghost+1:nx
    else
        # Parse the cells row by row, for the last 'nghost' rows at the top
        col_range = @i(1,ny-nghost+1):row_length:@i(1,ny)
        row_range = 1:nx
    end
    outer_rt_range = DomainRange(col_range, row_range)

    if params.single_comm_per_axis_pass
        r = params.extra_ring_width

        # Add 'r' columns/rows on each of the 4 sides
        full_range  = DomainRange(inflate(full_range.col,  r), inflate(full_range.row,  r))
        inner_range = DomainRange(inflate(inner_range.col, r), inflate(inner_range.row, r))

        if current_axis == X_axis
            # Shift the outer domain to the left and right by 'r' cells, and add 'r' rows at the top and bottom
            outer_lb_range = DomainRange(inflate(outer_lb_range.col, r), shift(outer_lb_range.row, -r))
            outer_rt_range = DomainRange(inflate(outer_rt_range.col, r), shift(outer_rt_range.row,  r))
        else
            # Shift the outer domain to the top and bottom by 'r' cells, and add 'r' columns at the left and right
            outer_lb_range = DomainRange(shift(outer_lb_range.col, -r), inflate(outer_lb_range.row, r))
            outer_rt_range = DomainRange(shift(outer_rt_range.col,  r), inflate(outer_rt_range.row, r))
        end
    end

    return DomainRanges(full_range, inner_range, outer_lb_range, outer_rt_range, current_axis)
end

#
# Reading/Writing
#

function write_data_to_file(params::ArmonParameters, data::ArmonData,
        col_range, row_range, file; direct_indexing=false, for_3D=true)
    @indexing_vars(params)

    vars_to_write = [data.x, data.y, data.rho, data.umat, data.vmat, data.pmat]

    p = params.output_precision
    format = Printf.Format(join(repeat(["%#$(p+7).$(p)e"], length(vars_to_write)), ", ") * "\n")

    for j in col_range
        for i in row_range
            if direct_indexing
                idx = i + j - 1
            else
                idx = @i(i, j)
            end

            Printf.format(file, format, getindex.(vars_to_write, idx)...)
        end
        for_3D && println(file)  # Separate rows to use pm3d plotting with gnuplot
    end
end


function build_file_path(params::ArmonParameters, file_name::String)
    (; output_dir, use_MPI, is_root, cart_coords) = params

    file_path = joinpath(output_dir, file_name)

    if is_root && !isdir(output_dir)
        mkpath(output_dir)
    end

    if use_MPI
        (cx, cy) = cart_coords
        params.use_MPI && (file_path *= "_$(cx)x$(cy)")
    end

    return file_path
end


function write_sub_domain_file(params::ArmonParameters, data::ArmonData, file_name::String; no_msg=false)
    (; silent, nx, ny, is_root, nghost) = params

    output_file_path = build_file_path(params, file_name)

    open(output_file_path, "w") do file
        col_range = 1:ny
        row_range = 1:nx
        if params.write_ghosts
            col_range = inflate(col_range, nghost)
            row_range = inflate(row_range, nghost)
        end

        write_data_to_file(params, data, col_range, row_range, file)
    end

    if !no_msg && is_root && silent < 2
        println("\nWrote to files $(output_file_path)_*x*")
    end
end


function write_slices_files(params::ArmonParameters, data::ArmonData, file_name::String; no_msg=false)
    (; output_dir, silent, nx, ny, use_MPI, is_root, cart_comm, global_grid, proc_dims, cart_coords) = params

    if is_root && !isdir(output_dir)
        mkpath(output_dir)
    end

    # Wait for the root command to complete
    use_MPI && MPI.Barrier(cart_comm)

    (g_nx, g_ny) = global_grid
    (px, py) = proc_dims
    (cx, cy) = cart_coords

    ((nx != ny) || (px != py)) && error("Domain slices are only implemented for square domains on a square process grid.")

    # Middle row
    cy_mid = cld(py, 2) - 1
    if cy == cy_mid
        y_mid = cld(g_ny, 2) - ny * cy + 1
        output_file_path_X = build_file_path(params, file_name * "_X")
        open(output_file_path_X, "w") do file
            col_range = y_mid:y_mid
            row_range = 1:nx
            write_data_to_file(params, data, col_range, row_range, file; for_3D=false)
        end
    end

    # Middle column
    cx_mid = cld(px, 2) - 1
    if cx == cx_mid
        x_mid = cld(g_nx, 2) - nx * cx + 1
        output_file_path_Y = build_file_path(params, file_name * "_Y")
        open(output_file_path_Y, "w") do file
            col_range = 1:ny
            row_range = x_mid:x_mid
            write_data_to_file(params, data, col_range, row_range, file; for_3D=false)
        end
    end

    # Diagonal
    if cx == cy
        output_file_path_D = build_file_path(params, file_name * "_D")
        open(output_file_path_D, "w") do file
            col_range = 1:1
            row_range = params.ideb:(params.row_length+1):(params.ifin+params.row_length+1)
            write_data_to_file(params, data, col_range, row_range, file; for_3D=false, direct_indexing=true)
        end
    end

    if !no_msg && is_root && silent < 2
        if params.use_MPI
            println("Wrote slices to files $(joinpath(output_dir, file_name))_*_*x*")
        else
            println("Wrote slices to files $(joinpath(output_dir, file_name))_*")
        end
    end
end


function read_data_from_file(params::ArmonParameters{T}, data::ArmonData{V},
        col_range, row_range, file; direct_indexing=false) where {T, V <: AbstractArray{T}}
    @indexing_vars(params)

    vars_to_read = [data.x, data.y, data.rho, data.umat, data.vmat, data.pmat]

    for j in col_range
        for i in row_range
            if direct_indexing
                idx = i + j - 1
            else
                idx = @i(i, j)
            end

            for var in vars_to_read[1:end-1]
                var[idx] = parse(T, readuntil(file, ','))
            end
            vars_to_read[end][idx] = parse(T, readuntil(file, '\n'))
        end
    end
end


function read_sub_domain_file!(params::ArmonParameters, data::ArmonData, file_name::String)
    (; nx, ny, nghost) = params

    file_path = build_file_path(params, file_name)

    open(file_path, "r") do file
        col_range = 1:ny
        row_range = 1:nx
        if params.write_ghosts
            col_range = inflate(col_range, nghost)
            row_range = inflate(row_range, nghost)
        end

        read_data_from_file(params, data, col_range, row_range, file)
    end
end

#
# Comparison functions
#

function compare_data(label::String, params::ArmonParameters{T}, 
        ref_data::ArmonData{V}, our_data::ArmonData{V}; mask=nothing) where {T, V <: AbstractArray{T}}
    (; row_length, nghost, nbcell, comparison_tolerance) = params
    different = false
    fields_to_compare = (:x, :y, :rho, :umat, :vmat, :pmat)
    for name in fields_to_compare
        ref_val = getfield(ref_data, name)
        our_val = getfield(our_data, name)

        diff_mask = .~ isapprox.(ref_val, our_val; atol=comparison_tolerance)
        !params.write_ghosts && (diff_mask .*= our_data.domain_mask)
        !isnothing(mask) && (diff_mask .*= mask)
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


function domain_mask_with_ghosts(params::ArmonParameters{T}, mask::V) where {T, V <: AbstractArray{T}}
    (; nbcell, nx, ny, nghost, row_length) = params

    r = params.extra_ring_width + nghost
    axis = params.current_axis

    for i in 1:nbcell
        ix = ((i-1) % row_length) - nghost
        iy = ((i-1) ÷ row_length) - nghost

        mask[i] = (
               (-r ≤ ix < nx+r && -r ≤ iy < ny+r)  # The sub-domain region plus a ring of ghost cells...
            && ( 0 ≤ ix < nx   ||  0 ≤ iy < ny  )  # ...while excluding the corners of the sub-domain...
            &&((axis == X_axis &&  0 ≤ iy < ny  )  # ...and excluding the ghost cells outside of the
            || (axis == Y_axis &&  0 ≤ ix < nx  )) # current axis
        ) ? 1 : 0
    end
end


function compare_with_file(params::ArmonParameters{T}, 
        data::ArmonData{V}, file_name::String, label::String) where {T, V <: AbstractArray{T}}
    ref_data = ArmonData(T, params.nbcell, params.comm_array_size)
    read_sub_domain_file!(params, ref_data, file_name)

    if params.use_MPI && params.write_ghosts
        domain_mask_with_ghosts(params, ref_data.domain_mask)
        different = compare_data(label, params, ref_data, data; mask=ref_data.domain_mask)
    else
        different = compare_data(label, params, ref_data, data)
    end

    if params.use_MPI
        different = MPI.Allreduce(different, |, params.cart_comm)
    end

    return different
end


function step_checkpoint(params::ArmonParameters{T}, 
        data::ArmonData{V}, cpu_data::ArmonData{W},
        step_label::String, cycle::Int, axis::Union{Axis, Nothing};
        dependencies=NoneEvent()) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    if params.compare
        wait(dependencies)

        if W != V
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

    total_mass::T = zero(T)
    total_energy::T = zero(T)

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
        total_mass   = MPI.Allreduce(total_mass,   MPI.SUM, params.cart_comm)
        total_energy = MPI.Allreduce(total_energy, MPI.SUM, params.cart_comm)
    end

    return total_mass, total_energy
end

#
# Main time loop
#

function time_loop(params::ArmonParameters{T}, data::ArmonData{V},
        cpu_data::ArmonData{W}) where {T, V <: AbstractArray{T}, W <: AbstractArray{T}}
    (; maxtime, maxcycle, nx, ny, silent, animation_step, is_root, dt_on_even_cycles) = params

    cycle  = 0
    t::T   = 0.
    next_dt::T = 0.
    prev_dt::T = 0.
    total_cycles_time::T = 0.

    t1 = time_ns()
    t_warmup = t1

    if params.use_MPI && params.use_gpu
        # Host version of temporary array used for MPI communications
        host_array = Vector{T}(undef, params.comm_array_size)
    else
        host_array = Vector{T}()
    end

    if params.async_comms
        # Disable multi-threading when computing the outer domains, since Polyester cannot run
        # multiple loops at the same time.
        outer_params = copy(params)
        outer_params.use_threading = false
    else
        outer_params = params
    end

    if silent <= 1
        initial_mass, initial_energy = conservation_vars(params, data)
    end

    update_axis_parameters(params, first(split_axes(params, cycle))[1])
    domain_ranges = compute_domain_ranges(params)

    prev_event = NoneEvent()

    # Finalize the initialisation by calling the EOS on the entire domain
    update_EOS!(params, data, full_domain(domain_ranges), :full) |> wait
    step_checkpoint(params, data, cpu_data, "update_EOS_init", cycle, params.current_axis) && @goto stop

    # Main solver loop
    while t < maxtime && cycle < maxcycle
        cycle_start = time_ns()

        if !dt_on_even_cycles || iseven(cycle)
            next_dt = dtCFL_MPI(params, data, prev_dt; dependencies=prev_event)
            prev_event = NoneEvent()

            if is_root && (!isfinite(next_dt) || next_dt <= 0.)
                error("Invalid dt for cycle $cycle: $next_dt")
            end

            if cycle == 0
                prev_dt = next_dt
            end
        end

        for (axis, dt_factor) in split_axes(params, cycle)
            update_axis_parameters(params, axis)
            domain_ranges = compute_domain_ranges(params)

            @perf_task "loop" "EOS+comms+fluxes" @time_expr_c "EOS+comms+fluxes" if params.async_comms
                @sync begin
                    @async begin
                        event_2 = update_EOS!(params, data, 
                            inner_domain(domain_ranges), :inner; dependencies=prev_event)
                        event_2 = numericalFluxes!(params, data, prev_dt * dt_factor, domain_ranges, :inner; dependencies=event_2)
                        wait(event_2)
                    end

                    @async begin
                        # Since the other async tack is the one who should be using all the threads,
                        # here we forcefully disable multi-threading.
                        no_threading = true

                        event_1 = update_EOS!(outer_params, data, outer_lb_domain(domain_ranges), :outer; 
                            dependencies=prev_event, no_threading)
                        event_1 = update_EOS!(outer_params, data, outer_rt_domain(domain_ranges), :outer; 
                            dependencies=event_1, no_threading)

                        event_1 = boundaryConditions!(outer_params, data, host_array, axis; 
                            dependencies=event_1, no_threading)

                        event_1 = numericalFluxes!(outer_params, data, prev_dt * dt_factor, 
                            domain_ranges, :outer_lb; dependencies=event_1, no_threading)
                        event_1 = numericalFluxes!(outer_params, data, prev_dt * dt_factor, 
                            domain_ranges, :outer_rt; dependencies=event_1, no_threading)
                        wait(event_1)
                    end
                end

                step_checkpoint(params, data, cpu_data, "EOS+comms+fluxes", cycle, axis) && @goto stop
                event = NoneEvent()
            else
                event = update_EOS!(params, data, full_domain(domain_ranges), :full; 
                    dependencies=prev_event)
                step_checkpoint(params, data, cpu_data, "update_EOS", cycle, axis; 
                    dependencies=event) && @goto stop

                event = boundaryConditions!(params, data, host_array, axis; dependencies=event)
                step_checkpoint(params, data, cpu_data, "boundaryConditions", cycle, axis; 
                    dependencies=event) && @goto stop

                event = numericalFluxes!(params, data, prev_dt * dt_factor, domain_ranges, :full; dependencies=event)
                step_checkpoint(params, data, cpu_data, "numericalFluxes", cycle, axis;
                    dependencies=event) && @goto stop

                params.measure_time && wait(event)
            end

            @perf_task "loop" "cellUpdate" event = cellUpdate!(params, data, prev_dt * dt_factor;
                dependencies=event)
            step_checkpoint(params, data, cpu_data, "cellUpdate", cycle, axis; 
                dependencies=event) && @goto stop

            @perf_task "loop" "euler_proj" event = projection_remap!(params, data, host_array,
                prev_dt * dt_factor; dependencies=event)
            step_checkpoint(params, data, cpu_data, "projection_remap", cycle, axis;
                dependencies=event) && @goto stop

            prev_event = event
        end

        if !is_warming_up()
            total_cycles_time += time_ns() - cycle_start
        end

        cycle += 1

        if is_root
            if silent <= 1
                wait(prev_event)
                current_mass, current_energy = conservation_vars(params, data)
                ΔM = abs(initial_mass - current_mass)     / initial_mass   * 100
                ΔE = abs(initial_energy - current_energy) / initial_energy * 100
                @printf("Cycle %4d: dt = %.18f, t = %.18f, |ΔM| = %#8.6g%%, |ΔE| = %#8.6g%%\n",
                    cycle, prev_dt, t, ΔM, ΔE)
            end
        elseif silent <= 1
            wait(prev_event)
            conservation_vars(params, data)
        end

        t += prev_dt
        prev_dt = next_dt

        if cycle == 5
            wait(prev_event)
            t_warmup = time_ns()
            set_warmup(false)
        end

        if animation_step != 0 && (cycle - 1) % animation_step == 0
            wait(prev_event)
            frame_index = (cycle - 1) ÷ animation_step
            frame_file = joinpath("anim", params.output_file) * "_" * @sprintf("%03d", frame_index)
            write_sub_domain_file(params, data, frame_file)
        end
    end

    wait(prev_event)

    @label stop

    t2 = time_ns()

    nb_cells = nx * ny
    grind_time = (t2 - t_warmup) / ((cycle - 5) * nb_cells)

    if is_root
        if params.compare
            # ignore timing errors when comparing
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
            println("Last cycle:  ", @sprintf("%.18f", t), " sec, Δt=", @sprintf("%.18f", next_dt), " sec")
        end
    end

    return next_dt, cycle, convert(T, 1 / grind_time), total_cycles_time
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
            rank, proc_size, local_rank, local_size, cart_coords[1], cart_coords[2], 
            minimum(getcpuids()), maximum(getcpuids()))
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
    @perf_task "init" "alloc" data = ArmonData(params)
    @perf_task "init" "init_test" wait(init_test(params, data))

    if params.use_gpu
        copy_time = @elapsed d_data = data_to_gpu(data, get_device_array(params))
        (is_root && silent <= 2) && @printf("Time for copy to device: %.3g sec\n", copy_time)

        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, d_data, data)

        data_from_gpu(data, d_data)
    else
        @pretty_time dt, cycles, cells_per_sec, total_time = time_loop(params, data, data)
    end

    if params.write_output
        write_sub_domain_file(params, data, params.output_file)
    end

    if params.write_slices
        write_slices_files(params, data, params.output_file)
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
                    step_label, step_time / 1e6, step_time / axis_total_time * 100, 
                    step_time / total_time * 100)
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

    if is_root && params.measure_hw_counters && !isempty(axis_hw_counters)
        sorted_counters = sort(collect(axis_hw_counters); lt=(a, b)->(a[1] < b[1]))
        sorted_counters = map((p)->(string(first(p)) => last(p)), sorted_counters)
        if params.silent < 3
            print_hardware_counters_table(stdout, params.hw_counters_options, sorted_counters)
        end
        if !isempty(params.hw_counters_output)
            open(params.hw_counters_output, "w") do file
                print_hardware_counters_table(file, params.hw_counters_options, sorted_counters; raw_print=true)
            end
        end
    end

    if params.return_data
        return data, dt, cycles, cells_per_sec, sorted_time_contrib, async_efficiency
    else
        return dt, cycles, cells_per_sec, sorted_time_contrib, async_efficiency
    end
end

end
