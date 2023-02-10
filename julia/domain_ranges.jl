
import Base: oneto, length, size, axes, isempty, first, last, in, show

#
# Range utilities
#

shift(r::AbstractUnitRange,   n::Int = 1) = r .+ n
shift(r::OrdinalRange,        n::Int = 1) = r .+ (step(r) * n)
expand(r::AbstractUnitRange,  n::Int = 1) = first(r):(last(r)+n)
expand(r::OrdinalRange,       n::Int = 1) = first(r):step(r):(last(r)+step(r)*n)
prepend(r::AbstractUnitRange, n::Int = 1) = (first(r)-n):last(r)
prepend(r::OrdinalRange,      n::Int = 1) = (first(r)-step(r)*n):step(r):last(r)
inflate(r::AbstractUnitRange, n::Int = 1) = (first(r)-n):(last(r)+n)
inflate(r::OrdinalRange,      n::Int = 1) = (first(r)-step(r)*n):step(r):(last(r)+step(r)*n)

function connect_ranges(r1::AbstractUnitRange, r2::AbstractUnitRange)
    last(r1) > first(r2) && return connect_ranges(r2, r1)
    !isempty(intersect(r1, r2)) && error("$r1 and $r2 intersect")
    last(r1) + 1 != first(r2) && error("$r1 doesn't touch $r2")
    return first(r1):last(r2)
end

function connect_ranges(r1::OrdinalRange, r2::OrdinalRange)
    last(r1) > first(r2) && return connect_ranges(r2, r1)
    !isempty(intersect(r1, r2)) && error("$r1 and $r2 intersect")
    step(r1) != step(r2) && error("$r1 and $r2 have different steps")
    last(r1) + step(r1) != first(r2) && error("$r1 doesn't touch $r2")
    return first(r1):step(r1):last(r2)
end

#
# DomainRange: Two dimensional range to index a 2D array stored with contiguous rows
#

struct DomainRange
    col::StepRange{Int, Int}
    row::StepRange{Int, Int}
end

length(dr::DomainRange) = length(dr.col) * length(dr.row)
size(dr::DomainRange) = (length(dr.col), length(dr.row))
axes(dr::DomainRange) = (oneto(length(dr.col)), oneto(length(dr.row)))

isempty(dr::DomainRange) = length(dr) == 0

first(dr::DomainRange) = first(dr.col) + first(dr.row) - 1
last(dr::DomainRange)  = last(dr.col)  + last(dr.row)  - 1

function in(x::Integer, dr::DomainRange)
    first(dr) <= x <= last(dr) || return false
    ix = x - first(dr.col) + 1
    id = fld(ix, step(dr.col))
    ix -= id * step(dr.col)
    return ix in dr.row
end

shift(dr::DomainRange,   n::Int = 1) = DomainRange(dr.col, shift(dr.row, n))
prepend(dr::DomainRange, n::Int = 1) = DomainRange(dr.col, prepend(dr.row, n))
expand(dr::DomainRange,  n::Int = 1) = DomainRange(dr.col, expand(dr.row, n))
inflate(dr::DomainRange, n::Int = 1) = DomainRange(dr.col, inflate(dr.row, n))

@inline function apply_along_direction(dr::DomainRange, dir::Axis, f, args...)
    if dir == X_axis
        return DomainRange(dr.col, f(dr.row, args...))
    else
        return DomainRange(f(dr.col, args...), dr.row)
    end
end

shift_dir(dr::DomainRange, dir::Axis, n::Int = 1)   = apply_along_direction(dr, dir, shift, n)
prepend_dir(dr::DomainRange, dir::Axis, n::Int = 1) = apply_along_direction(dr, dir, prepend, n)
expand_dir(dr::DomainRange, dir::Axis, n::Int = 1)  = apply_along_direction(dr, dir, expand, n)
inflate_dir(dr::DomainRange, dir::Axis, n::Int = 1) = apply_along_direction(dr, dir, inflate, n)

direction_length(dr::DomainRange, dir::Axis) = dir == X_axis ? length(dr.row) : length(dr.col)

linear_range(dr::DomainRange) = first(dr):last(dr)

show(io::IO, dr::DomainRange) = print(io, "DomainRange{$(dr.col), $(dr.row)}")

#
# Steps ranges
#

struct StepsRanges
    direction::Axis
    real_domain::DomainRange

    EOS::DomainRange
    fluxes::DomainRange
    cell_update::DomainRange
    advection::DomainRange
    projection::DomainRange

    outer_lb_EOS::DomainRange
    outer_rt_EOS::DomainRange
    outer_lb_fluxes::DomainRange
    outer_rt_fluxes::DomainRange
    inner_EOS::DomainRange
    inner_fluxes::DomainRange
end


function old_compute_domain_ranges(params::ArmonParameters, drs#= ::DomainRanges =#)
    (; nx, ny, nghost, row_length, current_axis) = params
    @indexing_vars(params)

    ax = current_axis

    real_range = drs.full

    EOS_range = real_range
    fluxes_range = inflate_dir(expand_dir(real_range, ax, 1), ax, 1)
    cell_update_range = real_range
    advection_range = inflate_dir(real_range, ax)
    projection_range = real_range
    outer_lb_EOS_range = drs.outer_lb
    outer_rt_EOS_range = drs.outer_rt
    outer_lb_fluxes_range = drs.outer_lb
    outer_rt_fluxes_range = shift_dir(drs.outer_rt, ax, 1)
    inner_EOS_range = drs.inner
    inner_fluxes_range = expand_dir(drs.inner, ax, 1)

    return StepsRanges(
        ax, real_range,
        EOS_range, fluxes_range, 
        cell_update_range, advection_range, projection_range,
        outer_lb_EOS_range, outer_rt_EOS_range,
        outer_lb_fluxes_range, outer_rt_fluxes_range,
        inner_EOS_range, inner_fluxes_range
    )
end


function steps_ranges(params::ArmonParameters)
    (; nx, ny, nghost, row_length, current_axis) = params
    @indexing_vars(params)

    ax = current_axis

    # Extra cells to compute in each step
    extra_FLX = 1
    extra_UP = 1

    if params.projection == :euler
        # No change
    elseif params.projection == :euler_2nd
        extra_FLX += 1
        extra_UP  += 1
    else
        error("Unknown scheme: $(params.projection)")
    end

    # Real domain
    col_range = @i(1,1):row_length:@i(1,ny)
    row_range = 1:nx
    real_range = DomainRange(col_range, row_range)

    # Steps ranges, computed so that there is no need for an extra BC step before the projection
    EOS_range = real_range  # The BC overwrites any changes to the ghost cells right after
    fluxes_range = inflate_dir(real_range, ax, extra_FLX)
    cell_update_range = inflate_dir(real_range, ax, extra_UP)
    advection_range = expand_dir(real_range, ax, 1)
    projection_range = real_range

    # Fluxes are computed between 'i-s' and 'i', we need one more cell on the right to have all fluxes
    fluxes_range = expand_dir(fluxes_range, ax, 1)

    # Inner ranges: real domain without sides
    inner_EOS_range = inflate_dir(EOS_range, ax, -nghost)
    inner_fluxes_range = inner_EOS_range

    rt_offset = direction_length(inner_EOS_range, ax)

    # Outer ranges: sides of the real domain
    if ax == X_axis
        outer_lb_EOS_range = DomainRange(col_range, row_range[1:nghost])
    else
        outer_lb_EOS_range = DomainRange(col_range[1:nghost], row_range)
    end

    outer_rt_EOS_range = shift_dir(outer_lb_EOS_range, ax, nghost + rt_offset)

    if rt_offset == 0
        # Correction when the side regions overlap
        overlap_width = direction_length(real_range, ax) - 2*nghost
        outer_rt_EOS_range = expand_dir(outer_rt_EOS_range, ax, overlap_width)
    end
    
    outer_lb_fluxes_range = prepend_dir(outer_lb_EOS_range, ax, extra_FLX)
    outer_rt_fluxes_range = expand_dir(outer_rt_EOS_range, ax, extra_FLX + 1)

    return StepsRanges(
        ax, real_range,
        EOS_range, fluxes_range, 
        cell_update_range, advection_range, projection_range,
        outer_lb_EOS_range, outer_rt_EOS_range,
        outer_lb_fluxes_range, outer_rt_fluxes_range,
        inner_EOS_range, inner_fluxes_range
    )
end


function boundary_conditions_indexes(params::ArmonParameters, side::Symbol)
    (; row_length, nx, ny) = params
    @indexing_vars(params)

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

    return i_start, loop_range, stride, d
end
