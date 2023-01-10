
import Base: oneto, length, size, axes, isempty, first, last, in

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

linear_range(dr::DomainRange) = first(dr):last(dr)

#
# DomainRanges: represents the different sub-domains of a domain for a sweep along a direction
#

struct DomainRanges
    full::DomainRange
    inner::DomainRange
    outer_lb::DomainRange  # left/bottom
    outer_rt::DomainRange  # right/top
    direction::Axis
end

full_domain(dr::DomainRanges) = dr.full
inner_domain(dr::DomainRanges) = dr.inner
outer_lb_domain(dr::DomainRanges) = dr.outer_lb
outer_rt_domain(dr::DomainRanges) = dr.outer_rt

# For fluxes only, we shift the computation to the right/top by one column/row since the flux at
# index 'i' is the flux between the cells 'i-s' and 'i', but we also need the fluxes on the other
# side, between the cells 'i' and 'i+s'. Therefore we need this shift to compute all fluxes needed 
# later, but only on outer right side.

# TODO: fix incorrect domain for GAD fluxes
full_fluxes_domain(dr::DomainRanges) = inflate_dir(expand_dir(dr.full, dr.direction, 1), dr.direction, 1)
inner_fluxes_domain(dr::DomainRanges) = expand_dir(dr.inner, dr.direction, 1)
outer_fluxes_lb_domain(dr::DomainRanges) = dr.outer_lb
outer_fluxes_rt_domain(dr::DomainRanges) = shift_dir(dr.outer_rt, dr.direction, 1)

full_domain_projection_advection(dr::DomainRanges) = inflate_dir(dr.full, dr.direction)
