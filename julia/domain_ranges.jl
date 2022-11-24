
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
# DomainRange: Two dimensional range to index a 2D array stored with contigous rows
#

DomainRange = @NamedTuple{col::StepRange{Int, Int}, row::StepRange{Int, Int}}

shift(dr::DomainRange,   n::Int = 1) = DomainRange((dr.col, shift(dr.row, n)))
prepend(dr::DomainRange, n::Int = 1) = DomainRange((dr.col, prepend(dr.row, n)))
expand(dr::DomainRange,  n::Int = 1) = DomainRange((dr.col, expand(dr.row, n)))
inflate(dr::DomainRange, n::Int = 1) = DomainRange((dr.col, inflate(dr.row, n)))

function shift_dir(dr::DomainRange, dir::Axis, n::Int = 1)
    if dir == X_axis
        return DomainRange((dr.col, shift(dr.row, n)))
    else
        return DomainRange((shift(dr.col, n), dr.row))
    end
end

function prepend_dir(dr::DomainRange, dir::Axis, n::Int = 1)
    if dir == X_axis
        return DomainRange((dr.col, prepend(dr.row, n)))
    else
        return DomainRange((prepend(dr.col, n), dr.row))
    end
end

function expand_dir(dr::DomainRange, dir::Axis, n::Int = 1)
    if dir == X_axis
        return DomainRange((dr.col, inflate(dr.row, n)))
    else
        return DomainRange((inflate(dr.col, n), dr.row))
    end
end

function inflate_dir(dr::DomainRange, dir::Axis, n::Int = 1)
    if dir == X_axis
        return DomainRange((dr.col, inflate(dr.row, n)))
    else
        return DomainRange((inflate(dr.col, n), dr.row))
    end
end

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
# index 'i' is the flux between the cells 'i-s' and 'i', but we also need the fluxes on the other
# side, between the cells 'i' and 'i+s'. Therefore we need this shift to compute all fluxes needed 
# later, but only on outer right side.

full_fluxes_domain(dr::DomainRanges) = expand_dir(dr.full, dr.direction, 1)
inner_fluxes_domain(dr::DomainRanges) = expand_dir(dr.inner, dr.direction, 1)
outer_fluxes_lb_domain(dr::DomainRanges) = dr.outer_lb
outer_fluxes_rt_domain(dr::DomainRanges) = shift_dir(dr.outer_rt, dr.direction, 1)

full_domain_projection_advection(dr::DomainRanges) = inflate_dir(dr.full, dr.direction)
