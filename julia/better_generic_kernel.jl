

function dummy_idx_expr(args...)
    esc(quote
        begin
            Expr(:meta, $(args...))
            @index(Global, Linear)
        end
    end)
end


macro index_1D_lin() dummy_idx_expr(:index_1D_lin) end
macro index_2D_lin() dummy_idx_expr(:index_2D_lin) end
macro iter_idx()     dummy_idx_expr(:iter_idx)     end
macro kernel_options(options...) 
    esc(quote
        begin
            Expr(:meta, :kernel_options, options...) 
        end
    end)
end
macro kernel_init(expr)
    esc(quote
        begin
            Expr(:meta, :kernel_init)
            $expr
        end
    end)
end


abstract type KernelContext end
struct CPU_kernel <: KernelContext end
struct GPU_kernel <: KernelContext end


function process_meta_block(state, expr, ::CPU_kernel, ::Val{:index_1D_lin})
    # TODO
end


function process_meta_block(state, expr, ::GPU_kernel, ::Val{:index_1D_lin})
    # TODO
end


function process_meta_block(state, expr, ::CPU_kernel, ::Val{:index_2D_lin})
    # TODO
end


function process_meta_block(state, expr, ::GPU_kernel, ::Val{:index_1D_lin})
    # TODO
end


function process_meta_block(state, expr, ::CPU_kernel, ::Val{:iter_idx})
    # TODO
end


function process_meta_block(state, expr, ::GPU_kernel, ::Val{:iter_idx})
    # TODO
end


function process_meta_block(state, expr, ::CPU_kernel, ::Val{:kernel_options})
    # TODO
end


function process_meta_block(state, expr, ::GPU_kernel, ::Val{:kernel_options})
    # TODO
end


function process_meta_block(state, expr, ::CPU_kernel, ::Val{:kernel_init})
    # TODO
end


function process_meta_block(state, expr, ::GPU_kernel, ::Val{:kernel_init})
    # TODO
end


function process_meta_block(state, expr, ::KernelContext, ::Val{S}) where S
    @debug "Generic kernel - At $(state.function_loc): Unknown meta expression '$S'"
    return expr  # Unknown meta expression
end


function parse_kernel_body(expr)
    # TODO: find all :meta expressions, and call 'process_meta_block' with the block containing the
    #  meta expression
end


struct KernelOptions{SIMD, Threading} end  # TODO: expandable options?

struct KernelOptions2{Options} end

const kernel_options = Dict{Symbol, Tuple{Type, Any}}()

function add_kernel_option(name::Symbol, ::T, default::T) where T
    !isbitstype(T) && error("All kernel options must be a bits type")
    haskey(kernel_options, name) && error("Duplicate option: $name")
    kernel_options[name] = (T, default)
end

add_kernel_option(:SIMD, Bool, true)
add_kernel_option(:Threading, Bool, true)


function build_kernel_options(; kwargs...)
    names = keys(kernel_options)
    types = Iterators.map(first, values(kwargs))
    values = Vector{Any}(undef, length(names))

    non_default_count = 0
    for (i, (name, (type, default))) in enumerate(kernel_options)
        if haskey(kwargs, name)
            non_default_count += 1
            value = kwargs[name]
            typeassert(value, type)
        else
            value = default
        end
        values[i] = value
    end

    if non_default_count < length(kwargs)
        unknown_options = filter(p -> !haskey(kernel_options, first(p)), kwargs)
        error("Unknown options: $unknown_options")
    end

    tuple_type = Tuple{types...}
    return NamedTuple{names, tuple_type}(values)
end


@generated function kernel_function(loop_body, ::KernelOptions{SIMD, Threading}) where {SIMD, Threading}
    loop_expr = quote
        for i in range
            $loop_body
        end
    end

    # TODO: parse expandable options + process method for each option

    if SIMD
        loop_expr = quote
            @fastmath @inbounds @simd ivdep $loop_expr
        end
    end

    if Threading
        loop_expr = quote
            @batch $loop_expr
        end
    end

    return loop_expr
end
