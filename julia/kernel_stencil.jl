
using StaticArrays


struct ImprintingArray{T, N} <: AbstractArray{T, N}
    dims::NTuple{N, Int}
    offset::NTuple{N, Int}
    min_get::MVector{N, Int}
    max_get::MVector{N, Int}
    min_set::MVector{N, Int}
    max_set::MVector{N, Int}
end


ImprintingArray{T}(dims::Int...) where T = ImprintingArray{T, length(dims)}(tuple(dims...))
ImprintingArray{T}(dims::NTuple{N, Int}) where {T, N} = ImprintingArray{T, N}(dims)
ImprintingArray{T}(dims::NTuple{N, Int}, offset::NTuple{N, Int}) where {T, N} = ImprintingArray{T, N}(dims, offset)
ImprintingArray{T, N}(dims::NTuple{N, Int}) where {T, N} = ImprintingArray{T, N}(dims, ntuple(Returns(0), Val(N)))
ImprintingArray{T, N}(dims::NTuple{N, Int}, offset::NTuple{N, Int}) where {T, N} =
    ImprintingArray{T, N}(
        dims, offset,
        MVector{N, Int}(ntuple(Returns(typemax(Int)), Val(N))),
        MVector{N, Int}(ntuple(Returns(typemin(Int)), Val(N))),
        MVector{N, Int}(ntuple(Returns(typemax(Int)), Val(N))),
        MVector{N, Int}(ntuple(Returns(typemin(Int)), Val(N)))
    )

Base.size(arr::ImprintingArray) = arr.dims
Base.length(arr::ImprintingArray) = prod(arr.dims)

Base.checkbounds(::Type{Bool}, ::ImprintingArray, I...) = true


function Base.copy(arr::ImprintingArray{T, N}) where {T, N}
    ImprintingArray{T, N}(
        arr.dims, arr.offset,
        copy(arr.min_get),
        copy(arr.max_get),
        copy(arr.min_set),
        copy(arr.max_set)
    )
end


idx_to_multi_dims(i::Int, _::NTuple{1, Int}, offset::NTuple{1, Int}) = i + offset[1]

function idx_to_multi_dims(i::Int, dims::NTuple{2, Int}, offset::NTuple{2, Int})
    iy, ix = divrem(i-1, dims[1])
    return ix + 1 + offset[1], iy + 1 + offset[2]
end

function idx_to_multi_dims(i::Int, dims::NTuple{3, Int}, offset::NTuple{3, Int})
    iy, ix = divrem(i-1, dims[1])
    iz, iy = divrem(iy, dims[2])
    return ix + 1 + offset[1], iy + 1 + offset[2], iz + 1 + offset[3]
end


function Base.getindex(arr::ImprintingArray{T, N}, i::Int) where {T, N}
    indexes = idx_to_multi_dims(i, arr.dims, arr.offset)
    arr.min_get .= min.(arr.min_get, indexes)
    arr.max_get .= max.(arr.max_get, indexes)
    return zero(T)
end


function Base.setindex!(arr::ImprintingArray{T, N}, _, i::Int) where {T, N}
    indexes = idx_to_multi_dims(i, arr.dims, arr.offset)
    arr.min_set .= min.(arr.min_set, indexes)
    arr.max_set .= max.(arr.max_set, indexes)
    return zero(T)
end


function Base.union(arr1::ImprintingArray{T, N}, arr2::ImprintingArray{T, N}) where {T, N}
    any(arr1.dims   .!= arr2.dims  ) && throw(DimensionMismatch("$(arr1.dims) != $(arr2.dims)"))
    any(arr1.offset .!= arr2.offset) && throw(DimensionMismatch("$(arr1.offset) != $(arr2.offset) (offset mismatch)"))
    return ImprintingArray{T, N}(
        arr1.dims, arr1.offset,
        MVector{N, Int}(min.(arr1.min_get, arr2.min_get)),
        MVector{N, Int}(max.(arr1.max_get, arr2.max_get)),
        MVector{N, Int}(min.(arr1.min_set, arr2.min_set)),
        MVector{N, Int}(max.(arr1.max_set, arr2.max_set))
    )
end


function Base.union!(arr1::ImprintingArray{T, N}, arr2::ImprintingArray{T, N}) where {T, N}
    any(arr1.dims   .!= arr2.dims  ) && throw(DimensionMismatch("$(arr1.dims) != $(arr2.dims)"))
    any(arr1.offset .!= arr2.offset) && throw(DimensionMismatch("$(arr1.offset) != $(arr2.offset) (offset mismatch)"))
    arr1.min_get .= min.(arr1.min_get, arr2.min_get)
    arr1.max_get .= max.(arr1.max_get, arr2.max_get)
    arr1.min_set .= min.(arr1.min_set, arr2.min_set)
    arr1.max_set .= max.(arr1.max_set, arr2.max_set)
    return arr1
end


Base.union(arrays::ImprintingArray{T, N}...) where {T, N} = foldl(union, arrays)
Base.union!(arrays::ImprintingArray{T, N}...) where {T, N} = foldl(union!, arrays)


was_written(arr::ImprintingArray) = !any(arr.min_set .== typemax(Int))
was_read(arr::ImprintingArray) = !any(arr.min_get .== typemax(Int))
was_accessed(arr::ImprintingArray) = was_written(arr) || was_read(arr)

function reset_gets(arr::ImprintingArray{T, N}) where {T, N}
    arr.min_get .= ntuple(Returns(typemax(Int)), Val(N))
    arr.max_get .= ntuple(Returns(typemin(Int)), Val(N))
end

function reset_sets(arr::ImprintingArray{T, N}) where {T, N}
    arr.min_set .= ntuple(Returns(typemax(Int)), Val(N))
    arr.max_set .= ntuple(Returns(typemin(Int)), Val(N))
end


function Base.show(io::IO, arr::ImprintingArray{T, N}) where {T, N}
    if !was_read(arr)
        get_range = "∅"
    else
        get_range = join(map(i -> arr.min_get[i]:arr.max_get[i], 1:N), ", ")
    end

    if !was_written(arr)
        set_range = "∅"
    else
        set_range = join(map(i -> arr.min_set[i]:arr.max_set[i], 1:N), ", ")
    end
    
    print(io, "ImprintingArray{get[$get_range],set[$set_range]}")
end


"""

Known problems:
 - branches are not supported

```
kernel_stencil(params, Armon.acoustic_GAD!; type_args=Dict{Symbol, Any}(
    :T => Float64,
    :LimiterType => Armon.MinmodLimiter
))

Dict{Symbol, Any} with 6 entries:
  :ustar => ImprintingArray{get[∅],set[1:1]}
  :cmat  => ImprintingArray{get[-1:2],set[∅]}
  :rho   => ImprintingArray{get[-1:2],set[∅]}
  :pstar => ImprintingArray{get[∅],set[1:1]}
  :pmat  => ImprintingArray{get[-1:2],set[∅]}
  :u     => ImprintingArray{get[-1:2],set[∅]}
```
"""
function kernel_stencil(params::ArmonParameters, kernel::Function;
        args::Dict{Symbol, Any}=Dict{Symbol, Any}(),
        type_args::Dict{Symbol, Any}=Dict{Symbol, Any}())
    kernel_module = parentmodule(kernel)

    # Get the name of the cpu version of the kernel
    kernel_name = string(nameof(kernel))
    if !startswith(kernel_name, "cpu_")
        kernel_name = "cpu_" * kernel_name
    end
    kernel_name = Symbol(kernel_name)

    cpu_kernel_func = getproperty(kernel_module, kernel_name)
    cpu_kernel = first(methods(cpu_kernel_func))

    # Get a DataType instance of the signature of the method
    args_data_type = Base.unwrap_unionall(cpu_kernel.sig)
    args_types = args_data_type.types
    args_types = args_types[2:end]  # Skip the method type

    # Get low level info about the method declaration
    _, decls, _, _ = Base.arg_decl_parts(cpu_kernel)
    args_names = decls[2:end]

    # Ignore the first 'params' argument
    args_types = args_types[2:end]
    args_names = args_names[2:end]

    # Ignore the SIMD and threading switches arguments
    args_types = args_types[1:end-2]
    args_names = args_names[1:end-2]

    kernel_ranges = []
    if args_names[1][1] == "loop_range"
        # kernel with 1D indexing
        push!(kernel_ranges, 1:1)

        args_types = args_types[2:end]
        args_names = args_names[2:end]
    else
        # kernel with 2D indexing
        push!(kernel_ranges, 1:1, 1:1)

        args_types = args_types[3:end]
        args_names = args_names[3:end]
    end

    imprinting_arrays = Dict{Symbol, Any}()

    # Build the array of arguments to pass to the kernel. All dynamic arrays types are replaced with
    # ImprintingArray.
    params_fields = fieldnames(typeof(params))
    args_instances = []
    for (arg_type, arg_name_info) in zip(args_types, args_names)
        name, _ = arg_name_info
        name = Symbol(name)

        if haskey(args, name)
            push!(args_instances, args[name])
            continue
        end

        if name in params_fields
            push!(args_instances, getfield(params, name))
            continue
        end

        arg_val = nothing

        if arg_type isa TypeVar
            if arg_type.ub isa UnionAll
                # arg_type = V <: AbstractArray{T, N}
                array_type = Base.unwrap_unionall(arg_type.ub)
                is_arg_abstract_array = array_type.name.name == :AbstractArray
            else
                if haskey(type_args, arg_type.name) && type_args[arg_type.name] == AbstractArray
                    # The type is not annotated as an abstract array ('V <: AbstractArray') but the
                    # user says so here.
                    array_type = Base.unwrap_unionall(AbstractArray)
                    is_arg_abstract_array = true
                else
                    is_arg_abstract_array = false
                end
            end

            if is_arg_abstract_array
                if array_type.parameters[1] isa TypeVar
                    # AbstractArray{T, N} -> Variadic type, by default we choose T=Float64
                    var_type_name = array_type.parameters[1].name
                    if haskey(type_args, var_type_name)
                        # This specific eltype is specified by the user
                        arg_eltype = type_args[var_type_name]
                    else
                        arg_eltype = Float64
                    end
                else
                    # AbstractArray{<fixed type>, N}
                    arg_eltype = array_type.parameters[1]
                end
                imprinting_arrays[name] = ImprintingArray(arg_eltype, params.global_grid)
                arg_val = imprinting_arrays[name]
            else
                if haskey(type_args, arg_type.name)
                    arg_type = type_args[arg_type.name]
                else
                    error("Missing default type value for method template: $arg_type")
                end
            end
        end

        if isnothing(arg_val)
            # Try to use the default value of the type: 0 for numbers, default constructor for other
            # types
            if arg_type <: Number
                arg_val = zero(arg_type)
            else
                arg_val = arg_type()
            end
        end

        push!(args_instances, arg_val)
    end

    cpu_kernel_func(params, kernel_ranges..., args_instances..., Armon.KernelWithoutThreading(), Armon.KernelWithoutSIMD())

    return imprinting_arrays
end
