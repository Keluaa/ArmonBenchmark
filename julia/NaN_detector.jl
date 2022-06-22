

# Use this flag to know when a NaN occured
NaNflag = false

function reset_nan_flag(value::Bool = false)
    global NaNflag = value
end

function nanfound(fncall)
    stk = stacktrace() # stacktrace(catch_backtrace())
    stk = relevantframes(stk)
    info = []
    for frame in stk
        push!(info, frameinfo(frame))
    end
    global NaNflag = true
    println((NaN_ERROR=fncall, stack=tuple(info...,)))
    return NaN
end

function frameinfo(frame)
   func = frame.func
   file = String(frame.file)
   line = frame.line
   result = ("$func in $file(#$line)")
   return result
end

function relevantframes(stack::Vector{Base.StackTraces.StackFrame}; framemin=3)
    framemax = nrelevantframes(stack)
    framemin = min(framemin, framemax)
    return framemax > 0 ? stack[framemin:framemax] : Vector{Base.StackTraces.StackFrame}()
end

function nrelevantframes(stack::Vector{Base.StackTraces.StackFrame})
    nframes = length(stack)
    iszero(nframes) && return 0
    framemax = 0
    for i=1:nframes
        if :eval === stack[i].func
            break
        else
            framemax = framemax + 1
        end
    end
    framemax = max(1, framemax)
    return framemax
end

# overload basic arithmetic to catch NaN generation
# IMPORTANT!! define these after the code supporting `nanfound`

function Base.:(+)(x::Float64, y::Float64)
    z = Core.Intrinsics.add_float(x, y)
    isnan(z) ? nanfound((:+, x, y)) : z
end

function Base.:(-)(x::Float64, y::Float64)
    z = Core.Intrinsics.sub_float(x, y)
    isnan(z) ? nanfound((:-, x, y)) : z
end

function Base.:(*)(x::Float64, y::Float64)
    z = Core.Intrinsics.mul_float(x, y)
    isnan(z) ? nanfound((:*, x, y)) : z
end

function Base.:(/)(x::Float64, y::Float64)
    z = Core.Intrinsics.div_float(x, y)
    isnan(z) ? nanfound((:/, x, y)) : z
end

function Base.:(%)(x::Float64, y::Float64)
    z = Core.Intrinsics.rem_float(x, y)
    isnan(z) ? nanfound((:%, x, y)) : z
end
