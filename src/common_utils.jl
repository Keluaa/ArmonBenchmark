
using Printf


function get_duration_string(duration_sec::Float64)
    hours = floor(duration_sec / 3600)
    duration_sec -= hours * 3600

    minutes = floor(duration_sec / 60)
    duration_sec -= minutes * 60

    seconds = floor(duration_sec)
    duration_sec -= seconds

    ms = floor(duration_sec * 1000)

    str = ""
    print_next = false
    for (duration, unit, force_print) in ((hours, 'h', false), (minutes, 'm', false), (seconds, 's', true))
        if print_next
            str *= @sprintf("%02d", duration) * unit
        elseif duration > 0 || force_print
            str *= @sprintf("%2d", duration) * unit
            print_next = true
        else
            str *= "   "
        end
    end
    
    if hours == 0 && minutes == 0
        str *= @sprintf("%03dms", ms)
    end

    return str
end


function duration_from_string(str)
    str = lowercase(str)
    if isnothing(match(r"^([0-9.]+[a-z]+)+$", str))
        error("'$str' doesn't match /^([0-9.]+[a-z]+)+\$/. All durations must have an unit.")
    end

    total_duration = 0
    for m in eachmatch(r"(?<duration>[0-9.]+)(?<unit>[a-z]+)", str)
        duration = parse(Float64, m["duration"])
        if m["unit"] == "d"
            duration *= 24 * 3600
        elseif m["unit"] == "h"
            duration *= 3600
        elseif m["unit"] in ("m", "min")
            duration *= 60
        elseif !(m["unit"] in ("s", "sec"))
            error("Unknown duration unit: $(m["unit"])")
        end
        total_duration += duration
    end

    return total_duration
end
