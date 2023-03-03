
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
