
using LinuxPerf
using Printf
using PrettyTables


function default_perf_options()
    return "(cache-misses,cache-references,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads,LLC-store-misses,LLC-stores)," *
           "(dTLB-loads,dTLB-load-misses)," *
           "(cpu-cycles,instructions,branch-instructions)"
end


function get_events_list(pstats_options::String)::Vector{Vector{LinuxPerf.EventTypeExt}}
    opts = LinuxPerf.parse_pstats_options(pstats_options)
    return opts.events
end


function get_event_name(event::LinuxPerf.EventTypeExt)
    return LinuxPerf.EVENT_TO_NAME[event]
end


function init_bench(pstats_options::String)
    event_groups = LinuxPerf.parse_groups(pstats_options)
    groups = LinuxPerf.set_default_spaces(event_groups, (true, false, false))
    bench = LinuxPerf.make_bench_threaded(groups, threads=true)
    return bench
end


function close_bench(bench::LinuxPerf.PerfBenchThreaded)
    stats = LinuxPerf.Stats(bench)
    close(bench)
    return stats
end


function enable_bench(bench::LinuxPerf.PerfBenchThreaded)
    enable!(bench)
end


function disable_bench(bench::LinuxPerf.PerfBenchThreaded)
    disable!(bench)
end


function sort_stats!(stats::LinuxPerf.Stats)
    return sort!(stats.threads; by=(t)->t.pid)
end


function add_counters(counter_1::LinuxPerf.Counter, counter_2::LinuxPerf.Counter)
    return LinuxPerf.Counter(
        counter_1.event,
        counter_1.value   + counter_2.value,
        counter_1.enabled + counter_2.enabled,
        counter_1.running + counter_2.running
    )
end


function reduce_threads_stats!(stats::LinuxPerf.Stats)
    counters = Dict{LinuxPerf.EventType, LinuxPerf.Counter}()
    for thread_stats in stats.threads, event_group in thread_stats.groups, event in event_group
        if haskey(counters, event.event)
            counters[event.event] = add_counters(counters[event.event], event)
        else
            counters[event.event] = event
        end
    end
    return counters
end


function sum_stats!(stats_1::Dict{LinuxPerf.EventType, LinuxPerf.Counter}, stats_2::Dict{LinuxPerf.EventType, LinuxPerf.Counter})
    foreach(stats_2) do (event, counter)
        if haskey(stats_1, event)
            stats_1[event] = add_counters(stats_1[event], counter)
        else
            stats_1[event] = counter
        end
    end
end


function sum_stats!(stats_1::LinuxPerf.Stats, stats_2::LinuxPerf.Stats)
    sort_stats!(stats_1)
    sort_stats!(stats_2)
    for (thread_stats_1, thread_stats_2) in zip(stats_1.threads, stats_2.threads)
        thread_stats_1.pid ≠ thread_stats_2.pid && error("Incoherent threads order")
        for (event_group_1, event_group_2) in zip(thread_stats_1.groups, thread_stats_2.groups)
            for (i, (event_1, event_2)) in enumerate(zip(event_group_1, event_group_2))
                event_1.event ≠ event_2.event && error("Incoherent events order")
                event_group_1[i] = LinuxPerf.Counter(
                    event_1.event, 
                    event_1.value + event_2.value, 
                    event_1.enabled, 
                    event_1.running
                )
            end
        end
    end
    return stats_1
end


function print_hardware_counters_table(io::IO, 
        counters::Vector{Pair{String, Dict{String, Dict{LinuxPerf.EventType, LinuxPerf.Counter}}}};
        raw_print=false)
    count_width = 12

    # Get a sorted list of events to display
    events = first(counters) |> last |> first |> last |> keys |> unique
    sort!(events; lt=(a, b)->(a.category < b.category || a.event < b.event))
    events_list = map((event) -> (LinuxPerf.EVENT_TO_NAME[event]), events)

    # Print header
    max_event_str_len = maximum(length.(events_list))
    header_lines = [[event_str[i:min(i+count_width-1, length(event_str))] for i in 1:count_width:max_event_str_len] 
                    for event_str in events_list]
    header_lines = hcat(repeat([""], length(first(header_lines))), header_lines...)
    header_lines = Tuple(header_lines[i, :] for i in axes(header_lines, 1))
    header_lines[end][1] = "Step name"

    # Build the data table for each axis
    first_axis = true
    for (axis, axis_steps) in counters
        header_lines[1][1] = String(axis)
        
        data = Vector{Vector{Union{String, UInt64}}}()
        for (step_name, step_counters) in sort(collect(axis_steps); lt=(a, b)->(a[1] < b[1]))
            row = Vector{Union{String, UInt64}}()
            push!(row, step_name)

            for event in events
                push!(row, step_counters[event].value)
            end

            push!(data, row)
        end
        data = permutedims(hcat(data...))

        if raw_print
            if first_axis
                # Print header line
                println(io, "steps, " * join(events_list, ", "))
            end

            for i in axes(data, 1)
                print(io, axis * "-" * data[i, 1] * ", ")
                println(io, join(data[i, 2:end], ", "))
            end
        else
            pretty_table(io, data; header = header_lines, formatters = ft_printf("%6.2g"))
        end

        first_axis = false
    end
end
