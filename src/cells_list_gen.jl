
struct Specs
    dim::Int
    mem::Int
    vars::Int
    val_bytes::Int
end


log_space(min_log, step_log, max_log) = 10 .^ (min_log:step_log:max_log)
to_domain_dims(cells, s::Specs) = cells .^ (1 / s.dim) .|> round .|> Int
closest_multiple(x, m) = Int(round(x / m) * m)
filter_max_mem!(cells, s::Specs) = filter!((c) -> (c^s.dim * s.val_bytes * s.vars < s.mem), cells)
to_domain_string(cells, s::Specs) = join((join(repeat(["$i"], s.dim), ',') for i in cells), s.dim > 1 ? "; " : ", ")


"""
    spread_cells(max_cells; 
        min_cells=10^4.5, step_log=log10(2), 
        dim=2, vars=16, type=Float64,
        device=:CPU, max_mem_per_device=nothing,
        max_mem=nothing,
        proc_per_device=1,
        devices=1
    )

`device` sets `max_mem`, and can be `:CPU` (250 GiB), `:A100` (40 GiB), `:MI100` (32 GiB),
`:MI250` (64 GiB), or `nothing` (custom max mem).

Each of the `devices` is considered to have `max_mem` memory. It is evenly distributed among the
`proc_per_device` processes on each device.

There will be `devices × proc_per_device` processes in total.
"""
function spread_cells(max_cells; 
    min_cells=10^4.5, step_log=log10(2), 
    dim=2, vars=16, type=Float64,
    device=:CPU, max_mem=nothing, max_mem_per_device=nothing,
    proc_per_device=1,
    devices=1
)
    if isnothing(max_mem)
        if !isnothing(max_mem_per_device)
            max_mem = max_mem_per_device
        elseif device == :CPU
            max_mem = 250 * 1024^3
        elseif device == :A100
            max_mem = 40 * 1024^3
        elseif device == :MI100
            max_mem = 32 * 1024^3
        elseif device == :MI250
            max_mem = 64 * 1024^3
        else
            error("Unknown device: " * device)
        end
        max_mem = max_mem / proc_per_device * devices * 0.95
        max_mem = floor(Int, max_mem)
    else
        device = "devices"
    end

    processes = proc_per_device * devices

    specs = Specs(dim, max_mem, vars, sizeof(type))

    cells = log_space(log10(min_cells), step_log, log10(max_cells))
    cells = to_domain_dims(cells, specs)
    cells = closest_multiple.(cells, processes)
    filter_max_mem!(cells, specs)

    max_used_mem = last(cells)^specs.dim * specs.vars * specs.val_bytes
    max_used_mem_proc = max_used_mem / processes

    max_used_mem = round(max_used_mem / 1e9; digits=1)
    max_used_mem_proc = round(max_used_mem_proc / 1e9; digits=1)
    max_mem = round(max_mem / 1e9; digits=1)

    println("Range from $(round(min_cells)) to $(round(max_cells)) in $(dim)D ($vars $type variables)")
    println("On $processes processes on $devices $device")
    println("Going up to $(join(repeat(["$(last(cells))"], dim), '×')) ($(round(last(cells)^dim/1e9; digits=1))×10⁹ cells)")
    println("Using up to $max_used_mem GB of the maximum $max_mem GB, or $max_used_mem_proc GB per process")
    println(to_domain_string(cells, specs))
    return cells, specs
end
