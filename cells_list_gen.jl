
struct Specs
    dim::Int
    mem::Int
    vars::Int
    val_bytes::Int
end


log_space(min_log, step_log, max_log) = 10 .^ (min_log:step_log:max_log)
to_domain_dims(cells, s::Specs) = cells .^ (1 / s.dim) .|> round .|> Int
closest_multiple(x, m) = round(x / m) * m
filter_max_mem!(cells, s::Specs) = filter!((c) -> (c^s.dim * s.val_bytes * s.vars < s.mem), cells)
to_domain_string(cells, s::Specs) = join((join(repeat(["$i"], s.dim), ',') for i in cells), s.dim > 1 ? "; " : ", ")


function spread_cells(max_cells; 
        min_cells=10^4.5, step_log=log10(2), 
        dim=2, vars=19, type=Float64,
        device=:CPU, max_mem=nothing,
        proc_per_device=1,
        devices=1)

    if isnothing(max_mem)
        if device == :CPU
            max_mem = 250e9
        elseif device == :CUDA
            max_mem = 42e9
        elseif device == :ROCM
            max_mem = 32e9
        else
            error("Unknown device: " * device)
        end
        max_mem = max_mem / proc_per_device * devices * 0.95
    else
        device = "?"
    end

    processes = proc_per_device * devices

    specs = Specs(dim, max_mem, vars, sizeof(type))

    cells = log_space(log10(min_cells), step_log, log10(max_cells))
    cells = to_domain_dims(cells, specs)
    cells = closest_multiple.(cells, processes)
    filter_max_mem!(cells, specs)

    println("From $(first(min_cells)) to $(max_cells) in $(dim)D on $(max_mem / 1e9) GB spread over $(processes) processes on $(devices) $(device):")

    to_domain_string(cells, specs)
end
