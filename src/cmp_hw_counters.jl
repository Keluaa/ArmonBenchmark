
using Printf
using PrettyTables
using CSV
using DataFrames
import Base: show


struct DiffValue
    value::Int64
    diff::Float64
end


function Base.show(io::IO, diff_value::DiffValue)
    diff = diff_value.diff
    exp = Int(round(log10(diff)))
    man = Int(round(diff / exp10(exp)))
    @printf(io, "%6.2g | %de%+d", diff_value.value, man, exp)
end


function cell_color(h, d, i, j)
    diff = d[i, j].diff
    if diff < 1e-1
        return Crayon(foreground = :light_green)
    elseif diff < 1
        return Crayon(foreground = :green)
    elseif diff < 1e1
        return Crayon()
    elseif diff < 1e2
        return Crayon(foreground = :red)
    else
        return Crayon(foreground = :red)
    end
end


function cmp_data(data::DataFrame, ref_data::DataFrame)
    output = Vector{Vector{Union{String, DiffValue}}}()
    push!(output, String.(data[:, "steps"]))
    
    col_names = names(ref_data)
    for counter_name in col_names[2:end]
        col_diffs = map(zip(data[:, counter_name], ref_data[:, counter_name])) do (val, ref_val)
            DiffValue(val, val / ref_val)
        end
        push!(output, col_diffs)
    end
    output = hcat(output...)                
    
    pretty_table(output;
                 alignment = [:l, repeat([:r], length(col_names) - 1)...],
                 header = col_names,
                 highlighters = (Highlighter((d, i, j) -> (j > 1), cell_color),))
end


function cmp_all_files(ref_data_file_name::String, cmp_data_file_names::Vector{String})
    ref_data = CSV.read(ref_data_file_name, DataFrame; stripwhitespace=true)
    for cmp_data_file_name in cmp_data_file_names
        data = CSV.read(cmp_data_file_name, DataFrame; stripwhitespace=true)
        cmp_data(data, ref_data)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    cmp_all_files(ARGS[1], ARGS[2:end])
end
