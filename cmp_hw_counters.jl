
using Printf
using PrettyTables
using CSV
using DataFrames
import Base: show


struct DiffValue
    value::Int
    diff::Int
end


function Base.show(io::IO, diff_value::DiffValue)
    print(io, @sprintf("%6.2g", diff_value.value))
    
    diff = diff_value.diff
    exp = Int(floor(log10(diff)))
    man = Int(floor(diff / exp10(exp)))
    diff_str = @sprintf(" * %de%+d", man, exp)

    if diff < 1e-1
        printstyled(io, diff_str; color=:green)
    elseif diff < 1
        printstyled(io, diff_str; color=:light_green)
    elseif diff < 1e1
        printstyled(io, diff_str; color=:light_red)
    else
        printstyled(io, diff_str; color=:red)
    end
end


function cmp_data(data::DataFrame, ref_data::DataFrame)
    output = Vector{Vector{Union{String, DiffValue}}}()
    push!(output, String.(data[2:end, "steps"]))
    
    col_names = names(ref_data)
    for counter_name in col_names[2:end]
        col_diffs = map(zip(cmp_data[:, counter_name], ref_data[:, counter_name])) do val, ref_val
            DiffValue(val, ref_val / val)
        end
        push!(output, col_diffs)
    end

    pretty_table(output; header = col_names)
end


function cmp_all_files(ref_data_file_name::String, cmp_data_file_names::Vector{String})
    ref_data = CSV.read(ref_data_file_name, DataFrame; stripwhitespace=true)
    for cmp_data_file_name in cmp_data_file_names
        cmp_data = CSV.read(cmp_data_file_name, DataFrame; stripwhitespace=true)
        cmp_data(cmp_data, ref_data)
    end
end


if abspath(PROGRAM_FILE) == @__FILE__
    cmp_all_files(ARGS[1], ARGS[2:end])
end
