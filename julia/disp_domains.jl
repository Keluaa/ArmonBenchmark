
using PrettyTables


function disp_array_domain_ranges(params::Armon.ArmonParameters, drs::Armon.DomainRanges, fluxes::Bool = false)
    inner    = fluxes ? Armon.inner_fluxes_domain(drs)    : Armon.inner_domain(drs)
    outer_lb = fluxes ? Armon.outer_fluxes_lb_domain(drs) : Armon.outer_lb_domain(drs)
    outer_rt = fluxes ? Armon.outer_fluxes_rt_domain(drs) : Armon.outer_rt_domain(drs)

    t = zeros(Int, params.nbcell)
    for i in 1:params.nbcell
        if i in inner
            t[i] += 1
        end
        if i in outer_lb
            t[i] += 2
        end
        if i in outer_rt
            t[i] += 4
        end
    end

    t = reshape(t, (params.row_length, params.col_length))'
    t = reverse(t, dims=1)

    return t
end


function disp_domain_ranges(params::Armon.ArmonParameters, disp_array::Matrix{Int64})
    (; nx, ny, nghost, row_length, col_length) = params

    real_domain(i, j) = 1 <= (i - nghost) <= nx && 1 <= (j - nghost) <= ny

    pretty_table(disp_array; 
        header     = (1:row_length) .- nghost, 
        row_labels = reverse((1:col_length) .- nghost),
        highlighters = (
            Highlighter((d,i,j) -> (real_domain(i, j) && d[i,j] == 1), Crayon(foreground=:black, background=:blue)),
            Highlighter((d,i,j) -> (real_domain(i, j) && d[i,j] == 2), Crayon(foreground=:black, background=:yellow)),
            Highlighter((d,i,j) -> (real_domain(i, j) && d[i,j] == 4), Crayon(foreground=:black, background=:green)),
            Highlighter((d,i,j) -> (d[i,j] == 1), Crayon(foreground=:red, background=:blue)),
            Highlighter((d,i,j) -> (d[i,j] == 2), Crayon(foreground=:red, background=:yellow)),
            Highlighter((d,i,j) -> (d[i,j] == 4), Crayon(foreground=:red, background=:green)),
            Highlighter((d,i,j) -> (d[i,j] != 0), Crayon(foreground=:black, background=:red))
        )
    )
end


function disp_domain_ranges(params::Armon.ArmonParameters, domain_ranges::Armon.DomainRanges, fluxes::Bool = false)
    disp_array = disp_array_domain_ranges(params, domain_ranges, fluxes)
    println("Domain separation for a $(params.current_axis) sweep" * (fluxes ? ", for fluxes:" : ":"))
    disp_domain_ranges(params, disp_array)
end


function disp_domain_ranges(params::Armon.ArmonParameters, fluxes::Bool = false)
    disp_domain_ranges(params, Armon.compute_domain_ranges(params), fluxes)
end
