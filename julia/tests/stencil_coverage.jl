
cd(@__DIR__)

if !@isdefined(Armon)
    include("../Armon.jl")
end

using .Armon
using Test


include("../kernel_stencil.jl")


lin_idx(p, i, j) = p.index_start + j * p.idx_row + i * p.idx_col


kernel_calls = [
    (p) -> kernel_stencil(p, Armon.acoustic!; 
        type_args=Dict{Symbol, Any}(:V => AbstractArray)),
    (p) -> kernel_stencil(p, Armon.acoustic_GAD!; 
        type_args=Dict{Symbol, Any}(:T => Float64, :LimiterType => Armon.MinmodLimiter)),
    (p) -> kernel_stencil(p, Armon.update_perfect_gas_EOS!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.update_bizarrium_EOS!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.cell_update!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.euler_projection!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.first_order_euler_remap!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.second_order_euler_remap!;
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => p.row_length, :i_start => lin_idx(p, 0, 1)-p.row_length, :d => 1),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => p.row_length, :i_start => lin_idx(p, p.nx+1, 1)-p.row_length, :d => -1),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => 1, :i_start => lin_idx(p, 1, p.ny+1)-1, :d => -p.row_length),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.boundaryConditions!;
        args=Dict{Symbol, Any}(:stride => 1, :i_start => lin_idx(p, 1, 0)-1, :d => p.row_length),
        type_args=Dict{Symbol, Any}(:T => Float64)),
    (p) -> kernel_stencil(p, Armon.read_border_array!;
        args=Dict{Symbol, Any}(:side_length => 1),
        type_args=Dict{Symbol, Any}(:V => AbstractArray)),
    (p) -> kernel_stencil(p, Armon.write_border_array!;
        args=Dict{Symbol, Any}(:side_length => 1),
        type_args=Dict{Symbol, Any}(:V => AbstractArray)),
    (p) -> kernel_stencil(p, Armon.init_test;
        type_args=Dict{Symbol, Any}(:Test => typeof(p.test))),  
]

kernel_names = [
    "acoustic!",
    "acoustic_GAD!",
    "update_perfect_gas_EOS!",
    "update_bizarrium_EOS!",
    "cell_update!",
    "euler_projection!",
    "first_order_euler_remap!",
    "second_order_euler_remap!",
    "boundaryConditions!_left",
    "boundaryConditions!_right",
    "boundaryConditions!_top",
    "boundaryConditions!_bottom",
    "read_border_array!",
    "write_border_array!",
    "init_test",
]


function get_all_kernel_stencils()
    p = ArmonParameters(;
        :ieee_bits => sizeof(Float64)*8,
        :test => :Sod, :scheme => :GAD, :projection => :euler_2nd, :riemann_limiter => :minmod,
        :nghost => 5, :nx => 100, :ny => 100,
        :cfl => 0,
        :silent => 5, :write_output => false, :measure_time => false,
        :use_MPI => false,
        :single_comm_per_axis_pass => false, :async_comms => false
    )

    kernel_stencils = Dict{String, Dict{Symbol, Any}}()
    
    for (kernel_name, kernel_call) in zip(kernel_names, kernel_calls)
        arrays_stencils = kernel_call(p)
        if haskey(kernel_stencils, kernel_name)
            prev_arrays_stencils = kernel_stencils[kernel_name]
            for (array_label, array_stencil) in prev_arrays_stencils
                prev_arrays_stencils[array_label] = union(array_stencil, arrays_stencils[array_label])
            end
        else
            kernel_stencils[kernel_name] = arrays_stencils
        end
    end

    return kernel_stencils
end
