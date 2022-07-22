
using Printf
using Dates


const USAGE = """
USAGE:
    julia build_cpu_gpu_diff_graph.jl <output_dir> <cpu_output_suffix> <gpu_output_suffix> <out_data_file_prefix> <title>
"""


if length(ARGS) != 5
    println(USAGE)
    exit()
end


output_dir = ARGS[1]
cpu_output_suffix = ARGS[2]
gpu_output_suffix = ARGS[3]
output_data_file_prefix = ARGS[4]
title = ARGS[5]

endswith(output_dir, '/') && (output_dir = output_dir[1:end-1])

cpu_stdout_file_name = "$output_dir/stdout_$cpu_output_suffix.txt"
gpu_stdout_file_name = "$output_dir/stdout_$gpu_output_suffix.txt"
cpu_stderr_file_name = "$output_dir/stderr_$cpu_output_suffix.txt"
gpu_stderr_file_name = "$output_dir/stderr_$gpu_output_suffix.txt"

plot_output_dir = joinpath(@__DIR__, "out")
mkpath(plot_output_dir)

output_data_file_name = "$plot_output_dir/$output_data_file_prefix.txt"
output_plot_script_file_name = "$plot_output_dir/$output_data_file_prefix.plot"
output_plot_file_name = "$plot_output_dir/$output_data_file_prefix.pdf"


get_accounting_cmd(job_ID) = `ccc_macct $job_ID`
slurm_accounting_cmd(job_ID) = `sacct -j $job_ID --noheader -o jobid,Elapsed%,ConsumedEnergyRaw%`
gnuplot_cmd(plot_script_file) = `gnuplot -c $plot_script_file`

gnuplot_script(data_file) = """
set terminal pdfcairo color size 10in, 6in
set output '$output_plot_file_name'
set multiplot layout 2,2 rowsfirst title "2D Eulerian mono-fluid hydrocode: $title"

# Throughput plot
set xlabel 'Cells'
set ylabel 'GPUs vs. CPU-only [Cells/sec ×10^9]'
set ytics nomirror
set mytics
set yrange [0:]
set y2tics
set my2tics
set y2range [0:20]
set y2label 'Speedup GPU vs CPU'
set key left top
set logscale x
plot "$data_file" using 1:2 axis x1y1 w lp title "GPU", "$data_file" using 1:3 axis x1y1 w lp title "CPU", "$data_file" using 1:4 axis x1y2 w lp dt 4 title "ratio GPU/CPU"

set ytics mirror
unset y2tics
unset my2tics
unset y2label
unset y2range

# MPI time plot
set xlabel 'Cells'
set ylabel 'MPI Time [%]'
set yrange [0:]
set key right top
set logscale x
plot "$data_file" using 1:5 w lp title "GPU", "$data_file" using 1:6 w lp title "CPU"

# Energy plot
set xlabel 'Cells'
set ylabel 'Energy consumption factor GPUs vs. CPU-only'
set yrange [0:]
set logscale x
plot "$data_file" using 1:7 w lp notitle

# Energy efficiency plot
set xlabel 'Cells'
set ylabel 'Energy consumption [µJ/cell/cycle]'
set yrange [0.1:]
set logscale xy
plot "$data_file" using 1:8 w lp title "GPU", "$data_file" using 1:9 w lp title "CPU"

unset multiplot
"""

time_str_to_sec(time_str) = Dates.value(DateTime(time_str, dateformat"HH:MM:SS") - DateTime("00:00:00", dateformat"HH:MM:SS")) / 1000


struct JobInfo
    elapsed_time::Float64  # In seconds
    joules::Float64
    cells::Float64
    throughput::Float64    # In cells/sec
    MPI_time::Float64      # In fraction of the total time
    cycles::Int
    ref_elapsed_time::Float64
    ref_joules::Float64
end


function get_power_consumption_of_job_steps(job_ID)    
    raw_job_accounting = read(slurm_accounting_cmd(job_ID), String)
    job_accounting = split(raw_job_accounting)

    first_step_pos = findfirst(v -> contains(v, ".ext"), job_accounting) + 3
    steps_elapsed_time = Float64[]
    steps_joules = Float64[]
    for i in first_step_pos:3:length(job_accounting)-3
        step_elapsed_time = time_str_to_sec(job_accounting[i+1])
        step_joules = parse(Float64, job_accounting[i+2])
        if step_joules > 3600*1000_000
            @warn "The energy counter of step $(length(steps_elapsed_time)) overflowed. Its energy amount will be estimated using the energy amount of the previous step."
            isempty(steps_joules) && error("There is no previous step.")
            step_joules = step_elapsed_time / last(steps_elapsed_time) * last(steps_joules)
        end
        push!(steps_elapsed_time, step_elapsed_time)
        push!(steps_joules, step_joules)
    end

    ref_job_step_pos = length(job_accounting)-2
    ref_elapsed_time = time_str_to_sec(job_accounting[ref_job_step_pos+1])
    ref_joules = parse(Float64, job_accounting[ref_job_step_pos+2])

    return ref_elapsed_time, ref_joules, steps_elapsed_time, steps_joules
end


function get_cycles_count_of_job_step(job_stderr, line_pos)
    cycle_pos = findnext(r"--cycle \K[0-9]+", job_stderr, line_pos)
    isnothing(cycle_pos) && error("Could not get the position of the cycles count option at index $line_pos")

    repeats_pos = findnext(r"--repeats \K[0-9]+", job_stderr, line_pos)
    isnothing(repeats_pos) && error("Could not get the position of the repeats count option at index $line_pos")

    cycles = parse(Int, job_stderr[cycle_pos])
    repeats = parse(Int, job_stderr[repeats_pos])

    return cycles * repeats, max(last(cycle_pos), last(repeats_pos))
end


function extract_info_of_job_step(job_stdout, line_pos)
    # Get cells count
    cells_pos = findnext(r"[0-9\.e+-]+ cells", job_stdout, last(line_pos))
    isnothing(cells_pos) && error("Could not get the position of the cells count")
    
    raw_cells = job_stdout[first(cells_pos):last(cells_pos)-length(" cells")]
    cells_count = parse(Float64, raw_cells)

    # Get cells throughput
    throughput_start = findnext("): ", job_stdout, last(line_pos))
    isnothing(throughput_start) && error("Could not get the start of the throughput")
    throughput_end = findnext("Giga", job_stdout, last(throughput_start))
    isnothing(throughput_end) && error("Could not get the end of the throughput")

    raw_throughput = job_stdout[last(throughput_start)+1:first(throughput_end)-1]
    throughput = parse(Float64, raw_throughput)

    # Get MPI time %
    mpi_time_end = findnext("% of MPI time", job_stdout, last(line_pos))
    isnothing(mpi_time_end) && error("Could not get the MPI time position")
    mpi_time_start = findprev(',', job_stdout, first(mpi_time_end))
    isnothing(mpi_time_start) && error("Could not get the MPI time position start")

    raw_mpi_time = job_stdout[mpi_time_start+2:first(mpi_time_end)-1]
    mpi_time = parse(Float64, raw_mpi_time) / 100

    return cells_count, throughput, mpi_time, last(mpi_time_end)
end


function extract_infos_of_job(job_stdout_file_name, job_stderr_file_name)
    job_stdout = read(job_stdout_file_name, String)
    job_stderr = read(job_stderr_file_name, String)

    job_ID_pos = findfirst(r"JobId=\K[0-9]+", job_stderr)
    isnothing(job_ID_pos) && error("Could not find the job ID in file '$job_stderr_file_name'")
    job_ID = job_stderr[job_ID_pos]

    ref_elapsed_time, ref_joules, steps_elapsed_time, steps_joules = get_power_consumption_of_job_steps(job_ID)

    info = JobInfo[]
    line_end = 1
    stderr_line_end = 1
    for step_i in 1:length(steps_elapsed_time)
        line_pos = findnext(" - ", job_stdout, line_end)
        cells, throughput, MPI_time, line_end = extract_info_of_job_step(job_stdout, last(line_pos))
        cycles, stderr_line_end = get_cycles_count_of_job_step(job_stderr, stderr_line_end)
        push!(info, JobInfo(steps_elapsed_time[step_i], steps_joules[step_i], cells, throughput, MPI_time, cycles, ref_elapsed_time, ref_joules))
    end

    return info
end    


function build_graph(data_file_name, plot_script_file_name)
    cpu_infos = extract_infos_of_job(cpu_stdout_file_name, cpu_stderr_file_name)
    gpu_infos = extract_infos_of_job(gpu_stdout_file_name, gpu_stderr_file_name)

    open(data_file_name, "w") do data_file
        for (i, (cpu_info, gpu_info)) in enumerate(zip(cpu_infos, gpu_infos))
            if cpu_info.cells != gpu_info.cells
                error("Cells of job $i don't match")
            end

            # gpu_joules = gpu_info.joules - gpu_info.ref_joules
            # cpu_joules = cpu_info.joules - cpu_info.ref_joules
            gpu_joules = gpu_info.joules
            cpu_joules = cpu_info.joules

            @printf(data_file, "%f, %f, %f, %f, %f, %f, %f, %f, %f\n", 
                cpu_info.cells,
                gpu_info.throughput,
                cpu_info.throughput,
                gpu_info.throughput / cpu_info.throughput,
                gpu_info.MPI_time * 100,
                cpu_info.MPI_time * 100,
                gpu_joules / cpu_joules,
                gpu_joules / gpu_info.cells / gpu_info.cycles * 1_000_000,
                cpu_joules / cpu_info.cells / cpu_info.cycles * 1_000_000,
            )
        end
    end

    write(plot_script_file_name, gnuplot_script(data_file_name))
end


build_graph(output_data_file_name, output_plot_script_file_name)
run(gnuplot_cmd(output_plot_script_file_name))
println("Created plot $output_plot_file_name")
