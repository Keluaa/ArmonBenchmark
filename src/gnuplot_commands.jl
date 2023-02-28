
base_gnuplot_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Giga Cells/sec'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set grid
set yrange [0:]
$(log_scale ? "set logscale x" : "")
plot """

base_gnuplot_histogram_script_commands(graph_file_name, title) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Total loop time (%)'
set title "$title"
set key left top
set style fill solid 1.00 border 0
set xtics rotate by 90 right
plot """

base_gnuplot_MPI_time_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Communications Time [sec]'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set grid
set ytics nomirror
set mytics
set yrange [0:]
set y2tics
set my2tics
set y2range [0:]
set y2label 'Communication Time / Total Time [%]'
$(log_scale ? "set logscale x" : "")
plot """

base_gnuplot_energy_script_commands(graph_file_name, energy_ref_file, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Energy [µJ/cell]'
set xlabel 'Number of cells'
set title "$title"
set key $legend_pos top
set grid
set yrange [0:]
$(log_scale ? "set logscale x" : "")
stats '$energy_ref_file' using 2 name 'REF'
plot """


function gp_perf_plot_cmd(file, legend, pt_idx;
        error_bars=false, mode=error_bars ? "yerrorlines" : "lp")
    "'$file' w $mode pt $pt_idx t '$legend'"
end


function gp_hist_plot_cmd(file, legend, color_idx)
    "'$file' u 2: xtic(1) w histogram lt $color_idx t '$legend'"
end


function gp_MPI_time_cmd(file, legend, color_idx, pt_idx)
    "'$file' u 1:2 axis x1y1 w lp lc $color_idx pt $pt_idx t '$legend'"
end


function gp_MPI_percent_cmd(file, legend, color_idx, pt_idx)
    "'$file' u 1:(\$2/\$3*100) axis x1y2 w lp lc $color_idx pt $(pt_idx-1) dt 4 t '$legend'"
end


function gp_energy_plot_cmd(file, legend, color_idx, pt_idx;
        error_bars=false, mode=error_bars ? "yerrorlines" : "lp")
    "'$file' u 1:(\$2/(\$1-REF_mean)*1e6) w $mode lc $color_idx pt $pt_idx t '$legend'"
end


function create_plot_file(measure::MeasureParams, plot_commands)
    open(measure.gnuplot_script, "w") do gnuplot_script
        print(gnuplot_script, base_gnuplot_script_commands(measure.plot_file, measure.plot_title, 
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end
end


function create_histogram_plot_file(measure::MeasureParams, plot_commands)
    open(measure.gnuplot_hist_script, "w") do gnuplot_script
        print(gnuplot_script, base_gnuplot_histogram_script_commands(measure.hist_plot_file, measure.plot_title))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end
end


function create_MPI_time_plot_file(measure::MeasureParams, plot_commands)
    open(measure.gnuplot_MPI_script, "w") do gnuplot_script
        plot_title = measure.plot_title * ", MPI communications time"
        print(gnuplot_script, base_gnuplot_MPI_time_script_commands(measure.time_MPI_plot_file, plot_title,
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end
end


function create_energy_plot_file(measure::MeasureParams, plot_commands, ref_file_name)
    open(measure.energy_script, "w") do gnuplot_script
        plot_title = measure.plot_title * ", Energy consumption"
        print(gnuplot_script, base_gnuplot_energy_script_commands(measure.energy_plot_file, ref_file_name, plot_title,
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end
end
