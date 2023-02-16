
base_gnuplot_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Giga Cells/sec'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set yrange [0:]
$(log_scale ? "set logscale x" : "")
`echo "$graph_file_name" >> $plots_update_file`
plot """

base_gnuplot_histogram_script_commands(graph_file_name, title) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Total loop time (%)'
set title "$title"
set key left top
set style fill solid 1.00 border 0
set xtics rotate by 90 right
`echo "$graph_file_name" >> $plots_update_file`
plot """

base_gnuplot_MPI_time_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Communications Time [sec]'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set ytics nomirror
set mytics
set yrange [0:]
set y2tics
set my2tics
set y2range [0:]
set y2label 'Communication Time / Total Time [%]'
$(log_scale ? "set logscale x" : "")
`echo "$graph_file_name" >> $plots_update_file`
plot """

base_gnuplot_energy_script_commands(graph_file_name, title, log_scale, legend_pos) = """
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Energy Consumption [J]'
set xlabel 'Cells count'
set title "$title"
set key $legend_pos top
set yrange [0:]
$(log_scale ? "set logscale x" : "")
`echo "$graph_file_name" >> $plots_update_file`
plot """

gnuplot_plot_command(data_file, legend_title, pt_index; mode="lp") = "'$(data_file)' w $(mode) pt $(pt_index) title '$(legend_title)'"
gnuplot_plot_command_errorbars(data_file, legend_title, pt_index) = gnuplot_plot_command(data_file, legend_title, pt_index; mode="yerrorlines")
gnuplot_hist_plot_command(data_file, legend_title, color_index) = "'$(data_file)' using 2: xtic(1) with histogram lt $(color_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_1(data_file, legend_title, color_index, pt_index) = "'$(data_file)' using 1:2 axis x1y1 w lp lc $(color_index) pt $(pt_index) title '$(legend_title)'"
gnuplot_MPI_plot_command_2(data_file, legend_title, color_index, pt_index) = "'$(data_file)' using 1:(\$2/\$3*100) axis x1y2 w lp lc $(color_index) pt $(pt_index-1) dt 4 title '$(legend_title)'"
gnuplot_energy_plot_command(data_file, legend_title, color_index, pt_index; mode="lp") = "'$(data_file)' using 1:2:3 w $(mode) lc $(color_index) pt $(pt_index) t '$(legend_title)'"
gnuplot_energy_plot_command_errorbars(data_file, legend_title, color_index, pt_index) = gnuplot_energy_plot_command(data_file, legend_title, color_index, pt_index; mode="yerrorlines")
gnuplot_energy_vals_plot_command(data_file, legend_title, color_index, pt_index, rep) = "'$(data_file)' u 1:$(rep+3) w lp lc $(color_index) dt 3 pt $(pt_index) t '$(legend_title)'"


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


function create_energy_plot_file(measure::MeasureParams, plot_commands)
    open(measure.energy_script, "w") do gnuplot_script
        plot_title = measure.plot_title * ", Energy consumption"
        print(gnuplot_script, base_gnuplot_energy_script_commands(measure.energy_plot_file, plot_title,
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", \\\n     ")
        println(gnuplot_script, plot_cmd)
    end
end
