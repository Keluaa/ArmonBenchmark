
# Small code used to get the directory of the gnuplot script, and cd to its parent.
# This allows to run this script from anywhere, and still be able to load all data files since their
# path is relative to the measurement directory.
# This works even when the script is loaded via the 'load' command, since ARG0 is always set to the
# path of the script being loaded by gnuplot.
const GP_CD_TO_MEASUREMENT_DIR = """
#!/usr/bin/gnuplot
p = strlen(ARG0); while (p > 0 && ARG0[p:p] ne "/") { p = p - 1 }; script_dir = ARG0[:p]
if (strlen(script_dir) > 0) { cd script_dir; cd ".." }
###"""

# Change the output of all plots to latex+eps files if `LATEX_PLOTS=1` alongside changes to improve
# the readability of the output. If `LATEX_TEST=1` then the output is in pdf but with the same
# settings.
const IF_LATEX_OUTPUT = """
IF_LATEX = system("echo \$LATEX_PLOTS")
IF_LATEX = IF_LATEX eq "1"
LATEX_TEST = system("echo \$LATEX_TEST")
LATEX_TEST = LATEX_TEST eq "1"
wrap_str(s) = (IF_LATEX ? sprintf("\$%s\$", s) : s)
texttt(s) = (IF_LATEX ? sprintf("\\\\texttt{%s}", s) : s)
###"""

const LOG_SCALE_CMD = """
set logscale x
set format x wrap_str("10^{%L}")
"""

base_gnuplot_script_commands(graph_file_name, title, log_scale, legend_pos) = """
$GP_CD_TO_MEASUREMENT_DIR
$IF_LATEX_OUTPUT
if (IF_LATEX) {
    if (LATEX_TEST) {
        set terminal pdfcairo color size 10cm, 8cm
        set output '$graph_file_name.tex.pdf'
    } else {
        set terminal epslatex color size 10cm, 8cm
        set output '$graph_file_name.tex'
    }
    set ylabel 'Performance [cell-cycles/s \$\\times 10^9\$]' offset 1,0,0
    unset title
    set key width -6  # Remove the blank region under the legend overlapping the grid
    set key samplen 2
} else {
    set terminal pdfcairo color size 10in, 6in
    set output '$graph_file_name.pdf'
    set ylabel 'Performance [cell-cycles/s ×10^9]'
    set title "$title"
}
set xlabel 'Cells count'
set key $legend_pos top
set grid
set yrange [0:]
ln_w = (IF_LATEX ? 3 : 1)
$(log_scale ? LOG_SCALE_CMD : "")
set format y wrap_str("%.2f")
plot """

base_gnuplot_histogram_script_commands(graph_file_name, title) = """
$GP_CD_TO_MEASUREMENT_DIR
set terminal pdfcairo color size 10in, 6in
set output '$graph_file_name'
set ylabel 'Total loop time (%)'
set title "$title"
set key left top
set style fill solid 1.00 border 0
set xtics rotate by 90 right
plot """

base_gnuplot_MPI_time_script_commands(graph_file_name, title, log_scale, legend_pos) = """
$GP_CD_TO_MEASUREMENT_DIR
$IF_LATEX_OUTPUT
if (IF_LATEX) {
    if (LATEX_TEST) {
        set terminal pdfcairo color size 10cm, 8cm
        set output '$graph_file_name.tex.pdf'
    } else {
        set terminal epslatex color size 10cm, 8cm
        set output '$graph_file_name.tex'
    }
    set ylabel 'Communications Time [sec]' offset 1,0,0
    set y2label 'Communication Time / Total Time [\\%]' offset -1,0,0
    unset title
    set key width -6  # Remove the blank region under the legend overlapping the grid
    set key samplen 2
} else {
    set terminal pdfcairo color size 10in, 6in
    set output '$graph_file_name.pdf'
    set ylabel 'Communications Time [sec]'
    set y2label 'Communication Time / Total Time [%]'
    set title "$title"
}
set xlabel 'Cells count'
set key $legend_pos top
set grid
set ytics nomirror
set mytics
set yrange [0:]
set y2tics
set my2tics
set y2range [0:]
ln_w = (IF_LATEX ? 3 : 1)
$(log_scale ? LOG_SCALE_CMD : "")
set format y wrap_str("%.2f")
set format y2 wrap_str("%.0f")
plot """

base_gnuplot_energy_script_commands(graph_file_name, ref_commands, title, log_scale, legend_pos) = """
$GP_CD_TO_MEASUREMENT_DIR
$IF_LATEX_OUTPUT
if (IF_LATEX) {
    if (LATEX_TEST) {
        set terminal pdfcairo color size 10cm, 8cm
        set output '$graph_file_name.tex.pdf'
    } else {
        set terminal epslatex color size 10cm, 8cm
        set output '$graph_file_name.tex'
    }
    set ylabel 'Energy consumption [µJ/cell-cycle]' offset 1,0,0
    unset title
    set key width -6  # Remove the blank region under the legend overlapping the grid
    set key samplen 2
} else {
    set terminal pdfcairo color size 10in, 6in
    set output '$graph_file_name.pdf'
    set ylabel 'Energy efficiency [µJ/cell-cycle]'
    set title "$title"
}
set xlabel 'Number of cells'
set key $legend_pos top
set grid
set yrange [0:]
ln_w = (IF_LATEX ? 3 : 1)
$(log_scale ? LOG_SCALE_CMD : "")
$ref_commands
plot """


const NEW_LINE = "\\\n     "


function gp_perf_plot_cmd(file, legend, pt_idx;
        error_bars=false, mode=error_bars ? "yerrorlines" : "lp")
    "'$file' $NEW_LINE  w $mode pt $pt_idx lw ln_w t '$legend'"
end


function gp_hist_plot_cmd(file, legend, color_idx)
    "'$file' $NEW_LINE  u 2: xtic(1) w histogram lt $color_idx t '$legend'"
end


function gp_MPI_time_cmd(file, legend, color_idx, pt_idx)
    "'$file' $NEW_LINE  u 1:(\$2/1e9) axis x1y1 w lp lc $color_idx lw ln_w pt $pt_idx t '$legend'"
end


function gp_MPI_percent_cmd(file, legend, color_idx, pt_idx)
    "'$file' $NEW_LINE  u 1:(\$2/\$3*100) axis x1y2 w lp lc $color_idx lw ln_w pt $(pt_idx-1) dt 4 t '$legend'"
end


function gp_energy_plot_cmd(file, legend, color_idx, pt_idx, ref_idx, cycles;
        error_bars=false, mode=error_bars ? "yerrorlines" : "lp")
    "'$file' $NEW_LINE  u 1:((\$2-REF$(ref_idx)_mean)/(\$1*$cycles)/\$3*1e6) w $mode lc $color_idx lw ln_w pt $pt_idx t '$legend'"
end


function gp_energy_ref_cmd(energy_ref_file, ref_idx)
    "stats '$energy_ref_file' using 2 name 'REF$(ref_idx)' nooutput"
end


function create_plot_file(measure::MeasureParams, plot_commands)
    script_path = joinpath(measure.script_dir, measure.gnuplot_script)
    open(script_path, "w") do gnuplot_script
        print(gnuplot_script, base_gnuplot_script_commands(
            measure.plot_file, measure.plot_title, 
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", $NEW_LINE")
        println(gnuplot_script, plot_cmd)
    end
end


function create_histogram_plot_file(measure::MeasureParams, plot_commands)
    script_path = joinpath(measure.script_dir, measure.gnuplot_hist_script)
    open(script_path, "w") do gnuplot_script
        print(gnuplot_script, base_gnuplot_histogram_script_commands(
            measure.hist_plot_file, measure.plot_title))
        plot_cmd = join(plot_commands, ", $NEW_LINE")
        println(gnuplot_script, plot_cmd)
    end
end


function create_MPI_time_plot_file(measure::MeasureParams, plot_commands)
    script_path = joinpath(measure.script_dir, measure.gnuplot_MPI_script)
    open(script_path, "w") do gnuplot_script
        plot_title = measure.plot_title * ", MPI communications time"
        print(gnuplot_script, base_gnuplot_MPI_time_script_commands(
            measure.time_MPI_plot_file, plot_title,
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", $NEW_LINE")
        println(gnuplot_script, plot_cmd)
    end
end


function create_energy_plot_file(measure::MeasureParams, plot_commands, ref_commands)
    script_path = joinpath(measure.script_dir, measure.energy_script)
    open(script_path, "w") do gnuplot_script
        plot_title = measure.plot_title * ", Energy consumption"
        ref_commands = join(ref_commands, "\n")
        print(gnuplot_script, base_gnuplot_energy_script_commands(
            measure.energy_plot_file, ref_commands, plot_title,
            measure.log_scale, measure.device == CPU ? "right" : "left"))
        plot_cmd = join(plot_commands, ", $NEW_LINE")
        println(gnuplot_script, plot_cmd)
    end
end
