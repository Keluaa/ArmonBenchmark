set terminal gif size 600, 450 animate delay 25
set output 'output.gif'

set xlabel "x"
set ylabel "y"
set zlabel "ρ"

set zrange [0:1]
set cbrange [0:1]
set xyplane 0

unset key

set view 32, 45

set xtics offset -0.5, -0.5, 0
set ytics offset  0.5, -0.5, 0

output_files = system("ls -1 anim/output_???")

do for [file in output_files] {
  splot file with pm3d
}
