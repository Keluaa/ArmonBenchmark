set terminal gif size 1000, 800 animate delay 100
set output 'output.gif'

set xlabel "x"
set ylabel "y"
set zlabel "ρ"

set view 90, 360

output_files = system("ls -1 anim/output_???")

do for [file in output_files] {
  splot file
}
