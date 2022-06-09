set terminal gif size 1000, 800 animate delay 25
set output 'output.gif'

set xlabel "x"
set ylabel "y"
set zlabel "ρ"

set zrange [0:1]
set cbrange [0:1]

# set view 90, 360

output_files = system("ls -1 anim/output_???")

do for [file in output_files] {
  splot file with pm3d
}
