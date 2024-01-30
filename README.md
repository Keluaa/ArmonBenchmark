# ArmonBenchmark.jl

Those scripts encapsulates the main benchmark for the [Armon.jl package](https://github.com/Keluaa/Armon.jl)
and [Armon-Kokkos](https://github.com/Keluaa/Armon-Kokkos).

The `src/batch_measure.jl` script uses a simple text file to create Slurm batch scripts for:
 - performance measurements
 - energy measurements
 - weak and strong scaling measurements
 
by changing the right values accross each job step.

Results are then saved in `.csv` files and plotted with Gnuplot with automatically generated scripts.


## Usage

Options are put in a text file, one per line. The text file is then given to `src/batch_measure.jl`,
which will then build the files and scripts.

Each "list of" is a set of possible values. Unless specified, values are separated with commas.
All combinaisons of the sets' values will be turned into individual job steps.

Lines starting with `#` will be ignored.


#### General options

 - `name`: name of the set of measurement (mandatory)
 - `backends`: list of backends (`julia` or `kokkos`, default: `julia`)
 - `device`: device type (`CPU`, `CUDA` or `ROCM`, default: `CPU`)
 - `make_sub_script`: create a Slurm batch script. Defaults to true.
 - `one_job_per_cell`: make one job step per cell count. Useful for energy measurements. Defaults to false.
 - `one_script_per_step`: split all job steps into separate batch scripts. Defaults to false.
 - `modules`: modules to load at the start of the job. Defaults to (`cuda, hwloc, mpi, cmake/3.22.2`).


#### Slurm options
 
 - `node`: Slurm partition (mandatory)
 - `processes`: list of total process count (across all nodes)
 - `node_count`: list of node count
 - `processes_per_node`: number of processes per node. If present, `node_count` is ignored and instead is deduced from `processes`
 - `distributions`: list of Slurm distributions, passed to the `--distribution` option (default: `block`)
 - `max_time`: maximum wall time of the job, in any format such as '30s', '1min12sec', '4h30m' or '8h'. Defaults to 1 hour.


#### `mpirun` options

 - `use_mpirun`: use the `mpirun` command to launch jobs, instead of Slurm. Appart of `processes`, all other Slurm options will be ignored if true.
 - `hostlist`: comma-separated values passed to the `--host` option of `mpirun`. If empty, the `--host` option is omitted and jobs will run locally.
 - `hosts_max_cores`: maximum number of cores in each node (mandatory)


#### MPI options

 - `use_MPI`: use MPI or not. Julia only. Defaults to `true`.
 - `async_comms`: list of bools, whether or not to enable asynchronous communications. Defaults to `false`.
 - `process_grids`: list of process grids, separated by `;`, one grid is expressed as `X_ranks,Y_ranks`, such that `X_ranks * Y_ranks == processes`. Defaults to `1,1`.
 - `process_grid_ratios`: list of process grids ratios, separated by `;`, one grid ratio is expressed as `X_ratio,Y_ratio`. A `1,1` ratio is an equal number of ranks in each direction, a `1,4` ratio is 4x more ranks along the Y direction, and a `2,5` ratio is `2/5` ranks along X with respect to Y. If given, `process_grids` is ignored.
 - `process_scaling`: If `true`, only the n-th domain in `domains` is done by the n-th process count in `processes`. Defaults to `false`. Useful for scaling analysis.


#### GPU, multithreading and vectorization options

 - `block_sizes`: list of block sizes for GPU devices. Defaults to 128.
 - `threads`: list of threads/cores to use. Defaults to 4.
 - `use_simd`: list of bools, whether or not to enable vectorisation. Defaults to `true`.
 - `jl_places`: list of places for thread binding, like OpenMP's `OMP_PLACES`, only for Julia. Defaults to `cores`.
 - `jl_proc_bind`: list of process binding modes, like OpenMP's `OMP_PROC_BIND`, only for Julia. Defaults to `close`.
 - `omp_places`: list of values for `OMP_PLACES`, only for Kokkos. Defaults to `cores`.
 - `omp_proc_bind`: list of values for `OMP_PROC_BIND`, only for Kokkos. Defaults to `close`.
 - `use_max_threads`: whether or not to always use the maximum number of cores available. If true, `threads` is ignored. Defaults to `false`.
 - `ieee_bits`: list of floating-point precision bits (either 32 or 64). Defaults to 64.


#### Solver options

 - `domains`: list of cell domains, separated by `;`, one domain is expressed as `X_cells,Y_cells` (mandatory)
 - `tests`: list of CFD tests to do (`Sod`, `Sod_y`, `Sod_circ`, `Sedov` or `Bizarrium`, defaults to `Sod`) (mandatory)
 - `splitting`: list of axis splitting method to use (`Sequential`, `Godunov`, `Strang`, `X_only`, `Y_only`, defaults to `Sequential`) (mandatory)
 - `armon`: list of command-line parameters to pass to the solver, separated by `;`
 - `legends`: list of legends to associate with each set of command-line parameters given via the `armon` option
 - `name_suffixes`: list of suffixes to data files and Gnuplot scripts for each set of command-line parameters given via the `armon` option
 - `repeats`: minimum number of repeats. Results are averaged over all repetitions. The total number of repetitions is added to results files. Defaults to 1.
 - `min_acquisition_time`: Until this time is reached, repeats the current measurement. Follows the same format as `max_time`. Defaults to `0`.
 - `cycles`: number of solver iterations to do. Defaults to 20.
 - `verbose`: whether or not to print additional solver info. Can hinder performance. Defaults to `false`.


#### Energy measurement options

 - `track_energy`: enable energy measurement. Requires `one_job_per_cell=true`. Defaults to `false`.
 - `energy_references`: number of energy measurements to do for energy overhead calculation. Defaults to `3`.


#### Kokkos options

 - `compilers`: list of C++ compilers (`gcc`, `clang`, `ICC`, `AOCC` or `ICX`, default: `GCC`)
 - `use_kokkos`: list of bools. If `true`, then Julia kernels are replaced by the same kernels of the C++ Kokkos solver, using [Kokkos.jl](https://github.com/Keluaa/Kokkos.jl). Defaults to `false`.
 - `use_md_iter`: list of integers. Which style of Kokkos kernels to use (`0`: single `parallel_for` with `RangePolicy`, `1`: `TeamPolicy` + `TeamThreadRange`, `2`: 2D `MDRangePolicy`, `3`: 2D `MDRangePolicy` + custom balencing). Defaults to `0`.
 - `cmake_options`: options to pass to CMake for the C++ Kokkos solver
 - `kokkos_backends`: list of Kokkos backends to use, separated by `;`. Defaults to `Serial,OpenMP`.
 - `kokkos_version`: Kokkos version to use. Defaults to `4.0.00`.


#### Plotting options

 - `title`: plot title
 - `log_scale`: whether or not to use a log scale in plots. Defaults to `true`.
 - `error_bars`: whether or not to plot error bars. Defaults to `false`.
 - `energy_plot`: whether or not to make energy plots. Defaults to `false`.
 - `perf_plot`: whether or not to make performance plots. Defaults to `true`.
 - `time_MPI_plot`: whether or not to make MPI communication time plots. Defaults to `false`.


## Utility scripts

`src/cells_list_gen.jl` is used to create lists of mesh sizes for measurements, depending on available
memory and the number of processes.

`src/slurm_topo_graph.jl` can be used to get and create the topology of the computing cluster using
Slurm, as well as create GraphViz visualisations.
