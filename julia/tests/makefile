
PROJECT_DIR = ../..
SRC_DIR = $(PROJECT_DIR)/julia
TEST_DIR = $(SRC_DIR)/tests
PERF_REF_DIR = $(TEST_DIR)/performance_data
DATA_REF_DIR = $(TEST_DIR)/reference_data

JULIA ?= julia
JULIA_ARGS ?= -O3 --check-bounds=no --project=$(PROJECT_DIR) -t auto

MPI_EXEC ?= mpiexecjl
MPI_EXEC_ARGS ?= --project=$(PROJECT_DIR) -n 25
MPI_JL_ARGS ?= --color=yes

main_test_file = $(TEST_DIR)/main.jl
create_perf_ref_file = $(PERF_REF_DIR)/create_performance_reference.jl
create_data_ref_file = $(DATA_REF_DIR)/create_references.jl


.PHONY: create_perf_reference create_solver_reference short perf mpi all

create_perf_reference:
	cd $(PERF_REF_DIR) && $(JULIA) $(JULIA_ARGS) $(create_perf_ref_file)

create_solver_reference:
	cd $(DATA_REF_DIR) && $(JULIA) $(JULIA_ARGS) $(create_data_ref_file)

short:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) short

perf:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) performance

mpi:
	$(MPI_EXEC) $(MPI_EXEC_ARGS) $(JULIA) $(JULIA_ARGS) $(MPI_JL_ARGS) $(main_test_file) mpi

all:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) all
