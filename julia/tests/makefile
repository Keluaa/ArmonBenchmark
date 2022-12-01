
PROJECT_DIR = ../..
SRC_DIR = $(PROJECT_DIR)/julia
TEST_DIR = $(SRC_DIR)/tests
PERF_REF_DIR = $(TEST_DIR)/performance_data
DATA_REF_DIR = $(TEST_DIR)/reference_data

JULIA ?= julia
JULIA_ARGS ?= -O3 --check-bounds=no --project=$(PROJECT_DIR)

main_test_file = $(TEST_DIR)/main.jl
create_perf_ref_file = $(PERF_REF_DIR)/create_performance_reference.jl
create_data_ref_file = $(DATA_REF_DIR)/create_references.jl


.PHONY: create_perf_reference create_solver_reference all

create_perf_reference:
	$(JULIA) $(JULIA_ARGS) $(create_perf_ref_file)

create_solver_reference:
	$(JULIA) $(JULIA_ARGS) $(create_data_ref_file)

short:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) short

perf:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) performance

all:
	$(JULIA) $(JULIA_ARGS) $(main_test_file) all
