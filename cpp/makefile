
CXXFLAGS = -O3 -fopenmp -Wall -std=c++17 -ffast-math
CFLAGS = -O3 -fopenmp -Wall -ffast-math

ifdef use_clang
	C = clang
	CXX = clang++
	CXXFLAGS += -march=native -mtune=native
	CFLAGS += -march=native -mtune=native
else ifdef use_icc
	C = icc
	CXX = icpc
else
	C = gcc
	CXX = g++
	CXXFLAGS += -march=native -mtune=native
	CFLAGS += -march=native -mtune=native
endif

use_simd ?= 1
use_threading ?= 1
use_single_precision ?= 0
use_std_vector ?= 0
debug ?= 0

ifeq ($(use_simd), 1)
	CFLAGS += -mavx2 -mavx
	CXXFLAGS += -mavx2 -mavx
else
	CFLAGS += -mno-avx2 -mno-avx -mno-sse4.2 -mno-sse4.1 -mno-sse3
	CXXFLAGS += -mno-avx2 -mno-avx -mno-sse4.2 -mno-sse4.1 -mno-sse3
endif

ifeq ($(debug), 1)
	CFLAGS := -g $(CFLAGS)
	CXXFLAGS := -g $(CXXFLAGS)
endif


.PHONY: clean

armon.exe: armon.cpp
	$(CXX) $(CXXFLAGS) -DUSE_SIMD=$(use_simd) -DUSE_THREADING=$(use_threading) -DUSE_SINGLE_PRECISION=$(use_single_precision) armon.cpp -o armon.exe

all: armon.exe

clean:
	rm -f ./armon.exe
