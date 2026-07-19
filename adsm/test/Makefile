# ADSM port faithfulness — equivalence test
#
#   make          build + run
#   make build    build only
#   make clean
#
# Standalone: no ROS, no external deps. Just a C++17 compiler.

CXX      ?= g++
CXXFLAGS ?= -O2 -std=c++17 -Wall

all: run

build: adsm_equivalence_test

adsm_equivalence_test: adsm_equivalence_test.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lm

run: adsm_equivalence_test
	./adsm_equivalence_test

clean:
	rm -f adsm_equivalence_test

.PHONY: all build run clean
