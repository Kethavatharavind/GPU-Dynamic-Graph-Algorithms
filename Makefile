NVCC     = nvcc
ARCH     = sm_80
NVFLAGS  = -O2 -std=c++17 -arch=$(ARCH) -allow-unsupported-compiler

all: group_project

group_project: src/main.cu src/triangle.cu src/components.cu src/cpu_algo.h
	$(NVCC) $(NVFLAGS) src/main.cu -o group_project

clean:
	rm -f group_project
