# Build with:  make deps   (once)  then  make
# ARCH: T4 (Colab) = sm_75. Check your lab GPU with nvidia-smi and change if needed (e.g. sm_86).
NVCC  ?= nvcc
ARCH  ?= sm_75
FLAGS  = -O2 -std=c++17 -arch=$(ARCH) -Iinclude
SRCS   = src/main.cu src/gpu_filters.cu src/cpu_filters.cpp src/stb_impl.cpp
TARGET = bin/pdc_image

all: $(TARGET)

$(TARGET): $(SRCS) include/common.h include/cuda_utils.h
	mkdir -p bin
	$(NVCC) $(FLAGS) $(SRCS) -o $(TARGET)

test: src/test_small.cu src/gpu_filters.cu src/cpu_filters.cpp
	mkdir -p bin
	$(NVCC) $(FLAGS) src/test_small.cu src/gpu_filters.cu src/cpu_filters.cpp -o bin/test_small

deps:
	wget -nc -P include https://raw.githubusercontent.com/nothings/stb/master/stb_image.h
	wget -nc -P include https://raw.githubusercontent.com/nothings/stb/master/stb_image_write.h

clean:
	rm -rf bin

.PHONY: all test deps clean
