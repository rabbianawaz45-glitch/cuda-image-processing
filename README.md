# GPU-Accelerated Image Processing using CUDA
Course: Parallel and Distributed Computing | Topic 02 | Track: GPU (CUDA)

Group: Rabbia Nawaz (BITF24M506), Malaika Usman (BITF24M518), Laiba Hanif (BITF24M521), Sadia Nadeem (BITF24M523)

Grayscale, 3x3 box blur and Sobel edge detection as CUDA kernels, compared against a sequential C++ CPU baseline.

## Project structure
```
include/common.h        shared contract: function signatures + rules (READ FIRST)
include/cuda_utils.h    CUDA_CHECK error macro
src/cpu_filters.cpp     CPU baseline                          (Member B)
src/gpu_filters.cu      CUDA kernels + wrappers + timing      (A: grayscale/timing, C: blur, D: sobel)
src/main.cu             batch loop, benchmark, CSV output     (D: batch, A: timing)
src/test_small.cu       5x5 hand-checked correctness test     (B)
src/stb_impl.cpp        stb_image implementation
scripts/make_test_images.py   generates test images
scripts/plot_results.py       makes report plots              (B)
results/                CSVs + plots
report/                 final report
```

## Conventions (everyone must follow, see include/common.h)
- Border rule: clamp coordinates in CPU and GPU. Blur: (sum+4)/9. Sobel: sqrt(gx^2+gy^2) clamped to 255.
- Max allowed CPU/GPU pixel difference: 1 (rounding). Wrap every CUDA call in CUDA_CHECK.
- Edit only your own file. `git pull` before work, `git push` after.

## Run on Google Colab (Runtime -> Change runtime type -> T4 GPU)
```
!git clone https://github.com/OWNER/cuda-image-processing.git
%cd cuda-image-processing
!nvidia-smi
!make deps
!make test && ./bin/test_small        # 5x5 correctness test
!python scripts/make_test_images.py
!make
!./bin/pdc_image images/sizes results/sizes.csv --trials 5 --save images/output
!./bin/pdc_image images/batch results/batch_1.csv   --batch 1
!./bin/pdc_image images/batch results/batch_10.csv  --batch 10
!./bin/pdc_image images/batch results/batch_50.csv  --batch 50
!./bin/pdc_image images/batch results/batch_100.csv --batch 100
!python scripts/plot_results.py
```
On a lab machine: `make ARCH=sm_XX` 
