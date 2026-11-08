// common.h -- shared contract for the whole group. Everyone includes this.
// RULES (agreed by all members):
//   * rgb images  : unsigned char, 3 channels interleaved (R,G,B,R,G,B...), size w*h*3
//   * gray images : unsigned char, 1 channel, size w*h
//   * blur and sobel take a GRAY image as input and write a GRAY image
//   * Border rule : CLAMP coordinates (x -> max(0, min(w-1, x))) in CPU AND GPU
//   * Blur        : 3x3 box blur, out = (sum + 4) / 9   (integer math)
//   * Sobel       : mag = sqrt(gx^2 + gy^2), clamp to 255, out = (unsigned char)(mag + 0.5f)
//   * Grayscale   : gray = (unsigned char)(0.299f*R + 0.587f*G + 0.114f*B + 0.5f)
#pragma once

// Times (milliseconds) measured with CUDA events for one GPU call
struct GpuTimes {
    float h2d_ms;     // host -> device copy
    float kernel_ms;  // kernel execution
    float d2h_ms;     // device -> host copy
    float total() const { return h2d_ms + kernel_ms + d2h_ms; }
};

// ---------- CPU baseline (cpu_filters.cpp) ----------
void grayscale_cpu(const unsigned char* rgb, unsigned char* gray, int w, int h);
void blur_cpu     (const unsigned char* gray, unsigned char* out, int w, int h);
void sobel_cpu    (const unsigned char* gray, unsigned char* out, int w, int h);

// ---------- GPU versions (gpu_filters.cu) ----------
// Each function takes HOST pointers, does cudaMalloc + copy + kernel + copy back,
// and returns the timing of each phase.
GpuTimes grayscale_gpu   (const unsigned char* h_rgb,  unsigned char* h_gray, int w, int h);
GpuTimes blur_gpu_global (const unsigned char* h_gray, unsigned char* h_out,  int w, int h);
GpuTimes blur_gpu_shared (const unsigned char* h_gray, unsigned char* h_out,  int w, int h);
GpuTimes sobel_gpu       (const unsigned char* h_gray, unsigned char* h_out,  int w, int h);
