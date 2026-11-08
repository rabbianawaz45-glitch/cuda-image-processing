// cpu_filters.cpp -- sequential CPU baseline (SISD). Owner: Member B.
// Plain nested loops, one thread. Must follow the rules in common.h exactly.
#include "common.h"
#include <cmath>

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void grayscale_cpu(const unsigned char* rgb, unsigned char* gray, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int i = y * w + x;
            float g = 0.299f * rgb[3 * i] + 0.587f * rgb[3 * i + 1] + 0.114f * rgb[3 * i + 2];
            gray[i] = (unsigned char)(g + 0.5f);
        }
    }
}

void blur_cpu(const unsigned char* in, unsigned char* out, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int sum = 0;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++)
                    sum += in[clampi(y + dy, 0, h - 1) * w + clampi(x + dx, 0, w - 1)];
            out[y * w + x] = (unsigned char)((sum + 4) / 9);
        }
    }
}

void sobel_cpu(const unsigned char* in, unsigned char* out, int w, int h) {
    auto p = [&](int x, int y) { return (int)in[clampi(y, 0, h - 1) * w + clampi(x, 0, w - 1)]; };
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int gx = -p(x - 1, y - 1) + p(x + 1, y - 1)
                     - 2 * p(x - 1, y) + 2 * p(x + 1, y)
                     - p(x - 1, y + 1) + p(x + 1, y + 1);
            int gy = -p(x - 1, y - 1) - 2 * p(x, y - 1) - p(x + 1, y - 1)
                     + p(x - 1, y + 1) + 2 * p(x, y + 1) + p(x + 1, y + 1);
            float mag = sqrtf((float)(gx * gx + gy * gy));
            if (mag > 255.0f) mag = 255.0f;
            out[y * w + x] = (unsigned char)(mag + 0.5f);
        }
    }
}
