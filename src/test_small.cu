// test_small.cu -- 5x5 hand-checkable correctness test (Member B). Build: make test
// Input pixel i = i*2 (0..48). By hand, the CENTER pixel (2,2) must give:
//   blur  = 24   (average of neighbours 12,14,16,22,24,26,32,34,36)
//   sobel = 82   (gx = 16, gy = 80, sqrt(16^2 + 80^2) = 81.58 -> 82)
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include "common.h"

static void print_img(const char* name, const unsigned char* a, int w, int h) {
    printf("%s:\n", name);
    for (int y = 0; y < h; y++) { for (int x = 0; x < w; x++) printf("%4d", a[y * w + x]); printf("\n"); }
}
static int diff(const unsigned char* a, const unsigned char* b, int n) {
    int m = 0; for (int i = 0; i < n; i++) m = std::max(m, std::abs((int)a[i] - (int)b[i])); return m;
}

int main() {
    const int w = 5, h = 5, n = w * h;
    unsigned char in[n], bc[n], bg[n], bs[n], sc[n], sg[n];
    for (int i = 0; i < n; i++) in[i] = (unsigned char)(i * 2);

    blur_cpu(in, bc, w, h);       blur_gpu_global(in, bg, w, h);
    blur_gpu_shared(in, bs, w, h);
    sobel_cpu(in, sc, w, h);      sobel_gpu(in, sg, w, h);

    print_img("input", in, w, h);
    print_img("blur CPU", bc, w, h);
    print_img("sobel CPU", sc, w, h);
    printf("\ncenter blur  CPU=%d GPU-global=%d GPU-shared=%d (expected 24)\n", bc[12], bg[12], bs[12]);
    printf("center sobel CPU=%d GPU=%d (expected 82)\n", sc[12], sg[12]);
    printf("max diff blur global: %d | blur shared: %d | sobel: %d\n",
           diff(bc, bg, n), diff(bc, bs, n), diff(sc, sg, n));
    return 0;
}
