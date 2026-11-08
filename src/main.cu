// main.cu -- batch pipeline + benchmark. Owners: Member D (batch loop), Member A (timing/CSV).
//
// Usage:
//   ./bin/pdc_image <input_dir> <results.csv> [--batch N] [--trials T] [--save <output_dir>]
//
// For every image: grayscale -> blur (global + shared) -> sobel, on CPU and GPU.
// blur and sobel are both fed the same grayscale image so CPU/GPU are compared fairly.
#include <chrono>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

#include "common.h"
#include "cuda_utils.h"
#include "stb_image.h"
#include "stb_image_write.h"

namespace fs = std::filesystem;
using Clock = std::chrono::high_resolution_clock;

static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

static int max_diff(const unsigned char* a, const unsigned char* b, size_t n) {
    int m = 0;
    for (size_t i = 0; i < n; i++) m = std::max(m, std::abs((int)a[i] - (int)b[i]));
    return m;
}

static std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("Usage: %s <input_dir> <results.csv> [--batch N] [--trials T] [--save out_dir]\n", argv[0]);
        return 1;
    }
    std::string in_dir = argv[1], csv_path = argv[2], save_dir;
    int batch = -1, trials = 5;
    for (int i = 3; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--batch" && i + 1 < argc)       batch = atoi(argv[++i]);
        else if (a == "--trials" && i + 1 < argc) trials = atoi(argv[++i]);
        else if (a == "--save" && i + 1 < argc)   save_dir = argv[++i];
    }

    // ---- collect image files (sorted, optionally only the first N = batch size)
    std::vector<fs::path> files;
    for (auto& e : fs::directory_iterator(in_dir)) {
        std::string ext = lower(e.path().extension().string());
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    if (batch > 0 && (size_t)batch < files.size()) files.resize(batch);
    if (files.empty()) { printf("No images found in %s\n", in_dir.c_str()); return 1; }

    // ---- GPU init (cudaFree(0) forces context creation so it is not counted in timings)
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    CUDA_CHECK(cudaFree(0));
    printf("GPU: %s | images: %zu | trials: %d\n\n", prop.name, files.size(), trials);

    if (!fs::path(csv_path).parent_path().empty()) fs::create_directories(fs::path(csv_path).parent_path());
    if (!save_dir.empty()) fs::create_directories(save_dir);
    FILE* csv = fopen(csv_path.c_str(), "w");
    if (!csv) { printf("Cannot open %s\n", csv_path.c_str()); return 1; }
    fprintf(csv, "image,width,height,filter,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms,speedup_kernel,speedup_total,max_diff\n");

    std::set<std::string> warmed;                         // warm-up each kernel once
    double tot_cpu = 0, tot_kernel = 0, tot_gpu = 0;      // whole-batch totals (pipeline = gray+blur_global+sobel)

    for (auto& path : files) {
        int w, h, c;
        unsigned char* rgb = stbi_load(path.string().c_str(), &w, &h, &c, 3);   // force 3 channels
        if (!rgb) { printf("Could not load %s\n", path.string().c_str()); continue; }
        size_t n = (size_t)w * h;
        std::string name = path.filename().string();

        std::vector<unsigned char> gray_cpu(n), gray_gpu(n), blur_c(n), blur_g(n), blur_s(n), sobel_c(n), sobel_g(n);

        auto bench = [&](const std::string& filter,
                         const std::function<void()>& cpu_fn,
                         const std::function<GpuTimes()>& gpu_fn,
                         const unsigned char* cpu_out, const unsigned char* gpu_out) {
            if (!warmed.count(filter)) { gpu_fn(); warmed.insert(filter); }       // warm-up (not timed)
            double cpu_sum = 0; GpuTimes g = {0, 0, 0};
            for (int t = 0; t < trials; t++) {
                auto t0 = Clock::now();
                cpu_fn();
                cpu_sum += ms_since(t0);
                GpuTimes r = gpu_fn();
                g.h2d_ms += r.h2d_ms; g.kernel_ms += r.kernel_ms; g.d2h_ms += r.d2h_ms;
            }
            double cpu = cpu_sum / trials;
            g.h2d_ms /= trials; g.kernel_ms /= trials; g.d2h_ms /= trials;
            double gtot = g.total();
            int diff = max_diff(cpu_out, gpu_out, n);
            printf("%-18s %5dx%-5d %-12s cpu %9.3f | gpu kernel %8.3f total %8.3f | speedup %7.2fx (kernel) %6.2fx (total) | diff %d%s\n",
                   name.c_str(), w, h, filter.c_str(), cpu, g.kernel_ms, gtot,
                   cpu / g.kernel_ms, cpu / gtot, diff, diff > 1 ? "  <-- CHECK!" : "");
            fprintf(csv, "%s,%d,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%d\n",
                    name.c_str(), w, h, filter.c_str(), cpu, g.h2d_ms, g.kernel_ms, g.d2h_ms, gtot,
                    cpu / g.kernel_ms, cpu / gtot, diff);
            if (filter != "blur_shared") { tot_cpu += cpu; tot_kernel += g.kernel_ms; tot_gpu += gtot; }
        };

        bench("grayscale",
              [&] { grayscale_cpu(rgb, gray_cpu.data(), w, h); },
              [&] { return grayscale_gpu(rgb, gray_gpu.data(), w, h); },
              gray_cpu.data(), gray_gpu.data());

        // blur & sobel use the CPU gray image as input on both sides
        bench("blur_global",
              [&] { blur_cpu(gray_cpu.data(), blur_c.data(), w, h); },
              [&] { return blur_gpu_global(gray_cpu.data(), blur_g.data(), w, h); },
              blur_c.data(), blur_g.data());

        bench("blur_shared",
              [&] { blur_cpu(gray_cpu.data(), blur_c.data(), w, h); },
              [&] { return blur_gpu_shared(gray_cpu.data(), blur_s.data(), w, h); },
              blur_c.data(), blur_s.data());

        bench("sobel",
              [&] { sobel_cpu(gray_cpu.data(), sobel_c.data(), w, h); },
              [&] { return sobel_gpu(gray_cpu.data(), sobel_g.data(), w, h); },
              sobel_c.data(), sobel_g.data());

        if (!save_dir.empty()) {                          // save GPU results as PNG
            std::string stem = path.stem().string();
            stbi_write_png((save_dir + "/" + stem + "_gray.png").c_str(),  w, h, 1, gray_gpu.data(), w);
            stbi_write_png((save_dir + "/" + stem + "_blur.png").c_str(),  w, h, 1, blur_g.data(),   w);
            stbi_write_png((save_dir + "/" + stem + "_sobel.png").c_str(), w, h, 1, sobel_g.data(),  w);
        }
        stbi_image_free(rgb);
    }
    fclose(csv);

    printf("\n===== BATCH SUMMARY (%zu images, pipeline: grayscale + blur + sobel) =====\n", files.size());
    printf("CPU total            : %10.2f ms\n", tot_cpu);
    printf("GPU kernels only     : %10.2f ms  -> speedup %.2fx\n", tot_kernel, tot_cpu / tot_kernel);
    printf("GPU incl. transfers  : %10.2f ms  -> speedup %.2fx\n", tot_gpu, tot_cpu / tot_gpu);
    printf("Results written to %s\n", csv_path.c_str());
    return 0;
}
