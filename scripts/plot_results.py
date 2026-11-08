"""Make the plots for the report. Run from the project root after the benchmarks:
   python scripts/plot_results.py
Inputs : results/sizes.csv, results/batch_<N>.csv
Outputs: results/speedup_vs_size_*.png, results/gpu_time_breakdown.png, results/batch_time.png, results/batch_speedup.png
"""
import glob
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("results", exist_ok=True)

# ---- 1. speedup vs image size
if os.path.exists("results/sizes.csv"):
    df = pd.read_csv("results/sizes.csv")
    df["pixels"] = df["width"] * df["height"]
    for col, title, out in [("speedup_kernel", "Kernel-only speedup", "speedup_vs_size_kernel.png"),
                            ("speedup_total", "Speedup including transfers", "speedup_vs_size_total.png")]:
        plt.figure(figsize=(7, 4.5))
        for name, g in df.groupby("filter"):
            g = g.sort_values("pixels")
            plt.plot(g["pixels"], g[col], marker="o", label=name)
        plt.axhline(1, ls="--", color="gray")
        plt.xscale("log")
        plt.xlabel("Image size (pixels)")
        plt.ylabel("Speedup (CPU time / GPU time)")
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"results/{out}", dpi=150)
        plt.close()

    # where does the GPU time go? (sobel as example)
    s = df[df["filter"] == "sobel"].sort_values("pixels")
    labels = s["width"].astype(str)
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, s["h2d_ms"], label="H2D copy")
    plt.bar(labels, s["kernel_ms"], bottom=s["h2d_ms"], label="Kernel")
    plt.bar(labels, s["d2h_ms"], bottom=s["h2d_ms"] + s["kernel_ms"], label="D2H copy")
    plt.xlabel("Image width (px)")
    plt.ylabel("GPU time (ms)")
    plt.title("Sobel: GPU time breakdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/gpu_time_breakdown.png", dpi=150)
    plt.close()

# ---- 2. batch size experiments (one CSV per batch size)
rows = []
for f in glob.glob("results/batch_*.csv"):
    d = pd.read_csv(f)
    d = d[d["filter"] != "blur_shared"]          # pipeline = grayscale + blur_global + sobel
    rows.append({"batch": d["image"].nunique(), "cpu": d["cpu_ms"].sum(),
                 "gpu_kernel": d["kernel_ms"].sum(), "gpu_total": d["gpu_total_ms"].sum()})
if rows:
    b = pd.DataFrame(rows).sort_values("batch")
    plt.figure(figsize=(7, 4.5))
    plt.plot(b["batch"], b["cpu"], marker="o", label="CPU")
    plt.plot(b["batch"], b["gpu_total"], marker="o", label="GPU (incl. transfers)")
    plt.plot(b["batch"], b["gpu_kernel"], marker="o", label="GPU (kernels only)")
    plt.xlabel("Batch size (images)")
    plt.ylabel("Total time (ms)")
    plt.title("Total time vs batch size")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/batch_time.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.plot(b["batch"], b["cpu"] / b["gpu_total"], marker="o", label="incl. transfers")
    plt.plot(b["batch"], b["cpu"] / b["gpu_kernel"], marker="o", label="kernels only")
    plt.axhline(1, ls="--", color="gray")
    plt.xlabel("Batch size (images)")
    plt.ylabel("Speedup")
    plt.title("Speedup vs batch size")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/batch_speedup.png", dpi=150)
    plt.close()
print("Plots saved in results/")
