"""
A/B interleaved benchmark: old vs optimized sicfun_gpu_kernel.dll

Methodology:
  - Swaps DLLs between every run to control for thermal/load drift
  - 10 reps per variant (20 total), interleaved old→new→old→new...
  - Reports mean, stddev, 95% CI, and Welch's t-test p-value (scipy)
  - Saves raw timings to JSON for reproducibility
"""
import subprocess, shutil, os, re, json, math, statistics
from scipy import stats as sp_stats

REPO = r"C:\Users\alexl\code\math\untitled"
BUILD_DIR = os.path.join(REPO, "src", "main", "native", "build")
DLL_NAME = "sicfun_gpu_kernel.dll"
DLL_PATH = os.path.join(BUILD_DIR, DLL_NAME)
OPTIMIZED_DLL = os.path.join(BUILD_DIR, "sicfun_gpu_kernel_optimized.bak")
OLD_DLL = os.path.join(BUILD_DIR, "sicfun_gpu_kernel_old.bak")
BENCH_EXE = os.path.join(REPO, "src", "main", "native", "bench", "native_baseline_bench.exe")
TSV_PATH = os.path.join(REPO, "validation-output", "native-benchmarks", "baseline.tsv")
RAW_JSON = os.path.join(REPO, "validation-output", "native-benchmarks", "ab_raw_timings.json")

REPS_PER_SIDE = 10


def setup_dlls():
    """Save both DLL versions."""
    shutil.copy2(DLL_PATH, OPTIMIZED_DLL)
    subprocess.run(["git", "checkout", "HEAD", "--", DLL_PATH], cwd=REPO, check=True)
    shutil.copy2(DLL_PATH, OLD_DLL)
    shutil.copy2(OPTIMIZED_DLL, DLL_PATH)
    old_sz = os.path.getsize(OLD_DLL)
    new_sz = os.path.getsize(OPTIMIZED_DLL)
    print(f"[setup] old={old_sz} bytes, optimized={new_sz} bytes")


def swap_dll(variant):
    src = OLD_DLL if variant == "old" else OPTIMIZED_DLL
    shutil.copy2(src, DLL_PATH)


def run_bench():
    if os.path.exists(TSV_PATH):
        os.remove(TSV_PATH)
    result = subprocess.run(
        [BENCH_EXE, "cuda", "quick"],
        capture_output=True, text=True, cwd=REPO, timeout=600,
    )
    timings = {}
    for line in (result.stdout + result.stderr).split("\n"):
        m = re.match(
            r"\[CUDA\]\s+(\S+)\s+batch=(\d+)"
            r"(?:\s+trials=(\d+))?"
            r"\s+\(\d+ reps\)\.\.\.\s+median=([\d.]+)ms",
            line,
        )
        if m:
            mode, batch, trials, median = m.groups()
            key = f"{mode}_b{batch}" + (f"_t{trials}" if trials else "")
            timings[key] = float(median)
    return timings


def main():
    print("=" * 80)
    print("A/B Interleaved Benchmark: old vs optimized sicfun_gpu_kernel.dll")
    print(f"Reps per variant: {REPS_PER_SIDE} | Total runs: {REPS_PER_SIDE * 2}")
    print(f"Statistical test: Welch's two-sample t-test (scipy.stats.ttest_ind)")
    print("=" * 80)

    setup_dlls()

    old_results = {}
    new_results = {}

    for i in range(REPS_PER_SIDE):
        for variant in ["old", "optimized"]:
            run_num = i * 2 + (0 if variant == "old" else 1) + 1
            print(f"[run {run_num:>2}/{REPS_PER_SIDE * 2}] {variant:<10}", end="", flush=True)
            swap_dll(variant)
            timings = run_bench()
            target = old_results if variant == "old" else new_results
            for k, v in timings.items():
                target.setdefault(k, []).append(v)
            # Print a quick summary of this run
            keys_preview = sorted(timings.keys())[:3]
            preview = ", ".join(f"{k}={timings[k]:.2f}" for k in keys_preview)
            print(f"  ({preview}, ...)")

    # Restore optimized DLL as final state
    shutil.copy2(OPTIMIZED_DLL, DLL_PATH)

    # Save raw data
    raw = {"old": {k: v for k, v in old_results.items()},
           "optimized": {k: v for k, v in new_results.items()},
           "reps_per_side": REPS_PER_SIDE}
    with open(RAW_JSON, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"\n[saved] Raw timings -> {RAW_JSON}")

    # Report
    print("\n" + "=" * 80)
    print("RESULTS (Welch's t-test, two-tailed, H0: old_mean == new_mean)")
    print("=" * 80)

    configs = sorted(set(list(old_results.keys()) + list(new_results.keys())))

    header = (
        f"{'Config':<25} | "
        f"{'Old (ms)':>10} {'±95CI':>7} | "
        f"{'New (ms)':>10} {'±95CI':>7} | "
        f"{'Δ%':>7} {'p':>10} {'Sig':>4}"
    )
    print(f"\n{header}")
    print("-" * len(header))

    sig_improvements = 0
    sig_regressions = 0

    for cfg in configs:
        old_t = old_results.get(cfg, [])
        new_t = new_results.get(cfg, [])
        if len(old_t) < 2 or len(new_t) < 2:
            continue

        old_mean = statistics.mean(old_t)
        new_mean = statistics.mean(new_t)
        n_old, n_new = len(old_t), len(new_t)

        # 95% CI using t-critical for actual df
        old_se = statistics.stdev(old_t) / math.sqrt(n_old)
        new_se = statistics.stdev(new_t) / math.sqrt(n_new)
        t_crit_old = sp_stats.t.ppf(0.975, df=n_old - 1)
        t_crit_new = sp_stats.t.ppf(0.975, df=n_new - 1)
        old_ci = old_se * t_crit_old
        new_ci = new_se * t_crit_new

        # Welch's t-test (scipy)
        t_stat, p_value = sp_stats.ttest_ind(old_t, new_t, equal_var=False)

        speedup = (old_mean - new_mean) / old_mean * 100 if old_mean > 0 else 0

        if p_value < 0.05:
            if speedup > 0:
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*"
                sig_improvements += 1
            else:
                sig = "REG"
                sig_regressions += 1
        else:
            sig = ""

        print(
            f"{cfg:<25} | "
            f"{old_mean:>8.2f}ms {old_ci:>5.2f}ms | "
            f"{new_mean:>8.2f}ms {new_ci:>5.2f}ms | "
            f"{speedup:>+6.1f}% {p_value:>10.6f} {sig:>4}"
        )

    print("-" * len(header))
    print(f"Significant improvements (p<0.05): {sig_improvements}")
    print(f"Significant regressions  (p<0.05): {sig_regressions}")

    # Cleanup
    for f in [OPTIMIZED_DLL, OLD_DLL]:
        if os.path.exists(f):
            os.remove(f)

    print(f"\nDone. Raw data: {RAW_JSON}")


if __name__ == "__main__":
    main()
