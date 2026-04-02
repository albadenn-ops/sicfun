// Phase 1 Baseline Benchmark for SICFUN Native Equity Computation
// Tests CPU, CUDA, and OpenCL paths via JNI DLL exports with MockJNIEnv.
//
// Build: powershell -ExecutionPolicy Bypass -File build_bench.ps1
// Run:   native_baseline_bench.exe [cpu|cuda|opencl|all] [quick]

#include <windows.h>
#include <jni.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <iomanip>
#include <functional>

// ── MockJNIEnv ──────────────────────────────────────────────────────────────
// Provides minimal JNI interface to call DLL-exported JNI functions without a JVM.
struct MockJNIEnv {
    struct MockArray {
        void* data;
        jsize length;
    };

    static jsize    JNICALL GetArrayLength(JNIEnv*, jarray a) { return reinterpret_cast<MockArray*>(a)->length; }
    static void     JNICALL GetIntArrayRegion(JNIEnv*, jintArray a, jsize s, jsize l, jint* b)       { memcpy(b, (jint*)reinterpret_cast<MockArray*>(a)->data + s, l * sizeof(jint)); }
    static void     JNICALL GetLongArrayRegion(JNIEnv*, jlongArray a, jsize s, jsize l, jlong* b)    { memcpy(b, (jlong*)reinterpret_cast<MockArray*>(a)->data + s, l * sizeof(jlong)); }
    static void     JNICALL GetFloatArrayRegion(JNIEnv*, jfloatArray a, jsize s, jsize l, jfloat* b) { memcpy(b, (jfloat*)reinterpret_cast<MockArray*>(a)->data + s, l * sizeof(jfloat)); }
    static void     JNICALL GetDoubleArrayRegion(JNIEnv*, jdoubleArray a, jsize s, jsize l, jdouble* b) { memcpy(b, (jdouble*)reinterpret_cast<MockArray*>(a)->data + s, l * sizeof(jdouble)); }
    static void     JNICALL SetIntArrayRegion(JNIEnv*, jintArray a, jsize s, jsize l, const jint* b)       { memcpy((jint*)reinterpret_cast<MockArray*>(a)->data + s, b, l * sizeof(jint)); }
    static void     JNICALL SetLongArrayRegion(JNIEnv*, jlongArray a, jsize s, jsize l, const jlong* b)    { memcpy((jlong*)reinterpret_cast<MockArray*>(a)->data + s, b, l * sizeof(jlong)); }
    static void     JNICALL SetFloatArrayRegion(JNIEnv*, jfloatArray a, jsize s, jsize l, const jfloat* b) { memcpy((jfloat*)reinterpret_cast<MockArray*>(a)->data + s, b, l * sizeof(jfloat)); }
    static void     JNICALL SetDoubleArrayRegion(JNIEnv*, jdoubleArray a, jsize s, jsize l, const jdouble* b) { memcpy((jdouble*)reinterpret_cast<MockArray*>(a)->data + s, b, l * sizeof(jdouble)); }
    static jboolean JNICALL ExceptionCheck(JNIEnv*) { return JNI_FALSE; }
    static void     JNICALL ExceptionClear(JNIEnv*) {}
    static jintArray JNICALL NewIntArray(JNIEnv*, jsize) { return nullptr; }
    // Stubs for JNI class/method/string lookups used by resolve_* config helpers
    static jclass     JNICALL FindClass(JNIEnv*, const char*) { return nullptr; }
    static jmethodID  JNICALL GetStaticMethodID(JNIEnv*, jclass, const char*, const char*) { return nullptr; }
    static void       JNICALL DeleteLocalRef(JNIEnv*, jobject) {}
    static jstring    JNICALL NewStringUTF(JNIEnv*, const char*) { return nullptr; }
    static jobject    JNICALL CallStaticObjectMethod(JNIEnv*, jclass, jmethodID, ...) { return nullptr; }
    static const char* JNICALL GetStringUTFChars(JNIEnv*, jstring, jboolean*) { return nullptr; }
    static void       JNICALL ReleaseStringUTFChars(JNIEnv*, jstring, const char*) {}
    static jsize      JNICALL GetStringUTFLength(JNIEnv*, jstring) { return 0; }

    JNINativeInterface_ fns;
    const JNINativeInterface_* fns_ptr;

    MockJNIEnv() {
        memset(&fns, 0, sizeof(fns));
        fns.GetArrayLength       = GetArrayLength;
        fns.GetIntArrayRegion    = GetIntArrayRegion;
        fns.GetLongArrayRegion   = GetLongArrayRegion;
        fns.GetFloatArrayRegion  = GetFloatArrayRegion;
        fns.GetDoubleArrayRegion = GetDoubleArrayRegion;
        fns.SetIntArrayRegion    = SetIntArrayRegion;
        fns.SetLongArrayRegion   = SetLongArrayRegion;
        fns.SetFloatArrayRegion  = SetFloatArrayRegion;
        fns.SetDoubleArrayRegion = SetDoubleArrayRegion;
        fns.ExceptionCheck       = ExceptionCheck;
        fns.ExceptionClear       = ExceptionClear;
        fns.NewIntArray          = NewIntArray;
        fns.FindClass            = FindClass;
        fns.GetStaticMethodID    = GetStaticMethodID;
        fns.DeleteLocalRef       = DeleteLocalRef;
        fns.NewStringUTF         = NewStringUTF;
        fns.CallStaticObjectMethod = CallStaticObjectMethod;
        fns.GetStringUTFChars    = GetStringUTFChars;
        fns.ReleaseStringUTFChars = ReleaseStringUTFChars;
        fns.GetStringUTFLength   = GetStringUTFLength;
        fns_ptr = &fns;
    }

    JNIEnv* env() { return reinterpret_cast<JNIEnv*>(&fns_ptr); }

    jarray wrap(void* data, jsize len) { return reinterpret_cast<jarray>(new MockArray{data, len}); }
    void   unwrap(jarray a) { delete reinterpret_cast<MockArray*>(a); }
};

// ── JNI function pointer types ──────────────────────────────────────────────

// computeBatch(env, cls, low_ids, high_ids, mode_code, trials, seeds, wins, ties, losses, stderrs)
typedef jint (JNICALL *FnComputeBatch)(JNIEnv*, jclass, jintArray, jintArray, jint, jint, jlongArray, jdoubleArray, jdoubleArray, jdoubleArray, jdoubleArray);

// computeBatchPacked(env, cls, packed_keys, mode_code, trials, mc_seed_base, key_material, wins_f, ties_f, losses_f, stderrs_f)
typedef jint (JNICALL *FnComputeBatchPacked)(JNIEnv*, jclass, jintArray, jint, jint, jlong, jlongArray, jfloatArray, jfloatArray, jfloatArray, jfloatArray);

// computeBatchPackedOnDevice(env, cls, device_index, packed_keys, mode_code, trials, mc_seed_base, key_material, wins_f, ties_f, losses_f, stderrs_f)
typedef jint (JNICALL *FnComputeBatchPackedOnDevice)(JNIEnv*, jclass, jint, jintArray, jint, jint, jlong, jlongArray, jfloatArray, jfloatArray, jfloatArray, jfloatArray);

// OpenCL computeBatch(env, cls, device_index, low_ids, high_ids, mode_code, trials, seeds, wins, ties, losses, stderrs)
typedef jint (JNICALL *FnOclComputeBatch)(JNIEnv*, jclass, jint, jintArray, jintArray, jint, jint, jlongArray, jdoubleArray, jdoubleArray, jdoubleArray, jdoubleArray);

// ── Statistics ──────────────────────────────────────────────────────────────

struct Stats {
    double mean, median, p95, stddev, min_val, max_val;
    int n;
};

Stats compute_stats(std::vector<double>& v) {
    Stats s{};
    s.n = (int)v.size();
    if (s.n == 0) return s;
    std::sort(v.begin(), v.end());
    double sum = 0;
    for (double x : v) sum += x;
    s.mean = sum / s.n;
    s.median = v[s.n / 2];
    s.p95 = v[(int)(s.n * 0.95)];
    s.min_val = v[0];
    s.max_val = v[s.n - 1];
    double sq = 0;
    for (double x : v) sq += (x - s.mean) * (x - s.mean);
    s.stddev = (s.n > 1) ? std::sqrt(sq / (s.n - 1)) : 0;
    return s;
}

// ── Data generation ─────────────────────────────────────────────────────────

static const int kIdBits = 11;

// Canonical pair ID for cards (c1, c2) where 0 <= c1 < c2 < 52
static int pair_id(int c1, int c2) {
    return c1 * (103 - c1) / 2 + c2 - c1 - 1;
}

void generate_matchups(int n, std::vector<jint>& low, std::vector<jint>& high,
                       std::vector<jint>& packed, std::vector<jlong>& seeds,
                       std::vector<jlong>& key_material) {
    low.resize(n); high.resize(n); packed.resize(n); seeds.resize(n); key_material.resize(n);

    // Build a table of guaranteed non-overlapping matchups.
    // Each uses 4 consecutive cards: hero=(4k, 4k+1), villain=(4k+2, 4k+3).
    // This gives 13 unique matchups (52/4). Cycle for larger batches.
    struct ValidMatchup { int hero_id, villain_id; };
    std::vector<ValidMatchup> valid;
    for (int k = 0; k < 13; k++) {
        int h = pair_id(4*k, 4*k+1);
        int v = pair_id(4*k+2, 4*k+3);
        valid.push_back({h, v});
    }

    std::mt19937_64 rng(42);
    for (int i = 0; i < n; i++) {
        auto& m = valid[i % valid.size()];
        int a = m.hero_id, b = m.villain_id;
        if (a > b) std::swap(a, b);
        low[i] = a;
        high[i] = b;
        packed[i] = (a << kIdBits) | b;
        seeds[i] = (jlong)(rng() & 0x7FFFFFFFFFFFFFFFULL);
        key_material[i] = seeds[i];
    }
}

// ── TSV writer ──────────────────────────────────────────────────────────────

void write_row(std::ofstream& out, const char* device, const char* mode, const char* func,
               int batch, int trials, const Stats& s) {
    auto row = [&](const char* metric, double val) {
        out << device << "\t" << mode << "\t" << func << "\t" << batch << "\t" << trials
            << "\t" << metric << "\t" << std::fixed << std::setprecision(4) << val << "\n";
        out.flush();
    };
    row("mean_ms", s.mean);
    row("median_ms", s.median);
    row("p95_ms", s.p95);
    row("stddev_ms", s.stddev);
    if (s.mean > 0) {
        row("matchups_s", batch * 1000.0 / s.mean);
        if (trials > 0)
            row("trials_s", (double)batch * trials * 1000.0 / s.mean);
    }
}

// ── DLL loading helpers ─────────────────────────────────────────────────────

HMODULE try_load(const char* primary, const char* fallback) {
    HMODULE h = LoadLibraryA(primary);
    if (!h && fallback) h = LoadLibraryA(fallback);
    return h;
}

template<typename F>
F get_fn(HMODULE dll, const char* name) {
    return reinterpret_cast<F>(GetProcAddress(dll, name));
}

// ── Benchmark runners ───────────────────────────────────────────────────────

void bench_cpu(MockJNIEnv& mock, HMODULE dll, std::ofstream& out, bool quick) {
    auto computeBatch = get_fn<FnComputeBatch>(dll, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatch");
    auto computeBatchPacked = get_fn<FnComputeBatchPacked>(dll, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPacked");

    if (!computeBatch) { std::cerr << "[CPU] computeBatch not found\n"; return; }
    if (!computeBatchPacked) { std::cerr << "[CPU] computeBatchPacked not found\n"; return; }

    const std::vector<int> batch_sizes = {1, 8, 64, 256, 1024, 4096, 16384};
    const std::vector<int> trial_counts = {100, 1000, 10000, 100000};
    const int reps = quick ? 3 : 10;

    for (int bs : batch_sizes) {
        std::vector<jint> low, high, packed;
        std::vector<jlong> seeds, key_mat;
        generate_matchups(bs, low, high, packed, seeds, key_mat);

        std::vector<jdouble> wins_d(bs), ties_d(bs), losses_d(bs), stderrs_d(bs);
        std::vector<jfloat>  wins_f(bs), ties_f(bs), losses_f(bs), stderrs_f(bs);

        auto j_low    = (jintArray)mock.wrap(low.data(), bs);
        auto j_high   = (jintArray)mock.wrap(high.data(), bs);
        auto j_packed = (jintArray)mock.wrap(packed.data(), bs);
        auto j_seeds  = (jlongArray)mock.wrap(seeds.data(), bs);
        auto j_km     = (jlongArray)mock.wrap(key_mat.data(), bs);
        auto j_wd     = (jdoubleArray)mock.wrap(wins_d.data(), bs);
        auto j_td     = (jdoubleArray)mock.wrap(ties_d.data(), bs);
        auto j_ld     = (jdoubleArray)mock.wrap(losses_d.data(), bs);
        auto j_sd     = (jdoubleArray)mock.wrap(stderrs_d.data(), bs);
        auto j_wf     = (jfloatArray)mock.wrap(wins_f.data(), bs);
        auto j_tf     = (jfloatArray)mock.wrap(ties_f.data(), bs);
        auto j_lf     = (jfloatArray)mock.wrap(losses_f.data(), bs);
        auto j_sf     = (jfloatArray)mock.wrap(stderrs_f.data(), bs);

        // Exact mode: only for small batches (takes ~6s per matchup on CPU)
        if (bs <= 8) {
            std::cout << "[CPU] exact batch=" << bs << " (" << reps << " reps)..." << std::flush;
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                jint st = computeBatch(mock.env(), nullptr, j_low, j_high, 0, 0, j_seeds, j_wd, j_td, j_ld, j_sd);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (st != 0) { std::cerr << " status=" << st; break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                auto s = compute_stats(times);
                write_row(out, "CPU", "exact", "computeBatch", bs, 0, s);
                std::cout << " median=" << s.median << "ms\n";
            }
        }

        // Monte Carlo: computeBatch
        for (int trials : trial_counts) {
            if (quick && bs >= 4096 && trials >= 100000) continue;
            std::cout << "[CPU] mc batch=" << bs << " trials=" << trials << " (" << reps << " reps)..." << std::flush;
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                jint st = computeBatch(mock.env(), nullptr, j_low, j_high, 1, trials, j_seeds, j_wd, j_td, j_ld, j_sd);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (st != 0) { std::cerr << " status=" << st; break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                auto s = compute_stats(times);
                write_row(out, "CPU", "mc", "computeBatch", bs, trials, s);
                std::cout << " median=" << s.median << "ms\n";
            }
        }

        // Monte Carlo: computeBatchPacked
        for (int trials : trial_counts) {
            if (quick && bs >= 4096 && trials >= 100000) continue;
            std::cout << "[CPU] mc_packed batch=" << bs << " trials=" << trials << " (" << reps << " reps)..." << std::flush;
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                jint st = computeBatchPacked(mock.env(), nullptr, j_packed, 1, trials, 12345LL, j_km, j_wf, j_tf, j_lf, j_sf);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (st != 0) { std::cerr << " status=" << st; break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                auto s = compute_stats(times);
                write_row(out, "CPU", "mc", "computeBatchPacked", bs, trials, s);
                std::cout << " median=" << s.median << "ms\n";
            }
        }

        mock.unwrap(j_low); mock.unwrap(j_high); mock.unwrap(j_packed);
        mock.unwrap(j_seeds); mock.unwrap(j_km);
        mock.unwrap(j_wd); mock.unwrap(j_td); mock.unwrap(j_ld); mock.unwrap(j_sd);
        mock.unwrap(j_wf); mock.unwrap(j_tf); mock.unwrap(j_lf); mock.unwrap(j_sf);
    }
}

void bench_cuda(MockJNIEnv& mock, HMODULE dll, std::ofstream& out, bool quick) {
    auto computeBatchPackedOnDevice = get_fn<FnComputeBatchPackedOnDevice>(
        dll, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPackedOnDevice");

    if (!computeBatchPackedOnDevice) { std::cerr << "[CUDA] computeBatchPackedOnDevice not found\n"; return; }

    // Warm up CUDA context
    {
        // hero=pair_id(0,1)=0, villain=pair_id(2,3)=101 — no card overlap
        std::vector<jint> p = {(0 << 11) | 101};
        std::vector<jlong> km = {42};
        std::vector<jfloat> w(1), t(1), l(1), s(1);
        auto jp = (jintArray)mock.wrap(p.data(), 1);
        auto jk = (jlongArray)mock.wrap(km.data(), 1);
        auto jw = (jfloatArray)mock.wrap(w.data(), 1);
        auto jt = (jfloatArray)mock.wrap(t.data(), 1);
        auto jl = (jfloatArray)mock.wrap(l.data(), 1);
        auto js = (jfloatArray)mock.wrap(s.data(), 1);
        std::cout << "[CUDA] warming up..." << std::flush;
        jint st = computeBatchPackedOnDevice(mock.env(), nullptr, 0, jp, 1, 100, 42LL, jk, jw, jt, jl, js);
        std::cout << " status=" << st << "\n";
        mock.unwrap(jp); mock.unwrap(jk); mock.unwrap(jw); mock.unwrap(jt); mock.unwrap(jl); mock.unwrap(js);
        if (st != 0) { std::cerr << "[CUDA] warmup failed, skipping CUDA benchmarks\n"; return; }
    }

    const std::vector<int> batch_sizes = {1, 8, 64, 256, 1024, 4096, 16384};
    const std::vector<int> trial_counts = {100, 1000, 10000, 100000};
    const int reps = quick ? 3 : 10;

    for (int bs : batch_sizes) {
        std::vector<jint> low, high, packed;
        std::vector<jlong> seeds, key_mat;
        generate_matchups(bs, low, high, packed, seeds, key_mat);
        std::vector<jfloat> wins(bs), ties(bs), losses(bs), stderrs(bs);

        auto j_packed = (jintArray)mock.wrap(packed.data(), bs);
        auto j_km     = (jlongArray)mock.wrap(key_mat.data(), bs);
        auto j_w      = (jfloatArray)mock.wrap(wins.data(), bs);
        auto j_t      = (jfloatArray)mock.wrap(ties.data(), bs);
        auto j_l      = (jfloatArray)mock.wrap(losses.data(), bs);
        auto j_s      = (jfloatArray)mock.wrap(stderrs.data(), bs);

        // Exact mode on CUDA: only for small batches
        if (bs <= 64) {
            std::cout << "[CUDA] exact batch=" << bs << " (" << reps << " reps)..." << std::flush;
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                jint st = computeBatchPackedOnDevice(mock.env(), nullptr, 0, j_packed, 0, 0, 0LL, j_km, j_w, j_t, j_l, j_s);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (st != 0) { std::cerr << " status=" << st; break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                auto s = compute_stats(times);
                write_row(out, "CUDA", "exact", "computeBatchPackedOnDevice", bs, 0, s);
                std::cout << " median=" << s.median << "ms\n";
            }
        }

        // Monte Carlo
        for (int trials : trial_counts) {
            if (quick && bs >= 4096 && trials >= 100000) continue;
            std::cout << "[CUDA] mc batch=" << bs << " trials=" << trials << " (" << reps << " reps)..." << std::flush;
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                jint st = computeBatchPackedOnDevice(mock.env(), nullptr, 0, j_packed, 1, trials, 12345LL, j_km, j_w, j_t, j_l, j_s);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (st != 0) { std::cerr << " status=" << st; break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                auto s = compute_stats(times);
                write_row(out, "CUDA", "mc", "computeBatchPackedOnDevice", bs, trials, s);
                std::cout << " median=" << s.median << "ms\n";
            }
        }

        mock.unwrap(j_packed); mock.unwrap(j_km);
        mock.unwrap(j_w); mock.unwrap(j_t); mock.unwrap(j_l); mock.unwrap(j_s);
    }
}

void bench_opencl(MockJNIEnv& mock, HMODULE dll, std::ofstream& out, bool quick) {
    auto computeBatch = get_fn<FnOclComputeBatch>(
        dll, "Java_sicfun_holdem_HeadsUpOpenCLNativeBindings_computeBatch");

    if (!computeBatch) { std::cerr << "[OpenCL] computeBatch not found\n"; return; }

    // Warm up
    {
        // pair_id(0,1)=0 vs pair_id(2,3)=101 — no card overlap
        std::vector<jint> lo = {0}, hi = {101};
        std::vector<jlong> se = {42};
        std::vector<jdouble> w(1), t(1), l(1), s(1);
        auto jlo = (jintArray)mock.wrap(lo.data(), 1);
        auto jhi = (jintArray)mock.wrap(hi.data(), 1);
        auto jse = (jlongArray)mock.wrap(se.data(), 1);
        auto jw = (jdoubleArray)mock.wrap(w.data(), 1);
        auto jt = (jdoubleArray)mock.wrap(t.data(), 1);
        auto jl = (jdoubleArray)mock.wrap(l.data(), 1);
        auto js = (jdoubleArray)mock.wrap(s.data(), 1);
        std::cout << "[OpenCL] warming up..." << std::flush;
        jint st = computeBatch(mock.env(), nullptr, 0, jlo, jhi, 1, 100, jse, jw, jt, jl, js);
        std::cout << " status=" << st << "\n";
        mock.unwrap(jlo); mock.unwrap(jhi); mock.unwrap(jse);
        mock.unwrap(jw); mock.unwrap(jt); mock.unwrap(jl); mock.unwrap(js);
        if (st != 0) { std::cerr << "[OpenCL] warmup failed, skipping OpenCL benchmarks\n"; return; }
    }

    const std::vector<int> batch_sizes = {1, 8, 64, 256, 1024, 4096, 16384};
    const std::vector<int> trial_counts = {100, 1000, 10000, 100000};
    const int reps = quick ? 3 : 10;

    for (int bs : batch_sizes) {
        std::vector<jint> low, high, packed;
        std::vector<jlong> seeds, key_mat;
        generate_matchups(bs, low, high, packed, seeds, key_mat);
        std::vector<jdouble> wins(bs), ties(bs), losses(bs), stderrs(bs);

        auto j_low   = (jintArray)mock.wrap(low.data(), bs);
        auto j_high  = (jintArray)mock.wrap(high.data(), bs);
        auto j_seeds = (jlongArray)mock.wrap(seeds.data(), bs);
        auto j_w     = (jdoubleArray)mock.wrap(wins.data(), bs);
        auto j_t     = (jdoubleArray)mock.wrap(ties.data(), bs);
        auto j_l     = (jdoubleArray)mock.wrap(losses.data(), bs);
        auto j_s     = (jdoubleArray)mock.wrap(stderrs.data(), bs);

        // OpenCL: MC only (no exact kernel)
        for (int trials : trial_counts) {
            if (quick && bs >= 4096 && trials >= 100000) continue;
            std::cout << "[OpenCL] mc batch=" << bs << " trials=" << trials << " (" << reps << " reps)..." << std::flush;
            std::vector<double> times;
            for (int r = 0; r < reps; r++) {
                auto t0 = std::chrono::high_resolution_clock::now();
                jint st = computeBatch(mock.env(), nullptr, 0, j_low, j_high, 1, trials, j_seeds, j_w, j_t, j_l, j_s);
                auto t1 = std::chrono::high_resolution_clock::now();
                if (st != 0) { std::cerr << " status=" << st; break; }
                times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
            }
            if (!times.empty()) {
                auto s = compute_stats(times);
                write_row(out, "OpenCL", "mc", "computeBatch", bs, trials, s);
                std::cout << " median=" << s.median << "ms\n";
            }
        }

        mock.unwrap(j_low); mock.unwrap(j_high); mock.unwrap(j_seeds);
        mock.unwrap(j_w); mock.unwrap(j_t); mock.unwrap(j_l); mock.unwrap(j_s);
    }
}

// ── Correctness validation (Phase 2) ────────────────────────────────────────

void validate_correctness(MockJNIEnv& mock,
                          FnComputeBatch cpuBatch,
                          FnComputeBatchPackedOnDevice cudaOnDevice,
                          FnOclComputeBatch oclBatch,
                          std::ofstream& out) {
    std::cout << "\n=== Phase 2: Correctness Validation ===\n";
    const int n = 100;
    const int trials = 10000;
    const double mc_tolerance = 0.05;

    std::vector<jint> low, high, packed;
    std::vector<jlong> seeds, key_mat;
    generate_matchups(n, low, high, packed, seeds, key_mat);

    // CPU results (reference)
    std::vector<jdouble> cpu_w(n), cpu_t(n), cpu_l(n), cpu_s(n);
    auto jl  = (jintArray)mock.wrap(low.data(), n);
    auto jh  = (jintArray)mock.wrap(high.data(), n);
    auto js  = (jlongArray)mock.wrap(seeds.data(), n);
    auto jw  = (jdoubleArray)mock.wrap(cpu_w.data(), n);
    auto jt  = (jdoubleArray)mock.wrap(cpu_t.data(), n);
    auto jlo = (jdoubleArray)mock.wrap(cpu_l.data(), n);
    auto jse = (jdoubleArray)mock.wrap(cpu_s.data(), n);

    std::cout << "[Validate] CPU MC n=" << n << " trials=" << trials << "..." << std::flush;
    jint st = cpuBatch(mock.env(), nullptr, jl, jh, 1, trials, js, jw, jt, jlo, jse);
    std::cout << " status=" << st << "\n";

    mock.unwrap(jl); mock.unwrap(jh); mock.unwrap(js);
    mock.unwrap(jw); mock.unwrap(jt); mock.unwrap(jlo); mock.unwrap(jse);

    if (st != 0) { std::cerr << "[Validate] CPU baseline failed\n"; return; }

    // CUDA comparison
    if (cudaOnDevice) {
        std::vector<jfloat> cuda_w(n), cuda_t(n), cuda_l(n), cuda_s(n);
        auto jp  = (jintArray)mock.wrap(packed.data(), n);
        auto jk  = (jlongArray)mock.wrap(key_mat.data(), n);
        auto jcw = (jfloatArray)mock.wrap(cuda_w.data(), n);
        auto jct = (jfloatArray)mock.wrap(cuda_t.data(), n);
        auto jcl = (jfloatArray)mock.wrap(cuda_l.data(), n);
        auto jcs = (jfloatArray)mock.wrap(cuda_s.data(), n);

        std::cout << "[Validate] CUDA MC n=" << n << " trials=" << trials << "..." << std::flush;
        st = cudaOnDevice(mock.env(), nullptr, 0, jp, 1, trials, 12345LL, jk, jcw, jct, jcl, jcs);
        std::cout << " status=" << st << "\n";

        mock.unwrap(jp); mock.unwrap(jk); mock.unwrap(jcw); mock.unwrap(jct); mock.unwrap(jcl); mock.unwrap(jcs);

        if (st == 0) {
            int mismatches = 0;
            for (int i = 0; i < n; i++) {
                double diff_w = std::abs(cpu_w[i] - (double)cuda_w[i]);
                double diff_l = std::abs(cpu_l[i] - (double)cuda_l[i]);
                if (cpu_w[i] > 0.01 && diff_w / cpu_w[i] > mc_tolerance) mismatches++;
                if (cpu_l[i] > 0.01 && diff_l / cpu_l[i] > mc_tolerance) mismatches++;
            }
            out << "VALIDATION\tCPU_vs_CUDA\tmc\t" << n << "\t" << trials
                << "\tmismatches\t" << mismatches << "\n";
            out.flush();
            std::cout << "[Validate] CPU vs CUDA: " << mismatches << "/" << (n*2) << " mismatches (tol=" << mc_tolerance << ")\n";
        }
    }

    // OpenCL comparison
    if (oclBatch) {
        std::vector<jdouble> ocl_w(n), ocl_t(n), ocl_l(n), ocl_s(n);
        auto jol  = (jintArray)mock.wrap(low.data(), n);
        auto joh  = (jintArray)mock.wrap(high.data(), n);
        auto jos  = (jlongArray)mock.wrap(seeds.data(), n);
        auto jow  = (jdoubleArray)mock.wrap(ocl_w.data(), n);
        auto jot  = (jdoubleArray)mock.wrap(ocl_t.data(), n);
        auto joll = (jdoubleArray)mock.wrap(ocl_l.data(), n);
        auto jose = (jdoubleArray)mock.wrap(ocl_s.data(), n);

        std::cout << "[Validate] OpenCL MC n=" << n << " trials=" << trials << "..." << std::flush;
        st = oclBatch(mock.env(), nullptr, 0, jol, joh, 1, trials, jos, jow, jot, joll, jose);
        std::cout << " status=" << st << "\n";

        mock.unwrap(jol); mock.unwrap(joh); mock.unwrap(jos);
        mock.unwrap(jow); mock.unwrap(jot); mock.unwrap(joll); mock.unwrap(jose);

        if (st == 0) {
            int mismatches = 0;
            for (int i = 0; i < n; i++) {
                double diff_w = std::abs(cpu_w[i] - ocl_w[i]);
                double diff_l = std::abs(cpu_l[i] - ocl_l[i]);
                if (cpu_w[i] > 0.01 && diff_w / cpu_w[i] > mc_tolerance) mismatches++;
                if (cpu_l[i] > 0.01 && diff_l / cpu_l[i] > mc_tolerance) mismatches++;
            }
            out << "VALIDATION\tCPU_vs_OpenCL\tmc\t" << n << "\t" << trials
                << "\tmismatches\t" << mismatches << "\n";
            out.flush();
            std::cout << "[Validate] CPU vs OpenCL: " << mismatches << "/" << (n*2) << " mismatches (tol=" << mc_tolerance << ")\n";
        }
    }
}

// ── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    std::string target = "all";
    bool quick = false;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "quick") quick = true;
        else if (arg == "cpu" || arg == "cuda" || arg == "opencl" || arg == "all" || arg == "validate") target = arg;
    }

    std::cout << "=== SICFUN Native Baseline Benchmark ===\n";
    std::cout << "Target: " << target << (quick ? " (quick)" : " (full)") << "\n\n";

    MockJNIEnv mock;
    // Use append mode — write header only if file is new/empty
    std::string outpath = "validation-output/native-benchmarks/baseline.tsv";
    bool file_exists = false;
    { std::ifstream check(outpath); file_exists = check.good() && check.peek() != std::ifstream::traits_type::eof(); }
    std::ofstream out(outpath, std::ios::app);
    if (!file_exists) out << "device\tmode\tfunction\tbatch_size\ttrials\tmetric\tvalue\n";

    HMODULE hCpu = nullptr, hCuda = nullptr, hOcl = nullptr;

    bool bench = (target != "validate");
    if (bench && (target == "all" || target == "cpu") || target == "validate") {
        hCpu = try_load("src/main/native/build/sicfun_native_cpu.dll",
                         "src/main/native/build-ddre-verify/sicfun_native_cpu.dll");
        if (!hCpu) std::cerr << "[CPU] DLL not found (error " << GetLastError() << ")\n";
    }
    if (bench && (target == "all" || target == "cuda") || target == "validate") {
        hCuda = try_load("src/main/native/build/sicfun_gpu_kernel.dll",
                          "src/main/native/build-ddre-verify-cuda/sicfun_gpu_kernel.dll");
        if (!hCuda) std::cerr << "[CUDA] DLL not found (error " << GetLastError() << ")\n";
    }
    if (bench && (target == "all" || target == "opencl") || target == "validate") {
        hOcl = try_load("src/main/native/build/sicfun_opencl_kernel.dll", nullptr);
        if (!hOcl) std::cerr << "[OpenCL] DLL not found (error " << GetLastError() << ")\n";
    }

    // Phase 1: Benchmarks
    if (bench && hCpu)  bench_cpu(mock, hCpu, out, quick);
    if (bench && hCuda) bench_cuda(mock, hCuda, out, quick);
    if (bench && hOcl)  bench_opencl(mock, hOcl, out, quick);

    // Phase 2: Correctness validation
    FnComputeBatch cpuFn = hCpu ? get_fn<FnComputeBatch>(hCpu, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatch") : nullptr;
    FnComputeBatchPackedOnDevice cudaFn = hCuda ? get_fn<FnComputeBatchPackedOnDevice>(hCuda, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPackedOnDevice") : nullptr;
    FnOclComputeBatch oclFn = hOcl ? get_fn<FnOclComputeBatch>(hOcl, "Java_sicfun_holdem_HeadsUpOpenCLNativeBindings_computeBatch") : nullptr;

    if (cpuFn && (cudaFn || oclFn)) {
        validate_correctness(mock, cpuFn, cudaFn, oclFn, out);
    }

    if (hCpu)  FreeLibrary(hCpu);
    if (hCuda) FreeLibrary(hCuda);
    if (hOcl)  FreeLibrary(hOcl);

    out.close();
    std::cout << "\nDone. Results in " << outpath << "\n";
    return 0;
}
