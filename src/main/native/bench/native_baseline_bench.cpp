
#include <windows.h>
#include <jni.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <random>

// Mock JNIEnv implementation to avoid starting a JVM
struct MockJNIEnv {
    struct MockArray {
        void* data;
        jsize length;
    };

    static jsize JNICALL GetArrayLength(JNIEnv* env, jarray array) {
        return reinterpret_cast<MockArray*>(array)->length;
    }

    static void JNICALL GetIntArrayRegion(JNIEnv* env, jintArray array, jsize start, jsize len, jint* buf) {
        auto mock = reinterpret_cast<MockArray*>(array);
        memcpy(buf, reinterpret_cast<jint*>(mock->data) + start, len * sizeof(jint));
    }

    static void JNICALL GetLongArrayRegion(JNIEnv* env, jlongArray array, jsize start, jsize len, jlong* buf) {
        auto mock = reinterpret_cast<MockArray*>(array);
        memcpy(buf, reinterpret_cast<jlong*>(mock->data) + start, len * sizeof(jlong));
    }

    static void JNICALL GetFloatArrayRegion(JNIEnv* env, jfloatArray array, jsize start, jsize len, jfloat* buf) {
        auto mock = reinterpret_cast<MockArray*>(array);
        memcpy(buf, reinterpret_cast<jfloat*>(mock->data) + start, len * sizeof(jfloat));
    }

    static void JNICALL GetDoubleArrayRegion(JNIEnv* env, jdoubleArray array, jsize start, jsize len, jdouble* buf) {
        auto mock = reinterpret_cast<MockArray*>(array);
        memcpy(buf, reinterpret_cast<jdouble*>(mock->data) + start, len * sizeof(jdouble));
    }

    static void JNICALL SetFloatArrayRegion(JNIEnv* env, jfloatArray array, jsize start, jsize len, const jfloat* buf) {
        auto mock = reinterpret_cast<MockArray*>(array);
        memcpy(reinterpret_cast<jfloat*>(mock->data) + start, buf, len * sizeof(jfloat));
    }

    static void JNICALL SetDoubleArrayRegion(JNIEnv* env, jdoubleArray array, jsize start, jsize len, const jdouble* buf) {
        auto mock = reinterpret_cast<MockArray*>(array);
        memcpy(reinterpret_cast<jdouble*>(mock->data) + start, buf, len * sizeof(jdouble));
    }

    static jboolean JNICALL ExceptionCheck(JNIEnv* env) { return JNI_FALSE; }
    static void JNICALL ExceptionClear(JNIEnv* env) {}

    JNINativeInterface_ functions;
    const JNINativeInterface_* functions_ptr;

    MockJNIEnv() {
        memset(&functions, 0, sizeof(functions));
        functions.GetArrayLength = GetArrayLength;
        functions.GetIntArrayRegion = GetIntArrayRegion;
        functions.GetLongArrayRegion = GetLongArrayRegion;
        functions.GetFloatArrayRegion = GetFloatArrayRegion;
        functions.GetDoubleArrayRegion = GetDoubleArrayRegion;
        functions.SetFloatArrayRegion = SetFloatArrayRegion;
        functions.SetDoubleArrayRegion = SetDoubleArrayRegion;
        functions.ExceptionCheck = ExceptionCheck;
        functions.ExceptionClear = ExceptionClear;
        functions_ptr = &functions;
    }

    JNIEnv* get() { return const_cast<JNIEnv*>(reinterpret_cast<const JNIEnv*>(&functions_ptr)); }

    jarray createArray(void* data, jsize length) {
        return reinterpret_cast<jarray>(new MockArray{data, length});
    }

    void destroyArray(jarray array) {
        delete reinterpret_cast<MockArray*>(array);
    }
};

typedef jint (JNICALL *ComputeBatchFunc)(JNIEnv*, jclass, jintArray, jintArray, jint, jint, jlongArray, jdoubleArray, jdoubleArray, jdoubleArray, jdoubleArray);
typedef jint (JNICALL *ComputeBatchPackedFunc)(JNIEnv*, jclass, jintArray, jint, jint, jlong, jlongArray, jfloatArray, jfloatArray, jfloatArray, jfloatArray);

struct BenchmarkConfig {
    std::string device;
    std::string mode; // "exact" or "mc"
    int batch_size;
    int trials;
};

void run_benchmark(
    MockJNIEnv& mock,
    const std::string& device_name,
    ComputeBatchFunc computeBatch,
    ComputeBatchPackedFunc computeBatchPacked,
    std::ofstream& out) {

    const std::vector<int> batch_sizes = {1, 8, 64, 256, 1024, 4096, 16384};
    const std::vector<int> trials_list = {100, 1000, 10000, 100000};
    const int repetitions = 3; // Reduced from 10

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> hole_dist(0, 1325);

    for (int batch_size : batch_sizes) {
        std::cout << "  Batch size: " << batch_size << std::endl;
        // Prepare data
        // ...
        // (rest of the prepare data block)
        std::vector<jint> low_ids(batch_size);
        std::vector<jint> high_ids(batch_size);
        std::vector<jint> packed_keys(batch_size);
        std::vector<jlong> seeds(batch_size);
        std::vector<jdouble> wins(batch_size), ties(batch_size), losses(batch_size), stderrs(batch_size);
        std::vector<jfloat> wins_f(batch_size), ties_f(batch_size), losses_f(batch_size), stderrs_f(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            low_ids[i] = 0;
            high_ids[i] = 1325;
            packed_keys[i] = (0 << 11) | 1325;
            seeds[i] = i;
        }

        jintArray j_low = (jintArray)mock.createArray(low_ids.data(), batch_size);
        jintArray j_high = (jintArray)mock.createArray(high_ids.data(), batch_size);
        jintArray j_packed = (jintArray)mock.createArray(packed_keys.data(), batch_size);
        jlongArray j_seeds = (jlongArray)mock.createArray(seeds.data(), batch_size);
        jdoubleArray j_wins = (jdoubleArray)mock.createArray(wins.data(), batch_size);
        jdoubleArray j_ties = (jdoubleArray)mock.createArray(ties.data(), batch_size);
        jdoubleArray j_losses = (jdoubleArray)mock.createArray(losses.data(), batch_size);
        jdoubleArray j_stderrs = (jdoubleArray)mock.createArray(stderrs.data(), batch_size);
        jfloatArray j_wins_f = (jfloatArray)mock.createArray(wins_f.data(), batch_size);
        jfloatArray j_ties_f = (jfloatArray)mock.createArray(ties_f.data(), batch_size);
        jfloatArray j_losses_f = (jfloatArray)mock.createArray(losses_f.data(), batch_size);
        jfloatArray j_stderrs_f = (jfloatArray)mock.createArray(stderrs_f.data(), batch_size);

        // Exact mode
        if (batch_size <= 64) { // Further limited
            std::cout << "    Exact mode..." << std::endl;
            for (int r = 0; r < repetitions; ++r) {
                auto start = std::chrono::high_resolution_clock::now();
                jint status = computeBatch(mock.get(), nullptr, j_low, j_high, 0, 0, j_seeds, j_wins, j_ties, j_losses, j_stderrs);
                auto end = std::chrono::high_resolution_clock::now();
                if (status != 0) std::cerr << "computeBatch exact failed with " << status << std::endl;
                double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
                out << device_name << "\texact\t" << batch_size << "\t0\tlatency_ms\t" << elapsed << std::endl;
            }
        }

        // Monte Carlo mode
        for (int trials : trials_list) {
            std::cout << "    MC trials: " << trials << std::endl;
            // computeBatch
            for (int r = 0; r < repetitions; ++r) {
                auto start = std::chrono::high_resolution_clock::now();
                jint status = computeBatch(mock.get(), nullptr, j_low, j_high, 1, trials, j_seeds, j_wins, j_ties, j_losses, j_stderrs);
                auto end = std::chrono::high_resolution_clock::now();
                if (status != 0) std::cerr << "computeBatch MC failed with " << status << " trials=" << trials << std::endl;
                double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
                out << device_name << "\tmc_batch\t" << batch_size << "\t" << trials << "\tlatency_ms\t" << elapsed << std::endl;
            }

            // computeBatchPacked
            for (int r = 0; r < repetitions; ++r) {
                auto start = std::chrono::high_resolution_clock::now();
                jint status = computeBatchPacked(mock.get(), nullptr, j_packed, 1, trials, 12345LL, j_seeds, j_wins_f, j_ties_f, j_losses_f, j_stderrs_f);
                auto end = std::chrono::high_resolution_clock::now();
                if (status != 0) std::cerr << "computeBatchPacked MC failed with " << status << " trials=" << trials << std::endl;
                double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
                out << device_name << "\tmc_packed\t" << batch_size << "\t" << trials << "\tlatency_ms\t" << elapsed << std::endl;
            }
        }

        mock.destroyArray(j_low);
        mock.destroyArray(j_high);
        mock.destroyArray(j_packed);
        mock.destroyArray(j_seeds);
        mock.destroyArray(j_wins);
        mock.destroyArray(j_ties);
        mock.destroyArray(j_losses);
        mock.destroyArray(j_stderrs);
        mock.destroyArray(j_wins_f);
        mock.destroyArray(j_ties_f);
        mock.destroyArray(j_losses_f);
        mock.destroyArray(j_stderrs_f);
    }
}

int main() {
    MockJNIEnv mock;
    std::ofstream out("validation-output/native-benchmarks/cpu_baseline.tsv");
    out << "device\tmode\tbatch_size\ttrials\tmetric\tvalue\n";

    // Load CPU DLL
    HMODULE hCpu = LoadLibraryA("src/main/native/build-ddre-verify/sicfun_native_cpu.dll");
    if (!hCpu) {
        std::cerr << "Failed to load sicfun_native_cpu.dll. Error: " << GetLastError() << std::endl;
        // Try other path if not found
        hCpu = LoadLibraryA("sicfun_native_cpu.dll");
    }

    if (hCpu) {
        std::cout << "Benchmarking CPU..." << std::endl;
        auto computeBatch = (ComputeBatchFunc)GetProcAddress(hCpu, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatch");
        auto computeBatchPacked = (ComputeBatchPackedFunc)GetProcAddress(hCpu, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPacked");

        if (computeBatch && computeBatchPacked) {
            run_benchmark(mock, "CPU", computeBatch, computeBatchPacked, out);
        } else {
            std::cerr << "Failed to find functions in sicfun_native_cpu.dll" << std::endl;
        }
        FreeLibrary(hCpu);
    } else {
        std::cerr << "Skipping CPU benchmark." << std::endl;
    }

    // Load GPU DLL
    HMODULE hGpu = LoadLibraryA("src/main/native/build-ddre-verify-cuda/sicfun_gpu_kernel.dll");
    if (!hGpu) {
        std::cerr << "Failed to load sicfun_gpu_kernel.dll. Error: " << GetLastError() << std::endl;
        hGpu = LoadLibraryA("sicfun_gpu_kernel.dll");
    }

    if (hGpu) {
        std::cout << "Benchmarking GPU..." << std::endl;
        auto computeBatch = (ComputeBatchFunc)GetProcAddress(hGpu, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatch");
        auto computeBatchPacked = (ComputeBatchPackedFunc)GetProcAddress(hGpu, "Java_sicfun_holdem_HeadsUpGpuNativeBindings_computeBatchPacked");

        if (computeBatch && computeBatchPacked) {
            run_benchmark(mock, "GPU", computeBatch, computeBatchPacked, out);
        } else {
            std::cerr << "Failed to find functions in sicfun_gpu_kernel.dll" << std::endl;
        }
        FreeLibrary(hGpu);
    } else {
        std::cerr << "Skipping GPU benchmark." << std::endl;
    }

    out.close();
    std::cout << "Done. Results written to validation-output/native-benchmarks/cpu_baseline.tsv" << std::endl;
    return 0;
}
