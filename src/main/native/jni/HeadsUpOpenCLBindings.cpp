/*
 * OpenCL host code for heads-up equity Monte Carlo computation.
 *
 * Dynamically loads OpenCL.dll at runtime via LoadLibraryA / GetProcAddress
 * so that the DLL can be built and loaded even when OpenCL is not installed;
 * calls that require OpenCL simply return status 200 in that case.
 *
 * JNI class: sicfun.holdem.HeadsUpOpenCLNativeBindings
 */

#include <jni.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

/* ── Minimal OpenCL type definitions for dynamic loading ────────── */

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

typedef int32_t   cl_int;
typedef uint32_t  cl_uint;
typedef uint64_t  cl_ulong;
typedef cl_uint   cl_bool;
typedef cl_ulong  cl_bitfield;
typedef cl_bitfield cl_device_type;
typedef cl_bitfield cl_command_queue_properties;
typedef cl_uint   cl_platform_info;
typedef cl_uint   cl_device_info;
typedef cl_uint   cl_program_build_info;
typedef cl_uint   cl_kernel_work_group_info;
typedef cl_bitfield cl_mem_flags;
typedef cl_uint   cl_mem_info;
typedef intptr_t  cl_context_properties;

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef void* cl_event;

#define CL_SUCCESS                    0
#define CL_DEVICE_TYPE_GPU            (1 << 2)
#define CL_DEVICE_TYPE_ALL            0xFFFFFFFF
#define CL_DEVICE_NAME                0x102B
#define CL_DEVICE_VENDOR              0x102C
#define CL_DEVICE_MAX_COMPUTE_UNITS   0x1002
#define CL_DEVICE_MAX_CLOCK_FREQUENCY 0x100C
#define CL_DEVICE_GLOBAL_MEM_SIZE     0x101F
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_MEM_READ_ONLY              (1 << 2)
#define CL_MEM_WRITE_ONLY             (1 << 1)
#define CL_MEM_READ_WRITE             (1 << 0)
#define CL_MEM_COPY_HOST_PTR          (1 << 5)
#define CL_PROGRAM_BUILD_LOG          0x1183
#define CL_KERNEL_WORK_GROUP_SIZE     0x11B0
#define CL_TRUE                       1
#define CL_FALSE                      0

/* ── Dynamic function pointers ──────────────────────────────────── */

namespace {

#define DECL_CL_FN(ret, name, args) typedef ret (CL_API_CALL *PFN_##name) args; PFN_##name fn_##name = nullptr;
#ifndef CL_API_CALL
#ifdef _WIN32
#define CL_API_CALL __stdcall
#else
#define CL_API_CALL
#endif
#endif

DECL_CL_FN(cl_int, clGetPlatformIDs, (cl_uint, cl_platform_id*, cl_uint*))
DECL_CL_FN(cl_int, clGetDeviceIDs, (cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*))
DECL_CL_FN(cl_int, clGetDeviceInfo, (cl_device_id, cl_device_info, size_t, void*, size_t*))
DECL_CL_FN(cl_context, clCreateContext, (const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_API_CALL*)(const char*,const void*,size_t,void*), void*, cl_int*))
DECL_CL_FN(cl_command_queue, clCreateCommandQueue, (cl_context, cl_device_id, cl_command_queue_properties, cl_int*))
DECL_CL_FN(cl_program, clCreateProgramWithSource, (cl_context, cl_uint, const char**, const size_t*, cl_int*))
DECL_CL_FN(cl_int, clBuildProgram, (cl_program, cl_uint, const cl_device_id*, const char*, void(CL_API_CALL*)(cl_program,void*), void*))
DECL_CL_FN(cl_int, clGetProgramBuildInfo, (cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*))
DECL_CL_FN(cl_kernel, clCreateKernel, (cl_program, const char*, cl_int*))
DECL_CL_FN(cl_int, clGetKernelWorkGroupInfo, (cl_kernel, cl_device_id, cl_kernel_work_group_info, size_t, void*, size_t*))
DECL_CL_FN(cl_mem, clCreateBuffer, (cl_context, cl_mem_flags, size_t, void*, cl_int*))
DECL_CL_FN(cl_int, clSetKernelArg, (cl_kernel, cl_uint, size_t, const void*))
DECL_CL_FN(cl_int, clEnqueueNDRangeKernel, (cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*))
DECL_CL_FN(cl_int, clEnqueueReadBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*))
DECL_CL_FN(cl_int, clEnqueueWriteBuffer, (cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*))
DECL_CL_FN(cl_int, clFinish, (cl_command_queue))
DECL_CL_FN(cl_int, clReleaseMemObject, (cl_mem))
DECL_CL_FN(cl_int, clReleaseKernel, (cl_kernel))
DECL_CL_FN(cl_int, clReleaseProgram, (cl_program))
DECL_CL_FN(cl_int, clReleaseCommandQueue, (cl_command_queue))
DECL_CL_FN(cl_int, clReleaseContext, (cl_context))

#ifdef _WIN32
static HMODULE g_opencl_lib = nullptr;
#else
static void* g_opencl_lib = nullptr;
#endif

bool load_opencl_library() {
  static std::once_flag flag;
  static bool loaded = false;
  std::call_once(flag, []() {
#ifdef _WIN32
    g_opencl_lib = LoadLibraryA("OpenCL.dll");
    if (!g_opencl_lib) return;
    #define LOAD_FN(name) fn_##name = reinterpret_cast<PFN_##name>(GetProcAddress(g_opencl_lib, #name))
#else
    g_opencl_lib = dlopen("libOpenCL.so", RTLD_LAZY);
    if (!g_opencl_lib) return;
    #define LOAD_FN(name) fn_##name = reinterpret_cast<PFN_##name>(dlsym(g_opencl_lib, #name))
#endif
    LOAD_FN(clGetPlatformIDs);
    LOAD_FN(clGetDeviceIDs);
    LOAD_FN(clGetDeviceInfo);
    LOAD_FN(clCreateContext);
    LOAD_FN(clCreateCommandQueue);
    LOAD_FN(clCreateProgramWithSource);
    LOAD_FN(clBuildProgram);
    LOAD_FN(clGetProgramBuildInfo);
    LOAD_FN(clCreateKernel);
    LOAD_FN(clGetKernelWorkGroupInfo);
    LOAD_FN(clCreateBuffer);
    LOAD_FN(clSetKernelArg);
    LOAD_FN(clEnqueueNDRangeKernel);
    LOAD_FN(clEnqueueReadBuffer);
    LOAD_FN(clEnqueueWriteBuffer);
    LOAD_FN(clFinish);
    LOAD_FN(clReleaseMemObject);
    LOAD_FN(clReleaseKernel);
    LOAD_FN(clReleaseProgram);
    LOAD_FN(clReleaseCommandQueue);
    LOAD_FN(clReleaseContext);
    #undef LOAD_FN
    loaded = fn_clGetPlatformIDs != nullptr;
  });
  return loaded;
}

/* ── OpenCL GPU device enumeration ──────────────────────────────── */

constexpr int kHoleCardsCount = 1326;
constexpr int kDeckSize = 52;
constexpr int kMaxChunkMatchups = 4096;

struct OCLGpuDevice {
  cl_device_id device;
  cl_platform_id platform;
  std::string name;
  std::string vendor;
  cl_uint compute_units;
  cl_uint clock_mhz;
  cl_ulong global_mem;
  size_t max_work_group_size;
};

std::vector<OCLGpuDevice>& cached_gpu_devices() {
  static std::vector<OCLGpuDevice> devices;
  static std::once_flag flag;
  std::call_once(flag, []() {
    if (!load_opencl_library()) return;

    cl_uint num_platforms = 0;
    if (fn_clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) return;
    std::vector<cl_platform_id> platforms(num_platforms);
    fn_clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    for (cl_uint pi = 0; pi < num_platforms; ++pi) {
      cl_uint num_devs = 0;
      if (fn_clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devs) != CL_SUCCESS) continue;
      if (num_devs == 0) continue;
      std::vector<cl_device_id> devs(num_devs);
      fn_clGetDeviceIDs(platforms[pi], CL_DEVICE_TYPE_GPU, num_devs, devs.data(), nullptr);
      for (cl_uint di = 0; di < num_devs; ++di) {
        OCLGpuDevice info{};
        info.device = devs[di];
        info.platform = platforms[pi];
        char buf[256] = {};
        fn_clGetDeviceInfo(devs[di], CL_DEVICE_NAME, sizeof(buf), buf, nullptr);
        info.name = buf;
        fn_clGetDeviceInfo(devs[di], CL_DEVICE_VENDOR, sizeof(buf), buf, nullptr);
        info.vendor = buf;
        fn_clGetDeviceInfo(devs[di], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(info.compute_units), &info.compute_units, nullptr);
        fn_clGetDeviceInfo(devs[di], CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(info.clock_mhz), &info.clock_mhz, nullptr);
        fn_clGetDeviceInfo(devs[di], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(info.global_mem), &info.global_mem, nullptr);
        fn_clGetDeviceInfo(devs[di], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(info.max_work_group_size), &info.max_work_group_size, nullptr);
        devices.push_back(info);
      }
    }
  });
  return devices;
}

/* ── Per-device compiled kernel cache ───────────────────────────── */

struct OCLDeviceContext {
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_mem hole_first_buf;
  cl_mem hole_second_buf;
  size_t max_wg_size;
  bool valid;
};

/* Embedded kernel source — generated at build time or included inline.
   The host code includes the .cl file as a string literal. */
static const char* kernel_source() {
  static const char* src =
#include "HeadsUpOpenCLKernel_embed.inc"
  ;
  return src;
}

/* Build the hole-cards lookup tables (same as CUDA version). */
void build_hole_cards_lookup(std::vector<uint8_t>& first, std::vector<uint8_t>& second) {
  first.resize(kHoleCardsCount);
  second.resize(kHoleCardsCount);
  int idx = 0;
  for (int f = 0; f < (kDeckSize - 1); ++f) {
    for (int s = f + 1; s < kDeckSize; ++s) {
      first[idx] = static_cast<uint8_t>(f);
      second[idx] = static_cast<uint8_t>(s);
      ++idx;
    }
  }
}

std::mutex g_ctx_mutex;
std::vector<OCLDeviceContext> g_device_contexts;
std::atomic<jint> g_last_engine_code(0);
constexpr jint kEngineOpenCL = 4;

OCLDeviceContext& get_or_create_context(int device_index) {
  std::lock_guard<std::mutex> guard(g_ctx_mutex);
  if (g_device_contexts.empty()) {
    g_device_contexts.resize(cached_gpu_devices().size());
    for (auto& c : g_device_contexts) c.valid = false;
  }
  auto& ctx = g_device_contexts[device_index];
  if (ctx.valid) return ctx;

  const auto& dev = cached_gpu_devices()[device_index];
  cl_int err = 0;

  ctx.context = fn_clCreateContext(nullptr, 1, &dev.device, nullptr, nullptr, &err);
  if (err != CL_SUCCESS) { ctx.valid = false; return ctx; }

  ctx.queue = fn_clCreateCommandQueue(ctx.context, dev.device, 0, &err);
  if (err != CL_SUCCESS) { fn_clReleaseContext(ctx.context); ctx.valid = false; return ctx; }

  const char* src = kernel_source();
  const size_t src_len = std::strlen(src);
  ctx.program = fn_clCreateProgramWithSource(ctx.context, 1, &src, &src_len, &err);
  if (err != CL_SUCCESS) {
    fn_clReleaseCommandQueue(ctx.queue);
    fn_clReleaseContext(ctx.context);
    ctx.valid = false;
    return ctx;
  }

  err = fn_clBuildProgram(ctx.program, 1, &dev.device, "-cl-std=CL1.2 -cl-mad-enable -cl-fast-relaxed-math", nullptr, nullptr);
  if (err != CL_SUCCESS) {
    size_t log_size = 0;
    fn_clGetProgramBuildInfo(ctx.program, dev.device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    if (log_size > 0) {
      std::vector<char> log_buf(log_size + 1, '\0');
      fn_clGetProgramBuildInfo(ctx.program, dev.device, CL_PROGRAM_BUILD_LOG, log_size, log_buf.data(), nullptr);
      std::fprintf(stderr, "[sicfun-opencl] kernel build log:\n%s\n", log_buf.data());
    }
    fn_clReleaseProgram(ctx.program);
    fn_clReleaseCommandQueue(ctx.queue);
    fn_clReleaseContext(ctx.context);
    ctx.valid = false;
    return ctx;
  }

  ctx.kernel = fn_clCreateKernel(ctx.program, "monte_carlo_kernel", &err);
  if (err != CL_SUCCESS) {
    fn_clReleaseProgram(ctx.program);
    fn_clReleaseCommandQueue(ctx.queue);
    fn_clReleaseContext(ctx.context);
    ctx.valid = false;
    return ctx;
  }

  fn_clGetKernelWorkGroupInfo(ctx.kernel, dev.device, CL_KERNEL_WORK_GROUP_SIZE,
      sizeof(ctx.max_wg_size), &ctx.max_wg_size, nullptr);
  if (ctx.max_wg_size == 0) ctx.max_wg_size = 64;
  ctx.max_wg_size = std::min(ctx.max_wg_size, static_cast<size_t>(256));

  /* Upload hole-cards lookup as constant buffers */
  std::vector<uint8_t> hc_first, hc_second;
  build_hole_cards_lookup(hc_first, hc_second);

  ctx.hole_first_buf = fn_clCreateBuffer(ctx.context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      hc_first.size(), hc_first.data(), &err);
  if (err != CL_SUCCESS) {
    fn_clReleaseKernel(ctx.kernel);
    fn_clReleaseProgram(ctx.program);
    fn_clReleaseCommandQueue(ctx.queue);
    fn_clReleaseContext(ctx.context);
    ctx.valid = false;
    return ctx;
  }

  ctx.hole_second_buf = fn_clCreateBuffer(ctx.context,
      CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      hc_second.size(), hc_second.data(), &err);
  if (err != CL_SUCCESS) {
    fn_clReleaseMemObject(ctx.hole_first_buf);
    fn_clReleaseKernel(ctx.kernel);
    fn_clReleaseProgram(ctx.program);
    fn_clReleaseCommandQueue(ctx.queue);
    fn_clReleaseContext(ctx.context);
    ctx.valid = false;
    return ctx;
  }

  ctx.valid = true;
  return ctx;
}

bool check_and_clear_exception(JNIEnv* env) {
  if (!env->ExceptionCheck()) return false;
  env->ExceptionClear();
  return true;
}

int compute_batch_opencl(
    const int device_index,
    const std::vector<jint>& low_buf,
    const std::vector<jint>& high_buf,
    const std::vector<jlong>& seed_buf,
    const jint trials,
    std::vector<jdouble>& win_buf,
    std::vector<jdouble>& tie_buf,
    std::vector<jdouble>& loss_buf,
    std::vector<jdouble>& stderr_buf) {

  const auto& devices = cached_gpu_devices();
  if (devices.empty()) return 201;
  if (device_index < 0 || device_index >= static_cast<int>(devices.size())) return 206;

  auto& ctx = get_or_create_context(device_index);
  if (!ctx.valid) return 202;

  const int n = static_cast<int>(low_buf.size());
  if (n <= 0) return 0;

  cl_int err = 0;

  /* Allocate device buffers */
  cl_mem d_low = fn_clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      n * sizeof(jint), const_cast<jint*>(low_buf.data()), &err);
  if (err != CL_SUCCESS) return 203;

  cl_mem d_high = fn_clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      n * sizeof(jint), const_cast<jint*>(high_buf.data()), &err);
  if (err != CL_SUCCESS) { fn_clReleaseMemObject(d_low); return 203; }

  cl_mem d_seeds = fn_clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
      n * sizeof(jlong), const_cast<jlong*>(seed_buf.data()), &err);
  if (err != CL_SUCCESS) { fn_clReleaseMemObject(d_high); fn_clReleaseMemObject(d_low); return 203; }

  cl_mem d_wins = fn_clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, n * sizeof(jdouble), nullptr, &err);
  cl_mem d_ties = fn_clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, n * sizeof(jdouble), nullptr, &err);
  cl_mem d_losses = fn_clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, n * sizeof(jdouble), nullptr, &err);
  cl_mem d_stderrs = fn_clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, n * sizeof(jdouble), nullptr, &err);

  int zero = 0;
  cl_mem d_status = fn_clCreateBuffer(ctx.context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
      sizeof(int), &zero, &err);

  if (!d_wins || !d_ties || !d_losses || !d_stderrs || !d_status) {
    if (d_wins) fn_clReleaseMemObject(d_wins);
    if (d_ties) fn_clReleaseMemObject(d_ties);
    if (d_losses) fn_clReleaseMemObject(d_losses);
    if (d_stderrs) fn_clReleaseMemObject(d_stderrs);
    if (d_status) fn_clReleaseMemObject(d_status);
    fn_clReleaseMemObject(d_seeds);
    fn_clReleaseMemObject(d_high);
    fn_clReleaseMemObject(d_low);
    return 203;
  }

  auto free_all = [&]() {
    fn_clReleaseMemObject(d_low);
    fn_clReleaseMemObject(d_high);
    fn_clReleaseMemObject(d_seeds);
    fn_clReleaseMemObject(d_wins);
    fn_clReleaseMemObject(d_ties);
    fn_clReleaseMemObject(d_losses);
    fn_clReleaseMemObject(d_stderrs);
    fn_clReleaseMemObject(d_status);
  };

  /* Chunked dispatch to prevent driver timeouts */
  const int max_chunk = kMaxChunkMatchups;

  for (int offset = 0; offset < n; offset += max_chunk) {
    const int chunk = std::min(max_chunk, n - offset);
    const int index_offset = offset;

    /* Reset status */
    zero = 0;
    err = fn_clEnqueueWriteBuffer(ctx.queue, d_status, CL_TRUE, 0, sizeof(int), &zero, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { free_all(); return 204; }

    /* Set kernel arguments */
    cl_uint arg = 0;
    cl_mem offset_low = d_low;      /* We pass the base buffer + offset via index_offset */
    cl_mem offset_high = d_high;
    cl_mem offset_seeds = d_seeds;
    cl_mem offset_wins = d_wins;
    cl_mem offset_ties = d_ties;
    cl_mem offset_losses = d_losses;
    cl_mem offset_stderrs = d_stderrs;

    /* The kernel gets the full buffers; we pass offset as index_offset
       and use sub-buffer addressing in the kernel via get_global_id + offset. */
    /* We create sub-buffers or pass offset — simpler: just pass offset arrays.
       Actually, OpenCL doesn't have pointer arithmetic like CUDA.
       Simplest: enqueue with global_work_offset. But clEnqueueNDRangeKernel
       doesn't have that in OpenCL 1.x for all devices reliably.
       Instead: we pass the offset as a kernel arg and the kernel adds it. */

    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_low);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_high);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_seeds);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(int), &chunk);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(int), &index_offset);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(int), &trials);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_wins);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_ties);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_losses);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_stderrs);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &d_status);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &ctx.hole_first_buf);
    fn_clSetKernelArg(ctx.kernel, arg++, sizeof(cl_mem), &ctx.hole_second_buf);

    /* Dispatch */
    const size_t wg_size = ctx.max_wg_size;
    const size_t global_size = ((static_cast<size_t>(chunk) + wg_size - 1) / wg_size) * wg_size;

    /* Use global_work_offset to index into the full buffers */
    const size_t global_offset = static_cast<size_t>(offset);

    err = fn_clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1,
        &global_offset, &global_size, &wg_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[sicfun-opencl] kernel enqueue failed: %d\n", err);
      free_all();
      return 204;
    }

    err = fn_clFinish(ctx.queue);
    if (err != CL_SUCCESS) {
      std::fprintf(stderr, "[sicfun-opencl] clFinish failed: %d\n", err);
      free_all();
      return 204;
    }

    /* Check kernel status */
    int kernel_status = 0;
    err = fn_clEnqueueReadBuffer(ctx.queue, d_status, CL_TRUE, 0, sizeof(int), &kernel_status, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) { free_all(); return 205; }
    if (kernel_status != 0) { free_all(); return kernel_status; }
  }

  /* Read results */
  err = fn_clEnqueueReadBuffer(ctx.queue, d_wins, CL_TRUE, 0, n * sizeof(jdouble), win_buf.data(), 0, nullptr, nullptr);
  if (err == CL_SUCCESS) err = fn_clEnqueueReadBuffer(ctx.queue, d_ties, CL_TRUE, 0, n * sizeof(jdouble), tie_buf.data(), 0, nullptr, nullptr);
  if (err == CL_SUCCESS) err = fn_clEnqueueReadBuffer(ctx.queue, d_losses, CL_TRUE, 0, n * sizeof(jdouble), loss_buf.data(), 0, nullptr, nullptr);
  if (err == CL_SUCCESS) err = fn_clEnqueueReadBuffer(ctx.queue, d_stderrs, CL_TRUE, 0, n * sizeof(jdouble), stderr_buf.data(), 0, nullptr, nullptr);
  if (err != CL_SUCCESS) { free_all(); return 205; }

  free_all();
  return 0;
}

}  // namespace

/* ── JNI exports ────────────────────────────────────────────────── */

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpOpenCLNativeBindings_openclDeviceCount(
    JNIEnv*,
    jclass) {
  if (!load_opencl_library()) return 0;
  return static_cast<jint>(cached_gpu_devices().size());
}

extern "C" JNIEXPORT jstring JNICALL
Java_sicfun_holdem_HeadsUpOpenCLNativeBindings_openclDeviceInfo(
    JNIEnv* env,
    jclass,
    jint device_index) {
  if (!load_opencl_library()) return env->NewStringUTF("");
  const auto& devices = cached_gpu_devices();
  if (device_index < 0 || device_index >= static_cast<jint>(devices.size())) {
    return env->NewStringUTF("");
  }
  const auto& dev = devices[device_index];
  const int mem_mb = static_cast<int>(dev.global_mem / (1024ULL * 1024ULL));
  char buf[512];
  std::snprintf(buf, sizeof(buf), "%s|%u|%u|%d|%s",
      dev.name.c_str(),
      dev.compute_units,
      dev.clock_mhz,
      mem_mb,
      dev.vendor.c_str());
  return env->NewStringUTF(buf);
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpOpenCLNativeBindings_computeBatch(
    JNIEnv* env,
    jclass,
    jint device_index,
    jintArray low_ids,
    jintArray high_ids,
    jint mode_code,
    jint trials,
    jlongArray seeds,
    jdoubleArray wins,
    jdoubleArray ties,
    jdoubleArray losses,
    jdoubleArray stderrs) {
  g_last_engine_code.store(0, std::memory_order_relaxed);

  if (!load_opencl_library()) return 200;

  if (low_ids == nullptr || high_ids == nullptr || seeds == nullptr ||
      wins == nullptr || ties == nullptr || losses == nullptr || stderrs == nullptr) {
    return 100;
  }

  const jsize n = env->GetArrayLength(low_ids);
  if (env->GetArrayLength(high_ids) != n || env->GetArrayLength(seeds) != n ||
      env->GetArrayLength(wins) != n || env->GetArrayLength(ties) != n ||
      env->GetArrayLength(losses) != n || env->GetArrayLength(stderrs) != n) {
    return 101;
  }
  if (mode_code == 0) {
    return 111;  /* Exact mode not supported on OpenCL */
  }
  if (mode_code != 1) {
    return 111;
  }
  if (trials <= 0) {
    return 126;
  }

  std::vector<jint> low_buf(static_cast<size_t>(n));
  std::vector<jint> high_buf(static_cast<size_t>(n));
  std::vector<jlong> seed_buf(static_cast<size_t>(n));
  std::vector<jdouble> win_buf(static_cast<size_t>(n));
  std::vector<jdouble> tie_buf(static_cast<size_t>(n));
  std::vector<jdouble> loss_buf(static_cast<size_t>(n));
  std::vector<jdouble> stderr_buf(static_cast<size_t>(n));

  env->GetIntArrayRegion(low_ids, 0, n, low_buf.data());
  env->GetIntArrayRegion(high_ids, 0, n, high_buf.data());
  env->GetLongArrayRegion(seeds, 0, n, seed_buf.data());
  if (check_and_clear_exception(env)) return 102;

  for (jsize i = 0; i < n; ++i) {
    if (low_buf[i] < 0 || low_buf[i] >= kHoleCardsCount ||
        high_buf[i] < 0 || high_buf[i] >= kHoleCardsCount) {
      return 125;
    }
  }

  const jint status = compute_batch_opencl(
      static_cast<int>(device_index),
      low_buf, high_buf, seed_buf, trials,
      win_buf, tie_buf, loss_buf, stderr_buf);
  if (status != 0) return status;

  g_last_engine_code.store(kEngineOpenCL, std::memory_order_relaxed);

  env->SetDoubleArrayRegion(wins, 0, n, win_buf.data());
  env->SetDoubleArrayRegion(ties, 0, n, tie_buf.data());
  env->SetDoubleArrayRegion(losses, 0, n, loss_buf.data());
  env->SetDoubleArrayRegion(stderrs, 0, n, stderr_buf.data());
  if (check_and_clear_exception(env)) {
    g_last_engine_code.store(0, std::memory_order_relaxed);
    return 124;
  }
  return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_sicfun_holdem_HeadsUpOpenCLNativeBindings_lastEngineCode(
    JNIEnv*,
    jclass) {
  return g_last_engine_code.load(std::memory_order_relaxed);
}
