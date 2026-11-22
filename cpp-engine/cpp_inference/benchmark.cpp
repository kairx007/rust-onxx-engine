#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr int64_t kBatch = 1;
constexpr int64_t kChannels = 1;
constexpr int64_t kTime = 100;
constexpr int64_t kFeatures = 40;
constexpr size_t kInputSize = static_cast<size_t>(kBatch * kChannels * kTime * kFeatures);

#ifdef _WIN32
std::wstring ToWide(const std::string& s) {
  if (s.empty()) {
    return std::wstring();
  }

  const int size_needed = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring result(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), result.data(), size_needed);
  return result;
}
#endif

double Mean(const std::vector<double>& v) {
  if (v.empty()) {
    return 0.0;
  }
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double Std(const std::vector<double>& v, double mean) {
  if (v.size() < 2) {
    return 0.0;
  }
  double accum = 0.0;
  for (double x : v) {
    const double d = x - mean;
    accum += d * d;
  }
  return std::sqrt(accum / static_cast<double>(v.size()));
}

double Percentile99(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const size_t idx = static_cast<size_t>(std::ceil(0.99 * static_cast<double>(v.size()))) - 1;
  return v[std::min(idx, v.size() - 1)];
}

std::vector<float> MakeInput() {
  std::vector<float> input(kInputSize);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>((i % 17) * 0.01f);
  }
  return input;
}

std::string ResolveModelPath(int argc, char** argv) {
  if (argc > 1) {
    return argv[1];
  }

  if (std::filesystem::exists("transformer.onnx")) {
    return "transformer.onnx";
  }
  if (std::filesystem::exists("deeplob.onnx")) {
    return "deeplob.onnx";
  }
  return "../deeplob.onnx";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string model_path = ResolveModelPath(argc, argv);
    const int warmup_runs = (argc > 2) ? std::max(0, std::stoi(argv[2])) : 100;
    const int timed_runs = (argc > 3) ? std::max(1, std::stoi(argv[3])) : 1000;

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "deeplob_benchmark");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
    const std::wstring model_path_w = ToWide(model_path);
    Ort::Session session(env, model_path_w.c_str(), session_options);
#else
    Ort::Session session(env, model_path.c_str(), session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);

    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    std::vector<float> input_data = MakeInput();
    const std::array<int64_t, 4> input_shape = {kBatch, kChannels, kTime, kFeatures};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        input_data.data(),
        input_data.size(),
        input_shape.data(),
        input_shape.size());

    const std::array<const char*, 1> input_names = {input_name};
    const std::array<const char*, 1> output_names = {output_name};

    for (int i = 0; i < warmup_runs; ++i) {
      (void)session.Run(
          Ort::RunOptions{nullptr},
          input_names.data(),
          &input_tensor,
          1,
          output_names.data(),
          1);
    }

    std::vector<double> latencies_ms;
    latencies_ms.reserve(static_cast<size_t>(timed_runs));

    for (int i = 0; i < timed_runs; ++i) {
      const auto t0 = std::chrono::steady_clock::now();
      auto outputs = session.Run(
          Ort::RunOptions{nullptr},
          input_names.data(),
          &input_tensor,
          1,
          output_names.data(),
          1);
      (void)outputs;
      const auto t1 = std::chrono::steady_clock::now();
      const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
      latencies_ms.push_back(ms);
    }

    const double mean = Mean(latencies_ms);
    const double std = Std(latencies_ms, mean);
    const double p99 = Percentile99(latencies_ms);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Model path       : " << model_path << "\n";
    std::cout << "Warmup runs      : " << warmup_runs << "\n";
    std::cout << "Timed runs       : " << timed_runs << "\n";
    std::cout << "Mean latency (ms): " << mean << "\n";
    std::cout << "Std latency (ms) : " << std << "\n";
    std::cout << "P99 latency (ms) : " << p99 << "\n";
    std::cout << "Target check     : " << ((mean < 1.0) ? "PASS (<1 ms mean)" : "FAIL (>=1 ms mean)") << "\n";

    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
