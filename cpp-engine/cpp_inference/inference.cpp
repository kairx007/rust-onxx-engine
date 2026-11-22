#include <algorithm>
#include <array>
#include <filesystem>
#include <iostream>
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

std::vector<float> MakeInput() {
  // Use deterministic synthetic input for smoke inference.
  std::vector<float> input(kInputSize);
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>((i % 23) * 0.01f);
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

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "deeplob_inference");

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

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(),
        &input_tensor,
        1,
        output_names.data(),
        1);

    float* logits = output_tensors.front().GetTensorMutableData<float>();
    const auto max_it = std::max_element(logits, logits + 3);
    const int pred_class = static_cast<int>(std::distance(logits, max_it));

    std::cout << "Model path: " << model_path << "\n";
    std::cout << "Logits: [" << logits[0] << ", " << logits[1] << ", " << logits[2] << "]\n";
    std::cout << "Predicted class: " << pred_class
              << " (0=down, 1=stationary, 2=up)\n";
    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
}
