#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

constexpr int64_t kChannels = 1;
constexpr int64_t kWindow = 100;
constexpr int64_t kFeatures = 40;
constexpr int kExpectedCsvColumns = 150;
constexpr int kLabelRawColumn = 145;  // raw CSV index (after python drops column 0 => 144)

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

std::string Trim(const std::string& s) {
  size_t start = 0;
  while (start < s.size() && std::isspace(static_cast<unsigned char>(s[start]))) {
    ++start;
  }
  size_t end = s.size();
  while (end > start && std::isspace(static_cast<unsigned char>(s[end - 1]))) {
    --end;
  }
  return s.substr(start, end - start);
}

std::string ResolvePath(const std::vector<std::string>& candidates) {
  for (const auto& c : candidates) {
    if (std::filesystem::exists(c)) {
      return c;
    }
  }
  return candidates.empty() ? std::string() : candidates.front();
}

std::string ResolveModelPath(const std::string& provided) {
  if (!provided.empty()) {
    return provided;
  }
  return ResolvePath({"transformer.onnx", "deeplob.onnx", "../transformer.onnx", "../deeplob.onnx"});
}

std::vector<double> ReadNpy1D(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) {
    throw std::runtime_error("Failed to open npy file: " + path);
  }

  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") {
    throw std::runtime_error("Invalid npy magic header: " + path);
  }

  char major = 0;
  char minor = 0;
  f.read(&major, 1);
  f.read(&minor, 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t hl16 = 0;
    f.read(reinterpret_cast<char*>(&hl16), 2);
    header_len = hl16;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }

  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  const bool is_f8 = header.find("'descr': '<f8'") != std::string::npos ||
                     header.find("\"descr\": \"<f8\"") != std::string::npos;
  const bool is_f4 = header.find("'descr': '<f4'") != std::string::npos ||
                     header.find("\"descr\": \"<f4\"") != std::string::npos;

  if (!is_f8 && !is_f4) {
    throw std::runtime_error("Unsupported npy dtype (expected <f4 or <f8): " + path);
  }

  const auto l = header.find('(');
  const auto r = header.find(')', l == std::string::npos ? 0 : l + 1);
  if (l == std::string::npos || r == std::string::npos) {
    throw std::runtime_error("Failed to parse npy shape from header: " + path);
  }

  const std::string shape = header.substr(l + 1, r - l - 1);
  std::stringstream ss(shape);
  std::string first_dim;
  std::getline(ss, first_dim, ',');
  const int n = std::stoi(Trim(first_dim));
  if (n <= 0) {
    throw std::runtime_error("Invalid npy shape length in: " + path);
  }

  std::vector<double> out(static_cast<size_t>(n));
  if (is_f8) {
    std::vector<double> tmp(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(double)));
    out = std::move(tmp);
  } else {
    std::vector<float> tmp(static_cast<size_t>(n));
    f.read(reinterpret_cast<char*>(tmp.data()), static_cast<std::streamsize>(tmp.size() * sizeof(float)));
    for (size_t i = 0; i < tmp.size(); ++i) {
      out[i] = static_cast<double>(tmp[i]);
    }
  }

  return out;
}

struct CsvData {
  std::vector<std::array<float, kFeatures>> rows;
  std::vector<int> labels;  // remapped to 0/1/2 when valid, else -1
};

CsvData LoadCsv(const std::string& csv_path) {
  std::ifstream f(csv_path);
  if (!f) {
    throw std::runtime_error("Failed to open CSV: " + csv_path);
  }

  CsvData data;

  std::string line;
  // Skip metadata/header line as in Python loader (skiprows=1).
  if (!std::getline(f, line)) {
    throw std::runtime_error("CSV appears empty: " + csv_path);
  }

  size_t line_num = 1;
  while (std::getline(f, line)) {
    ++line_num;
    if (line.empty()) {
      continue;
    }

    std::array<float, kFeatures> feat{};
    int label = -1;

    size_t start = 0;
    int col = 0;
    int feat_col = 0;

    while (start <= line.size()) {
      const size_t comma = line.find(',', start);
      const size_t end = (comma == std::string::npos) ? line.size() : comma;
      const std::string token = Trim(line.substr(start, end - start));

      // Raw CSV col0 is index/garbage. We need cols 1..40 for features.
      if (col >= 1 && col <= 40) {
        if (feat_col < kFeatures) {
          feat[feat_col++] = token.empty() ? 0.0f : std::stof(token);
        }
      }

      if (col == kLabelRawColumn) {
        if (!token.empty()) {
          const int raw = static_cast<int>(std::lround(std::stod(token)));
          const int remapped = raw - 1;
          if (remapped >= 0 && remapped < 3) {
            label = remapped;
          }
        }
      }

      ++col;
      if (comma == std::string::npos) {
        break;
      }
      start = comma + 1;
    }

    if (col < kExpectedCsvColumns) {
      continue;
    }
    if (feat_col != kFeatures) {
      throw std::runtime_error("Failed to parse 40 features at CSV line " + std::to_string(line_num));
    }

    data.rows.push_back(feat);
    data.labels.push_back(label);
  }

  return data;
}

struct CliArgs {
  std::string model_path;
  std::string csv_path;
  std::string mu_path;
  std::string sigma_path;
  int64_t batch_size = 1;
  int64_t max_windows = -1;
  int warmup_runs = 100;
};

CliArgs ParseArgs(int argc, char** argv) {
  CliArgs args;

  args.model_path = ResolveModelPath(argc > 1 ? argv[1] : std::string());
  args.csv_path = (argc > 2)
      ? argv[2]
      : ResolvePath({"data/raw/FI2010_test.csv", "../data/raw/FI2010_test.csv"});
  args.mu_path = (argc > 3)
      ? argv[3]
      : ResolvePath({"data/processed/mu.npy", "../data/processed/mu.npy"});
  args.sigma_path = (argc > 4)
      ? argv[4]
      : ResolvePath({"data/processed/sigma.npy", "../data/processed/sigma.npy"});

  if (argc > 5) {
    args.batch_size = std::max<int64_t>(1, std::stoll(argv[5]));
  }
  if (argc > 6) {
    args.max_windows = std::stoll(argv[6]);
  }
  if (argc > 7) {
    args.warmup_runs = std::max(0, std::stoi(argv[7]));
  }

  return args;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    CliArgs args = ParseArgs(argc, argv);

    // Current ONNX exports may still behave as static batch=1 in some runtimes.
    if (args.batch_size != 1) {
      std::cout << "Warning: forcing batch_size=1 for robust ONNX compatibility in C++ inference.\n";
      args.batch_size = 1;
    }

    if (args.model_path.empty() || !std::filesystem::exists(args.model_path)) {
      throw std::runtime_error("Model file not found. Provide model path as arg1.");
    }
    if (args.csv_path.empty() || !std::filesystem::exists(args.csv_path)) {
      throw std::runtime_error("CSV file not found. Provide csv path as arg2.");
    }
    if (args.mu_path.empty() || !std::filesystem::exists(args.mu_path)) {
      throw std::runtime_error("mu.npy not found. Provide mu path as arg3.");
    }
    if (args.sigma_path.empty() || !std::filesystem::exists(args.sigma_path)) {
      throw std::runtime_error("sigma.npy not found. Provide sigma path as arg4.");
    }

    const std::vector<double> mu = ReadNpy1D(args.mu_path);
    const std::vector<double> sigma = ReadNpy1D(args.sigma_path);
    if (mu.size() != kFeatures || sigma.size() != kFeatures) {
      throw std::runtime_error("mu/sigma shape mismatch. Expected 40 features.");
    }

    CsvData csv = LoadCsv(args.csv_path);
    if (csv.rows.size() <= static_cast<size_t>(kWindow)) {
      throw std::runtime_error("Not enough rows for windowed inference (need >100 rows).");
    }

    int64_t total_windows = static_cast<int64_t>(csv.rows.size()) - kWindow;
    if (args.max_windows > 0) {
      total_windows = std::min<int64_t>(total_windows, args.max_windows);
    }

    std::cout << "Model path       : " << args.model_path << "\n";
    std::cout << "CSV path         : " << args.csv_path << "\n";
    std::cout << "mu/sigma         : " << args.mu_path << " | " << args.sigma_path << "\n";
    std::cout << "Rows loaded      : " << csv.rows.size() << "\n";
    std::cout << "Windows used     : " << total_windows << "\n";
    std::cout << "Batch size       : " << args.batch_size << "\n";
    std::cout << "Warmup runs      : " << args.warmup_runs << "\n";

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "deeplob_full_inference");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
    const std::wstring model_path_w = ToWide(args.model_path);
    Ort::Session session(env, model_path_w.c_str(), session_options);
#else
    Ort::Session session(env, args.model_path.c_str(), session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_alloc = session.GetInputNameAllocated(0, allocator);
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* input_name = input_name_alloc.get();
    const char* output_name = output_name_alloc.get();

    const std::array<const char*, 1> input_names = {input_name};
    const std::array<const char*, 1> output_names = {output_name};

    // Warm-up with first batch.
    {
      const int64_t warmup_bs = std::min<int64_t>(args.batch_size, total_windows);
      const size_t warmup_size = static_cast<size_t>(warmup_bs * kChannels * kWindow * kFeatures);
      std::vector<float> warmup_input(warmup_size);

      for (int64_t b = 0; b < warmup_bs; ++b) {
        const int64_t w = b;
        for (int64_t t = 0; t < kWindow; ++t) {
          const auto& row = csv.rows[static_cast<size_t>(w + t)];
          for (int64_t fidx = 0; fidx < kFeatures; ++fidx) {
            const double x = static_cast<double>(row[static_cast<size_t>(fidx)]);
            const double z = (x - mu[static_cast<size_t>(fidx)]) / (sigma[static_cast<size_t>(fidx)] + 1e-8);
            const size_t idx = static_cast<size_t>(b * kWindow * kFeatures + t * kFeatures + fidx);
            warmup_input[idx] = static_cast<float>(z);
          }
        }
      }

      const std::array<int64_t, 4> warmup_shape = {warmup_bs, kChannels, kWindow, kFeatures};
      Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value warmup_tensor = Ort::Value::CreateTensor<float>(
          mem,
          warmup_input.data(),
          warmup_input.size(),
          warmup_shape.data(),
          warmup_shape.size());

      for (int i = 0; i < args.warmup_runs; ++i) {
        (void)session.Run(
            Ort::RunOptions{nullptr},
            input_names.data(),
            &warmup_tensor,
            1,
            output_names.data(),
            1);
      }
    }

    int64_t processed = 0;
    int64_t correct = 0;
    int64_t labeled = 0;

    const auto t0 = std::chrono::steady_clock::now();

    while (processed < total_windows) {
      const int64_t bs = std::min<int64_t>(args.batch_size, total_windows - processed);
      const size_t input_size = static_cast<size_t>(bs * kChannels * kWindow * kFeatures);
      std::vector<float> input(input_size);

      for (int64_t b = 0; b < bs; ++b) {
        const int64_t w = processed + b;
        for (int64_t t = 0; t < kWindow; ++t) {
          const auto& row = csv.rows[static_cast<size_t>(w + t)];
          for (int64_t fidx = 0; fidx < kFeatures; ++fidx) {
            const double x = static_cast<double>(row[static_cast<size_t>(fidx)]);
            const double z = (x - mu[static_cast<size_t>(fidx)]) / (sigma[static_cast<size_t>(fidx)] + 1e-8);
            const size_t idx = static_cast<size_t>(b * kWindow * kFeatures + t * kFeatures + fidx);
            input[idx] = static_cast<float>(z);
          }
        }
      }

      const std::array<int64_t, 4> input_shape = {bs, kChannels, kWindow, kFeatures};
      Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          mem,
          input.data(),
          input.size(),
          input_shape.data(),
          input_shape.size());

      auto outputs = session.Run(
          Ort::RunOptions{nullptr},
          input_names.data(),
          &input_tensor,
          1,
          output_names.data(),
          1);

      float* logits = outputs.front().GetTensorMutableData<float>();
      for (int64_t b = 0; b < bs; ++b) {
        float* row_logits = logits + (b * 3);
        const int pred = static_cast<int>(std::distance(row_logits, std::max_element(row_logits, row_logits + 3)));

        const int64_t y_idx = processed + b + kWindow;
        if (y_idx < static_cast<int64_t>(csv.labels.size()) && csv.labels[static_cast<size_t>(y_idx)] >= 0) {
          ++labeled;
          if (pred == csv.labels[static_cast<size_t>(y_idx)]) {
            ++correct;
          }
        }
      }

      processed += bs;
    }

    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mean_ms = total_ms / static_cast<double>(std::max<int64_t>(1, total_windows));
    const double throughput = (static_cast<double>(total_windows) * 1000.0) / std::max(1.0, total_ms);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Total infer time : " << total_ms << " ms\n";
    std::cout << "Mean per window  : " << mean_ms << " ms\n";
    std::cout << "Throughput       : " << throughput << " windows/s\n";

    if (labeled > 0) {
      const double acc = 100.0 * static_cast<double>(correct) / static_cast<double>(labeled);
      std::cout << "Accuracy         : " << acc << "% (" << correct << "/" << labeled << ")\n";
    } else {
      std::cout << "Accuracy         : N/A (labels unavailable)\n";
    }

    return 0;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return 1;
   } catch (const std::exception& e) {
     std::cerr << "Error: " << e.what() << "\n";
     return 1;
   }
 }
