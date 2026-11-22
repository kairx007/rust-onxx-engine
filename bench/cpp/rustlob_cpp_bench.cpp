// C++ / ONNX Runtime engine adapter for the RustLOB Phase 2 benchmark harness.
//
// Mirrors the session setup of cpp-engine/cpp_inference/benchmark.cpp (C++ API, CPU
// execution provider, zero-copy CreateTensor over the caller's buffer). benchmark.cpp sets
// ORT_ENABLE_ALL explicitly; that is ONNX Runtime's default level, so this adapter leaves it
// unset, like the Rust and Python engines, and the three engines share one configuration label.
// and implements the protocol and report schema documented in bench/README.md:
// it reads the exact tensors exported by `lob-inference export-windows`.
//
// Usage:
//   rustlob_cpp_bench bench     <model.onnx> <windows.npy> <model-name> <threads|default> <first|cycle> <warmup> <iters> <out.json>
//   rustlob_cpp_bench logits    <model.onnx> <windows.npy> <threads|default> <out.npy>
//   rustlob_cpp_bench coldstart <model.onnx> <windows.npy>
//
// "predict" path timed per call (samples_ms): CreateTensor view over the window,
// Session::Run, read 3 logits, finite check, argmax. session_run_samples_ms: Session::Run alone.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace {

using Clock = std::chrono::steady_clock;
constexpr size_t kWindowValues = 1 * 100 * 40;

struct Windows {
  std::vector<float> data;
  size_t count = 0;
};

// Minimal NPY v1/v2 reader for a little-endian float32 C-order array [N,1,100,40].
Windows ReadWindows(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + path);
  char magic[6];
  f.read(magic, 6);
  if (std::string(magic, 6) != "\x93NUMPY") throw std::runtime_error("not an NPY file: " + path);
  unsigned char version[2];
  f.read(reinterpret_cast<char*>(version), 2);
  uint32_t header_len = 0;
  if (version[0] == 1) {
    uint16_t len16 = 0;
    f.read(reinterpret_cast<char*>(&len16), 2);
    header_len = len16;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);
  if (header.find("'<f4'") == std::string::npos) throw std::runtime_error("expected <f4 dtype in " + path);
  if (header.find("'fortran_order': False") == std::string::npos) throw std::runtime_error("expected C order in " + path);
  const auto l = header.find('(');
  const auto r = header.find(')', l);
  std::stringstream shape(header.substr(l + 1, r - l - 1));
  std::vector<size_t> dims;
  std::string token;
  while (std::getline(shape, token, ',')) {
    if (token.find_first_not_of(' ') != std::string::npos) dims.push_back(std::stoul(token));
  }
  if (dims.size() != 4 || dims[1] != 1 || dims[2] != 100 || dims[3] != 40) {
    throw std::runtime_error("expected shape [N,1,100,40] in " + path);
  }
  Windows w;
  w.count = dims[0];
  w.data.resize(w.count * kWindowValues);
  f.read(reinterpret_cast<char*>(w.data.data()), static_cast<std::streamsize>(w.data.size() * sizeof(float)));
  if (!f) throw std::runtime_error("short read in " + path);
  return w;
}

void WriteLogitsNpy(const std::string& path, const std::vector<float>& logits, size_t rows) {
  std::ifstream exists(path);
  if (exists.good()) throw std::runtime_error("refusing to overwrite " + path);
  std::string header = "{'descr': '<f4', 'fortran_order': False, 'shape': (" + std::to_string(rows) + ", 3), }";
  const size_t total = 10 + header.size() + 1;
  header.append((64 - total % 64) % 64, ' ');
  header.push_back('\n');
  std::ofstream f(path, std::ios::binary);
  f.write("\x93NUMPY\x01\x00", 8);
  const uint16_t len = static_cast<uint16_t>(header.size());
  f.write(reinterpret_cast<const char*>(&len), 2);
  f.write(header.data(), static_cast<std::streamsize>(header.size()));
  f.write(reinterpret_cast<const char*>(logits.data()), static_cast<std::streamsize>(logits.size() * sizeof(float)));
}

std::string Fnv1a64(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  uint64_t hash = 0xcbf29ce484222325ULL;
  char buffer[16384];
  while (f.read(buffer, sizeof(buffer)) || f.gcount() > 0) {
    for (std::streamsize i = 0; i < f.gcount(); ++i) {
      hash ^= static_cast<unsigned char>(buffer[i]);
      hash *= 0x100000001b3ULL;
    }
  }
  std::ostringstream out;
  out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0') << hash;
  return out.str();
}

std::string SessionOptionsText(const std::string& threads) {
  return "intra_op_threads=" + (threads == "default" ? std::string("ort-default") : threads) +
         "; inter_op_threads=ort-default; graph_optimization=ort-default; intra_op_spinning=ort-default; device_policy=cpu";
}

struct Engine {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "rustlob_cpp_bench"};
  Ort::Session session{nullptr};
  Ort::MemoryInfo memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::string input_name, output_name;
  double load_ms = 0.0;

  Engine(const std::string& model, const std::string& threads) {
    Ort::SessionOptions options;
    if (threads != "default") options.SetIntraOpNumThreads(std::stoi(threads));
    const auto t0 = Clock::now();
    session = Ort::Session(env, model.c_str(), options);
    load_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    Ort::AllocatorWithDefaultOptions allocator;
    input_name = session.GetInputNameAllocated(0, allocator).get();
    output_name = session.GetOutputNameAllocated(0, allocator).get();
  }

  // Returns argmax; fills logits[3]; run_ms = Session::Run only.
  int Predict(float* window, float logits[3], double& run_ms) {
    const std::array<int64_t, 4> shape = {1, 1, 100, 40};
    Ort::Value input = Ort::Value::CreateTensor<float>(memory, window, kWindowValues, shape.data(), shape.size());
    const char* in_names[] = {input_name.c_str()};
    const char* out_names[] = {output_name.c_str()};
    const auto t0 = Clock::now();
    auto outputs = session.Run(Ort::RunOptions{nullptr}, in_names, &input, 1, out_names, 1);
    run_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    const auto info = outputs.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementCount() != 3) throw std::runtime_error("expected 3 output values");
    const float* out = outputs.front().GetTensorData<float>();
    for (int c = 0; c < 3; ++c) {
      if (!std::isfinite(out[c])) throw std::runtime_error("non-finite logit");
      logits[c] = out[c];
    }
    return static_cast<int>(std::max_element(logits, logits + 3) - logits);
  }
};

void JsonArray(std::ostream& out, const std::vector<double>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) out << (i ? "," : "") << std::setprecision(17) << values[i];
  out << "]";
}

int Bench(int argc, char** argv) {
  if (argc != 10) throw std::runtime_error("bench needs 8 arguments");
  const std::string model = argv[2], windows_path = argv[3], model_name = argv[4], threads = argv[5],
                    policy = argv[6], out_path = argv[9];
  const int warmup = std::stoi(argv[7]), iters = std::stoi(argv[8]);
  if (std::ifstream(out_path).good()) throw std::runtime_error("refusing to overwrite " + out_path);
  Windows w = ReadWindows(windows_path);
  const size_t used = policy == "first" ? 1 : w.count;
  Engine engine(model, threads);
  float logits[3];
  double run_ms = 0.0;
  for (int i = 0; i < warmup; ++i) engine.Predict(&w.data[(i % used) * kWindowValues], logits, run_ms);
  std::vector<double> samples, run_samples;
  samples.reserve(iters);
  run_samples.reserve(iters);
  const auto trial_start = Clock::now();
  for (int i = 0; i < iters; ++i) {
    const auto t0 = Clock::now();
    engine.Predict(&w.data[(i % used) * kWindowValues], logits, run_ms);
    samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    run_samples.push_back(run_ms);
  }
  const double elapsed = std::chrono::duration<double>(Clock::now() - trial_start).count();
  const std::string options = SessionOptionsText(threads);
  std::ofstream out(out_path);
  out << "{\"metadata\":{"
      << "\"model\":\"" << model_name << "\",\"model_file\":\"" << model << "\",\"model_file_hash\":\""
      << Fnv1a64(model) << "\",\"mode\":\"inference\",\"input\":\"" << windows_path
      << "\",\"input_shape\":\"[1, 1, 100, 40]\",\"input_type\":\"float32\",\"execution_provider\":\"cpu\","
      << "\"runtime_initialization_ms\":0.0,\"initial_model_load_ms\":" << engine.load_ms
      << ",\"warmup_iterations\":" << warmup << ",\"measured_iterations\":" << iters << ",\"trial_count\":1,"
      << "\"runtime_version\":\"onnxruntime C++ " << OrtGetApiBase()->GetVersionString() << "\","
      << "\"runtime_build_info\":\"" << Ort::GetBuildInfoString() << "\","
      << "\"rust_version\":null,\"build_mode\":\"release (CMake Release, -O3)\",\"operating_system\":\"macos aarch64\","
      << "\"cpu\":null,\"ram_bytes\":null,\"thread_configuration\":\"" << options << "\",\"git_commit\":null,"
      << "\"timestamp_unix_seconds\":" << std::time(nullptr) << ",\"memory\":null,\"cpu_utilization_percent\":null,"
      << "\"cpu_peak_percent\":null,\"session_options\":\"" << options << "\",\"input_policy\":\"" << policy
      << "\",\"windows_used\":" << used << ",\"load_average\":null},"
      << "\"trials\":[{\"elapsed_seconds\":" << std::setprecision(17) << elapsed
      << ",\"completed_windows\":" << iters << ",\"throughput_per_second\":" << (iters / elapsed)
      << ",\"samples_ms\":";
  JsonArray(out, samples);
  out << ",\"session_run_samples_ms\":";
  JsonArray(out, run_samples);
  out << ",\"statistics\":null,\"session_run_statistics\":null,\"preprocessing_stages\":null}],\"aggregate\":null}\n";
  return 0;
}

int Logits(int argc, char** argv) {
  if (argc != 6) throw std::runtime_error("logits needs 4 arguments");
  Windows w = ReadWindows(argv[3]);
  Engine engine(argv[2], argv[4]);
  std::vector<float> all(w.count * 3);
  double run_ms = 0.0;
  for (size_t i = 0; i < w.count; ++i) engine.Predict(&w.data[i * kWindowValues], &all[i * 3], run_ms);
  WriteLogitsNpy(argv[5], all, w.count);
  std::cout << "saved logits (" << w.count << ", 3) -> " << argv[5] << "\n";
  return 0;
}

int ColdStart(int argc, char** argv) {
  if (argc != 4) throw std::runtime_error("coldstart needs 2 arguments");
  Windows w = ReadWindows(argv[3]);
  Engine engine(argv[2], "default");
  float logits[3];
  double run_ms = 0.0;
  const int pred = engine.Predict(w.data.data(), logits, run_ms);
  std::cout << "{\"model_load_ms\":" << engine.load_ms << ",\"first_inference_ms\":" << run_ms
            << ",\"class\":" << pred << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "";
    if (mode == "bench") return Bench(argc, argv);
    if (mode == "logits") return Logits(argc, argv);
    if (mode == "coldstart") return ColdStart(argc, argv);
    std::cerr << "usage: rustlob_cpp_bench bench|logits|coldstart ...\n";
    return 2;
  } catch (const Ort::Exception& e) {
    std::cerr << "ONNX Runtime error: " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
