// The engine's own executor (Backend::kNative): Fullerene's convolutional
// networks on the CPU kernels of carbon/cpu.h, read from their ONNX files.
#pragma once

#include <filesystem>
#include <memory>

#include "carbon/cpu.h"
#include "carbon/session.h"

namespace carbon::net {

// Throws carbon::Error for a file that does not parse or a node the executor
// does not run.  One thread a run: a worker per core (carbon/parallel.h).  An
// order model's weights (order.native.bin, order_net.cpp) open as that network.
// Files in FP32, in int8 (Q/DQ) and in FP16 (float16 weights, input and output:
// <model>.fp16.onnx) run; the session's input and outputs are float32 for every
// one of them.  An FP16 file keeps its tensors in FP16 between layers where the
// instruction set sums in FP16 (cpu::f16_arithmetic), else in FP32;
// CARBON_FP16_ACTIVATIONS=1 or 0 in the environment chooses (net.cpp).
std::unique_ptr<Session> open_native_session(const std::filesystem::path& model, const SessionOptions& options);

// The order model's weights file (order_net.cpp): whether `file` is one, and its network.
bool is_order_net(const std::filesystem::path& file);
std::unique_ptr<Session> open_order_net(const std::filesystem::path& file, cpu::Isa isa);

}  // namespace carbon::net
