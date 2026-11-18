// A transformer language model file with random weights (carbon/tlm.h's layout), for the tests of its CPU and
// GPU runs: the same model, for a seed, in each precision.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "carbon/cpu.h"
#include "carbon/tlm.h"

namespace carbon {

// The file in the temp folder: `classes` the recognizer's (tokens classes + 1), its vocabulary hash `vocab_hash`.
// INT8's inputs take fixed ranges, about the random model's own: the layer norms' outputs and the attention's
// within +-4, GELU's within -0.2 .. 5 (values past them are clipped, as a calibration's would clip some).
inline std::filesystem::path random_model(const std::string& name, int d, int layers, int heads, int ffn, int ctx,
                                          int classes, uint32_t seed, uint64_t vocab_hash = 0,
                                          TlmPrecision precision = TlmPrecision::kF32) {
  TlmHeader h{};
  std::memcpy(h.magic, kTlmMagic, sizeof(kTlmMagic));
  h.version = precision == TlmPrecision::kF32 ? 1 : 2;
  h.precision = precision;
  h.d = d, h.layers = layers, h.heads = heads, h.ffn = ffn, h.ctx = ctx;
  h.tokens = classes + 1, h.classes = classes;
  h.vocab_hash = vocab_hash;
  const bool f16 = precision == TlmPrecision::kF16, i8 = precision == TlmPrecision::kInt8;
  std::mt19937 g(seed);
  std::normal_distribution<float> n(0.0f, 1.0f);
  std::vector<char> body;
  size_t params = 0;
  auto put = [&](const void* p, size_t bytes) {
    const char* c = static_cast<const char*>(p);
    body.insert(body.end(), c, c + bytes);
  };
  auto draw = [&](size_t count, float scale, float mean = 0.0f) {
    std::vector<float> v(count);
    for (float& x : v) x = mean + scale * n(g);
    params += count;
    return v;
  };
  auto plain = [&](const std::vector<float>& v) {
    if (f16) {
      for (const float x : v) {
        const uint16_t b = cpu::f32_to_f16(x);
        put(&b, 2);
      }
    } else {
      put(v.data(), 4 * v.size());
    }
  };
  // a weight [k][m], INT8's columns symmetric (max |w| / 127), then its input's (scale, zero point) for [lo, hi]
  auto weight = [&](const std::vector<float>& v, size_t k, size_t m, float lo, float hi) {
    if (!i8) return plain(v);
    std::vector<float> s(m, 0.0f);
    for (size_t r = 0; r < k; ++r) {
      for (size_t j = 0; j < m; ++j) s[j] = std::max(s[j], std::abs(v[r * m + j]));
    }
    for (float& x : s) x = std::max(x / 127.0f, 1e-8f);
    std::vector<int8_t> q(k * m);
    for (size_t r = 0; r < k; ++r) {
      for (size_t j = 0; j < m; ++j) {
        q[r * m + j] = static_cast<int8_t>(std::clamp(std::nearbyint(v[r * m + j] / s[j]), -127.0f, 127.0f));
      }
    }
    put(q.data(), q.size());
    put(s.data(), 4 * s.size());
    const float lo0 = std::min(lo, 0.0f), hi0 = std::max(hi, 0.0f);
    const float sc = std::max((hi0 - lo0) / 255.0f, 1e-8f);
    const float iq[2] = {sc, std::clamp(std::nearbyint(-lo0 / sc), 0.0f, 255.0f)};
    put(iq, sizeof(iq));
  };
  const size_t D = static_cast<size_t>(d), F = static_cast<size_t>(ffn), T = static_cast<size_t>(h.tokens);
  const float sd = 1.0f / std::sqrt(static_cast<float>(d)), sf = 1.0f / std::sqrt(static_cast<float>(ffn));
  const std::vector<float> emb = draw(T * D, 0.5f);
  if (i8) {  // per token: the head's output channels
    std::vector<float> s(T);
    std::vector<int8_t> q(T * D);
    for (size_t t = 0; t < T; ++t) {
      float a = 0;
      for (size_t j = 0; j < D; ++j) a = std::max(a, std::abs(emb[t * D + j]));
      s[t] = std::max(a / 127.0f, 1e-8f);
      for (size_t j = 0; j < D; ++j) {
        q[t * D + j] = static_cast<int8_t>(std::clamp(std::nearbyint(emb[t * D + j] / s[t]), -127.0f, 127.0f));
      }
    }
    put(q.data(), q.size());
    put(s.data(), 4 * s.size());
  } else {
    plain(emb);
  }
  plain(draw(static_cast<size_t>(ctx) * D, 0.2f));  // the positions
  for (int l = 0; l < layers; ++l) {
    plain(draw(D, 0.1f, 1.0f)), plain(draw(D, 0.1f));                         // ln1
    weight(draw(D * 3 * D, sd), D, 3 * D, -4, 4), plain(draw(3 * D, 0.05f));  // qkv
    weight(draw(D * D, sd), D, D, -4, 4), plain(draw(D, 0.05f));              // out
    plain(draw(D, 0.1f, 1.0f)), plain(draw(D, 0.1f));                         // ln2
    weight(draw(D * F, sd), D, F, -4, 4), plain(draw(F, 0.05f));              // ffn1
    weight(draw(F * D, sf), F, D, -0.2f, 5), plain(draw(D, 0.05f));           // ffn2
  }
  plain(draw(D, 0.1f, 1.0f)), plain(draw(D, 0.1f));  // the final ln
  if (i8) {  // the head's input, within +-4
    const float sc = 8.0f / 255.0f, iq[2] = {sc, std::nearbyint(4.0f / sc)};
    put(iq, sizeof(iq));
  }
  h.params = params;
  const auto file = std::filesystem::temp_directory_path() /
                    ("carbon-" + name + "." + precision_name(precision) + ".tlm.bin");
  std::ofstream f(file, std::ios::binary);
  f.write(reinterpret_cast<const char*>(&h), sizeof(h));
  f.write(body.data(), static_cast<std::streamsize>(body.size()));
  return file;
}

}  // namespace carbon
