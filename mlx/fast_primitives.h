// Copyright © 2024 Apple Inc.

#include <optional>

#include "mlx/primitives.h"
#include <iostream>

namespace mlx::core::fast {

// Custom primitive accepts a fallback function which it uses for
// transformations. Transformations are virtual so that derived classes may
// override the default behavior.
class Custom : public Primitive {
 public:
  explicit Custom(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback)
      : Primitive(stream), fallback_(fallback) {}

  virtual std::pair<std::vector<array>, std::vector<int>> vmap(
      const std::vector<array>& inputs,
      const std::vector<int>& axes) override;

  virtual std::vector<array> jvp(
      const std::vector<array>& primals,
      const std::vector<array>& tangents,
      const std::vector<int>& argnums) override;

  virtual std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

 private:
  std::function<std::vector<array>(std::vector<array>)> fallback_;
};

class RMSNorm : public Custom {
 public:
  RMSNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps) {}

  static bool use_fallback(Stream stream);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(RMSNorm)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()

  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class RMSNormVJP : public Custom {
 public:
  RMSNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(RMSNormVJP)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class LayerNorm : public Custom {
 public:
  LayerNorm(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps) {}

  static bool use_fallback(Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(LayerNorm)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class LayerNormVJP : public Custom {
 public:
  LayerNormVJP(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      float eps)
      : Custom(stream, fallback), eps_(eps) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(LayerNormVJP)
  bool is_equivalent(const Primitive& other) const override;
  auto state() const {
    return std::make_pair(nullptr, eps_);
  }

 private:
  float eps_;
};

class RoPE : public Custom {
 public:
  RoPE(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int dims,
      bool traditional,
      float base,
      float scale,
      bool forward)
      : Custom(stream, fallback),
        dims_(dims),
        traditional_(traditional),
        base_(base),
        scale_(scale),
        forward_(forward) {}

  static bool use_fallback(Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  std::vector<array> vjp(
      const std::vector<array>& primals,
      const std::vector<array>& cotangents,
      const std::vector<int>& argnums,
      const std::vector<array>& outputs) override;

  DEFINE_PRINT(RoPE)
  bool is_equivalent(const Primitive& other) const override;
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(
        nullptr, dims_, traditional_, base_, scale_, forward_);
  }

 private:
  int dims_;
  bool traditional_;
  float base_;
  float scale_;
  bool forward_;
};

class ScaledDotProductAttention : public Custom {
 public:
  explicit ScaledDotProductAttention(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      const float scale,
      const bool do_causal)
      : Custom(stream, fallback), scale_(scale), do_causal_(do_causal) {}

  static bool use_fallback(
      const array& q,
      const array& k,
      const array& v,
      bool has_mask,
      bool has_arr_mask,
      bool do_causal,
      Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    eval_gpu(inputs, outputs[0]);
  }

  void eval_gpu(const std::vector<array>& inputs, array& out);
  bool is_equivalent(const Primitive& other) const override;

  DEFINE_PRINT(ScaledDotProductAttention);
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, scale_, do_causal_);
  }

 private:
  float scale_;
  bool do_causal_;
};

class InfLLMV2AttentionStage1 : public Custom {
 public:
  explicit InfLLMV2AttentionStage1(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      const float scale,
      const bool do_causal)
      : Custom(stream, fallback), scale_(scale), do_causal_(do_causal) {}

  static bool use_fallback(
      const array& q,
      const array& k,
      const array& v,
      bool has_mask,
      bool has_arr_mask,
      bool do_causal,
      Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    std::cout << "[DEBUG ZWL] " << __FILE__ << " : " << __LINE__ << " InfLLMV2AttentionStage1::eval_gpu" << std::endl;
    eval_gpu(inputs, outputs[0]);
  }

  void eval_gpu(const std::vector<array>& inputs, array& out);
  bool is_equivalent(const Primitive& other) const override;

  DEFINE_PRINT(InfLLMV2AttentionStage1);
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, scale_, do_causal_);
  }

 private:
  float scale_;
  bool do_causal_;
};

class InfLLMV2AttentionStage2 : public Custom {
 public:
  explicit InfLLMV2AttentionStage2(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      const array& cu_seqlens_q,
      const array& cu_seqlens_k,
      const int max_seqlen_q,
      const int max_seqlen_k,
      const int window_size_left,
      const int window_size_right,
      const array& blockmask_uint64,
      const int block_window_size,
      const float scale,
      const bool do_causal)
      : Custom(stream, fallback), scale_(scale), do_causal_(do_causal), cu_seqlens_q_(cu_seqlens_q), cu_seqlens_k_(cu_seqlens_k), max_seqlen_q_(max_seqlen_q), max_seqlen_k_(max_seqlen_k), window_size_left_(window_size_left), window_size_right_(window_size_right), blockmask_uint64_(blockmask_uint64), block_window_size_(block_window_size) {}

  static bool use_fallback(
      const array& q,
      const array& k,
      const array& v,
      const array& cu_seqlens_q,
      const array& cu_seqlens_k,
      const int max_seqlen_q,
      const int max_seqlen_k,
      const int window_size_left,
      const int window_size_right,
      const array& blockmask_uint64,
      const int block_window_size,
      bool has_mask,
      bool has_arr_mask,
      bool do_causal,
      Stream s);

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("NYI");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    // std::cout << "[DEBUG ZWL] " << __FILE__ << " : " << __LINE__ << " InfLLMV2AttentionStage2::eval_gpu" << std::endl;
    eval_gpu(inputs, outputs[0]);
  }

  void eval_gpu(const std::vector<array>& inputs, array& out);
  bool is_equivalent(const Primitive& other) const override;

  DEFINE_PRINT(InfLLMV2AttentionStage2);
  DEFINE_INPUT_OUTPUT_SHAPE()
  auto state() const {
    return std::make_tuple(nullptr, cu_seqlens_q_, cu_seqlens_k_, max_seqlen_q_, max_seqlen_k_, window_size_left_, window_size_right_, blockmask_uint64_, block_window_size_, scale_, do_causal_);
  }

 private:
  const array& cu_seqlens_q_;
  const array& cu_seqlens_k_;
  const int max_seqlen_q_;
  const int max_seqlen_k_;
  const int window_size_left_;
  const int window_size_right_;
  const array& blockmask_uint64_;
  const int block_window_size_;
  float scale_;
  bool do_causal_;
};

class AffineQuantize : public Custom {
 public:
  explicit AffineQuantize(
      Stream stream,
      std::function<std::vector<array>(std::vector<array>)> fallback,
      int group_size,
      int bits,
      bool dequantize)
      : Custom(stream, fallback),
        group_size_(group_size),
        bits_(bits),
        dequantize_(dequantize) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(AffineQuantize);

  bool is_equivalent(const Primitive& other) const override;
  std::vector<Shape> output_shapes(const std::vector<array>& inputs) override;
  auto state() const {
    return std::make_tuple(nullptr, group_size_, bits_, dequantize_);
  }

 private:
  int group_size_;
  int bits_;
  bool dequantize_;
};

struct CustomKernelShapeInfo {
  bool shape = false;
  bool strides = false;
  bool ndim = false;
};

class CustomKernel : public Primitive {
 public:
  CustomKernel(
      Stream stream,
      std::string name,
      std::string source,
      std::tuple<int, int, int> grid,
      std::tuple<int, int, int> threadgroup,
      std::vector<CustomKernelShapeInfo> shape_infos,
      bool ensure_row_contiguous,
      std::optional<float> init_value)
      : Primitive(stream),
        source_(std::move(source)),
        name_(std::move(name)),
        grid_(grid),
        threadgroup_(threadgroup),
        shape_infos_(std::move(shape_infos)),
        ensure_row_contiguous_(ensure_row_contiguous),
        init_value_(init_value) {}

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override {
    throw std::runtime_error("Custom Metal kernels only run on GPU.");
  }

  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs)
      override;

  DEFINE_PRINT(CustomKernel);

 private:
  std::string source_;
  std::string name_;
  std::tuple<int, int, int> grid_;
  std::tuple<int, int, int> threadgroup_;
  std::vector<CustomKernelShapeInfo> shape_infos_;
  bool ensure_row_contiguous_;
  std::optional<float> init_value_;
};

} // namespace mlx::core::fast
