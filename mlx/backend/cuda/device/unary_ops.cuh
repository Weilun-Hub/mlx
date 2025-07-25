// Copyright © 2025 Apple Inc.

#pragma once

#include "mlx/backend/cuda/device/cexpf.cuh"
#include "mlx/backend/cuda/device/cucomplex_math.cuh"
#include "mlx/backend/cuda/device/fp16_math.cuh"
#include "mlx/backend/cuda/device/utils.cuh"

#include <math_constants.h>

namespace mlx::core::cu {

struct Abs {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x;
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {sqrt(cuCrealf(x) * cuCrealf(x) + cuCimagf(x) * cuCimagf(x)), 0};
    } else {
      return abs(x);
    }
  }
};

struct ArcCos {
  template <typename T>
  __device__ T operator()(T x) {
    return acos(x);
  }

  __device__ cuComplex operator()(cuComplex x);
};

struct ArcCosh {
  template <typename T>
  __device__ T operator()(T x) {
    return acosh(x);
  }
};

struct ArcSin {
  template <typename T>
  __device__ T operator()(T x) {
    return asin(x);
  }

  __device__ cuComplex operator()(cuComplex x);
};

struct ArcSinh {
  template <typename T>
  __device__ T operator()(T x) {
    return asinh(x);
  }
};

struct ArcTan {
  template <typename T>
  __device__ T operator()(T x) {
    return atan(x);
  }

  __device__ cuComplex operator()(cuComplex x);
};

struct ArcTanh {
  template <typename T>
  __device__ T operator()(T x) {
    return atanh(x);
  }
};

struct BitwiseInvert {
  template <typename T>
  __device__ T operator()(T x) {
    return ~x;
  }
};

struct Ceil {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else {
      return ceil(x);
    }
  }
};

struct Conjugate {
  __device__ cuComplex operator()(cuComplex x) {
    return {cuCrealf(x), -cuCimagf(x)};
  }
};

struct Cos {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          cos(cuCrealf(x)) * cosh(cuCimagf(x)),
          -sin(cuCrealf(x)) * sinh(cuCimagf(x))};
    } else {
      return cos(x);
    }
  }
};

struct Cosh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          cosh(cuCrealf(x)) * cos(cuCimagf(x)),
          sinh(cuCrealf(x)) * sin(cuCimagf(x))};
    } else {
      return cosh(x);
    }
  }
};

struct Erf {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erf(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erf(__bfloat162float(x));
    } else {
      return erf(x);
    }
  }
};

struct ErfInv {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return erfinv(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return erfinv(__bfloat162float(x));
    } else {
      return erfinv(x);
    }
  }
};

struct Exp {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return detail::cexpf(x);
    } else {
      return exp(x);
    }
  }
};

struct Expm1 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, __half>) {
      return expm1(__half2float(x));
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return expm1(__bfloat162float(x));
    } else {
      return expm1(x);
    }
  }
};

struct Floor {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_integral_v<T>) {
      return x;
    } else {
      return floor(x);
    }
  }
};

struct Imag {
  __device__ float operator()(cuComplex x) {
    return cuCimagf(x);
  }
};

struct Log {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      auto r = log(cuCrealf(Abs{}(x)));
      auto i = atan2f(cuCimagf(x), cuCrealf(x));
      return {r, i};
    } else {
      return log(x);
    }
  }
};

struct Log2 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      auto y = Log{}(x);
      return {cuCrealf(y) / CUDART_LN2_F, cuCimagf(y) / CUDART_LN2_F};
    } else {
      return log2(x);
    }
  }
};

struct Log10 {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      auto y = Log{}(x);
      return {cuCrealf(y) / CUDART_LNT_F, cuCimagf(y) / CUDART_LNT_F};
      return y;
    } else {
      return log10(x);
    }
  }
};

struct Log1p {
  template <typename T>
  __device__ T operator()(T z) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      float x = cuCrealf(z);
      float y = cuCimagf(z);
      float zabs = cuCrealf(Abs{}(z));
      float theta = atan2f(y, x + 1);
      if (zabs < 0.5f) {
        float r = x * (2 + x) + y * y;
        if (r == 0) { // handle underflow
          return {x, theta};
        }
        return {0.5f * log1pf(r), theta};
      } else {
        float z0 = hypotf(x + 1, y);
        return {logf(z0), theta};
      }
    } else {
      return log1p(z);
    }
  }
};

struct LogicalNot {
  __device__ bool operator()(bool x) {
    return !x;
  }
};

struct Negative {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return 0 - x;
    } else {
      return -x;
    }
  }
};

struct Real {
  __device__ float operator()(cuComplex x) {
    return cuCrealf(x);
  }
};

struct Round {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {rint(cuCrealf(x)), rint(cuCimagf(x))};
    } else {
      return rint(x);
    }
  }
};

struct Sigmoid {
  template <typename T>
  __device__ T operator()(T x) {
    T y = 1 / (1 + exp(-abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

struct Sign {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_unsigned_v<T>) {
      return x != 0;
    } else if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      if (cuCrealf(x) == 0 && cuCimagf(x) == 0) {
        return x;
      } else {
        return x / Abs()(x);
      }
    } else if constexpr (cuda::std::is_same_v<T, __nv_bfloat16>) {
      return static_cast<float>((x > T(0.f)) - (x < T(0.f)));
    } else {
      return (x > T(0)) - (x < T(0));
    }
  }
};

struct Sin {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          sin(cuCrealf(x)) * cosh(cuCimagf(x)),
          cos(cuCrealf(x)) * sinh(cuCimagf(x))};
    } else {
      return sin(x);
    }
  }
};

struct Sinh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      return {
          sinh(cuCrealf(x)) * cos(cuCimagf(x)),
          cosh(cuCrealf(x)) * sin(cuCimagf(x))};
    } else {
      return sinh(x);
    }
  }
};

struct Square {
  template <typename T>
  __device__ T operator()(T x) {
    return x * x;
  }
};

struct Sqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return sqrt(x);
  }

  __device__ cuComplex operator()(cuComplex x) {
    auto xr = cuCrealf(x);
    auto xi = cuCimagf(x);
    if (xr == 0.0f && xi == 0.0f) {
      return {0.0f, 0.0f};
    }
    auto r = cuCrealf(Abs{}(x));
    auto a = sqrt((r + xr) / 2.0f);
    auto b_abs = sqrt((r - xr) / 2.0f);
    auto b = copysign(b_abs, xi);
    return {a, b};
  }
};

struct Rsqrt {
  template <typename T>
  __device__ T operator()(T x) {
    return rsqrt(x);
  }
  __device__ cuComplex operator()(cuComplex x) {
    return 1.0f / Sqrt{}(x);
  }
};

struct Tan {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      float tan_a = tan(cuCrealf(x));
      float tanh_b = tanh(cuCimagf(x));
      float t1 = tan_a * tanh_b;
      float denom = 1. + t1 * t1;
      return {(tan_a - tanh_b * t1) / denom, (tanh_b + tan_a * t1) / denom};
    } else {
      return tan(x);
    }
  }
};

struct Tanh {
  template <typename T>
  __device__ T operator()(T x) {
    if constexpr (cuda::std::is_same_v<T, cuComplex>) {
      float tanh_a = tanh(cuCrealf(x));
      float tan_b = tan(cuCimagf(x));
      float t1 = tanh_a * tan_b;
      float denom = 1. + t1 * t1;
      return {(tanh_a + tan_b * t1) / denom, (tan_b - tanh_a * t1) / denom};
    } else {
      return tanh(x);
    }
  }
};

inline __device__ cuComplex ArcCos::operator()(cuComplex x) {
  auto i = cuComplex{0.0, 1.0};
  auto y = Log{}(x + i * Sqrt{}(1.0 - x * x));
  return {cuCimagf(y), -cuCrealf(y)};
};

inline __device__ cuComplex ArcSin::operator()(cuComplex x) {
  auto i = cuComplex{0.0f, 1.0f};
  auto y = Log{}(i * x + Sqrt{}(1.0f - x * x));
  return {cuCimagf(y), -cuCrealf(y)};
};

inline __device__ cuComplex ArcTan::operator()(cuComplex x) {
  auto i = cuComplex{0.0f, 1.0f};
  auto ix = i * x;
  return (1.0f / cuComplex{0.0f, 2.0f}) * Log{}((1.0f + ix) / (1.0f - ix));
};

} // namespace mlx::core::cu
