// Copyright 2021 Yan Yan
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Fused sparse depthwise convolution CUDA kernels.
//
// Single pass gather -> per-channel multiply -> scatter-add, no [nhot, C]
// intermediate buffer. For a single kernel offset the (input -> output) map is
// a translation, hence injective: within one launch every written address is
// distinct, so the forward and grad_input scatters need NO atomics
// (accumulation across offsets is safe because launches are sequential on the
// stream). Only the weight gradient (a reduction over all pairs into C values)
// uses atomicAdd.
//
// This single .cu is shared by the JIT path (torch.utils.cpp_extension.load)
// and the AOT build (setuptools CUDAExtension), via TORCH_EXTENSION_NAME.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/Atomic.cuh>

template <typename scalar_t>
__global__ void dw_fwd_kernel(scalar_t* __restrict__ out,
                              const scalar_t* __restrict__ feat,
                              const scalar_t* __restrict__ w,
                              const int* __restrict__ in_inds,
                              const int* __restrict__ out_inds,
                              long nhot, long C) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = nhot * C;
  if (tid >= total) return;
  long j = tid / C;
  long c = tid - j * C;
  long ii = in_inds[j];
  long oo = out_inds[j];
  out[oo * C + c] += feat[ii * C + c] * w[c];  // distinct addr per launch
}

template <typename scalar_t>
__global__ void dw_bwd_kernel(scalar_t* __restrict__ grad_feat,
                              scalar_t* __restrict__ grad_w,
                              const scalar_t* __restrict__ feat,
                              const scalar_t* __restrict__ grad_out,
                              const scalar_t* __restrict__ w,
                              const int* __restrict__ in_inds,
                              const int* __restrict__ out_inds,
                              long nhot, long C) {
  long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long total = nhot * C;
  if (tid >= total) return;
  long j = tid / C;
  long c = tid - j * C;
  long ii = in_inds[j];
  long oo = out_inds[j];
  scalar_t go = grad_out[oo * C + c];
  grad_feat[ii * C + c] += go * w[c];               // in_idx distinct per launch
  gpuAtomicAdd(&grad_w[c], feat[ii * C + c] * go);  // reduction -> atomic
}

static inline int mirror_nhot(const int* pnum, int kv, int kv_center,
                              bool subm, int i) {
  int nhot = pnum[i];
  if (subm && i > kv_center) nhot = pnum[kv - i - 1];
  return nhot;
}

void depthwise_forward(at::Tensor out, at::Tensor feat, at::Tensor filt,
                       at::Tensor pair_in, at::Tensor pair_out,
                       at::Tensor pnum_cpu, bool subm) {
  int kv = pair_in.size(0);
  long maxp = pair_in.size(1);
  long C = feat.size(1);
  int kv_center = kv / 2;
  const int* pnum = pnum_cpu.data_ptr<int>();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* pin = pair_in.data_ptr<int>();
  const int* pout = pair_out.data_ptr<int>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, feat.scalar_type(),
      "depthwise_forward", [&] {
        for (int i = 0; i < kv; ++i) {
          if (subm && i == kv_center) continue;
          int nhot = mirror_nhot(pnum, kv, kv_center, subm, i);
          if (nhot <= 0) continue;
          long total = (long)nhot * C;
          const int threads = 256;
          long blocks = (total + threads - 1) / threads;
          dw_fwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              out.data_ptr<scalar_t>(), feat.data_ptr<scalar_t>(),
              filt.data_ptr<scalar_t>() + (long)i * C, pin + (long)i * maxp,
              pout + (long)i * maxp, nhot, C);
        }
      });
}

void depthwise_backward(at::Tensor grad_feat, at::Tensor grad_filt,
                        at::Tensor feat, at::Tensor grad_out, at::Tensor filt,
                        at::Tensor pair_in, at::Tensor pair_out,
                        at::Tensor pnum_cpu, bool subm) {
  int kv = pair_in.size(0);
  long maxp = pair_in.size(1);
  long C = feat.size(1);
  int kv_center = kv / 2;
  const int* pnum = pnum_cpu.data_ptr<int>();
  auto stream = at::cuda::getCurrentCUDAStream();
  const int* pin = pair_in.data_ptr<int>();
  const int* pout = pair_out.data_ptr<int>();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, feat.scalar_type(),
      "depthwise_backward", [&] {
        for (int i = 0; i < kv; ++i) {
          if (subm && i == kv_center) continue;
          int nhot = mirror_nhot(pnum, kv, kv_center, subm, i);
          if (nhot <= 0) continue;
          long total = (long)nhot * C;
          const int threads = 256;
          long blocks = (total + threads - 1) / threads;
          dw_bwd_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
              grad_feat.data_ptr<scalar_t>(),
              grad_filt.data_ptr<scalar_t>() + (long)i * C,
              feat.data_ptr<scalar_t>(), grad_out.data_ptr<scalar_t>(),
              filt.data_ptr<scalar_t>() + (long)i * C, pin + (long)i * maxp,
              pout + (long)i * maxp, nhot, C);
        }
      });
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("depthwise_forward", &depthwise_forward, "fused depthwise forward");
  m.def("depthwise_backward", &depthwise_backward, "fused depthwise backward");
}
