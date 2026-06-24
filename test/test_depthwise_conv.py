# Copyright 2021 Yan Yan
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Validate the ultra fast sparse depthwise convolution.

A depthwise convolution is mathematically identical to a regular (dense
channel mixing) convolution whose per-kernel-offset weight matrix is the
diagonal ``diag(w)``. We therefore validate ``SubMConvDepthwise3d`` /
``SparseConvDepthwise3d`` against the already well tested ``SubMConv3d`` /
``SparseConv3d`` configured with such a block-diagonal weight. This checks both
the forward pass and (via backprop) the gradients, while reusing the exact same
rulebook so the comparison is apples to apples.
"""

import unittest

import numpy as np
import torch

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.constants import ALL_WEIGHT_IS_KRSC
from spconv.test_utils import TestCase, generate_sparse_data


def _diag_weight_from_depthwise(depth_weight: torch.Tensor) -> torch.Tensor:
    """[C, *ksize, 1] depthwise weight -> [C, *ksize, C] block-diagonal."""
    C = depth_weight.shape[0]
    ksize = list(depth_weight.shape[1:-1])
    kv = int(np.prod(ksize))
    flat = depth_weight.reshape(C, kv)  # [C, kv]
    reg = torch.zeros((C, kv, C), dtype=depth_weight.dtype,
                      device=depth_weight.device)
    idx = torch.arange(C, device=depth_weight.device)
    # reg[o, i, o] = flat[o, i]
    reg[idx, :, idx] = flat
    return reg.reshape(C, *ksize, C).contiguous()


class SparseDepthwiseConv3dTest(TestCase):
    def _run_case(self, subm: bool, stride=1, padding=1, channels=16,
                  kernel_size=3):
        if not torch.cuda.is_available():
            self.skipTest("depthwise conv comparison needs CUDA")
        if not ALL_WEIGHT_IS_KRSC:
            self.skipTest("reference comparison assumes KRSC weight layout")
        device = torch.device("cuda:0")
        np.random.seed(484)
        torch.manual_seed(484)
        shape = [20, 20, 20]
        num_points = [1500]
        data = generate_sparse_data(shape, num_points, channels,
                                    with_dense=False)
        features = torch.from_numpy(data["features"]).to(device).float()
        indices = torch.from_numpy(data["indices"]).to(device).int()
        bs = 1

        if subm:
            dw = spconv.SubMConvDepthwise3d(channels, kernel_size,
                                            padding=padding, bias=True,
                                            indice_key="dw").to(device)
            ref = spconv.SubMConv3d(channels, channels, kernel_size,
                                    padding=padding, bias=True,
                                    indice_key="ref",
                                    algo=ConvAlgo.Native).to(device)
        else:
            dw = spconv.SparseConvDepthwise3d(channels, kernel_size,
                                              stride=stride, padding=padding,
                                              bias=True).to(device)
            ref = spconv.SparseConv3d(channels, channels, kernel_size,
                                      stride=stride, padding=padding,
                                      bias=True,
                                      algo=ConvAlgo.Native).to(device)

        # share weights: ref gets the block-diagonal of the depthwise weight.
        with torch.no_grad():
            ref.weight.copy_(_diag_weight_from_depthwise(dw.weight.data))
            ref.bias.copy_(dw.bias.data)

        feat_dw = features.clone().requires_grad_(True)
        feat_ref = features.clone().requires_grad_(True)

        x_dw = spconv.SparseConvTensor(feat_dw, indices, shape, bs)
        x_ref = spconv.SparseConvTensor(feat_ref, indices, shape, bs)

        out_dw = dw(x_dw)
        out_ref = ref(x_ref)

        # output indices must match (same rulebook / spatial structure)
        self.assertAllEqual(out_dw.indices.cpu().numpy(),
                            out_ref.indices.cpu().numpy())
        self.assertAllClose(out_dw.features.detach().cpu().numpy(),
                            out_ref.features.detach().cpu().numpy(),
                            atol=1e-4)

        # backward
        g = torch.randn_like(out_dw.features)
        out_dw.features.backward(g)
        out_ref.features.backward(g.clone())

        self.assertAllClose(feat_dw.grad.cpu().numpy(),
                            feat_ref.grad.cpu().numpy(), atol=1e-4)
        # depthwise weight grad equals the diagonal of the reference weight grad
        ref_diag = _diag_weight_from_depthwise(
            torch.ones_like(dw.weight)) * ref.weight.grad
        C = channels
        kv = int(np.prod(dw.weight.shape[1:-1]))
        idx = torch.arange(C, device=device)
        ref_w_grad_diag = ref.weight.grad.reshape(C, kv, C)[idx, :, idx]
        self.assertAllClose(dw.weight.grad.reshape(C, kv).cpu().numpy(),
                            ref_w_grad_diag.cpu().numpy(), atol=1e-4)
        self.assertAllClose(dw.bias.grad.cpu().numpy(),
                            ref.bias.grad.cpu().numpy(), atol=1e-4)

    def test_subm_depthwise(self):
        self._run_case(subm=True, stride=1, padding=1)

    def test_sparse_depthwise_stride1(self):
        self._run_case(subm=False, stride=1, padding=1)

    def test_sparse_depthwise_stride2(self):
        self._run_case(subm=False, stride=2, padding=1)


if __name__ == "__main__":
    unittest.main()
