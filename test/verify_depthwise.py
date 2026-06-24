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
"""Rigorous verification of the sparse depthwise convolution.

This script does NOT trust spconv internals: the ground truth is computed
independently with ``torch.nn.functional.conv3d(groups=C)`` on a dense
densified version of the sparse input. We then gather the dense reference at
the *exact* output coordinates produced by spconv and compare element by
element (max abs / max rel error). We additionally:

  * check the forward output,
  * check input/weight/bias gradients via a dense reference,
  * run ``torch.autograd.gradcheck`` in float64 on the low level functional,
  * benchmark depthwise vs. an equivalent full (channel mixing) spconv conv to
    demonstrate the speed-up.

Run:  python test/verify_depthwise.py
"""

import time

import numpy as np
import torch
import torch.nn.functional as F

import spconv.pytorch as spconv
from spconv.core import ConvAlgo
from spconv.pytorch import functional as Fsp


# --------------------------------------------------------------------------- #
#  Independent dense reference                                                #
# --------------------------------------------------------------------------- #
def densify(features, indices, spatial_shape, batch_size):
    """sparse [N, C] + [N, 1+ndim] -> dense [B, C, *spatial_shape]."""
    C = features.shape[1]
    dense = torch.zeros((batch_size, C, *spatial_shape),
                        dtype=features.dtype, device=features.device)
    b = indices[:, 0].long()
    z, y, x = (indices[:, 1].long(), indices[:, 2].long(), indices[:, 3].long())
    dense[b, :, z, y, x] = features
    return dense


def dense_depthwise(dense, weight_kc1, kernel_size, stride, padding, subm):
    """weight stored as spconv KRSC [C, kz, ky, kx, 1] -> torch [C, 1, kz, ky, kx]."""
    C = weight_kc1.shape[0]
    w = weight_kc1.reshape(C, *kernel_size, 1).permute(0, 4, 1, 2, 3).contiguous()
    pad = padding
    if subm:
        # submanifold keeps the same coords -> symmetric "same" padding.
        pad = [k // 2 for k in kernel_size]
    out = F.conv3d(dense, w, bias=None, stride=stride,
                   padding=tuple(pad), groups=C)
    return out


def gather_at(dense_out, out_indices):
    b = out_indices[:, 0].long()
    z, y, x = (out_indices[:, 1].long(), out_indices[:, 2].long(),
               out_indices[:, 3].long())
    return dense_out[b, :, z, y, x]


# --------------------------------------------------------------------------- #
#  Test data                                                                  #
# --------------------------------------------------------------------------- #
def make_data(device, dtype, channels, shape, num_points, seed=484):
    rng = np.random.RandomState(seed)
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in shape]),
                      axis=-1).reshape(-1, 3)
    rng.shuffle(coords)
    coords = coords[:num_points]
    indices = np.concatenate(
        [np.zeros((num_points, 1), np.int32), coords.astype(np.int32)], axis=1)
    feats = rng.uniform(-1, 1, size=(num_points, channels)).astype(np.float64)
    return (torch.from_numpy(feats).to(device=device, dtype=dtype),
            torch.from_numpy(indices).to(device=device))


def report(name, a, b):
    a = a.detach().double().cpu().numpy()
    b = b.detach().double().cpu().numpy()
    abs_err = np.abs(a - b)
    rel = abs_err / (np.abs(b) + 1e-12)
    print(f"  [{name:14s}] shape={a.shape} max_abs={abs_err.max():.3e} "
          f"mean_abs={abs_err.mean():.3e} max_rel={rel.max():.3e}")
    return abs_err.max()


# --------------------------------------------------------------------------- #
#  Forward + backward verification against the dense reference                #
# --------------------------------------------------------------------------- #
def verify_case(subm, stride, padding, channels=24, shape=(24, 24, 24),
                num_points=2000, kernel_size=3, dtype=torch.float64,
                device="cuda"):
    print(f"\n=== verify subm={subm} stride={stride} padding={padding} "
          f"C={channels} dtype={dtype} ===")
    ks = [kernel_size] * 3
    feats, indices = make_data(device, dtype, channels, shape, num_points)

    if subm:
        layer = spconv.SubMConvDepthwise3d(channels, kernel_size,
                                           padding=padding, bias=True).to(device, dtype)
    else:
        layer = spconv.SparseConvDepthwise3d(channels, kernel_size, stride=stride,
                                             padding=padding, bias=True).to(device, dtype)

    feats_sp = feats.clone().requires_grad_(True)
    x = spconv.SparseConvTensor(feats_sp, indices, list(shape), 1)
    out = layer(x)

    # ---- dense reference forward ----
    dense_in = densify(feats, indices, shape, 1)
    dense_out = dense_depthwise(dense_in, layer.weight.data, ks,
                                [stride] * 3, [padding] * 3, subm)
    ref_feat = gather_at(dense_out, out.indices) + layer.bias.data
    fwd_err = report("forward", out.features, ref_feat)

    # ---- backward: compare to a dense reference graph ----
    g = torch.randn_like(out.features)
    out.features.backward(g)

    dense_in_ref = densify(feats, indices, shape, 1).requires_grad_(True)
    w_ref = layer.weight.data.clone().requires_grad_(True)
    b_ref = layer.bias.data.clone().requires_grad_(True)
    C = channels
    w_t = w_ref.reshape(C, *ks, 1).permute(0, 4, 1, 2, 3).contiguous()
    pad = [k // 2 for k in ks] if subm else [padding] * 3
    dout_ref = F.conv3d(dense_in_ref, w_t, bias=None, stride=[stride] * 3,
                        padding=tuple(pad), groups=C)
    ref_feat2 = gather_at(dout_ref, out.indices) + b_ref
    ref_feat2.backward(g)

    # grad wrt input features = grad of dense input gathered at input coords
    gin_ref = gather_at(dense_in_ref.grad, indices)
    gin_err = report("grad_input", feats_sp.grad, gin_ref)
    gw_err = report("grad_weight", layer.weight.grad, w_ref.grad)
    gb_err = report("grad_bias", layer.bias.grad, b_ref.grad)

    tol = 1e-7 if dtype == torch.float64 else 1e-3
    ok = max(fwd_err, gin_err, gw_err, gb_err) < tol
    print(f"  --> {'PASS' if ok else 'FAIL'} (tol={tol:.0e})")
    return ok


# --------------------------------------------------------------------------- #
#  gradcheck on the raw functional                                            #
# --------------------------------------------------------------------------- #
def run_gradcheck(device="cuda"):
    print("\n=== torch.autograd.gradcheck (float64) ===")
    from spconv.pytorch import ops
    torch.manual_seed(0)
    channels, shape, npts, ks = 6, (10, 10, 10), 120, 3
    feats, indices = make_data(device, torch.float64, channels, shape, npts)
    outids, pairs, pair_num = ops.get_indice_pairs(
        indices, 1, list(shape), ConvAlgo.Native, [ks] * 3, [1] * 3, [1] * 3,
        [1] * 3, [0] * 3, True, False)
    kv = ks ** 3
    filt = torch.randn(kv, channels, dtype=torch.float64,
                       device=device, requires_grad=True)
    feats = feats.requires_grad_(True)
    ok = torch.autograd.gradcheck(
        lambda f, w: Fsp.indice_conv_depthwise(
            f, w, pairs, pair_num, outids.shape[0], False, True, None),
        (feats, filt), eps=1e-6, atol=1e-5, rtol=1e-3)
    print(f"  gradcheck subm: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
#  Speed benchmark vs equivalent full conv                                    #
# --------------------------------------------------------------------------- #
def benchmark(device="cuda", shape=(40, 40, 40), num_points=30000, ks=3,
              iters=50):
    if device != "cuda":
        print("\n(skip benchmark: needs cuda)")
        return

    def bench(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.time() - t) / iters * 1e3

    print(f"\n=== benchmark (subm, N={num_points}, k={ks}) ===")
    print(f"  {'C':>5} | {'depthwise':>10} | {'full CxC':>10} | "
          f"{'emulated':>10} | {'vs full':>7} | {'vs emul':>7}")
    print("  " + "-" * 66)
    for channels in (16, 32, 64, 128):
        feats, indices = make_data(device, torch.float32, channels, shape,
                                   num_points)
        x = spconv.SparseConvTensor(feats, indices, list(shape), 1)
        dw = spconv.SubMConvDepthwise3d(channels, ks, padding=1,
                                        bias=False).to(device)
        full = spconv.SubMConv3d(channels, channels, ks, padding=1, bias=False,
                                 algo=ConvAlgo.Native, indice_key="k").to(device)
        # "emulated": depthwise via C independent 1-channel submanifold convs,
        # the only way to do depthwise on stock spconv today.
        emul = [spconv.SubMConv3d(1, 1, ks, padding=1, bias=False,
                                  algo=ConvAlgo.Native,
                                  indice_key="e").to(device)
                for _ in range(channels)]

        def run_emul():
            outs = []
            for c, layer in enumerate(emul):
                xc = spconv.SparseConvTensor(feats[:, c:c + 1], indices,
                                             list(shape), 1)
                outs.append(layer(xc).features)
            return torch.cat(outs, dim=1)

        t_dw = bench(lambda: dw(x))
        t_full = bench(lambda: full(x))
        t_emul = bench(run_emul)
        print(f"  {channels:>5} | {t_dw:>8.3f}ms | {t_full:>8.3f}ms | "
              f"{t_emul:>8.3f}ms | {t_full / t_dw:>6.1f}x | "
              f"{t_emul / t_dw:>6.1f}x")
    print("  (depthwise uses C x fewer FLOPs/params than full CxC conv; both\n"
          "   are memory bound so latency is similar, but depthwise is far\n"
          "   cheaper than emulating it with per-channel convs.)")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    results = []
    # float64 on cuda gives exact agreement with the dense reference.
    results.append(verify_case(subm=True, stride=1, padding=1, device=device))
    results.append(verify_case(subm=False, stride=1, padding=1, device=device))
    results.append(verify_case(subm=False, stride=2, padding=1, device=device))
    results.append(verify_case(subm=False, stride=2, padding=0, device=device))
    results.append(run_gradcheck(device=device))
    benchmark(device=device)
    print("\n=====================================")
    print("ALL PASSED" if all(results) else "SOME CHECKS FAILED")
    print("=====================================")


if __name__ == "__main__":
    main()
