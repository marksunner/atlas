// SPDX-License-Identifier: AGPL-3.0-only

// SwiGLU pre-activation clamp (Step 3.7 `swiglu_limits` / `swiglustep`).
//
// The reference computes silu(gate).clamp(max=limit) * up.clamp(-limit, limit)
// (vLLM SwigluStepAndMul). Atlas fuses silu*up inside the expert down
// kernels, so instead of threading a limit through every fused variant this
// kernel clamps the gate/up GEMM outputs in place BEFORE the fused
// activation:
//
//   gate: clamp(silu(g), max=L) == silu(min(g, x*)) with silu(x*) = L,
//         exact because silu is monotone on x >= 0 and x* > 0.
//   up:   plain clamp to [-L, L].
//
// `gate_max` is the pre-activation threshold x* (computed host-side),
// NOT the limit L itself.
//
// Grid: (ceil(n / 256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void swiglu_clamp_bf16(
    __nv_bfloat16* __restrict__ gate,  // [n] in place
    __nv_bfloat16* __restrict__ up,    // [n] in place
    float gate_max,                    // pre-silu threshold x*
    float up_limit,                    // symmetric up clamp L
    unsigned int n
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = __bfloat162float(gate[idx]);
    if (g > gate_max) gate[idx] = __float2bfloat16(gate_max);

    float u = __bfloat162float(up[idx]);
    float uc = fminf(fmaxf(u, -up_limit), up_limit);
    if (uc != u) up[idx] = __float2bfloat16(uc);
}
