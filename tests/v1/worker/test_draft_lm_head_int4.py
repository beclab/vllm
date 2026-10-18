# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Int4 copy of a shared lm_head for the drafter (draft_lm_head_quantization)."""

import pytest
import torch

from vllm.model_executor.layers.quantization.online.lm_head import (
    INT4_GROUP_SIZE,
    _LMHeadInt4Method,
    dequantize_int4_gptq,
    quantize_int4_gptq,
)
from vllm.platforms import current_platform


def test_int4_quantize_round_trip_within_half_step():
    torch.manual_seed(0)
    w = torch.randn(300, 512, dtype=torch.float16)
    qweight, scales = quantize_int4_gptq(w, chunk_rows=128)
    assert qweight.shape == (512 // 8, 300) and qweight.dtype == torch.int32
    assert scales.shape == (512 // INT4_GROUP_SIZE, 300)
    w_hat = dequantize_int4_gptq(qweight, scales)
    step = scales.t().float().repeat_interleave(INT4_GROUP_SIZE, dim=1)
    assert ((w_hat.float() - w.float()).abs() <= step / 2 + 1e-3).all()


def test_int4_packing_is_low_nibble_first_uint4b8():
    w = torch.zeros(1, INT4_GROUP_SIZE, dtype=torch.float16)
    w[0, 0], w[0, 1], w[0, 7] = 7.0, -7.0, 1.0  # scale 1 -> codes 15, 1, 9
    qweight, _ = quantize_int4_gptq(w)
    word = int(qweight[0, 0]) & 0xFFFFFFFF
    assert [(word >> (4 * i)) & 0xF for i in range(8)] == [15, 1, 8, 8, 8, 8, 8, 9]


@pytest.mark.skipif(not current_platform.is_xpu(), reason="XPU W4A16 kernel")
@pytest.mark.parametrize("num_tokens", [1, 5, 40])
def test_int4_lm_head_kernel_matches_dequantized_reference(num_tokens):
    torch.manual_seed(0)
    vocab, hidden = 4096 + 64, 1024
    w = (torch.randn(vocab, hidden, device="xpu") * 0.02).half()
    x = torch.randn(num_tokens, hidden, device="xpu").half()
    layer = torch.nn.Module()
    method = _LMHeadInt4Method(layer, w)
    out = method.apply(layer, x)
    qweight, scales = quantize_int4_gptq(w)
    ref = x.float() @ dequantize_int4_gptq(qweight, scales).float().t()
    assert out.shape == (num_tokens, vocab)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)
    agree = (out.float().argmax(-1) == ref.argmax(-1)).float().mean()
    assert agree >= 0.95
