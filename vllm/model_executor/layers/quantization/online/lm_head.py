# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online-quantized copies of an already-loaded lm_head (draft use only)."""

from typing import Literal

import torch
from torch.nn import Module, Parameter

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

INT4_GROUP_SIZE = 128


def quantize_int4_gptq(
    weight: torch.Tensor, group_size: int = INT4_GROUP_SIZE, chunk_rows: int = 8192
) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric round-to-nearest int4 of the rows of `weight` [N, K].

    Returns the GPTQ checkpoint layout the W4A16 kernels load: `qweight`
    [K // 8, N] int32 (uint4b8 values packed along K, low nibble first) and
    `scales` [K // group_size, N] in the weight dtype.
    """
    n, k = weight.shape
    assert k % group_size == 0 and group_size % 8 == 0
    qweight = torch.empty(n, k // 8, dtype=torch.int32, device=weight.device)
    scales = torch.empty(n, k // group_size, dtype=weight.dtype, device=weight.device)
    shifts = torch.arange(0, 32, 4, device=weight.device, dtype=torch.int64)
    for start in range(0, n, chunk_rows):
        rows = slice(start, start + chunk_rows)
        w = weight[rows].float().reshape(-1, k // group_size, group_size)
        s = (w.abs().amax(dim=-1, keepdim=True) / 7.0).clamp(min=1e-8)
        q = (torch.round(w / s).clamp(-8, 7) + 8).to(torch.int64)
        packed = (q.reshape(-1, k // 8, 8) << shifts).sum(dim=-1)
        packed = torch.where(packed >= 2**31, packed - 2**32, packed)
        qweight[rows] = packed.to(torch.int32)
        scales[rows] = s.squeeze(-1).to(weight.dtype)
    return qweight.t(), scales.t().contiguous()


def dequantize_int4_gptq(
    qweight: torch.Tensor, scales: torch.Tensor, group_size: int = INT4_GROUP_SIZE
) -> torch.Tensor:
    """Inverse of `quantize_int4_gptq`, returning the [N, K] weight."""
    shifts = torch.arange(0, 32, 4, device=qweight.device, dtype=torch.int32)
    q = (qweight.t().unsqueeze(-1) >> shifts) & 0xF  # [N, K // 8, 8]
    q = q.reshape(q.shape[0], -1).float() - 8
    s = scales.t().float().repeat_interleave(group_size, dim=1)
    return (q * s).to(scales.dtype)


class _LMHeadInt4Method:
    """Weight-only int4 lm_head on the platform's mixed-precision kernel."""

    def __init__(self, layer: Module, weight: torch.Tensor) -> None:
        n, k = weight.shape
        config = MPLinearLayerConfig(
            full_weight_shape=(k, n),
            partition_weight_shape=(k, n),
            weight_type=scalar_types.uint4b8,
            act_type=weight.dtype,
            group_size=INT4_GROUP_SIZE,
            zero_points=False,
            has_g_idx=False,
        )
        kernel_type = choose_mp_linear_kernel(config)
        qweight, scales = quantize_int4_gptq(weight)
        layer.register_parameter("qweight", Parameter(qweight, requires_grad=False))
        layer.register_parameter("scales", Parameter(scales, requires_grad=False))
        self.kernel = kernel_type(
            config,
            w_q_param_name="qweight",
            w_s_param_name="scales",
            w_zp_param_name="qzeros",
            w_gidx_param_name="g_idx",
        )
        self.kernel.process_weights_after_loading(layer)
        logger.info(
            "Quantized an int4 copy of the lm_head (%d x %d per rank) with %s.",
            n,
            k,
            kernel_type.__name__,
        )

    def apply(
        self, layer: Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return self.kernel.apply_weights(layer, x, bias)


def quantized_lm_head_copy(
    lm_head: ParallelLMHead, quantization: Literal["int4"]
) -> ParallelLMHead:
    """A ParallelLMHead holding this rank's shard of `lm_head`, quantized.

    `lm_head` is left unchanged and no bf16 copy of its weight is made.
    """
    if quantization != "int4":
        raise ValueError(f"Unsupported lm_head quantization {quantization!r}.")
    weight = lm_head.weight
    if (
        not isinstance(lm_head.quant_method, UnquantizedEmbeddingMethod)
        or getattr(lm_head, "bias", None) is not None
        or weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        raise ValueError(
            "Quantizing an lm_head copy needs an unquantized fp16/bf16 lm_head "
            "without bias."
        )
    with torch.device("meta"):
        head = ParallelLMHead(
            lm_head.num_embeddings,
            lm_head.embedding_dim,
            params_dtype=weight.dtype,
            org_num_embeddings=lm_head.org_vocab_size,
            padding_size=lm_head.padding_size,
            prefix=getattr(lm_head, "prefix", ""),
        )
    # Same vocab sharding as the source; only the weight representation differs.
    head.shard_indices = lm_head.shard_indices
    del head.weight
    with torch.no_grad():
        head.quant_method = _LMHeadInt4Method(head, weight.data)
    return head
