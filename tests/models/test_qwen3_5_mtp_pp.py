# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Qwen3.5 MTP drafters declare SupportsPP (#46994), so a PP > 1 target
passes the draft-model registry check."""

from vllm.model_executor.models.interfaces import supports_pp
from vllm.model_executor.models.qwen3_5_mtp import Qwen3_5MoeMTP, Qwen3_5MTP


def test_qwen3_5_mtp_supports_pp():
    assert supports_pp(Qwen3_5MTP)
    assert supports_pp(Qwen3_5MoeMTP)
