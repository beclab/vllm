# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""On XPU with TP > 1 the MRV2 drafter must not capture graphs: oneCCL
collectives replayed from the drafter's XPU graphs give wrong drafts and
corrupt the target output."""

from types import SimpleNamespace

import pytest

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as mod


@pytest.fixture
def captured_modes(monkeypatch):
    modes = []

    class FakeManager:
        def __init__(self, vllm_config, device, cudagraph_mode, decode_query_len):
            modes.append(cudagraph_mode)

    monkeypatch.setattr(mod, "SpeculatorCudaGraphManager", FakeManager)
    return modes


def _speculator(tp_size):
    return SimpleNamespace(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(tensor_parallel_size=tp_size)
        ),
        device="xpu",
        num_speculative_steps=3,
    )


@pytest.mark.parametrize(
    "is_xpu,tp_size,expect_eager",
    [(True, 2, True), (True, 1, False), (False, 2, False)],
)
def test_drafter_graph_mode(monkeypatch, captured_modes, is_xpu, tp_size, expect_eager):
    monkeypatch.setattr(mod.current_platform, "is_xpu", lambda: is_xpu)
    mod.AutoRegressiveSpeculator.init_cudagraph_manager(
        _speculator(tp_size), CUDAGraphMode.FULL_DECODE_ONLY
    )
    if expect_eager:
        assert captured_modes == [CUDAGraphMode.NONE, CUDAGraphMode.NONE]
    else:
        assert captured_modes == [
            CUDAGraphMode.FULL_DECODE_ONLY,
            CUDAGraphMode.FULL_DECODE_ONLY,
        ]
