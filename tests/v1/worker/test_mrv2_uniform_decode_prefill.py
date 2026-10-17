# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A prompt of exactly 1 + num_spec_tokens tokens must not be treated as a
uniform decode batch (it would replay the spec-decode graph, #51865)."""

from types import SimpleNamespace

import numpy as np

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def _runner(prefill_len, num_computed_prefill):
    req_states = SimpleNamespace(
        req_id_to_index={"a": 0, "b": 1},
        prefill_len=SimpleNamespace(np=np.array(prefill_len, dtype=np.int32)),
        num_computed_prefill_tokens=np.array(num_computed_prefill, dtype=np.int32),
    )
    return SimpleNamespace(req_states=req_states)


def _sched(*req_ids):
    return SimpleNamespace(num_scheduled_tokens={r: 5 for r in req_ids})


def test_prefill_request_is_detected():
    # "a" is a fresh 5-token prompt, same shape as a 1 + 4 spec decode row.
    runner = _runner(prefill_len=[5, 20], num_computed_prefill=[0, 20])
    assert GPUModelRunner._batch_has_prefill(runner, _sched("a"))
    assert GPUModelRunner._batch_has_prefill(runner, _sched("a", "b"))


def test_decode_only_batch():
    runner = _runner(prefill_len=[5, 20], num_computed_prefill=[5, 20])
    assert not GPUModelRunner._batch_has_prefill(runner, _sched("a", "b"))
