# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Which rows the PP sampled-token broadcast must carry."""

import os
import tempfile
from unittest.mock import Mock

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.worker.gpu import pp_utils


def _batch(num_computed, prefill_len, num_scheduled):
    return Mock(
        num_reqs=len(num_computed),
        num_computed_tokens_np=np.array(num_computed, dtype=np.int32),
        prefill_len_np=np.array(prefill_len, dtype=np.int32),
        num_scheduled_tokens=np.array(num_scheduled, dtype=np.int32),
    )


def test_excludes_non_final_prefill_chunks():
    """Unchanged behaviour: a chunk that does not finish its prefill is skipped."""
    # Row 0 is a middle prefill chunk and produces no sample; row 1 finishes its
    # prefill this step and therefore does.
    batch = _batch(
        num_computed=[512, 1000],
        prefill_len=[4096, 1004],
        num_scheduled=[448, 4],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [False, True]


def test_none_when_no_row_samples():
    """Unchanged behaviour: an all-prefill batch needs no broadcast at all."""
    batch = _batch(
        num_computed=[0, 512],
        prefill_len=[4096, 4096],
        num_scheduled=[448, 448],
    )

    assert pp_utils.compute_need_sampled_mask(batch) is None


def test_keeps_decoding_request_past_its_length_cap():
    """A decoding request must never be dropped from the broadcast.

    Speculative decoding advances `num_computed_tokens` several tokens per step,
    so it can overrun `prompt_len + max_tokens` while the scheduler is still
    running the request. Predicting "this one is finishing" and skipping its
    broadcast freezes the earlier pipeline stages' `last_sampled_tokens` and
    `draft_tokens` while the last rank keeps advancing its own, and the stages
    then diverge permanently.
    """
    batch = _batch(
        # 14176 computed tokens is already past this request's own
        # prompt_len + max_tokens; the scheduler is still running it.
        num_computed=[14176],
        prefill_len=[12175],
        num_scheduled=[8],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True]


def test_decode_row_ahead_of_a_prefill_chunk():
    """Row order does not matter: only whether the row finishes its prefill."""
    batch = _batch(
        num_computed=[10, 512],
        prefill_len=[8, 4096],
        num_scheduled=[1, 448],
    )

    mask = pp_utils.compute_need_sampled_mask(batch)

    assert mask is not None
    assert mask.tolist() == [True, False]


class _Ready:
    def synchronize(self) -> None:
        pass


def _host_broadcast_worker(rank: int, init_file: str, steps: list) -> None:
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    host = pp_utils._HostBroadcaster(src=1, group=dist.group.WORLD)
    if rank == 1:
        for packed, drafts in steps:
            host.send(packed.clone(), _Ready())
            host.send(drafts.clone(), _Ready())
        host.submit(lambda: None).result(timeout=30)
    else:
        futures = [
            host.recv(torch.empty_like(packed), torch.empty_like(drafts))
            for packed, drafts in steps
        ]
        for fut, (packed, drafts) in zip(futures, steps):
            got_packed, got_drafts = fut.result(timeout=30)
            assert torch.equal(got_packed, packed)
            assert torch.equal(got_drafts, drafts)
    dist.destroy_process_group()


def test_host_broadcaster_matches_steps_in_order():
    """The XPU gloo path pairs per-step (sampled, drafts) sends with receives
    posted ahead of time, in submission order, with different batch sizes."""
    steps = [
        (torch.arange(n * 7, dtype=torch.int64).view(n, 7) + 100 * i,
         torch.arange(n * 4, dtype=torch.int64).view(n, 4) - 100 * i)
        for i, n in enumerate([3, 1, 5])
    ]
    with tempfile.TemporaryDirectory() as d:
        init_file = os.path.join(d, "init")
        mp.start_processes(
            _host_broadcast_worker,
            args=(init_file, steps),
            nprocs=2,
            join=True,
            start_method="spawn",
        )
