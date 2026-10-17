# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pipeline Parallelism utils for V2 Model Runner."""

import queue
import threading
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

import numpy as np
import torch

import vllm.envs as envs
from vllm.distributed.parallel_state import get_pp_group
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.worker.gpu.buffer_utils import async_copy_to_gpu
from vllm.v1.worker.gpu.input_batch import InputBatch


@dataclass
class PendingRecv:
    """Per-step slot data for a deferred postprocess on the main stream."""

    # None when the outputs arrive through `host_future` (host broadcast).
    event: torch.cuda.Event | None

    sampled_tokens: torch.Tensor | None  # [num_reqs, max_sample_len]
    num_sampled: torch.Tensor | None  # [num_reqs]
    num_rejected: torch.Tensor | None  # [num_reqs]
    idx_mapping: torch.Tensor  # [num_reqs]
    idx_mapping_np: np.ndarray  # [num_reqs]
    # Records which rows need a deferred postprocess (bool).
    need_sampled_mask: np.ndarray  # [num_reqs]
    # Snapshot of slot generation counters at receive time, used to
    # detect requests aborted since then.
    gen_at_receive_np: np.ndarray  # [num_reqs]
    draft_tokens: torch.Tensor | None = None  # [num_reqs, num_speculative_steps]
    # Host broadcast: resolves to (packed[, drafts]) CPU tensors, where
    # packed = [sampled_tokens | num_sampled | num_rejected].
    host_future: Future | None = None


class _HostBroadcaster:
    """Runs gloo broadcasts in submission order on one daemon thread.

    Both PP ends submit their broadcasts in the same per-step order, so the
    single FIFO thread keeps them matched without blocking the main thread.
    """

    def __init__(self, src: int, group: torch.distributed.ProcessGroup):
        self.src = src
        self.group = group
        self.tasks: queue.SimpleQueue[tuple[Callable[[], object], Future]] = (
            queue.SimpleQueue()
        )
        self.thread = threading.Thread(
            target=self._run, name="pp_host_broadcast", daemon=True
        )
        self.thread.start()

    def _run(self) -> None:
        while True:
            fn, fut = self.tasks.get()
            try:
                fut.set_result(fn())
            except BaseException as e:  # noqa: BLE001
                fut.set_exception(e)

    def submit(self, fn: Callable[[], object]) -> Future:
        fut: Future = Future()
        self.tasks.put((fn, fut))
        return fut

    def send(self, tensor: torch.Tensor, ready: torch.cuda.Event) -> None:
        """Broadcast a pinned CPU tensor once `ready` (its D2H copy) completes."""

        def _send() -> None:
            ready.synchronize()
            torch.distributed.broadcast(tensor, src=self.src, group=self.group)

        self.submit(_send)

    def recv(self, *tensors: torch.Tensor) -> Future:
        def _recv() -> tuple[torch.Tensor, ...]:
            for t in tensors:
                torch.distributed.broadcast(t, src=self.src, group=self.group)
            return tensors

        return self.submit(_recv)


def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
    """Return a bool array of shape `[input_batch.num_reqs]` marking requests
    that produce a sampled token this step, and therefore must have that token
    (and the draft block proposed from it) propagated to the earlier PP stages.
    Returns None if no request in the batch produces a sample."""

    old_computed = input_batch.num_computed_tokens_np
    prefill_len = input_batch.prefill_len_np
    # Exclude non-final prefill chunks (they don't produce a sample).
    produces_sample = old_computed + input_batch.num_scheduled_tokens >= prefill_len
    return produces_sample if produces_sample.any() else None


class PPHandler:
    """Runs the PP sampled-token broadcast/recv on a side stream so the
    default stream isn't gated by the matching peer call. Step T's recv is
    consumed at step T+pp_size via `get_prev_sampled_outputs`.

    Uses a dedicated NCCL communicator (sibling of the PP `device_group`)
    for the broadcast so it does not serialize on the wire with the
    inter-stage hidden-state p2p send/recv ops.
    """

    def __init__(
        self, max_num_reqs: int, num_speculative_steps: int, device: torch.device
    ):
        self.is_last_rank = get_pp_group().is_last_rank
        self.last_rank = get_pp_group().last_rank
        self.max_sample_len = num_speculative_steps + 1
        self.num_speculative_steps = num_speculative_steps
        self.device = device
        self.main_stream = torch.cuda.current_stream(device)
        self.broadcast_stream = torch.cuda.Stream(device)

        # On non-last ranks, a FIFO with one entry per in-flight step: the entry
        # pushed by step T's `receive` is consumed pp_size steps later. Pre-seeded
        # with pp_size None placeholders so the first pp_size consumes are no-ops.
        # None means no postprocess is pending for that step (broadcast skipped).
        # Only XPU can disable microbatching via VLLM_XPU_PP_MICROBATCH.
        ring_depth = get_pp_group().world_size
        if current_platform.is_xpu() and not envs.VLLM_XPU_PP_MICROBATCH:
            ring_depth = 1
        self.queue: deque[PendingRecv | None] = (
            deque() if self.is_last_rank else deque([None] * ring_depth)
        )

        # Per req-index generation counter, incremented every time a request
        # index is freed in RequestStats. Used for invalidating freed req data
        # between PP decodes.
        self.req_idx_gen_np = np.zeros(max_num_reqs, dtype=np.int32)

        # On XPU (oneCCL 2021.15), a second xccl communicator for these small
        # broadcasts deadlocks with the hidden-state send/recv under load, so
        # they go over a gloo group from a helper thread instead.
        self.host: _HostBroadcaster | None = None
        if current_platform.is_xpu() and envs.VLLM_XPU_PP_HOST_BROADCAST:
            self.pin_memory = is_pin_memory_available()
            # Keeps pinned buffers alive until their async H2D/D2H copies finish.
            self.host_buffers: deque[torch.Tensor] = deque(maxlen=64)
            self.host = _HostBroadcaster(
                self.last_rank,
                get_pp_group().make_sibling_cpu_group(group_desc="pp_broadcast"),
            )
            return

        # Dedicated subgroup for the sampled-token broadcast.
        self.broadcast_group = get_pp_group().make_sibling_device_group(
            group_desc="pp_broadcast"
        )

    def on_req_idx_freed(self, req_idx: int) -> None:
        self.req_idx_gen_np[req_idx] += 1

    def get_prev_sampled_outputs(
        self, draft_tokens_to_update: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor] | None:
        """Consume the entry from pp_size steps ago and wait for its recv event,
        then filter out entries whose request was freed since `receive`.
        """
        if not self.queue:
            return None
        slot = self.queue.popleft()
        # Reserve this step's slot; `receive` overwrites it if applicable.
        self.queue.append(None)
        if slot is None:
            return None

        # Skip requests which did not need sampled output and/or those already
        # finished. The post_update kernel skips the -1 entries.
        freed = self.req_idx_gen_np[slot.idx_mapping_np] != slot.gen_at_receive_np
        exclude_mask = freed | ~slot.need_sampled_mask
        idx_mapping = slot.idx_mapping
        if exclude_mask.any():
            if exclude_mask.all():
                # No states require update anymore.
                return None
            # Filter excluded request indices.
            idx_mapping_np = np.where(exclude_mask, -1, slot.idx_mapping_np)
            idx_mapping = async_copy_to_gpu(idx_mapping_np, device=self.device)

        if slot.host_future is not None:
            self._materialize_host_slot(slot)
        else:
            self.main_stream.wait_event(slot.event)
        if slot.draft_tokens is not None and draft_tokens_to_update is not None:
            draft_tokens = slot.draft_tokens
            draft_idx_mapping = slot.idx_mapping
            if exclude_mask.any():
                keep = ~exclude_mask
                keep_t = torch.as_tensor(keep, device=self.device)
                draft_tokens = draft_tokens[keep_t]
                draft_idx_mapping = async_copy_to_gpu(
                    slot.idx_mapping_np[keep], device=self.device
                )
            draft_tokens_to_update[draft_idx_mapping] = draft_tokens

        return dict(
            sampled_tokens=slot.sampled_tokens,
            num_sampled=slot.num_sampled,
            num_rejected=slot.num_rejected,
            idx_mapping=idx_mapping,
        )

    def _to_host(self, tensor: torch.Tensor) -> torch.Tensor:
        """Async D2H copy into a pinned buffer; returns the buffer."""
        buf = torch.empty(
            tensor.shape, dtype=tensor.dtype, pin_memory=self.pin_memory
        )
        buf.copy_(tensor, non_blocking=True)
        return buf

    def _host_send(self, tensor: torch.Tensor) -> None:
        assert self.host is not None
        buf = self._to_host(tensor)
        ready = self.main_stream.record_event()
        self.host.send(buf, ready)

    def _materialize_host_slot(self, slot: PendingRecv) -> None:
        assert slot.host_future is not None
        received = slot.host_future.result()
        packed_cpu = received[0]
        drafts_cpu = received[1] if len(received) > 1 else None
        packed = packed_cpu.to(self.device, non_blocking=True)
        self.host_buffers.append(packed_cpu)
        n = self.max_sample_len
        slot.sampled_tokens = packed[:, :n].contiguous()
        slot.num_sampled = packed[:, n].to(torch.int32)
        slot.num_rejected = packed[:, n + 1].to(torch.int32)
        if drafts_cpu is not None:
            slot.draft_tokens = drafts_cpu.to(self.device, non_blocking=True)
            self.host_buffers.append(drafts_cpu)
        slot.host_future = None

    def broadcast_drafts(
        self, draft_tokens: torch.Tensor, input_batch: InputBatch
    ) -> None:
        """Broadcast draft proposals so non-last ranks can embed real token ids."""
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            return
        if self.host is not None:
            self._host_send(draft_tokens[input_batch.idx_mapping])
            return
        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send = draft_tokens[input_batch.idx_mapping].contiguous()
            # Must record the idx_mapping tensor since it was allocated
            # on the main stream.
            input_batch.idx_mapping.record_stream(self.broadcast_stream)
            torch.distributed.broadcast(
                send, src=self.last_rank, group=self.broadcast_group
            )

    def receive(self, input_batch: InputBatch) -> bool:
        """Returns True iff sampled tokens need to be gathered from *all*
        requests in the batch."""
        assert not self.is_last_rank
        need_sampled_mask = compute_need_sampled_mask(input_batch)
        if need_sampled_mask is None:
            # Leave this step's reserved slot as None.
            return False

        # Snapshot the per-slot generation counter so a later free of any of
        # these RequestStates request indices is detectable at consume time.
        gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]

        num_reqs = input_batch.num_reqs
        if self.host is not None:
            packed = torch.empty(
                num_reqs,
                self.max_sample_len + 2,
                dtype=torch.int64,
                pin_memory=self.pin_memory,
            )
            recv_tensors = [packed]
            if self.num_speculative_steps > 0:
                recv_tensors.append(
                    torch.empty(
                        num_reqs,
                        self.num_speculative_steps,
                        dtype=torch.int64,
                        pin_memory=self.pin_memory,
                    )
                )
            host_future = self.host.recv(*recv_tensors)
            self.queue[-1] = PendingRecv(
                None,
                None,
                None,
                None,
                input_batch.idx_mapping,
                input_batch.idx_mapping_np,
                need_sampled_mask,
                gen_at_receive_np,
                host_future=host_future,
            )
            return bool(need_sampled_mask.all())

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            sampled_tokens = torch.empty(
                num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
            )
            combined = torch.empty(2, num_reqs, dtype=torch.int32, device=self.device)
            torch.distributed.broadcast(
                sampled_tokens, src=self.last_rank, group=self.broadcast_group
            )
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            draft_tokens = None
            if self.num_speculative_steps > 0:
                draft_tokens = torch.empty(
                    num_reqs,
                    self.num_speculative_steps,
                    dtype=torch.int64,
                    device=self.device,
                )
                torch.distributed.broadcast(
                    draft_tokens, src=self.last_rank, group=self.broadcast_group
                )
            event = self.broadcast_stream.record_event()
            num_sampled, num_rejected = combined.unbind(dim=0)
            # Must record_stream since these were allocated on broadcast stream but
            # later used on the main stream.
            sampled_tokens.record_stream(self.main_stream)
            combined.record_stream(self.main_stream)
            if draft_tokens is not None:
                draft_tokens.record_stream(self.main_stream)
        self.queue[-1] = PendingRecv(
            event,
            sampled_tokens,
            num_sampled,
            num_rejected,
            input_batch.idx_mapping,
            input_batch.idx_mapping_np,
            need_sampled_mask,
            gen_at_receive_np,
            draft_tokens,
        )
        return bool(need_sampled_mask.all())

    def broadcast(
        self,
        sampled_token_ids: torch.Tensor,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        input_batch: InputBatch,
    ) -> None:
        assert self.is_last_rank
        if compute_need_sampled_mask(input_batch) is None:
            # No request needs sampled outputs for a subsequent decode step.
            return

        assert sampled_token_ids.dtype == torch.int64

        if self.host is not None:
            pad = self.max_sample_len - sampled_token_ids.shape[-1]
            packed = torch.cat(
                (
                    torch.nn.functional.pad(sampled_token_ids, (0, pad)),
                    num_sampled.to(torch.int64).unsqueeze(1),
                    num_rejected.to(torch.int64).unsqueeze(1),
                ),
                dim=1,
            )
            self._host_send(packed)
            return

        if current_platform.is_xpu():
            self.main_stream.synchronize()

        with torch.cuda.stream(self.broadcast_stream):
            self.broadcast_stream.wait_stream(self.main_stream)
            send_tokens = torch.nn.functional.pad(
                sampled_token_ids,
                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
            )
            torch.distributed.broadcast(
                send_tokens.contiguous(),
                src=self.last_rank,
                group=self.broadcast_group,
            )
            combined = torch.stack((num_sampled, num_rejected), dim=0)
            torch.distributed.broadcast(
                combined, src=self.last_rank, group=self.broadcast_group
            )
            for tensor in (sampled_token_ids, num_sampled, num_rejected):
                tensor.record_stream(self.broadcast_stream)
