"""TileRT PD producer connector for vLLM prefill (model-agnostic framework).

Loaded into vLLM via the official plugin surface:

  --kv-transfer-config '{
      "kv_connector": "TileRTConnector",
      "kv_connector_module_path": "tilert.pd_vllm.prefill_connector",
      "kv_role": "kv_producer",
      "kv_connector_extra_config": {"tilert_host": "<decode-node-ip>",
                                     "tilert_ctrl_port": 5556,
                                     "tilert_model": "glm5"}
  }'

Claim discipline (MultiConnector-safe): only requests whose
``kv_transfer_params`` carry ``tilert_host`` are claimed; everything else is a
strict no-op so a native connector can coexist.

The connector owns the model-agnostic plumbing (claim, chunked-prefill
tracking, worker init, staging, background send, TCP handshake); all per-model
extraction / layout / RDMA planning is delegated to the selected model profile
(``tilert_model``, default ``glm5``).
"""

import logging
import queue
import threading
from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    SupportsHMA,
)

from tilert.pd_vllm import wire
from tilert.pd_vllm.profiles import base as profiles
from tilert.pd_vllm.wire import derive_rid

logger = logging.getLogger("pd_vllm.connector")


@dataclass
class _ReqMeta:
    req_id: str
    rid: str
    num_tokens: int
    last_prompt_token: int
    block_ids_per_group: list
    tilert_host: str
    tilert_ctrl_port: int
    sampling: dict | None = None


@dataclass
class TileRTMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)


@dataclass
class _Pending:
    """Scheduler-side chunked-prefill accumulation."""

    req_id: str
    prompt_token_ids: list
    total_tokens: int
    block_ids_per_group: list
    params: dict


class TileRTConnector(KVConnectorBase_V1, SupportsHMA):
    # ══════════════════════════ init ══════════════════════════

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._default_host = extra.get("tilert_host")
        self._default_port = int(extra.get("tilert_ctrl_port", 5556))
        self._sync_send = bool(extra.get("tilert_sync_send", False))
        self._max_seq = int(extra.get("tilert_max_seq_len", vllm_config.model_config.max_model_len))
        self._profile = profiles.get_profile(extra.get("tilert_model", "glm5"))
        self._transport_name = extra.get("tilert_transport", "mooncake")

        # scheduler-side
        self._pending: dict[str, _Pending] = {}

        # worker-side (lazy)
        self._kv_caches: dict = {}
        self._reg = None  # profile registration (layer map)
        self._tp_rank: int | None = None
        self._transport = None
        self._staging = None
        self._send_q: queue.Queue = queue.Queue()
        self._sender_thread: threading.Thread | None = None

        logger.info(
            "TileRTConnector: role=%s profile=%s target=%s:%s sync=%s",
            role,
            self._profile.name,
            self._default_host,
            self._default_port,
            self._sync_send,
        )

    # ══════════════════════ scheduler side ═════════════════════

    @staticmethod
    def _claim(params) -> dict | None:
        """Return kv_transfer_params if this request is ours, else None."""
        if params and isinstance(params, dict) and params.get("tilert_host"):
            return params
        return None

    def _params_of(self, new_req) -> dict | None:
        sp = getattr(new_req, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) if sp is not None else None
        if extra:
            return self._claim(extra.get("kv_transfer_params"))
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        pass

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        meta = TileRTMetadata()
        num_sched = scheduler_output.num_scheduled_tokens or {}

        for req_id in scheduler_output.finished_req_ids:
            self._pending.pop(req_id, None)
        for req_id in getattr(scheduler_output, "preempted_req_ids", None) or []:
            self._pending.pop(req_id, None)

        for new_req in scheduler_output.scheduled_new_reqs:
            params = self._params_of(new_req)
            if params is None:
                continue  # not ours — strict no-op (MultiConnector safety)
            token_ids = list(new_req.prompt_token_ids or [])
            if not token_ids:
                continue
            groups = [list(g) for g in new_req.block_ids]
            n = num_sched.get(new_req.req_id, 0)
            if new_req.num_computed_tokens + n >= len(token_ids):
                meta.requests.append(self._emit(new_req.req_id, token_ids, groups, params))
            else:
                self._pending[new_req.req_id] = _Pending(
                    req_id=new_req.req_id,
                    prompt_token_ids=token_ids,
                    total_tokens=len(token_ids),
                    block_ids_per_group=groups,
                    params=params,
                )

        cached = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(getattr(cached, "req_ids", []) or []):
            p = self._pending.get(req_id)
            if p is None:
                continue
            new_blocks = cached.new_block_ids[i]
            if new_blocks is not None:
                for gi, g in enumerate(new_blocks):
                    if gi < len(p.block_ids_per_group) and g:
                        p.block_ids_per_group[gi].extend(g)
            n = num_sched.get(req_id, 0)
            if cached.num_computed_tokens[i] + n >= p.total_tokens:
                meta.requests.append(
                    self._emit(req_id, p.prompt_token_ids, p.block_ids_per_group, p.params)
                )
                del self._pending[req_id]
        return meta

    def _emit(self, req_id, token_ids, groups, params) -> _ReqMeta:
        host = params.get("tilert_host") or self._default_host
        assert host is not None, "claimed a request with no tilert_host"
        m = _ReqMeta(
            req_id=req_id,
            rid=derive_rid(req_id),
            num_tokens=len(token_ids),
            last_prompt_token=int(token_ids[-1]),
            block_ids_per_group=groups,
            tilert_host=host,
            tilert_ctrl_port=int(params.get("tilert_ctrl_port", self._default_port)),
            sampling=params.get("sampling"),
        )
        logger.info(
            "claimed %s (rid=%s, %d tokens) -> %s:%d",
            req_id,
            m.rid,
            m.num_tokens,
            m.tilert_host,
            m.tilert_ctrl_port,
        )
        return m

    def request_finished(self, request, block_ids):
        self._pending.pop(getattr(request, "request_id", ""), None)
        return False, None

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids)

    # ══════════════════════ worker side ════════════════════════

    def register_kv_caches(self, kv_caches):
        self._kv_caches = kv_caches
        cfg = getattr(self, "_kv_cache_config", None)
        self._reg = self._profile.classify_layers(kv_caches, cfg)

    def _ensure_worker_ready(self) -> None:
        if self._transport is not None:
            return
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        self._tp_rank = int(get_tensor_model_parallel_rank())

        from tilert.pd_vllm.transport import make_transport

        hostname = wire.local_ip()
        total = self._profile.staging_bytes(self._reg, self._tp_rank, self._max_seq)
        dev = torch.cuda.current_device()
        self._staging = torch.zeros(total, dtype=torch.uint8, device=f"cuda:{dev}")

        self._transport = make_transport(self._transport_name)
        self._transport.init(hostname)
        self._transport.register(self._staging.data_ptr(), total, dev)

        if not self._sync_send:
            self._sender_thread = threading.Thread(
                target=self._sender_loop, name="tilert-pd-sender", daemon=True
            )
            self._sender_thread.start()
        logger.info(
            "worker ready: rank=%d transport=%s staging=%.1f MB profile=%s",
            self._tp_rank,
            self._transport.name,
            total / 1e6,
            self._profile.name,
        )

    def start_load_kv(self, forward_context, **kwargs):
        pass

    def wait_for_layer_load(self, layer_name):
        pass

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs):
        pass

    def wait_for_save(self):
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, TileRTMetadata) or not metadata.requests:
            return
        self._ensure_worker_ready()
        assert self._tp_rank is not None  # set by _ensure_worker_ready
        if self._tp_rank not in self._profile.sender_ranks:
            return  # this rank does not participate (e.g. replicated MLA)
        for m in metadata.requests:
            try:
                sections = self._profile.extract(
                    self._reg, m, self._tp_rank, self._staging, self._max_seq
                )
            except Exception:
                logger.exception("extraction failed for %s", m.rid)
                continue
            job = {"meta": m, "sections": sections, "seq": sections["seq"]}
            if self._sync_send:
                self._send(job)
            else:
                self._send_q.put(job)

    def get_finished(self, finished_req_ids):
        return None, None

    # ── background send ──

    def _sender_loop(self) -> None:
        while True:
            job = self._send_q.get()
            try:
                self._send(job)
            except Exception:
                logger.exception("send failed for %s", job["meta"].rid)

    def _send(self, job: dict) -> None:
        import socket as _socket
        import time as _time

        # _send only runs after wait_for_save() -> _ensure_worker_ready()
        assert self._transport is not None and self._staging is not None
        assert self._tp_rank is not None
        m: _ReqMeta = job["meta"]
        seq = job["seq"]
        t0 = _time.time()
        conn = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        try:
            conn.setsockopt(_socket.IPPROTO_TCP, _socket.TCP_NODELAY, 1)
            conn.settimeout(60)
            conn.connect((m.tilert_host, m.tilert_ctrl_port))
            hello = wire.recv_msg(conn)
            assert hello.get("magic") == wire.MAGIC, f"bad hello: {hello}"
            assert hello.get("layout_version") == self._profile.layout_version, (
                f"layout version mismatch: {hello.get('layout_version')} "
                f"vs {self._profile.layout_version}"
            )
            assert hello.get("transport") == self._transport.name, (
                f"transport mismatch: decode={hello.get('transport')} "
                f"vs prefill={self._transport.name}"
            )
            remote_max_seq = int(hello["max_seq_len"])
            assert seq <= remote_max_seq, f"seq {seq} exceeds decode max_seq_len {remote_max_seq}"

            wire.send_msg(
                conn,
                {
                    "rid": m.rid,
                    "rank": self._tp_rank,
                    "seq_len": seq,
                    "last_prompt_token": m.last_prompt_token,
                    "sampling": m.sampling,
                },
            )

            base = self._staging.data_ptr()
            srcs, dsts, lens = self._profile.rdma_plan(
                hello, job["sections"], self._tp_rank, seq, base
            )
            self._transport.write(hello, srcs, dsts, lens)

            wire.send_msg(conn, {"done": True, "rid": m.rid, "rank": self._tp_rank})
            logger.info(
                "sent %s: rank=%d seq=%d %.1f MB in %.1f ms",
                m.rid,
                self._tp_rank,
                seq,
                sum(lens) / 1e6,
                1000 * (_time.time() - t0),
            )
        finally:
            conn.close()
