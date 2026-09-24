import logging
import os
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
_SENT = "sent"
_REJECTED_TRANSIENT = "rejected_transient"
_REJECTED_PERMANENT = "rejected_permanent"
_TRANSIENT_REJECTS = frozenset({"busy", "cancelling"})
_ADMISSION_ATTEMPTS = 5
_ADMISSION_BACKOFF_S = 0.2


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
    prompt_token_ids: list = field(default_factory=list)


@dataclass
class TileRTMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)


@dataclass
class _Pending:
    req_id: str
    prompt_token_ids: list
    total_tokens: int
    block_ids_per_group: list
    params: dict


class TileRTConnector(KVConnectorBase_V1, SupportsHMA):

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        self._default_host = extra.get("tilert_host")
        self._default_port = int(extra.get("tilert_ctrl_port", 5556))
        self._admission_attempts = int(extra.get("tilert_admission_attempts", _ADMISSION_ATTEMPTS))
        self._max_seq = int(extra.get("tilert_max_seq_len", vllm_config.model_config.max_model_len))
        self._profile = profiles.get_profile(extra.get("tilert_model", "glm5"))
        self._transport_name = extra.get("tilert_transport", "mooncake")
        self._pd_buffer_device = str(
            extra.get("tilert_pd_buffer_device")
            or os.environ.get("TILERT_PD_BUFFER_DEVICE")
            or "cuda"
        ).lower()
        self._pending: dict[str, _Pending] = {}
        self._kv_caches: dict = {}
        self._reg = None
        self._tp_rank: int | None = None
        self._transport = None
        self._staging = None
        self._send_q: queue.Queue = queue.Queue()
        self._sender_thread: threading.Thread | None = None
        logger.info(
            "TileRTConnector: role=%s profile=%s target=%s:%s",
            role,
            self._profile.name,
            self._default_host,
            self._default_port,
        )

    @staticmethod
    def _claim(params) -> dict | None:
        if params and isinstance(params, dict) and params.get("tilert_host"):
            return params
        return None

    def _params_of(self, new_req) -> dict | None:
        sp = getattr(new_req, "sampling_params", None)
        extra = getattr(sp, "extra_args", None) if sp is not None else None
        if extra:
            claimed = self._claim(extra.get("kv_transfer_params"))
            if claimed is not None:
                claimed = dict(claimed)
                claimed["_wants_prompt_ids"] = wire.wants_prompt_token_ids(
                    {"repetition_penalty": getattr(sp, "repetition_penalty", 1.0)}
                )
            return claimed
        return None

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return (0, False)

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
                continue
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
        m = _ReqMeta(
            req_id=req_id,
            rid=derive_rid(req_id),
            num_tokens=len(token_ids),
            last_prompt_token=int(token_ids[-1]),
            prompt_token_ids=list(token_ids) if params.get("_wants_prompt_ids") else [],
            block_ids_per_group=groups,
            tilert_host=params.get("tilert_host") or self._default_host,
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
        return (False, None)

    def request_finished_all_groups(self, request, block_ids):
        return self.request_finished(request, block_ids)

    def register_kv_caches(self, kv_caches):
        self._kv_caches = kv_caches
        cfg = getattr(self, "_kv_cache_config", None)
        self._reg = self._profile.classify_layers(kv_caches, cfg)
        self._collect_kv_scales()

    def _collect_kv_scales(self) -> None:
        if not getattr(self._profile, "mla_rocm_fp8", False):
            return
        ctx = self._vllm_config.compilation_config.static_forward_context
        scales = []
        for lid, name, _t, _gi in self._reg.mla_layers:
            layer = ctx.get(name)
            k = getattr(layer, "_k_scale", None)
            if k is None:
                raise RuntimeError(
                    f"MLA layer {name!r} has no _k_scale; the flat ROCm fp8 KV layout "
                    "cannot be dequantised without it"
                )
            scales.append(float(k.item()) if hasattr(k, "item") else float(k))
        self._profile.set_kv_scales(scales)

    def _ensure_worker_ready(self) -> None:
        if self._transport is not None:
            return
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank

        self._tp_rank = get_tensor_model_parallel_rank()
        from tilert.pd_vllm.transport import make_transport

        hostname = wire.local_ip()
        dev = torch.cuda.current_device()
        host = self._pd_buffer_device == "cpu"
        senders = len(self._profile.sender_ranks)
        nshards = int((os.environ.get("TILERT_PD_SHARDS") or "8").strip() or 8)
        nshards = max(1, min(nshards, torch.cuda.device_count() or 1))
        if host and nshards > 1:
            logger.info("staging buffer is host-resident: ignoring TILERT_PD_SHARDS=%d", nshards)
            nshards = 1
        if senders > 1:
            if host:
                raise RuntimeError(
                    "TILERT_PD_SENDERS>1 and a host-resident staging buffer are mutually "
                    "exclusive: multi-sender exists to keep every shard on its own card"
                )
            nshards = senders
        total = self._profile.staging_bytes(self._reg, self._tp_rank, self._max_seq, nshards)
        if host and total > 4:
            from tilert.pd_vllm.transport import alloc_pinned_huge

            self._staging = alloc_pinned_huge(total)
            logger.info(
                "staging buffer: %.2f GiB in pinned host DRAM (rank=%d)",
                total / 1024**3,
                self._tp_rank,
            )
        elif senders > 1 and total > 4:
            own = torch.zeros(total, dtype=torch.uint8, device=f"cuda:{dev}")
            self._staging = [own if i == self._tp_rank else None for i in range(nshards)]
            logger.info(
                "staging buffer: own shard only, %.2f GB on cuda:%d (rank=%d of %d senders)",
                total / 1024**3,
                dev,
                self._tp_rank,
                senders,
            )
        elif nshards > 1 and total > 4:
            self._staging = [
                torch.zeros(total, dtype=torch.uint8, device=f"cuda:{i}") for i in range(nshards)
            ]
            logger.info(
                "staging buffer: %d shards x %.2f GB on cuda:0..%d (rank=%d)",
                nshards,
                total / 1024**3,
                nshards - 1,
                self._tp_rank,
            )
        else:
            host = False
            self._staging = torch.zeros(total, dtype=torch.uint8, device=f"cuda:{dev}")
        self._transport = make_transport(self._transport_name)
        self._transport.init(hostname)
        if isinstance(self._staging, list):
            rails = self._transport.rails()
            for i, t in enumerate(self._staging):
                if t is None:
                    continue
                self._transport.register(
                    t.data_ptr(), t.numel(), i, f"hip:{i % rails}" if rails else None
                )
            logger.info("staging shards pinned to %s rails", rails or "auto")
        else:
            self._transport.register(self._staging.data_ptr(), total, dev, host=host)
        self._sender_thread = threading.Thread(
            target=self._sender_loop, name="tilert-pd-sender", daemon=True
        )
        self._sender_thread.start()
        logger.info(
            "worker ready: rank=%d transport=%s staging=%.1f MB profile=%s",
            self._tp_rank,
            self._transport.name,
            total / 1000000.0,
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
        if self._tp_rank not in self._profile.sender_ranks:
            return
        for m in metadata.requests:
            try:
                sections = self._profile.extract(
                    self._reg, m, self._tp_rank, self._staging, self._max_seq
                )
            except Exception:
                logger.exception("extraction failed for %s", m.rid)
                continue
            self._send_q.put({"meta": m, "sections": sections, "seq": sections["seq"]})

    def get_finished(self, finished_req_ids):
        return (None, None)

    def _sender_loop(self) -> None:
        while True:
            job = self._send_q.get()
            try:
                self._send_with_retry(job)
            except Exception:
                logger.exception("send failed for %s", job["meta"].rid)

    def _send(self, job: dict) -> None:
        import socket as _socket
        import time as _time

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
            remote_proto = hello.get("protocol_version", 1)
            assert (
                remote_proto == wire.PROTOCOL_VERSION
            ), f"control-plane protocol mismatch: decode={remote_proto} vs prefill={wire.PROTOCOL_VERSION}; upgrade both ends together"
            assert (
                hello.get("layout_version") == self._profile.layout_version
            ), f"layout version mismatch: {hello.get('layout_version')} vs {self._profile.layout_version}"
            remote_senders = int(hello.get("senders", 1) or 1)
            assert remote_senders == len(self._profile.sender_ranks), (
                f"sender-count mismatch: decode expects {remote_senders} sender rank(s) but prefill has "
                f"{len(self._profile.sender_ranks)}; set TILERT_PD_SENDERS to the same value on both roles"
            )
            assert (
                hello.get("transport") == self._transport.name
            ), f"transport mismatch: decode={hello.get('transport')} vs prefill={self._transport.name}"
            remote_max_seq = int(hello["max_seq_len"])
            assert seq <= remote_max_seq, f"seq {seq} exceeds decode max_seq_len {remote_max_seq}"
            msg = {
                "rid": m.rid,
                "rank": self._tp_rank,
                "seq_len": seq,
                "last_prompt_token": m.last_prompt_token,
                "sampling": m.sampling,
                "admission_window_s": self._admission_window(),
            }
            if getattr(self._profile, "kv_scales", None) is not None:
                msg["kv_scales"] = self._profile.kv_scales
            if self._tp_rank == 0 and m.prompt_token_ids:
                msg["prompt_token_ids"] = m.prompt_token_ids
            wire.send_msg(conn, msg)
            ack = wire.recv_msg(conn)
            if not ack.get("accepted"):
                reason = ack.get("error")
                logger.warning("decode node refused %s rank=%d: %s", m.rid, self._tp_rank, ack)
                return _REJECTED_TRANSIENT if reason in _TRANSIENT_REJECTS else _REJECTED_PERMANENT
            if (
                ack.get("rid") != m.rid
                or ack.get("rank") != self._tp_rank
                or (not isinstance(ack.get("generation"), int))
            ):
                logger.error(
                    "discarding %s rank=%d: admission does not match the request (%s)",
                    m.rid,
                    self._tp_rank,
                    ack,
                )
                return _REJECTED_PERMANENT
            generation = ack["generation"]
            base = (
                [(t.data_ptr() if t is not None else 0) for t in self._staging]
                if isinstance(self._staging, list)
                else self._staging.data_ptr()
            )
            srcs, dsts, lens = self._profile.rdma_plan(
                hello, job["sections"], self._tp_rank, seq, base
            )
            self._transport.write(hello, srcs, dsts, lens)
            wire.send_msg(conn, wire.done_msg(m.rid, self._tp_rank, generation))
            logger.info(
                "sent %s: rank=%d seq=%d gen=%d %.1f MB in %.1f ms",
                m.rid,
                self._tp_rank,
                seq,
                generation,
                sum(lens) / 1000000.0,
                1000 * (_time.time() - t0),
            )
            return _SENT
        finally:
            conn.close()

    def _admission_window(self) -> float:
        return _ADMISSION_BACKOFF_S * (2 ** max(0, self._admission_attempts - 1) - 1)

    def _send_with_retry(self, job: dict) -> None:
        import time as _time

        m = job["meta"]
        delay = _ADMISSION_BACKOFF_S
        for attempt in range(1, self._admission_attempts + 1):
            outcome = self._send(job)
            if outcome != _REJECTED_TRANSIENT:
                return
            if attempt == self._admission_attempts:
                break
            logger.info(
                "retrying admission for %s rank=%d in %.1fs (attempt %d/%d)",
                m.rid,
                self._tp_rank,
                delay,
                attempt,
                self._admission_attempts,
            )
            _time.sleep(delay)
            delay *= 2
        logger.error(
            "gave up admitting %s rank=%d after %d attempts: its shard was NOT transferred, so the decode node will wait out its kv_transfer_timeout for this request",
            m.rid,
            self._tp_rank,
            self._admission_attempts,
        )
