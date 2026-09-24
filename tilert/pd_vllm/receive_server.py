import logging
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass, field

import torch

from tilert.pd_vllm import wire

logger = logging.getLogger("pd_vllm.receive")
FREE = "free"
RESERVED = "reserved"
TRANSFERRING = "transferring"
COMPLETE = "complete"
CANCELLING = "cancelling"


@dataclass
class ReceivedRequest:
    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    prompt_token_ids: list = field(default_factory=list)
    done_ranks: set = field(default_factory=set)
    t_first_conn: float = 0.0
    t_complete: float = 0.0
    generation: int = 0
    state: str = RESERVED
    active_writers: int = 0

    @property
    def has_live_writer(self) -> bool:
        return self.active_writers > 0


class ReceiveServer:

    def __init__(
        self,
        profile,
        max_seq_len: int,
        ctrl_port: int = 5556,
        hostname: str | None = None,
        device: str = "cuda:0",
        request_timeout: float = 120.0,
        transport: str = "mooncake",
        buffer_device: str = "cuda:0",
    ):
        self.profile = profile
        self.max_seq_len = max_seq_len
        self.ctrl_port = ctrl_port
        self.device = device
        self.request_timeout = request_timeout
        total = profile.buffer_bytes(max_seq_len)
        self.buffer_device = buffer_device
        self.host_buffer = buffer_device == "cpu"
        nshards = int((os.environ.get("TILERT_PD_SHARDS") or "8").strip() or 8)
        nshards = max(1, min(nshards, torch.cuda.device_count() or 1))
        if self.host_buffer and nshards > 1:
            logger.info(
                "receive buffer is host-resident: ignoring TILERT_PD_SHARDS=%d "
                "(a DRAM region has no per-card pressure to spread)",
                nshards,
            )
            nshards = 1
        shard_bytes = 0
        if self.host_buffer:
            from tilert.pd_vllm.transport import alloc_pinned_huge

            logger.info(
                "allocating receive buffer: %.2f GiB in pinned host DRAM (profile=%s)",
                total / 1024**3,
                profile.name,
            )
            self.buffer = alloc_pinned_huge(total)
            self.base_ptr = self.buffer.data_ptr()
        elif nshards > 1:
            shard_bytes = profile.shard_bytes(max_seq_len, nshards)
            logger.info(
                "allocating receive buffer: %d shards x %.2f GB on cuda:0..%d (profile=%s)",
                nshards,
                shard_bytes / 1024**3,
                nshards - 1,
                profile.name,
            )
            self.buffer = [
                torch.zeros(shard_bytes, dtype=torch.uint8, device=f"cuda:{i}")
                for i in range(nshards)
            ]
            self.base_ptr = [b.data_ptr() for b in self.buffer]
        else:
            logger.info(
                "allocating receive buffer: %.2f GB on %s (profile=%s)",
                total / 1024**3,
                device,
                profile.name,
            )
            self.buffer = torch.zeros(total, dtype=torch.uint8, device=device)
            self.base_ptr = self.buffer.data_ptr()
        self._hello_layout = profile.hello_layout(self.base_ptr, max_seq_len)
        from tilert.pd_vllm.transport import make_transport

        if hostname is None:
            hostname = wire.local_ip()
        dev_id = torch.device(device).index or 0
        self._transport = make_transport(transport)
        self._transport.init(hostname)
        if nshards > 1:
            rails = self._transport.rails()
            for i, ptr in enumerate(self.base_ptr):
                self._transport.register(ptr, shard_bytes, i, f"hip:{i % rails}" if rails else None)
            logger.info("receive buffer shards pinned to %s rails", rails or "auto")
        else:
            self._transport.register(self.base_ptr, total, dev_id, host=self.host_buffer)
        self._transport_meta = self._transport.local_meta()
        logger.info(
            "transport=%s ready, buffer registered (%.2f GB)", self._transport.name, total / 1024**3
        )
        self._lock = threading.Lock()
        self._current: ReceivedRequest | None = None
        self._cancelled: dict[str, float] = {}
        self._generation = 0
        self.completed: queue.Queue[ReceivedRequest] = queue.Queue()
        self._srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self._srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        self._srv.bind(("::", ctrl_port))
        self._srv.listen(32)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._accept_loop, name="pd-recv-accept", daemon=True
        )
        self._thread.start()
        logger.info("control plane listening on :%d", ctrl_port)

    def release(self, rid: str) -> None:
        with self._lock:
            self._tombstone(rid)
            cur = self._current
            if cur is None:
                return
            if cur.rid != rid:
                logger.info("release(%s) ignored: the slot now holds %s", rid, cur.rid)
                return
            if cur.has_live_writer and cur.state != COMPLETE:
                cur.state = CANCELLING
                logger.warning(
                    "release(%s) with %d writer(s) still active: slot stays claimed (cancelling) until they stop",
                    cur.rid,
                    cur.active_writers,
                )
                return
            self._current = None

    def _tombstone(self, rid: str) -> None:
        now = time.time()
        self._cancelled = {r: t for r, t in self._cancelled.items() if t > now}
        self._cancelled[rid] = now + self.request_timeout

    def _extend_tombstone(self, rid: str, window_s) -> None:
        try:
            window = float(window_s)
        except (TypeError, ValueError):
            return
        if window <= 0:
            return
        deadline = time.time() + window
        if deadline > self._cancelled.get(rid, 0.0):
            self._cancelled[rid] = deadline

    def expect(self, rid: str) -> None:
        with self._lock:
            if self._cancelled.pop(rid, None) is not None:
                logger.info("request %s re-announced; its tombstone is dropped", rid)

    def _is_tombstoned(self, rid: str) -> bool:
        deadline = self._cancelled.get(rid)
        if deadline is None:
            return False
        if deadline <= time.time():
            del self._cancelled[rid]
            return False
        return True

    def _next_generation(self) -> int:
        self._generation += 1
        return self._generation

    def _reusable(self, cur: ReceivedRequest) -> bool:
        if cur.state == COMPLETE:
            return True
        if cur.has_live_writer:
            return False
        if cur.state in (FREE, CANCELLING):
            return True
        return time.time() - cur.t_first_conn >= self.request_timeout

    def state_snapshot(self) -> dict:
        with self._lock:
            cur = self._current
            if cur is None:
                return {"state": FREE, "rid": None, "generation": None, "active_writers": 0}
            return {
                "state": cur.state,
                "rid": cur.rid,
                "generation": cur.generation,
                "active_writers": cur.active_writers,
            }

    def close(self) -> None:
        self._stop.set()
        try:
            self._srv.close()
        except OSError:
            pass

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                break
            t = threading.Thread(target=self._handle, args=(conn, addr), daemon=True)
            t.start()

    def _admit(self, req: dict, rid: str, rank: int) -> dict:
        if req.get("kv_scales") and getattr(self.profile, "kv_scales", None) is None:
            self.profile.set_kv_scales(req["kv_scales"])
        with self._lock:
            if self._is_tombstoned(rid):
                self._extend_tombstone(rid, req.get("admission_window_s"))
                logger.warning(
                    "refusing %s rank %d for now: no consumer is waiting for this request",
                    rid,
                    rank,
                )
                return wire.reject_msg("cancelling", rid=rid, detail="no_consumer")
            cur = self._current
            if cur is not None and cur.rid == rid:
                if cur.state == CANCELLING:
                    return wire.reject_msg("cancelling", rid=rid)
                if rank in cur.done_ranks:
                    return wire.reject_msg("duplicate_rank", rid=rid, rank=rank)
            elif cur is not None and (not self._reusable(cur)):
                logger.warning(
                    "rejecting %s rank %d (busy with %s, state=%s, writers=%d)",
                    rid,
                    rank,
                    cur.rid,
                    cur.state,
                    cur.active_writers,
                )
                return wire.reject_msg("busy", busy_rid=cur.rid, busy_state=cur.state)
            else:
                self._current = cur = ReceivedRequest(
                    rid=rid,
                    seq_len=int(req["seq_len"]),
                    last_prompt_token=int(req.get("last_prompt_token", 0)),
                    first_token_id=req.get("first_token_id"),
                    sampling=req.get("sampling"),
                    prompt_token_ids=list(req.get("prompt_token_ids") or []),
                    t_first_conn=time.time(),
                    generation=self._next_generation(),
                    state=RESERVED,
                )
                logger.info(
                    "request %s: seq_len=%d (generation %d)", rid, cur.seq_len, cur.generation
                )
            if not cur.prompt_token_ids and req.get("prompt_token_ids"):
                cur.prompt_token_ids = list(req["prompt_token_ids"])
                logger.info(
                    "request %s: prompt bitmap seeded from rank %d (%d ids)",
                    rid,
                    rank,
                    len(cur.prompt_token_ids),
                )
            cur.state = TRANSFERRING
            cur.active_writers += 1
            return wire.accept_msg(rid, rank, cur.generation)

    def _writer_left(self, rid: str, generation: int) -> None:
        with self._lock:
            cur = self._current
            if cur is None or cur.generation != generation:
                return
            cur.active_writers = max(0, cur.active_writers - 1)
            if cur.state == CANCELLING and cur.active_writers == 0:
                logger.info("request %s drained; receive slot free", cur.rid)
                self._current = None

    def _handle(self, conn: socket.socket, addr) -> None:
        admitted: tuple[str, int] | None = None
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(self.request_timeout)
            with self._lock:
                advisory_busy = self._current is not None and (not self._reusable(self._current))
            wire.send_msg(
                conn,
                wire.hello_msg(
                    self._transport.name,
                    self._transport_meta,
                    self.max_seq_len,
                    self.profile.layout_version,
                    self._hello_layout,
                    busy=advisory_busy,
                ),
            )
            req = wire.recv_msg(conn)
            rid, rank = (req["rid"], int(req["rank"]))
            if req.get("seq_len", 0) > self.max_seq_len:
                wire.send_msg(
                    conn,
                    wire.reject_msg(
                        "seq_len exceeds max_seq_len", rid=rid, max_seq_len=self.max_seq_len
                    ),
                )
                return
            reply = self._admit(req, rid, rank)
            wire.send_msg(conn, reply)
            if not reply.get("accepted"):
                return
            admitted = (rid, reply["generation"])
            done = wire.recv_msg(conn)
            if not done.get("done"):
                logger.warning("rank %d sent non-done message: %s", rank, done)
                return
            with self._lock:
                cur = self._current
                if cur is None or cur.rid != rid or cur.generation != admitted[1]:
                    logger.warning(
                        "ignoring done from %s rank %d: it was admitted to generation %d and the slot has moved on",
                        rid,
                        rank,
                        admitted[1],
                    )
                    return
                if cur.state == CANCELLING:
                    logger.warning(
                        "ignoring done from %s rank %d: request was abandoned", rid, rank
                    )
                    return
                cur.done_ranks.add(rank)
                logger.info(
                    "request %s: rank %d done (%d/%d)",
                    rid,
                    rank,
                    len(cur.done_ranks),
                    len(self.profile.sender_ranks),
                )
                if cur.done_ranks >= set(self.profile.sender_ranks):
                    cur.state = COMPLETE
                    cur.t_complete = time.time()
                    self.completed.put(cur)
                    logger.info(
                        "request %s: all ranks done in %.1f ms",
                        rid,
                        1000 * (cur.t_complete - cur.t_first_conn),
                    )
        except Exception:
            logger.exception("connection from %s failed", addr)
        finally:
            if admitted is not None:
                self._writer_left(*admitted)
            conn.close()
