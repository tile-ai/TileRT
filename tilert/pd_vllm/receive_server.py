"""Decode-side receive server (W4): Mooncake buffer + TCP control plane."""

import contextlib
import logging
import queue
import socket
import threading
import time
from dataclasses import dataclass, field

import torch

from tilert.pd_vllm import wire

logger = logging.getLogger("pd_vllm.receive")


@dataclass
class ReceivedRequest:
    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    done_ranks: set = field(default_factory=set)
    t_first_conn: float = 0.0
    t_complete: float = 0.0


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
    ):
        self.profile = profile
        self.max_seq_len = max_seq_len
        self.ctrl_port = ctrl_port
        self.device = device
        self.request_timeout = request_timeout

        total = profile.buffer_bytes(max_seq_len)
        logger.info(
            "allocating receive buffer: %.2f GB on %s (profile=%s)",
            total / 1024**3,
            device,
            profile.name,
        )
        self.buffer = torch.zeros(total, dtype=torch.uint8, device=device)
        self.base_ptr = self.buffer.data_ptr()
        self._hello_layout = profile.hello_layout(self.base_ptr, max_seq_len)

        # RDMA transport (mooncake default / nixl), single cuda:0 registration
        from tilert.pd_vllm.transport import make_transport

        if hostname is None:
            hostname = wire.local_ip()
        dev_id = torch.device(device).index or 0
        self._transport = make_transport(transport)
        self._transport.init(hostname)
        self._transport.register(self.base_ptr, total, dev_id)
        self._transport_meta = self._transport.local_meta()
        logger.info(
            "transport=%s ready, buffer registered (%.2f GB)", self._transport.name, total / 1024**3
        )

        self._lock = threading.Lock()
        self._current: ReceivedRequest | None = None
        self.completed: queue.Queue[ReceivedRequest] = queue.Queue()

        # dual-stack: accept IPv4 (v4-mapped) and IPv6, incl. link-local peers
        # (e.g. an IPv6-only decode node reached over fe80::.../bond0)
        self._srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with contextlib.suppress(OSError):
            self._srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        self._srv.bind(("::", ctrl_port))
        self._srv.listen(32)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._accept_loop, name="pd-recv-accept", daemon=True
        )
        self._thread.start()
        logger.info("control plane listening on :%d", ctrl_port)

    # ── public ───────────────────────────────────────────────────────────

    def release(self) -> None:
        """Mark the single receive slot free (call after inject/decode)."""
        with self._lock:
            self._current = None

    def close(self) -> None:
        self._stop.set()
        with contextlib.suppress(OSError):
            self._srv.close()

    # ── accept / per-connection handling ─────────────────────────────────

    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except OSError:
                break
            t = threading.Thread(target=self._handle, args=(conn, addr), daemon=True)
            t.start()

    def _handle(self, conn: socket.socket, addr) -> None:
        try:
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            conn.settimeout(self.request_timeout)
            # `busy` in hello is advisory (a same-rid rank must still proceed);
            # the authoritative accept/reject happens once the rid is known.
            with self._lock:
                advisory_busy = self._current is not None and self._current.t_complete == 0.0
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
            rid, rank = req["rid"], int(req["rank"])
            if req.get("seq_len", 0) > self.max_seq_len:
                wire.send_msg(conn, {"error": "seq_len exceeds max_seq_len"})
                return

            with self._lock:
                cur = self._current
                if cur is None or cur.rid != rid:
                    if (
                        cur is not None
                        and cur.t_complete == 0.0
                        and time.time() - cur.t_first_conn < self.request_timeout
                    ):
                        # busy with a different in-flight rid
                        wire.send_msg(conn, {"error": "busy", "busy_rid": cur.rid})
                        logger.warning("rejecting %s (busy with %s)", rid, cur.rid)
                        return
                    self._current = cur = ReceivedRequest(
                        rid=rid,
                        seq_len=int(req["seq_len"]),
                        last_prompt_token=int(req.get("last_prompt_token", 0)),
                        first_token_id=req.get("first_token_id"),
                        sampling=req.get("sampling"),
                        t_first_conn=time.time(),
                    )
                    logger.info("request %s: seq_len=%d", rid, cur.seq_len)

            # wait for this rank's done (RDMA happens meanwhile)
            done = wire.recv_msg(conn)
            if not done.get("done"):
                logger.warning("rank %d sent non-done message: %s", rank, done)
                return
            with self._lock:
                cur = self._current
                if cur is None or cur.rid != rid:
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
            conn.close()
