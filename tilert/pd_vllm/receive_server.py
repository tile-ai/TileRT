"""Decode-side receive server (W4): Mooncake buffer + TCP control plane.

Owns one big cuda:0 receive buffer (region layout supplied by the model
profile), registered with a local Mooncake TransferEngine. Runs a TCP
server; each participating prefill rank connects per request, gets the hello
(session id + buffer base addresses), sends its request metadata, **waits to be
admitted**, then RDMA-writes its sections and reports ``done``. When all 8 ranks
report, the assembled request is handed to the consumer via a queue.

bs=1: one in-flight request, so the buffer has exactly one tenant at a time and
admission is what enforces it. A rank asking for a different rid while the
current tenant is still in use is rejected (``accepted: false``) and must not
write; ``busy`` in the hello is advisory only, because the rid is not known yet.
The router's gated dispatch should make a rejection rare, but it is the
correctness fence, not an optimisation: a write that ignores it lands inside
another request's KV.

Tenancy is tracked explicitly (``FREE``/``RESERVED``/``TRANSFERRING``/
``COMPLETE``/``CANCELLING``) with a monotonic ``generation`` and a count of
ranks that may still be writing. The buffer is never handed to a different
request while that count is non-zero -- the previous version inferred ownership
from a wall-clock comparison and replaced timed-out requests whose ranks were
still mid-RDMA.
"""

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


# Receive-buffer tenancy states. The buffer holds one request at a time, so
# "who owns it right now" has to be explicit -- the previous code inferred it
# from `t_complete == 0.0` plus a wall-clock comparison, and that inference is
# what allowed a timed-out request to be replaced while its ranks were still
# writing into the buffer.
FREE = "free"  # no tenant; admissible
RESERVED = "reserved"  # rid claimed, no rank writing yet
TRANSFERRING = "transferring"  # >=1 rank admitted and writing
COMPLETE = "complete"  # every sender rank reported done; queued
CANCELLING = "cancelling"  # abandoned; NOT reusable until writers stop


@dataclass
class ReceivedRequest:
    rid: str
    seq_len: int
    last_prompt_token: int
    first_token_id: int | None
    sampling: dict | None
    # Full prompt ids when the prefill connector sent them (rank 0, penalty requests
    # only); empty otherwise -- the engine then leaves the prompt bitmap clear.
    prompt_token_ids: list = field(default_factory=list)
    done_ranks: set = field(default_factory=set)
    t_first_conn: float = 0.0
    t_complete: float = 0.0
    # Monotonic tenancy id. Echoed on accept and checked on done, so a message
    # from a previous tenant cannot be counted towards the current one.
    generation: int = 0
    state: str = RESERVED
    # Ranks admitted but not yet known to have stopped writing. The buffer must
    # not be handed to another request while this is non-zero.
    active_writers: int = 0

    @property
    def has_live_writer(self) -> bool:
        """Whether a rank may still be RDMA-writing into the buffer.

        The hard half of the reuse rule: while this is true the buffer cannot be
        handed to another request under any circumstance, timeout included. The
        soft half (has this tenancy finished, or has it aged out?) is policy and
        lives in :meth:`ReceiveServer._reusable`, which owns the timeout.
        """
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
        # Tombstones: rid -> deadline. A rid whose consumer has let go must not
        # be admitted again, and until now nothing recorded that. See release().
        self._cancelled: dict[str, float] = {}
        # Monotonic tenancy counter. Never reused, so a message from an earlier
        # tenant is always distinguishable from the current one.
        self._generation = 0
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

    def release(self, rid: str) -> None:
        """Give up the receive slot held by ``rid``.

        Scoped to a rid ON PURPOSE. A caller only ever knows about its own
        request, and the slot it once held may since have been handed to another:
        a transfer that arrives after its consumer gave up is enqueued with no
        one waiting for it, and the NEXT request drains it as unmatched. An
        unscoped release there would free whatever tenancy is current -- that
        next request's own -- and it would then never complete, because its
        ranks' `done` messages are dropped once the tenancy is cancelled.
        Releasing a rid that no longer owns the slot is a no-op.

        Only actually frees the buffer when nothing is writing into it. A caller
        that gives up on a request whose ranks are still mid-RDMA (a rejected
        request, a drained entry) moves it to ``CANCELLING`` instead: the slot
        stays claimed until the writers stop, because freeing it there is
        precisely how one request's KV lands inside another's buffer. Every
        sender connection carries ``request_timeout`` as its socket timeout, so
        the drain is bounded without needing to force it.

        Always leaves a TOMBSTONE for ``rid``, whatever the slot turns out to
        hold. The ``CANCELLING`` state above only refuses senders while a writer
        is still live; it says nothing about a rank that has not connected YET.
        A cancel that wins before any sender arrives leaves ``_current`` None,
        so without a tombstone the late rank finds a free buffer, is admitted,
        and holds it while the next request's ranks are turned away "busy" until
        they exhaust their admission retries -- that request then waits out its
        whole kv_transfer_timeout. The same hole is open after a normal
        completion: a rank that never reported done is not in ``done_ranks``, so
        it would open a fresh tenancy rather than be refused as a duplicate.

        ``request_timeout`` is the right lifetime because it is the senders'
        socket timeout: past it, no rank can still be trying to join this rid.
        """
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
                    "release(%s) with %d writer(s) still active: slot stays "
                    "claimed (cancelling) until they stop",
                    cur.rid,
                    cur.active_writers,
                )
                return
            self._current = None

    def _tombstone(self, rid: str) -> None:
        """Record ``rid`` as done with, and drop the tombstones that expired.

        Caller holds ``self._lock``. Pruning here rather than on a timer keeps
        the map bounded without a second thread to reason about.
        """
        now = time.time()
        self._cancelled = {r: t for r, t in self._cancelled.items() if t > now}
        self._cancelled[rid] = now + self.request_timeout

    def _extend_tombstone(self, rid: str, window_s) -> None:
        """Push ``rid``'s tombstone out to cover ``window_s`` from now.

        Caller holds ``self._lock``. Never shortens one: a second rank
        declaring a smaller budget must not expose the request to the first.
        """
        try:
            window = float(window_s)
        except (TypeError, ValueError):
            return  # older sender: keep the default
        if window <= 0:
            return
        deadline = time.time() + window
        if deadline > self._cancelled.get(rid, 0.0):
            self._cancelled[rid] = deadline

    def expect(self, rid: str) -> None:
        """Announce that a consumer is now waiting for ``rid``.

        Clears any tombstone for it. Called from ``/pd/decode`` on admission,
        which is precisely the event that distinguishes the two things a
        tombstone cannot tell apart: a request everyone has given up on, and the
        same request coming back because vLLM rescheduled it. The first is never
        re-announced; the second always is, and its senders' admission retries
        are what carry them across the gap.
        """
        with self._lock:
            if self._cancelled.pop(rid, None) is not None:
                logger.info("request %s re-announced; its tombstone is dropped", rid)

    def _is_tombstoned(self, rid: str) -> bool:
        """Caller holds ``self._lock``."""
        deadline = self._cancelled.get(rid)
        if deadline is None:
            return False
        if deadline <= time.time():
            del self._cancelled[rid]
            return False
        return True

    def _next_generation(self) -> int:
        """Caller holds ``self._lock``."""
        self._generation += 1
        return self._generation

    def _reusable(self, cur: ReceivedRequest) -> bool:
        """Whether ``cur``'s buffer may be given to a different rid.

        Three clauses, and the order matters:

        1. ``COMPLETE`` is reusable whatever ``active_writers`` says. A rank sends
           ``done`` only AFTER its RDMA write has returned, so once every sender
           rank is done nothing is writing; a non-zero count there is just
           sockets that have not closed yet. ``release()`` makes the same
           exception, and the two must agree -- if this clause moved below the
           writer check, a request could be released by one rule and refused by
           the other.
        2. **Otherwise, never under a live writer.** No timeout overrides this.
           Freeing the buffer while a rank is mid-RDMA is what puts one request's
           KV inside another's, and no elapsed time makes that safe.
        3. Reusable once abandoned (``CANCELLING``) or **aged out** -- a tenancy
           still marked ``RESERVED``/``TRANSFERRING`` whose writers have all gone
           without completing it. That happens when every sender dies
           mid-transfer, and without the age-out the slot would be held for good:
           nothing calls ``release()`` for a rid whose ``/pd/decode`` never
           arrived (the router gave up on the prefill leg, say). The original code
           aged out on wall clock ALONE, which is how a live writer got
           overwritten; this keeps the bound and drops the hazard.
        """
        if cur.state == COMPLETE:
            return True
        if cur.has_live_writer:
            return False
        if cur.state in (FREE, CANCELLING):
            return True
        return (time.time() - cur.t_first_conn) >= self.request_timeout

    def state_snapshot(self) -> dict:
        """Current tenancy, for tests and the decode server's status endpoint."""
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

    def _admit(self, req: dict, rid: str, rank: int) -> dict:
        """Decide whether ``rank`` of ``rid`` may write, under the lock.

        Returns the message to send back: an accept carrying the tenancy
        generation, or a reject. Admission is granted only for a tenancy this
        rank is actually joining, so the sender can key its RDMA on the reply.
        """
        with self._lock:
            if self._is_tombstoned(rid):
                # Checked BEFORE anything about the current tenancy: this rid is
                # dead whatever the slot holds now, and admitting it would let a
                # request nobody is waiting for occupy the buffer.
                #
                # RETRYABLE on purpose. A rid is not unique to a tenancy -- it
                # is derived from the vLLM request id, so a preempted request
                # rescheduled by vLLM sends again under the SAME rid. That retry
                # must not lose its shard to a tombstone left by the attempt
                # before it, and it announces itself by calling /pd/decode
                # again, which is what clears the tombstone (see expect()).
                # Refusing transiently lets the sender wait for that to happen;
                # a rid nobody re-announces just exhausts its retries, which
                # costs a dead request nothing.
                #
                # Reported as `cancelling`, a reason senders ALREADY retry,
                # rather than a new one. A new reason is permanent to any
                # connector built before it: during a rolling upgrade it would
                # drop the shard, and the rescheduled request would then wait
                # out its kv_transfer timeout. The distinction is only useful
                # in a log, so it goes in `detail`.
                #
                # Extend to outlast THIS sender's remaining retries. The
                # default lifetime is request_timeout, which is one
                # connection's socket timeout -- but every retry opens a new
                # connection, so a sender configured with enough attempts can
                # still be trying after the tombstone has expired, and would
                # then be admitted for a request nobody wants. Only the sender
                # knows its budget, so it declares it.
                self._extend_tombstone(rid, req.get("admission_window_s"))
                logger.warning(
                    "refusing %s rank %d for now: no consumer is " "waiting for this request",
                    rid,
                    rank,
                )
                return wire.reject_msg("cancelling", rid=rid, detail="no_consumer")
            cur = self._current
            if cur is not None and cur.rid == rid:
                if cur.state == CANCELLING:
                    return wire.reject_msg("cancelling", rid=rid)
                if rank in cur.done_ranks:
                    # A duplicate for a rank that already finished would be
                    # counted twice and could complete the request early.
                    return wire.reject_msg("duplicate_rank", rid=rid, rank=rank)
            elif cur is not None and not self._reusable(cur):  # noqa: R505 (exclusive branches)
                # Busy with a DIFFERENT rid whose buffer is still in use. This is
                # the reply the sender used to ignore, writing anyway.
                logger.warning(
                    "rejecting %s rank %d (busy with %s, state=%s, " "writers=%d)",
                    rid,
                    rank,
                    cur.rid,
                    cur.state,
                    cur.active_writers,
                )
                return wire.reject_msg("busy", busy_rid=cur.rid, busy_state=cur.state)
            else:
                # Free, or the previous tenant is finished/drained: new tenancy.
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

            # Absorb the prompt ids from whichever rank carries them. Only
            # rank 0 sends them (prefill_connector._send), but ranks connect
            # in ARBITRARY order and only the first one to arrive builds the
            # ReceivedRequest -- so keying this off the creation path would
            # drop the ids ~7/8 of the time, non-deterministically.
            if not cur.prompt_token_ids and req.get("prompt_token_ids"):
                cur.prompt_token_ids = list(req["prompt_token_ids"])
                logger.info(
                    "request %s: prompt bitmap seeded from rank %d " "(%d ids)",
                    rid,
                    rank,
                    len(cur.prompt_token_ids),
                )

            cur.state = TRANSFERRING
            cur.active_writers += 1
            return wire.accept_msg(rid, rank, cur.generation)

    def _writer_left(self, rid: str, generation: int) -> None:
        """One admitted rank has stopped writing (done, error, or disconnect).

        Called from the connection's ``finally`` so a sender that dies mid-RDMA
        still releases its claim. Bounded by the socket timeout, which is why the
        cancelling drain needs no forced override.
        """
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
            # `busy` in hello is advisory only (a same-rid rank must still
            # proceed, and the rid is not known yet); the authoritative decision
            # is the accept/reject below.
            with self._lock:
                advisory_busy = self._current is not None and not self._reusable(self._current)
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

            # wait for this rank's done (RDMA happens meanwhile)
            done = wire.recv_msg(conn)
            if not done.get("done"):
                logger.warning("rank %d sent non-done message: %s", rank, done)
                return
            with self._lock:
                cur = self._current
                if cur is None or cur.rid != rid or cur.generation != admitted[1]:
                    # The tenancy this rank was admitted against is gone. Its
                    # bytes went into a buffer that has since been reassigned or
                    # abandoned, so counting the done would attribute them to
                    # whoever holds the slot now.
                    logger.warning(
                        "ignoring done from %s rank %d: it was admitted to "
                        "generation %d and the slot has moved on",
                        rid,
                        rank,
                        admitted[1],
                    )
                    return
                if cur.state == CANCELLING:
                    logger.warning(
                        "ignoring done from %s rank %d: request was " "abandoned", rid, rank
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
