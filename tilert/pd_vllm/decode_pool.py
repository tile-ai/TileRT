"""Decode nodes: who is free, who holds one, and who tells one to stop.

A TileRT decode engine serves ONE sequence at a time, so a node is a reservation
rather than a connection: the router hands it out, the request holds it for its
whole life, and it goes back exactly once. Two rules make that safe, and both
were learned the hard way:

* release EXACTLY once, on every exit -- including the ones that are not
  exceptions. A cancelled task (a client hanging up mid-stream) does not raise
  ``Exception``, so a handler that only caught that leaked the node until
  restart.
* cancel the node if, and only if, a request went out to it and it has not
  reported ``done``. It admits the request before answering and then holds its
  slot until its OWN timeout, so a POST that failed while waiting for headers
  still needs cancelling -- while cancelling one the node never saw is harmless
  but cancelling one that finished is not, it can land on the NEXT request.

:class:`NodeLease` is those two rules in one object, because they were spelled
out at five call sites across two handlers and each fix landed at some of them.
"""

from __future__ import annotations

import logging
import threading
import time

import requests

from tilert.pd_vllm.capabilities import NodeCapabilities

logger = logging.getLogger("pd_vllm.pool")

__all__ = ["DecodeNode", "NodeLease", "Pool", "acquire_lease", "cancel_decode"]

QUEUE_LOG_SECONDS = 0.1


class DecodeNode:
    def __init__(self, host: str, ctrl_port: int, http_port: int):
        self.host = host
        self.ctrl_port = ctrl_port
        self.http_port = http_port
        self.busy = False
        # Declared capabilities, probed lazily. None = not established yet.
        self.caps: NodeCapabilities | None = None
        self.caps_at: float = 0.0

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class Pool:
    """Decode-node reservation.

    ``queue_timeout`` > 0 makes ``acquire`` wait for a node instead of failing
    fast. A decode engine serves one sequence at a time, so a client that puts
    more than one request in flight per node — a multi-turn agentic session
    fanning out into concurrent sub-conversations, for instance — otherwise
    gets 429s for load the pool can serve a moment later. 0 keeps the
    fail-fast behaviour.
    """

    # How long a probed capability set is trusted. Bounded so a node that is
    # restarted onto a newer engine is picked up without restarting the router,
    # which the deployment contract promises ("三个组件可独立重启").
    CAPS_TTL_S = 60.0
    CAPS_TIMEOUT_S = 2.0

    def __init__(self, nodes: list[DecodeNode], queue_timeout: float = 0.0):
        self.nodes = nodes
        self.queue_timeout = queue_timeout
        self._cv = threading.Condition()
        self._caps_lock = threading.Lock()

    def acquire(self) -> DecodeNode | None:
        """Reserve a node, or None once ``queue_timeout`` elapses.

        Blocks while waiting; both call sites already hop off the event loop
        via ``run_in_threadpool``, so other streams keep being served.
        """
        deadline = time.monotonic() + self.queue_timeout
        with self._cv:
            while True:
                for n in self.nodes:
                    if not n.busy:
                        n.busy = True
                        return n
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(remaining)

    def release(self, node: DecodeNode) -> None:
        with self._cv:
            node.busy = False
            self._cv.notify()

    # ── capability probing ───────────────────────────────────────────────
    def _node_caps(self, node: DecodeNode) -> NodeCapabilities:
        """One node's declared capabilities, cached for ``CAPS_TTL_S``.

        A probe that fails returns "nothing supported" WITHOUT caching, so the
        answer is conservative right now and re-probed on the next request
        rather than pinned for a whole TTL. Probing lazily (not at startup)
        keeps the router startable before any decode node exists.
        """
        with self._caps_lock:
            if node.caps is not None and time.time() - node.caps_at < self.CAPS_TTL_S:
                return node.caps
        try:
            r = requests.get(f"{node.http_base}/capabilities", timeout=self.CAPS_TIMEOUT_S)
            r.raise_for_status()
            caps = NodeCapabilities.from_payload(r.json())
        except Exception as e:
            # Includes a node predating /capabilities (404): it cannot declare
            # support, so it does not get credit for any.
            logger.warning(
                "capability probe failed for %s (%s); treating "
                "every optional field as unsupported",
                node.http_base,
                e,
            )
            return NodeCapabilities()
        with self._caps_lock:
            node.caps, node.caps_at = caps, time.time()
        logger.info("capabilities for %s: %s", node.http_base, caps.to_payload())
        return caps

    def capabilities(self) -> NodeCapabilities:
        """What a request may rely on whichever node serves it.

        The intersection across the pool, because validation runs before a node
        is chosen. An empty pool yields no support, which is also correct: there
        is nothing that could execute the field.
        """
        result: NodeCapabilities | None = None
        for n in self.nodes:
            caps = self._node_caps(n)
            result = caps if result is None else result.intersect(caps)
        return result or NodeCapabilities()


def cancel_decode(node, rid: str) -> None:
    """Tell a decode node to stop working on ``rid``.

    Best effort: the node may already have finished, and a failed POST has no
    remedy. Callers run it off the response path so a wedged node cannot delay
    the reply.
    """
    try:
        requests.post(f"{node.http_base}/pd/cancel", json={"rid": rid}, timeout=5)
    except Exception:
        logger.warning("cancel POST failed for %s", rid)


class NodeLease:
    """One node, held for one request.

    ``rid`` and ``dispatched`` are set as the request learns them: the rid comes
    from the prefill reply, and ``dispatched`` goes true the moment a POST leaves
    for the node -- not when it succeeds, since a timeout while waiting for
    headers may still have been admitted.

    ``release`` is idempotent and takes the one fact the lease cannot know: did
    the NODE say it was done. Cancelling runs on a plain thread, off the response
    path, so a wedged node cannot delay the reply, and off the event loop, where
    an ``await`` could be cancelled before it fires.
    """

    def __init__(self, pool: Pool, node: DecodeNode):
        self.pool = pool
        self.node = node
        self.rid: str | None = None
        self.dispatched = False
        self._released = False

    def release(self, *, terminated: bool = False) -> None:
        if self._released:
            return
        self._released = True
        self.pool.release(self.node)
        if self.dispatched and not terminated and self.rid is not None:
            threading.Thread(target=cancel_decode, args=(self.node, self.rid), daemon=True).start()

    def __enter__(self) -> NodeLease:
        return self

    def __exit__(self, *exc) -> None:
        # A path that knows the node terminated releases explicitly before
        # leaving; this is the backstop for every other exit.
        self.release()


def acquire_lease(pool: Pool) -> tuple[NodeLease | None, float]:
    """Reserve a node, and report how long the caller queued for it."""
    t0 = time.monotonic()
    node = pool.acquire()
    waited = time.monotonic() - t0
    if node is None:
        return None, waited
    if waited >= QUEUE_LOG_SECONDS:
        logger.info("queued %.1fs for decode node %s", waited, node.host)
    return NodeLease(pool, node), waited
