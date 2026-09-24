from __future__ import annotations

import logging
import threading
import time

import requests

from tilert.pd_vllm.capabilities import NodeCapabilities, engine_capabilities

logger = logging.getLogger("pd_vllm.pool")
__all__ = ["DecodeNode", "NodeLease", "Pool", "acquire_lease", "cancel_decode"]
QUEUE_LOG_SECONDS = 0.1


class DecodeNode:

    def __init__(self, host: str, ctrl_port: int, http_port: int):
        self.host = host
        self.ctrl_port = ctrl_port
        self.http_port = http_port
        self.busy = False
        self.caps: NodeCapabilities | None = None
        self.caps_at: float = 0.0

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class Pool:
    CAPS_TTL_S = 60.0
    CAPS_TIMEOUT_S = 2.0

    def __init__(self, nodes: list[DecodeNode], queue_timeout: float = 0.0):
        self.nodes = nodes
        self.queue_timeout = queue_timeout
        self._cv = threading.Condition()
        self._caps_lock = threading.Lock()

    def acquire(self) -> DecodeNode | None:
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

    def _node_caps(self, node: DecodeNode) -> NodeCapabilities:
        with self._caps_lock:
            if node.caps is not None and time.time() - node.caps_at < self.CAPS_TTL_S:
                return node.caps
        try:
            r = requests.get(f"{node.http_base}/capabilities", timeout=self.CAPS_TIMEOUT_S)
            r.raise_for_status()
            caps = NodeCapabilities.from_payload(r.json())
        except Exception as e:
            logger.warning(
                "capability probe failed for %s (%s); treating every optional field as unsupported",
                node.http_base,
                e,
            )
            return NodeCapabilities()
        with self._caps_lock:
            node.caps, node.caps_at = (caps, time.time())
        logger.info("capabilities for %s: %s", node.http_base, caps.to_payload())
        return caps

    def capabilities(self) -> NodeCapabilities:
        result: NodeCapabilities | None = None
        for n in self.nodes:
            caps = self._node_caps(n)
            result = caps if result is None else result.intersect(caps)
        return result or NodeCapabilities()


def cancel_decode(node, rid: str) -> None:
    try:
        requests.post(f"{node.http_base}/pd/cancel", json={"rid": rid}, timeout=5)
    except Exception:
        logger.warning("cancel POST failed for %s", rid)


class NodeLease:

    def __init__(self, pool: Pool, node: DecodeNode):
        self.pool = pool
        self.node = node
        self.rid: str | None = None
        self.dispatched = False
        self._released = False

    @property
    def released(self) -> bool:
        """Whether this lease has already gone back to the pool. Read by the
        backstop that reclaims a lease whose handler never released it, so the
        normal path is not logged as a leak."""
        return self._released

    def release(self, *, terminated: bool = False) -> None:
        if self._released:
            return
        self._released = True
        self.pool.release(self.node)
        if self.dispatched and (not terminated) and (self.rid is not None):
            threading.Thread(target=cancel_decode, args=(self.node, self.rid), daemon=True).start()

    def __enter__(self) -> NodeLease:
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def acquire_lease(pool: Pool) -> tuple[NodeLease | None, float]:
    t0 = time.monotonic()
    node = pool.acquire()
    waited = time.monotonic() - t0
    if node is None:
        return (None, waited)
    if waited >= QUEUE_LOG_SECONDS:
        logger.info("queued %.1fs for decode node %s", waited, node.host)
    return (NodeLease(pool, node), waited)
