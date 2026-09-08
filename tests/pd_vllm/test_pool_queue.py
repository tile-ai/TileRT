"""A decode node must be waited for, not refused, when ``queue_timeout`` > 0.

A TileRT decode engine serves one sequence at a time, so the router's pool
is as deep as the node count. An agentic session fans a single conversation
out into concurrent sub-conversations, which puts more requests in flight
than the pool has nodes -- with fail-fast reservation the surplus turns
into ``429``s even though the pool can serve them a moment later.
``queue_timeout`` makes the surplus wait instead; ``0`` keeps the original
fail-fast behaviour, so fixed-sequence-length runs are unaffected.

The waiting side matters as much as the timeout: ``release`` has to wake a
waiter, otherwise a queued request sleeps out its full timeout even though a
node freed up immediately.

No GPU, no tilert, no vllm: ``Pool`` is pure ``threading``.

Run:
  CUDA_VISIBLE_DEVICES= python -m pytest \
      tests/pd_vllm/test_pool_queue.py -v
"""

from __future__ import annotations

import threading
import time

from tilert.pd_vllm.decode_pool import DecodeNode, Pool

TIMEOUT = 0.3


def _pool(queue_timeout: float, nodes: int = 1) -> Pool:
    return Pool(
        [DecodeNode("h%d" % i, 5556 + i, 5557 + i) for i in range(nodes)],
        queue_timeout=queue_timeout,
    )


def test_fail_fast_is_the_default() -> None:
    """Constructed without the argument, the pool refuses as it always did."""
    pool = Pool([DecodeNode("h0", 5556, 5557)])
    assert pool.queue_timeout == 0.0
    assert pool.acquire() is not None
    t0 = time.monotonic()
    assert pool.acquire() is None
    assert time.monotonic() - t0 < TIMEOUT, "fail-fast must not block"


def test_timeout_zero_still_hands_out_free_nodes() -> None:
    pool = _pool(0.0, nodes=2)
    first, second = pool.acquire(), pool.acquire()
    assert first is not None and second is not None and first is not second
    assert pool.acquire() is None
    pool.release(first)
    assert pool.acquire() is first


def test_waits_until_a_node_is_released() -> None:
    """The queued caller gets the node a releaser frees, not a 429."""
    pool = _pool(30.0)
    held = pool.acquire()
    assert held is not None

    def _release_soon() -> None:
        time.sleep(0.05)
        pool.release(held)

    threading.Thread(target=_release_soon, daemon=True).start()
    t0 = time.monotonic()
    got = pool.acquire()
    waited = time.monotonic() - t0
    assert got is held, "the freed node must be handed to the waiter"
    assert waited < 30.0, "release must wake the waiter, not let it time out"


def test_gives_up_after_the_timeout() -> None:
    """Nothing frees up, so the caller is refused -- but only after waiting."""
    pool = _pool(TIMEOUT)
    assert pool.acquire() is not None
    t0 = time.monotonic()
    assert pool.acquire() is None
    assert time.monotonic() - t0 >= TIMEOUT


def test_every_waiter_is_served_when_nodes_come_back() -> None:
    """One release must not wake a waiter that then loses the node again."""
    pool = _pool(30.0, nodes=2)
    held = [pool.acquire(), pool.acquire()]
    assert all(n is not None for n in held)
    got: list[DecodeNode | None] = []
    lock = threading.Lock()

    def _waiter() -> None:
        n = pool.acquire()
        with lock:
            got.append(n)

    threads = [threading.Thread(target=_waiter, daemon=True) for _ in range(2)]
    for t in threads:
        t.start()
    for n in held:
        time.sleep(0.05)
        assert n is not None
        pool.release(n)
    for t in threads:
        t.join(timeout=10)
    assert len(got) == 2 and all(n is not None for n in got)
    assert len({id(n) for n in got}) == 2, "two waiters must not share one node"
