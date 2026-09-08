"""PD decode server: HTTP orchestration around receive -> convert -> inject -> decode.

Internal token-level API (the client-facing OpenAI layer lives in pd_router /
a later serving layer):

  POST /pd/decode   {rid, first_token_id, max_tokens, sampling?, timeout_s?}
      Waits for the wire transfer of `rid` to complete, converts, injects
      into the engine, decodes, returns {"rid", "token_ids", "timing_ms"}.
      Refuses (501) before the wire-wait if `sampling` asks for something this
      engine cannot execute -- see /capabilities.
  GET  /health          {"status": "ok"}
  GET  /capabilities    which optional sampling params this engine honours; the
                        router reads it to refuse such a request before the
                        prefill instance has run the prompt
  GET  /decode_status   {"status": "idle"|"busy", "current_rid": ...}

bs=1: a busy server answers 429 immediately (the router's gated dispatch
should make that unreachable).

Run (stub engine, plumbing test):
  python -m tilert.pd_vllm.decode_server \
      --engine stub --max-seq-len 4096 --ctrl-port 5556 --http-port 5557
"""

import argparse
import contextlib
import json
import logging
import os
import queue as queue_mod
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from tilert.pd_vllm.capabilities import (
    CapabilityError,
    engine_capabilities,
    validate_generation_request,
)
from tilert.pd_vllm.grammar_spec import (
    GrammarError,
    GrammarViolationError,
)
from tilert.pd_vllm.receive_server import ReceiveServer

logger = logging.getLogger("pd_vllm.decode_server")

# How long to wait for the KV of a request we are rejecting after its prefill has
# already run, before giving up and releasing the receive slot anyway. The
# transfer is already in flight when we reject, so this is normally milliseconds;
# it only bites if the prefill instance died mid-push.
_ABANDON_DRAIN_S = 30.0

# Returned by _drain_own_kv when the wire-wait was cancelled rather than timing
# out. A distinct sentinel because the two mean opposite things to an operator:
# a timeout points at the RDMA path, a cancel means the client left.
_CANCELLED = object()


class LogprobsUnavailable(Exception):
    """The active engine cannot produce logprobs -> HTTP 501.

    Refused rather than answered without the field: a caller cannot tell that
    apart from the model having had nothing to report.
    """


class DecodeBody(BaseModel):
    rid: str
    first_token_id: int
    max_tokens: int = 256
    sampling: dict | None = None
    timeout_s: float = 120.0
    stream: bool = False
    # Constrained decoding: engine grammar spec ({"type","value"}) + thinking
    # gate. None -> unconstrained (the router omits it for plain requests).
    grammar_spec: dict | None = None
    enable_thinking: bool = True
    # Per-token log probabilities. None -> not requested (the router omits it),
    # 0 -> the chosen token's logprob only, N -> N candidates per position.
    top_logprobs: int | None = None


DECODE_POLL_S = max(0.0, float(os.environ.get("TILERT_DECODE_POLL_MS") or "200")) / 1000.0


def build_app(server: ReceiveServer, engine) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()
    state: dict[str, Any] = {"current_rid": None}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities():
        """Which optional generation parameters this node can execute.

        The router polls this so it can refuse an unsupported request before
        prefilling it. Without it the same request still fails -- but only after
        vLLM has run the prompt and pushed its KV over RDMA, which is the whole
        cost of the request for none of the answer.

        Sourced from the live engine (see ``engine_capabilities``), so a probe
        demotion inside the adapter shows up here too. ``logprobs`` is reported
        alongside for symmetry with the 501 ``/pd/decode`` already returns.
        """
        caps = engine_capabilities(engine)
        payload = caps.to_payload()
        payload["logprobs"] = bool(getattr(engine, "supports_logprobs", lambda: False)())
        return {
            "profile": getattr(server.profile, "name", None),
            "engine": type(engine).__name__,
            "capabilities": payload,
        }

    @app.get("/decode_status")
    def decode_status():
        busy = lock.locked()
        return {"status": "busy" if busy else "idle", "current_rid": state["current_rid"]}

    @app.post("/pd/cancel")
    def pd_cancel(body: dict):
        """Explicit kill switch: cancel the in-flight decode for `rid`.

        Deterministic cancel path — dead-connection detection at the
        transport layer is unreliable (asyncio writes to a closed socket
        do not raise), so the router calls this on client disconnect.
        """
        rid = body.get("rid")
        ev = state.get("cancel_event")
        if rid and rid == state["current_rid"] and ev is not None:
            ev.set()
            logger.info("cancel requested for %s", rid)
            return {"cancelled": rid}
        return JSONResponse(
            {"error": "no matching in-flight request", "current_rid": state["current_rid"]},
            status_code=404,
        )

    def _cleanup():
        try:
            engine.reset()
        except Exception:
            logger.exception("engine reset failed")
        # Scoped to OUR rid: by now the slot may hold a later request whose
        # transfer started while this one was giving up.
        if state["current_rid"]:
            server.release(state["current_rid"])
        state["current_rid"] = None
        state["cancel_event"] = None
        lock.release()

    def _drain_own_kv(rid: str, timeout_s: float, cancel=None):
        """Pop from server.completed until OUR rid surfaces.

        Returns the request, ``None`` on timeout, or ``_CANCELLED`` if ``cancel``
        was set while waiting.

        Observing ``cancel`` is the point: a client that hangs up during the KV
        transfer used to leave this loop running to the full timeout_s (120 s by
        default), and since bs=1 the whole node was unavailable for that long.
        The cancel is checked every poll, so it takes effect within one 0.5 s
        tick rather than two minutes.

        Entries for other rids are stale transfers whose consumer never called
        /pd/decode (or was rejected before it could) — drop and release them so
        they stop holding the single receive slot.
        """
        deadline = time.time() + timeout_s
        while True:
            if cancel is not None and cancel.is_set():
                return _CANCELLED
            remaining = deadline - time.time()
            if remaining <= 0:
                return None
            try:
                cand = server.completed.get(timeout=min(remaining, 0.5))
            except queue_mod.Empty:
                continue
            if cand.rid == rid:
                return cand
            logger.warning("dropping unmatched request %s (waiting for %s)", cand.rid, rid)
            # The dropped entry's rid, not ours and not "whatever is current":
            # this transfer arrived after its own consumer gave up, and the
            # tenancy may already belong to a later request.
            server.release(cand.rid)

    def _abandon_pending_kv(rid: str, cancel=None) -> None:
        """Release the receive slot for a request we are rejecting post-prefill.

        The prefill request always runs before /pd/decode, so by the time we reject a
        request (bad grammar spec, missing backend, unexpected prep failure) vLLM
        has already pushed its KV, or is pushing it. Dropping only the lock
        leaves that transfer owning the receive server's single slot: the NEXT
        request's ranks are turned away with "busy", its KV never lands, and its
        /pd/decode blocks until the 120 s kv_transfer_timeout — one bad grammar
        stalls the following request for two minutes.

        So drain our own entry before releasing. The transfer is already under
        way, so this normally costs milliseconds; the bound stops a prefill that
        died mid-push from holding the slot indefinitely.

        ``cancel`` is the SAME event /pd/cancel sets, and this drain has to watch
        it like the phase-1 one does. Without it a cancel arriving here is
        answered 200 -- the event is armed and this rid is still current -- while
        the drain runs on to _ABANDON_DRAIN_S, so the client is told the request
        was cancelled and the next one is refused 429 for another 30 s. The
        bound alone is not enough: it is sized for a transfer already in flight,
        not for one whose prefill leg is never coming.
        """
        try:
            if _drain_own_kv(rid, _ABANDON_DRAIN_S, cancel) is None:
                logger.warning(
                    "abandoned %s: its KV did not arrive within " "%.0fs", rid, _ABANDON_DRAIN_S
                )
        except Exception:
            logger.exception("draining KV for abandoned %s failed", rid)
        finally:
            _cleanup()

    def _log_reqstat(body, req, n_tokens, timing):
        logger.info(
            "REQSTAT rid=%s seq=%d completion=%d %s",
            body.rid,
            req.seq_len,
            n_tokens,
            " ".join(f"{k}={v}" for k, v in timing.items()),
        )

    @app.post("/pd/decode")
    def pd_decode(body: DecodeBody):
        if not lock.acquire(blocking=False):
            return JSONResponse(
                {"error": "busy", "current_rid": state["current_rid"]}, status_code=429
            )
        state["current_rid"] = body.rid
        # We are the consumer this rid was waiting for, so drop any tombstone a
        # previous attempt at the same request left behind -- vLLM reuses the
        # request id when it reschedules, and its senders would otherwise be
        # refused until the tombstone aged out.
        server.expect(body.rid)
        # Armed HERE, not once decoding starts. /pd/cancel needs something to
        # set from the moment the request is admitted: the wire-wait below can
        # be the longest phase of all, and a cancel arriving during it used to
        # find cancel_event still None and answer 404 while the slot stayed
        # held.
        cancel = threading.Event()
        state["cancel_event"] = cancel
        t0 = time.time()

        # phase 0: compile the grammar BEFORE convert / inject, so a bad spec (or
        # a missing backend) fails without touching the GPU. Fail-closed
        # classification survives the HTTP hop via error_type (the router
        # propagates the status verbatim). Both error paths must still hand back
        # the receive slot -- see _abandon_pending_kv.
        # Same fail-fast point as the grammar spec: refuse before the wire-wait
        # and any GPU work if this engine cannot produce what was asked for.
        if (
            body.top_logprobs is not None
            and not getattr(engine, "supports_logprobs", lambda: False)()
        ):
            logger.info(
                "logprobs requested but unsupported by %s (rid=%s)", type(engine).__name__, body.rid
            )
            _abandon_pending_kv(body.rid, cancel)
            return JSONResponse(
                {
                    "error": f"{type(engine).__name__} does not produce logprobs",
                    "error_type": "logprobs_unavailable",
                },
                status_code=501,
            )
        # Same fail-fast point for the sampling params this engine cannot apply.
        # The router normally refuses these before prefilling (it reads
        # /capabilities), but /pd/decode is directly reachable and the streaming
        # branch cannot report a status once its headers are out — so the check
        # belongs here, ahead of the wire-wait, rather than inside decode().
        try:
            validate_generation_request(body.sampling or {}, engine_capabilities(engine))
        except CapabilityError as e:
            logger.info("sampling rejected for %s: %s (%s)", body.rid, e, e.error_type)
            _abandon_pending_kv(body.rid, cancel)
            return JSONResponse(e.to_payload(), status_code=e.http_status)
        try:
            grammar_session = engine.prepare_grammar(body.grammar_spec, body.enable_thinking)
        except GrammarError as e:
            logger.info("grammar rejected for %s: %s (%s)", body.rid, e, e.error_type)
            _abandon_pending_kv(body.rid, cancel)
            return JSONResponse(e.to_payload(), status_code=e.http_status)
        except Exception as e:
            logger.exception("grammar prepare failed for %s", body.rid)
            _abandon_pending_kv(body.rid, cancel)
            return JSONResponse({"error": str(e)}, status_code=500)

        # phase 1: wire wait + convert + inject (common to both modes)
        try:
            req = _drain_own_kv(body.rid, body.timeout_s, cancel)
            if req is _CANCELLED:
                logger.info("cancelled during KV transfer for %s", body.rid)
                _cleanup()
                # 499, nginx's "client closed request": the caller asked us to
                # stop, so this is neither our failure (5xx) nor a bad request
                # (4xx). The router has stopped reading by now; the status is
                # for a direct caller and for the log.
                return JSONResponse(
                    {
                        "error": "cancelled during KV transfer",
                        "error_type": "request_cancelled",
                        "rid": body.rid,
                    },
                    status_code=499,
                )
            if req is None:
                _cleanup()
                return JSONResponse(
                    {"error": "kv_transfer_timeout", "rid": body.rid}, status_code=504
                )
            t_recv = time.time()
            conv = server.profile.convert(
                server.buffer, server.base_ptr, server.max_seq_len, req, server.profile.num_ranks
            )
            t_conv = time.time()
            engine.inject(conv)
            t_inj = time.time()
        except Exception as e:
            logger.exception("prepare failed for %s", body.rid)
            _cleanup()
            return JSONResponse({"error": str(e), "rid": body.rid}, status_code=500)

        pre_timing = {
            "wire_wait": round(1000 * (t_recv - t0), 1),
            "convert": round(1000 * (t_conv - t_recv), 1),
            "inject": round(1000 * (t_inj - t_conv), 1),
        }

        # phase 2: decode (cancel was armed at admission)

        # Logprobs sink. The callback appends BEFORE the token goes on the
        # queue, so anything the generator dequeues already has its entry --
        # indexing by emitted count needs no second lock.
        want_lp = body.top_logprobs is not None
        lp_sink: list[tuple[float | None, list]] = []

        def _emit(tok, logprob=None, candidates=None):
            if want_lp:
                if logprob is None:
                    if lp_sink:
                        # Past position 0 a bare token is an engine fault:
                        # recording None would reach the client as the
                        # "very unlikely" sentinel, indistinguishable from a
                        # measurement.
                        raise LogprobsUnavailable(f"engine emitted token {tok} without a logprob")
                    # Position 0 is first_token_id, sampled by prefill, so no
                    # decode-side value exists. Hold the slot to keep one entry
                    # per token and send null; the router fills it.
                    lp_sink.append((None, []))
                    return tok
                lp_sink.append((float(logprob), list(candidates or ())))
            return tok

        def _lp_slice(start: int, count: int) -> dict:
            """The `lp`/`tp` fields for tokens [start, start+count)."""
            rows = lp_sink[start : start + count]
            return {
                "lp": [r[0] for r in rows],
                "tp": [[list(c) for c in r[1]] for r in rows],
            }

        if not body.stream:
            try:
                tokens = engine.decode(
                    first_token_id=body.first_token_id,
                    max_tokens=body.max_tokens,
                    sampling=body.sampling,
                    on_token=_emit if want_lp else None,
                    cancel_event=cancel,
                    grammar_session=grammar_session,
                    **({"top_logprobs": body.top_logprobs} if want_lp else {}),
                )
                timing = {
                    **pre_timing,
                    "decode": round(1000 * (time.time() - t_inj), 1),
                    **getattr(engine, "last_stats", {}),
                }
                _log_reqstat(body, req, len(tokens), timing)
                out = {
                    "rid": body.rid,
                    "token_ids": tokens,
                    "seq_len": req.seq_len,
                    "timing_ms": timing,
                }
                if want_lp:
                    if len(lp_sink) != len(tokens):
                        raise LogprobsUnavailable(
                            f"engine returned {len(lp_sink)} logprob entries "
                            f"for {len(tokens)} tokens"
                        )
                    out["logprobs"] = _lp_slice(0, len(tokens))
                return out
            except LogprobsUnavailable as e:
                logger.info("logprobs unavailable for %s: %s", body.rid, e)
                return JSONResponse(
                    {"error": str(e), "error_type": "logprobs_unavailable"}, status_code=501
                )
            except GrammarViolationError as e:
                logger.info("grammar violation for %s: %s", body.rid, e)
                return JSONResponse(e.to_payload(), status_code=e.http_status)
            except Exception as e:
                logger.exception("decode failed for %s", body.rid)
                return JSONResponse({"error": str(e), "rid": body.rid}, status_code=500)
            finally:
                _cleanup()

        # streaming: ndjson lines {"t":[ids...]}* then {"done":true,...};
        # lock/engine ownership transfers to the generator.
        q: queue_mod.Queue = queue_mod.Queue()
        fin: dict = {"loop": None, "ev": None}

        def _signal_done() -> None:
            loop, ev = fin["loop"], fin["ev"]
            if loop is not None and ev is not None:
                loop.call_soon_threadsafe(ev.set)

        def _run():
            try:
                tokens = engine.decode(
                    first_token_id=body.first_token_id,
                    max_tokens=body.max_tokens,
                    sampling=body.sampling,
                    on_token=(lambda *a: q.put(_emit(*a))) if want_lp else q.put,
                    cancel_event=cancel,
                    grammar_session=grammar_session,
                    **({"top_logprobs": body.top_logprobs} if want_lp else {}),
                )
                q.put(("done", tokens))
                _signal_done()
            except GrammarViolationError as e:
                # 200 headers may already be sent; signal a typed error so the
                # router can emit an SSE error event + [DONE] (fail-closed).
                logger.info("stream grammar violation for %s: %s", body.rid, e)
                q.put(("error", e.to_payload()))
                _signal_done()
            except LogprobsUnavailable as e:
                # Typed, like the blocking branch: the engine cannot produce what
                # was asked for, which is a capability answer (501), not a broken
                # component (502). Falling into the generic handler below dropped
                # the type, so adding a stop string -- which is what puts a
                # non-streaming request on this protocol -- silently changed the
                # status for the same inability.
                logger.info("stream logprobs unavailable for %s: %s", body.rid, e)
                q.put(("error", {"error": str(e), "error_type": "logprobs_unavailable"}))
                _signal_done()
            except Exception as e:  # pragma: no cover
                logger.exception("stream decode failed for %s", body.rid)
                q.put(("error", {"error": str(e)}))
                _signal_done()

        worker = threading.Thread(target=_run, name="pd-decode", daemon=True)

        async def _gen():
            # MUST be an async generator: on client disconnect starlette
            # cancels the response task, and only async generators get the
            # cancellation delivered into their frame so `finally` runs
            # (a sync generator is silently abandoned -> the engine slot
            # leaks forever; found by the streaming-cancel drill).
            import asyncio

            import anyio
            from starlette.concurrency import run_in_threadpool

            fin["loop"] = asyncio.get_running_loop()
            fin["ev"] = asyncio.Event()
            worker.start()
            try:
                batch: list[int] = []
                n_emitted = 0
                done_msg = None
                last_activity = time.time()

                while done_msg is None:
                    try:
                        first = q.get_nowait()
                    except queue_mod.Empty:
                        if time.time() - last_activity > 600:
                            yield json.dumps({"error": "decode stalled"}) + "\n"
                            return
                        await asyncio.sleep(0.001)
                        continue
                    if isinstance(first, int):
                        line = {"t": [first]}
                        if want_lp:
                            line.update(_lp_slice(0, 1))
                        n_emitted = 1
                        yield json.dumps(line) + "\n"
                    else:
                        done_msg = first
                    last_activity = time.time()
                    break

                while done_msg is None:
                    drained = False
                    while True:
                        try:
                            item = q.get_nowait()
                        except queue_mod.Empty:
                            break
                        drained = True
                        if isinstance(item, int):
                            batch.append(item)
                        else:
                            done_msg = item
                            break
                    if batch:
                        line = {"t": batch}
                        if want_lp:
                            # lp[i] / tp[i] line up with t[i]; written before
                            # the queue put, so visible for everything dequeued.
                            line.update(_lp_slice(n_emitted, len(batch)))
                        n_emitted += len(batch)
                        yield json.dumps(line) + "\n"
                        batch = []
                    if done_msg is None:
                        if drained:
                            last_activity = time.time()
                        elif time.time() - last_activity > 600:  # noqa: R505 (exclusive branches)
                            yield json.dumps({"error": "decode stalled"}) + "\n"
                            return
                        else:
                            with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                                await asyncio.wait_for(fin["ev"].wait(), timeout=DECODE_POLL_S)
                kind, payload = done_msg
                if kind == "done":
                    timing = {
                        **pre_timing,
                        "decode": round(1000 * (time.time() - t_inj), 1),
                        **getattr(engine, "last_stats", {}),
                    }
                    _log_reqstat(body, req, len(payload), timing)
                    yield json.dumps(
                        {
                            "done": True,
                            "n": len(payload),
                            "seq_len": req.seq_len,
                            "finish_reason": timing.get("finish_reason", "stop"),
                            "timing_ms": timing,
                        }
                    ) + "\n"
                else:
                    # payload is a typed dict {"error", ["error_type"]}; emit
                    # verbatim so the router can classify (grammar_violation).
                    yield json.dumps(payload) + "\n"
            finally:
                cancel.set()
                # shield: cleanup must complete even inside a cancelled scope,
                # and the worker must be joined before engine.reset() (the
                # engine may be mid-decode_mtp on the GPU).
                with anyio.CancelScope(shield=True):
                    await run_in_threadpool(worker.join, 120)
                if worker.is_alive():
                    logger.error("decode worker failed to stop for %s", body.rid)
                _cleanup()

        return StreamingResponse(_gen(), media_type="application/x-ndjson")

    return app  # noqa: R504 (assembled across the function)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["stub", "tilert"], default="stub")
    ap.add_argument(
        "--model",
        default="glm5",
        help="model profile (glm5 / glm5_2 / glm5_3 / dsv32); "
        "must match the prefill side's tilert_model",
    )
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--ctrl-port", type=int, default=5556)
    ap.add_argument("--http-port", type=int, default=5557)
    ap.add_argument("--model-weights-dir", default="")
    ap.add_argument("--with-mtp", action="store_true")
    ap.add_argument(
        "--transport",
        choices=["mooncake", "nixl"],
        default="mooncake",
        help="RDMA data-plane backend " "(must match prefill's tilert_transport)",
    )
    ap.add_argument(
        "--kv-cache-dtype",
        default="fp8_ds_mla",
        help="MLA cache dtype (must match vLLM prefill); " "MLA-family profiles only",
    )
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_parser().parse_args()

    from tilert.pd_vllm.profiles import base as profiles

    profile = profiles.get_profile(args.model)
    # MLA-family profiles (glm5/glm5_2/dsv32) need the cache dtype to size the
    # receive buffer; profiles without the knob skip it.
    if hasattr(profile, "configure"):
        profile.configure(args.kv_cache_dtype)
        logger.info(
            "profile %s MLA cache dtype = %s (layout v%d)",
            profile.name,
            args.kv_cache_dtype,
            profile.layout_version,
        )
    # Depth-from-checkpoint members must resolve their layer count before the
    # receive buffer is sized off it.
    if hasattr(profile, "configure_weights") and args.model_weights_dir:
        profile.configure_weights(args.model_weights_dir)

    if args.engine == "stub":
        from tilert.pd_vllm.engine_iface import StubEngine

        engine: Any = StubEngine()
    else:
        logger.info(
            "loading TileRT engine (profile=%s, weights=%s)...",
            profile.name,
            args.model_weights_dir,
        )
        engine = profile.build_engine(
            model_weights_dir=args.model_weights_dir,
            max_seq_len=args.max_seq_len,
            with_mtp=args.with_mtp,
            ar_steps=8,
        )
        logger.info("TileRT engine ready (cache window %d)", engine.max_seq_len)

    server = ReceiveServer(
        profile, max_seq_len=args.max_seq_len, ctrl_port=args.ctrl_port, transport=args.transport
    )
    app = build_app(server, engine)
    logger.info(
        "decode server on :%d (profile=%s, engine=%s, ctrl=:%d)",
        args.http_port,
        profile.name,
        args.engine,
        args.ctrl_port,
    )
    # Bind dual-stack (IPv4 + IPv6) explicitly. uvicorn's host="::" is
    # IPv6-only under some uvicorn/OS combinations, which leaves the decode
    # HTTP endpoint unreachable from an IPv4 router. Mirror the control plane
    # (receive_server) by clearing IPV6_V6ONLY on an AF_INET6 socket.
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", args.http_port))
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])


if __name__ == "__main__":
    main()
