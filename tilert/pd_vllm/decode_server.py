import argparse
import contextlib
import json
import logging
import os
import queue as queue_mod
import socket
import sys
import threading
import time

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from tilert.pd_vllm.capabilities import (
    CapabilityError,
    engine_capabilities,
    validate_generation_request,
)
from tilert.pd_vllm.grammar_spec import GrammarError, GrammarViolationError
from tilert.pd_vllm.receive_server import ReceiveServer

logger = logging.getLogger("pd_vllm.decode_server")
_ABANDON_DRAIN_S = 30.0
_CANCELLED = object()


class LogprobsUnavailable(Exception):
    pass


class DecodeBody(BaseModel):
    rid: str
    first_token_id: int
    max_tokens: int = 256
    sampling: dict | None = None
    timeout_s: float = 120.0
    stream: bool = False
    grammar_spec: dict | None = None
    enable_thinking: bool = True
    top_logprobs: int | None = None


DECODE_POLL_S = max(0.0, float(os.environ.get("TILERT_DECODE_POLL_MS") or "200")) / 1000.0


def build_app(server: ReceiveServer, engine) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()
    state = {"current_rid": None}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/capabilities")
    def capabilities():
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
        rid = body.get("rid")
        ev = state.get("cancel_event")
        if rid and rid == state["current_rid"] and (ev is not None):
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
        if state["current_rid"]:
            server.release(state["current_rid"])
        state["current_rid"] = None
        state["cancel_event"] = None
        lock.release()

    def _drain_own_kv(rid: str, timeout_s: float, cancel=None):
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
            server.release(cand.rid)

    def _abandon_pending_kv(rid: str, cancel=None) -> None:
        try:
            if _drain_own_kv(rid, _ABANDON_DRAIN_S, cancel) is None:
                logger.warning(
                    "abandoned %s: its KV did not arrive within %.0fs", rid, _ABANDON_DRAIN_S
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
            " ".join((f"{k}={v}" for k, v in timing.items())),
        )

    @app.post("/pd/decode")
    def pd_decode(body: DecodeBody):
        if not lock.acquire(blocking=False):
            return JSONResponse(
                {"error": "busy", "current_rid": state["current_rid"]}, status_code=429
            )
        state["current_rid"] = body.rid
        server.expect(body.rid)
        cancel = threading.Event()
        state["cancel_event"] = cancel
        t0 = time.time()
        if body.top_logprobs is not None and (
            not getattr(engine, "supports_logprobs", lambda: False)()
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
        try:
            req = _drain_own_kv(body.rid, body.timeout_s, cancel)
            if req is _CANCELLED:
                logger.info("cancelled during KV transfer for %s", body.rid)
                _cleanup()
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
        want_lp = body.top_logprobs is not None
        lp_sink: list[tuple[float | None, list]] = []

        def _emit(tok, logprob=None, candidates=None):
            if want_lp:
                if logprob is None:
                    if lp_sink:
                        raise LogprobsUnavailable(f"engine emitted token {tok} without a logprob")
                    lp_sink.append((None, []))
                    return tok
                lp_sink.append((float(logprob), list(candidates or ())))
            return tok

        def _lp_slice(start: int, count: int) -> dict:
            rows = lp_sink[start : start + count]
            return {"lp": [r[0] for r in rows], "tp": [[list(c) for c in r[1]] for r in rows]}

        if not body.stream:
            try:
                tokens = engine.decode(
                    first_token_id=body.first_token_id,
                    max_tokens=body.max_tokens,
                    sampling=body.sampling,
                    on_token=_emit if want_lp else None,
                    cancel_event=cancel,
                    grammar_session=grammar_session,
                    **{"top_logprobs": body.top_logprobs} if want_lp else {},
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
                            f"engine returned {len(lp_sink)} logprob entries for {len(tokens)} tokens"
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
        q: queue_mod.Queue = queue_mod.Queue()
        fin: dict = {"loop": None, "ev": None}

        def _signal_done() -> None:
            loop, ev = (fin["loop"], fin["ev"])
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
                    **{"top_logprobs": body.top_logprobs} if want_lp else {},
                )
                q.put(("done", tokens))
                _signal_done()
            except GrammarViolationError as e:
                logger.info("stream grammar violation for %s: %s", body.rid, e)
                q.put(("error", e.to_payload()))
                _signal_done()
            except LogprobsUnavailable as e:
                logger.info("stream logprobs unavailable for %s: %s", body.rid, e)
                q.put(("error", {"error": str(e), "error_type": "logprobs_unavailable"}))
                _signal_done()
            except Exception as e:
                logger.exception("stream decode failed for %s", body.rid)
                q.put(("error", {"error": str(e)}))
                _signal_done()

        worker = threading.Thread(target=_run, name="pd-decode", daemon=True)

        async def _gen():
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
                            yield (json.dumps({"error": "decode stalled"}) + "\n")
                            return
                        await asyncio.sleep(0.001)
                        continue
                    if isinstance(first, int):
                        line = {"t": [first]}
                        if want_lp:
                            line.update(_lp_slice(0, 1))
                        n_emitted = 1
                        yield (json.dumps(line) + "\n")
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
                            line.update(_lp_slice(n_emitted, len(batch)))
                        n_emitted += len(batch)
                        yield (json.dumps(line) + "\n")
                        batch = []
                    if done_msg is None:
                        if drained:
                            last_activity = time.time()
                        elif time.time() - last_activity > 600:  # noqa: R505 (exclusive branches)
                            yield (json.dumps({"error": "decode stalled"}) + "\n")
                            return
                        else:
                            try:
                                await asyncio.wait_for(fin["ev"].wait(), timeout=DECODE_POLL_S)
                            except TimeoutError:
                                pass
                kind, payload = done_msg
                if kind == "done":
                    timing = {
                        **pre_timing,
                        "decode": round(1000 * (time.time() - t_inj), 1),
                        **getattr(engine, "last_stats", {}),
                    }
                    _log_reqstat(body, req, len(payload), timing)
                    yield (
                        json.dumps(
                            {
                                "done": True,
                                "n": len(payload),
                                "seq_len": req.seq_len,
                                "finish_reason": timing.get("finish_reason", "stop"),
                                "timing_ms": timing,
                            }
                        )
                        + "\n"
                    )
                else:
                    yield (json.dumps(payload) + "\n")
            finally:
                cancel.set()
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
    ap.add_argument("--model", default="glm5", help="model profile")
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--ctrl-port", type=int, default=5556)
    ap.add_argument("--http-port", type=int, default=5557)
    ap.add_argument("--model-weights-dir", default="")
    ap.add_argument("--with-mtp", action="store_true")
    ap.add_argument(
        "--num-mtp",
        type=int,
        choices=(3,),
        default=3,
        help="MTP draft depth for speculative decoding; needs --with-mtp",
    )
    ap.add_argument(
        "--transport",
        choices=["mooncake", "nixl"],
        default="mooncake",
        help="RDMA data-plane backend (must match prefill's tilert_transport)",
    )
    ap.add_argument(
        "--kv-cache-dtype",
        default="fp8_ds_mla",
        help="MLA cache dtype (must match vLLM prefill); MLA-family profiles only",
    )
    ap.add_argument(
        "--pd-buffer-device",
        choices=["cuda", "cpu"],
        default=(os.environ.get("TILERT_PD_BUFFER_DEVICE") or "cuda").lower(),
        help="where the PD receive buffer lives; 'cpu' = pinned host memory registered with the transport (frees buffer_bytes(max_seq_len) of VRAM, adds one H2D copy per request), 'cuda' = sharded over the visible cards per TILERT_PD_SHARDS",
    )
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = build_parser().parse_args()
    from tilert.pd_vllm.profiles import base as profiles

    profile = profiles.get_profile(args.model)
    num_mtp = profiles.resolve_num_mtp(profile, args.num_mtp, with_mtp=args.with_mtp)
    if hasattr(profile, "configure"):
        profile.configure(args.kv_cache_dtype)
        logger.info(
            "profile %s MLA cache dtype = %s (layout v%d)",
            profile.name,
            args.kv_cache_dtype,
            profile.layout_version,
        )
    if hasattr(profile, "configure_weights") and args.model_weights_dir:
        profile.configure_weights(args.model_weights_dir)
    if args.engine == "stub":
        from tilert.pd_vllm.engine_iface import StubEngine

        engine = StubEngine()
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
            num_mtp=num_mtp,
        )
        logger.info(
            "TileRT engine ready (cache window %d, num_mtp %d)", engine.max_seq_len, num_mtp
        )
    server = ReceiveServer(
        profile,
        max_seq_len=args.max_seq_len,
        ctrl_port=args.ctrl_port,
        transport=args.transport,
        buffer_device="cpu" if args.pd_buffer_device == "cpu" else "cuda:0",
    )
    app = build_app(server, engine)
    logger.info(
        "decode server on :%d (profile=%s, engine=%s, ctrl=:%d)",
        args.http_port,
        profile.name,
        args.engine,
        args.ctrl_port,
    )
    sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    with contextlib.suppress(OSError):
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    sock.bind(("::", args.http_port))
    uvicorn.Server(uvicorn.Config(app, log_level="warning")).run(sockets=[sock])


if __name__ == "__main__":
    main()
