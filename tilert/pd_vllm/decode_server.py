"""PD decode server (W6): HTTP orchestration around receive -> convert -> inject -> decode.

Internal token-level API (the client-facing OpenAI layer lives in pd_router /
a later serving layer):

  POST /pd/decode   {rid, first_token_id, max_tokens, sampling?, timeout_s?}
      Waits for the wire transfer of `rid` to complete, converts, injects
      into the engine, decodes, returns {"rid", "token_ids", "timing_ms"}.
  GET  /health          {"status": "ok"}
  GET  /decode_status   {"status": "idle"|"busy", "current_rid": ...}

bs=1: a busy server answers 429 immediately (the router's gated dispatch
should make that unreachable).
"""

import argparse
import contextlib
import json
import logging
import queue as queue_mod
import socket
import threading
import time
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from tilert.pd_vllm.receive_server import ReceiveServer

logger = logging.getLogger("pd_vllm.decode_server")


class DecodeBody(BaseModel):
    rid: str
    first_token_id: int
    max_tokens: int = 256
    sampling: dict | None = None
    timeout_s: float = 120.0
    stream: bool = False


def build_app(server: ReceiveServer, engine) -> FastAPI:
    app = FastAPI()
    lock = threading.Lock()
    state: dict[str, Any] = {"current_rid": None}

    @app.get("/health")
    def health():
        return {"status": "ok"}

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
        server.release()
        state["current_rid"] = None
        state["cancel_event"] = None
        lock.release()

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
        t0 = time.time()
        # phase 1: wire wait + convert + inject (common to both modes)
        try:
            # Drain until OUR rid arrives; drop stale completed entries
            # (e.g. a transfer whose consumer never called /pd/decode).
            req = None
            deadline = time.time() + body.timeout_s
            while time.time() < deadline:
                try:
                    cand = server.completed.get(timeout=max(0.1, deadline - time.time()))
                except queue_mod.Empty:
                    break
                if cand.rid == body.rid:
                    req = cand
                    break
                logger.warning(
                    "dropping unmatched request %s " "(waiting for %s)", cand.rid, body.rid
                )
                server.release()
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

        # phase 2: decode
        cancel = threading.Event()
        state["cancel_event"] = cancel

        if not body.stream:
            try:
                tokens = engine.decode(
                    first_token_id=body.first_token_id,
                    max_tokens=body.max_tokens,
                    sampling=body.sampling,
                    cancel_event=cancel,
                )
                timing = {
                    **pre_timing,
                    "decode": round(1000 * (time.time() - t_inj), 1),
                    **getattr(engine, "last_stats", {}),
                }
                _log_reqstat(body, req, len(tokens), timing)
                return {
                    "rid": body.rid,
                    "token_ids": tokens,
                    "seq_len": req.seq_len,
                    "timing_ms": timing,
                }
            except Exception as e:
                logger.exception("decode failed for %s", body.rid)
                return JSONResponse({"error": str(e), "rid": body.rid}, status_code=500)
            finally:
                _cleanup()

        # streaming: ndjson lines {"t":[ids...]}* then {"done":true,...};
        # lock/engine ownership transfers to the generator.
        q: queue_mod.Queue = queue_mod.Queue()

        def _run():
            try:
                tokens = engine.decode(
                    first_token_id=body.first_token_id,
                    max_tokens=body.max_tokens,
                    sampling=body.sampling,
                    on_token=q.put,
                    cancel_event=cancel,
                )
                q.put(("done", tokens))
            except Exception as e:  # pragma: no cover
                logger.exception("stream decode failed for %s", body.rid)
                q.put(("error", str(e)))

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

            worker.start()
            try:
                batch: list[int] = []
                done_msg = None
                last_activity = time.time()
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
                        yield json.dumps({"t": batch}) + "\n"
                        batch = []
                    if done_msg is None:
                        if drained:
                            last_activity = time.time()
                        elif time.time() - last_activity > 600:  # noqa: R505 (exclusive branches)
                            yield json.dumps({"error": "decode stalled"}) + "\n"
                            return
                        else:
                            await asyncio.sleep(0.005)
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
                    yield json.dumps({"error": payload}) + "\n"
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["stub", "tilert"], default="stub")
    ap.add_argument("--model", default="glm5", help="model profile")
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
    args = ap.parse_args()

    from tilert.pd_vllm.profiles import base as profiles

    profile = profiles.get_profile(args.model)
    # MLA-family profiles (glm5/dsv32) need the cache dtype to size the receive
    # buffer.
    if hasattr(profile, "configure"):
        profile.configure(args.kv_cache_dtype)
        logger.info(
            "profile %s MLA cache dtype = %s (layout v%d)",
            profile.name,
            args.kv_cache_dtype,
            profile.layout_version,
        )

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
