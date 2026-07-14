"""PD router (W6): client-facing entry over vLLM prefill + TileRT decode.

Does OpenAI-semantics output parsing (reasoning + tool calls), streaming and
non-streaming.

Flow per request (phase-1 hybrid, see design doc):
  1. pick a free decode node (in-memory busy tracking; all busy -> 429)
  2. forward to vLLM with max_tokens=1 + logprobs and inject
     kv_transfer_params {tilert_host, tilert_ctrl_port} — the connector
     claims the request and RDMA-sends state to the decode node
  3. extract rid + first_token_id from the vLLM response
     (requires vLLM serve launched with --return-tokens-as-token-ids)
  4. call the decode node (/pd/decode; stream or not) and assemble the
     OpenAI response: reasoning_content / content / tool_calls via the
     vLLM parser engine (decision B1 — this process's env has vllm
     installed, CPU-only; the decode node does not).

Environment: run in a vllm-equipped env with CUDA_VISIBLE_DEVICES="" (the
router must never touch GPUs). --parser none falls back to raw passthrough.

Run:
  CUDA_VISIBLE_DEVICES= python -m tilert.pd_vllm.pd_router \
      --vllm-url http://prefill-node:8000 \
      --decode decode-node:5556:5557 --port 23333 \
      --model-path /path/to/GLM-5.1 --parser glm47
"""

import argparse
import json
import logging
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tilert.pd_vllm.wire import derive_rid

logger = logging.getLogger("pd_vllm.router")


class DecodeNode:
    def __init__(self, host: str, ctrl_port: int, http_port: int):
        self.host = host
        self.ctrl_port = ctrl_port
        self.http_port = http_port
        self.busy = False

    @property
    def http_base(self) -> str:
        return f"http://{self.host}:{self.http_port}"


class Pool:
    def __init__(self, nodes: list[DecodeNode]):
        self.nodes = nodes
        self._lock = threading.Lock()

    def acquire(self) -> DecodeNode | None:
        with self._lock:
            for n in self.nodes:
                if not n.busy:
                    n.busy = True
                    return n
        return None

    def release(self, node: DecodeNode) -> None:
        with self._lock:
            node.busy = False


def first_token_from_logprobs(resp: dict, is_chat: bool) -> int:
    """Parse 'token_id:N' (vLLM --return-tokens-as-token-ids) from logprobs."""
    choice = resp["choices"][0]
    lp = choice.get("logprobs") or {}
    tok: str | None = None
    if is_chat:
        content = lp.get("content") or []
        if content:
            tok = content[0].get("token")
    else:
        toks = lp.get("tokens") or []
        if toks:
            tok = toks[0]
    if tok and tok.startswith("token_id:"):
        return int(tok.split(":", 1)[1])
    raise ValueError(
        f"cannot extract first token id from logprobs ({tok!r}); launch vLLM "
        f"with --return-tokens-as-token-ids and request logprobs"
    )


def _thinking_enabled(body: dict) -> bool:
    ctk = body.get("chat_template_kwargs") or {}
    return bool(ctk.get("enable_thinking", True))


class RouterCtx:
    """Immutable per-process context (tokenizer, parser factory, config)."""

    def __init__(self, vllm_url: str, pool: Pool, tokenizer, parser_name: str):
        self.vllm_url = vllm_url
        self.pool = pool
        self.tokenizer = tokenizer
        self.parser_name = parser_name
        self._parsers = {}
        if parser_name != "none":
            if tokenizer is None:
                raise SystemExit("--parser requires --model-path (tokenizer)")
            from tilert.pd_vllm.oai_parser import make_parser

            self._parsers[True] = make_parser(parser_name, tokenizer, thinking=True)
            self._parsers[False] = self._parsers[True].with_thinking(False)
            logger.info("parser '%s' ready (thinking variants cached)", parser_name)

    def parser(self, thinking: bool):
        return self._parsers.get(thinking)


def build_app(ctx: RouterCtx) -> FastAPI:
    app = FastAPI()
    pool = ctx.pool

    @app.get("/health")
    def health():
        return {"status": "ok", "decode_free": sum(1 for n in pool.nodes if not n.busy)}

    @app.get("/pool_status")
    def pool_status():
        return {"nodes": [{"host": n.host, "busy": n.busy} for n in pool.nodes]}

    # ── shared prefill step ──────────────────────────────────────────────
    def _prefill(path, body, node):
        prefill_body = dict(body)
        prefill_body["max_tokens"] = 1
        prefill_body["stream"] = False
        if path.endswith("chat/completions"):
            prefill_body["logprobs"] = True
            prefill_body["top_logprobs"] = 1
        else:
            prefill_body["logprobs"] = 1
        prefill_body["kv_transfer_params"] = {
            "tilert_host": node.host,
            "tilert_ctrl_port": node.ctrl_port,
        }
        r = requests.post(f"{ctx.vllm_url}{path}", json=prefill_body, timeout=600)
        r.raise_for_status()
        return r.json()

    def _sampling_of(body):
        return {k: body[k] for k in ("temperature", "top_p", "top_k") if k in body}

    def _max_tokens_of(body):
        return int(body.get("max_tokens") or body.get("max_completion_tokens") or 256)

    # ── non-streaming ────────────────────────────────────────────────────
    def _handle(path: str, body: dict):
        is_chat = path.endswith("chat/completions")
        node = pool.acquire()
        if node is None:
            return JSONResponse({"error": "all decode nodes busy"}, status_code=429)
        t0 = time.time()
        try:
            prefill = _prefill(path, body, node)
            t_prefill = time.time()
            rid = derive_rid(prefill["id"])
            first_token_id = first_token_from_logprobs(prefill, is_chat)

            dr = requests.post(
                f"{node.http_base}/pd/decode",
                json={
                    "rid": rid,
                    "first_token_id": first_token_id,
                    "max_tokens": _max_tokens_of(body),
                    "sampling": _sampling_of(body),
                },
                timeout=600,
            )
            dr.raise_for_status()
            decode = dr.json()
            token_ids = decode["token_ids"]
            timing = decode.get("timing_ms", {})
            finish = timing.get("finish_reason", "stop")
            if finish == "cancelled":
                finish = "stop"

            choice: dict = {"index": 0, "finish_reason": finish}
            parser = ctx.parser(_thinking_enabled(body)) if is_chat else None
            if parser is not None:
                text = ctx.tokenizer.decode(token_ids, skip_special_tokens=False)
                parsed = parser.parse_complete(text)
                msg = {"role": "assistant", "content": parsed.content or ""}
                if parsed.reasoning_content:
                    msg["reasoning_content"] = parsed.reasoning_content
                if parsed.tool_calls:
                    msg["tool_calls"] = [c.to_openai(i) for i, c in enumerate(parsed.tool_calls)]
                    choice["finish_reason"] = "tool_calls"
                choice["message"] = msg
            else:
                text = (
                    ctx.tokenizer.decode(token_ids, skip_special_tokens=True)
                    if ctx.tokenizer
                    else None
                )
                if is_chat:
                    choice["message"] = {"role": "assistant", "content": text}
                else:
                    choice["text"] = text
                    choice["token_ids"] = token_ids

            return JSONResponse(
                {
                    "id": prefill["id"],
                    "object": "chat.completion" if is_chat else "text_completion",
                    "created": int(time.time()),
                    "model": prefill.get("model"),
                    "choices": [choice],
                    "usage": {
                        "prompt_tokens": (prefill.get("usage") or {}).get("prompt_tokens"),
                        "completion_tokens": len(token_ids),
                    },
                    "pd_timing_ms": {
                        "prefill": round(1000 * (t_prefill - t0), 1),
                        **timing,
                    },
                }
            )
        except Exception as e:
            logger.exception("pd request failed")
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            pool.release(node)

    # ── streaming (chat only) ────────────────────────────────────────────
    async def _handle_stream(path: str, body: dict, request: Request):
        from starlette.concurrency import run_in_threadpool

        node = pool.acquire()
        if node is None:
            return JSONResponse({"error": "all decode nodes busy"}, status_code=429)

        try:
            prefill = await run_in_threadpool(_prefill, path, body, node)
            rid = derive_rid(prefill["id"])
            first_token_id = first_token_from_logprobs(prefill, True)
        except Exception as e:
            pool.release(node)
            logger.exception("pd stream request failed before streaming")
            return JSONResponse({"error": str(e)}, status_code=502)

        chunk_id = prefill["id"]
        model = prefill.get("model")
        prompt_tokens = (prefill.get("usage") or {}).get("prompt_tokens")
        parser = ctx.parser(_thinking_enabled(body))

        def _chunk(delta: dict, finish=None, usage=None) -> str:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }
            if usage is not None:
                payload["usage"] = usage
            return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        def _event_delta(ev: dict) -> dict:
            if ev["kind"] == "reasoning":
                return {"reasoning_content": ev["text"]}
            if ev["kind"] == "content":
                return {"content": ev["text"]}
            return {
                "tool_calls": [
                    {
                        "index": ev["index"],
                        "id": ev["id"],
                        "type": "function",
                        "function": {"name": ev["name"], "arguments": ev["arguments"]},
                    }
                ]
            }

        def _fire_cancel():
            try:
                requests.post(f"{node.http_base}/pd/cancel", json={"rid": rid}, timeout=5)
            except Exception:
                logger.warning("cancel POST failed for %s", rid)

        async def _gen():
            import anyio
            import httpx

            from tilert.pd_vllm.oai_parser import IncrementalDetok

            n_tokens = 0
            saw_tool = False
            finish_reason = "stop"
            client_gone = False
            completed_ok = False
            detok = IncrementalDetok(ctx.tokenizer)
            sess = parser.stream() if parser else None
            client = httpx.AsyncClient(timeout=httpx.Timeout(600, read=600))
            try:
                yield _chunk({"role": "assistant"})
                async with client.stream(
                    "POST",
                    f"{node.http_base}/pd/decode",
                    json={
                        "rid": rid,
                        "first_token_id": first_token_id,
                        "max_tokens": _max_tokens_of(body),
                        "sampling": _sampling_of(body),
                        "stream": True,
                    },
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        # Deterministic client-liveness check: writes to a
                        # dead socket do NOT raise (verified by drill), so
                        # poll the ASGI disconnect state every line.
                        if await request.is_disconnected():
                            client_gone = True
                            logger.info("client disconnected, cancelling %s", rid)
                            break
                        if not line:
                            continue
                        msg = json.loads(line)
                        if "t" in msg:
                            n_tokens += len(msg["t"])
                            text = detok.push(msg["t"])
                            if not text:
                                continue
                            if sess is None:
                                yield _chunk({"content": text})
                                continue
                            for ev in sess.feed(text):
                                if ev["kind"] == "tool":
                                    saw_tool = True
                                yield _chunk(_event_delta(ev))
                        elif "done" in msg:
                            finish_reason = msg.get("finish_reason", "stop")
                            if finish_reason == "cancelled":
                                finish_reason = "stop"
                        elif "error" in msg:
                            yield _chunk({"content": f"\n[decode error: {msg['error']}]"})
                            finish_reason = "stop"
                if client_gone:
                    logger.info("client gone mid-stream for %s", rid)
                    return  # finally fires the cancel
                if sess is not None:
                    for ev in sess.finish():
                        if ev["kind"] == "tool":
                            saw_tool = True
                        yield _chunk(_event_delta(ev))
                if saw_tool:
                    finish_reason = "tool_calls"
                yield _chunk(
                    {},
                    finish=finish_reason,
                    usage={
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": n_tokens,
                    },
                )
                yield "data: [DONE]\n\n"
                completed_ok = True
            except Exception:
                logger.exception("stream failed mid-flight for %s", rid)
            finally:
                # Runs under cancellation too (client disconnect cancels this
                # task). Order matters: release first (sync, can't be
                # cancelled), then best-effort cancel via a plain thread
                # (an await here could be cancelled before firing), then a
                # shielded aclose.
                pool.release(node)
                if not completed_ok:
                    threading.Thread(target=_fire_cancel, daemon=True).start()
                with anyio.CancelScope(shield=True):
                    await client.aclose()

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        from starlette.concurrency import run_in_threadpool

        body = await request.json()
        if body.get("stream"):
            return await _handle_stream("/v1/chat/completions", body, request)
        # blocking work off the event loop (decode can take minutes)
        return await run_in_threadpool(_handle, "/v1/chat/completions", body)

    @app.post("/v1/completions")
    async def completions(request: Request):
        from starlette.concurrency import run_in_threadpool

        body = await request.json()
        if body.get("stream"):
            return JSONResponse(
                {"error": "streaming is supported on /v1/chat/completions"}, status_code=400
            )
        return await run_in_threadpool(_handle, "/v1/completions", body)

    return app  # noqa: R504 (assembled across the function)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--vllm-url", required=True)
    ap.add_argument(
        "--decode", nargs="+", required=True, help="decode nodes as host:ctrl_port:http_port"
    )
    ap.add_argument("--host", default="0.0.0.0")  # nosec B104 (bind-all by design)
    ap.add_argument("--port", type=int, default=23333)
    ap.add_argument(
        "--model-path", default="", help="tokenizer path (required unless --parser none)"
    )
    ap.add_argument(
        "--parser",
        choices=["glm47", "none"],
        default="glm47",
        help="output parser (reasoning + tool calls)",
    )
    args = ap.parse_args()

    nodes = []
    for spec in args.decode:
        host, cport, hport = spec.rsplit(":", 2)
        nodes.append(DecodeNode(host, int(cport), int(hport)))

    tokenizer = None
    if args.model_path:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )  # nosec B615

    ctx = RouterCtx(args.vllm_url, Pool(nodes), tokenizer, args.parser)
    app = build_app(ctx)
    logger.info(
        "router on :%d -> vllm=%s, %d decode node(s), parser=%s",
        args.port,
        args.vllm_url,
        len(nodes),
        args.parser,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
