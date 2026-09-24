import argparse
import contextlib
import functools
import json
import logging
import os
import threading
import time

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import TypeAdapter, ValidationError

from tilert.pd_vllm import generation_defaults
from tilert.pd_vllm.capabilities import (
    CapabilityError,
    CapabilityUnavailable,
    InvalidParameter,
    NodeCapabilities,
    validate_generation_request,
)
from tilert.pd_vllm.decode_pool import DecodeNode, NodeLease, Pool, acquire_lease, cancel_decode
from tilert.pd_vllm.decode_response import (
    BUSY,
    OK,
    PROPAGATE,
    PROPAGATED_ERROR_TYPES,
    REFUSED,
    RETRY,
    SERVER_ERROR,
    TRUNCATED,
    TYPED_ERROR,
    UNTYPED_ERROR,
    DecodeReader,
    classify_decode_status,
    decode_refusal,
    terminal_verdict,
)
from tilert.pd_vllm.generation_defaults import GenerationDefaults, UnsupportedGenerationDefault
from tilert.pd_vllm.grammar_spec import GrammarError, extract_request_grammar_spec
from tilert.pd_vllm.logprobs import LogprobsUnsupported, resolve_logprobs_request
from tilert.pd_vllm.openai_params import resolve_max_tokens
from tilert.pd_vllm.presentation import SseWriter, blocking_choice, blocking_envelope, collect
from tilert.pd_vllm.presentation import finish_reason as reply_finish_reason
from tilert.pd_vllm.presentation import sse_chunk, sse_delta, textless_choice, usage_chunk
from tilert.pd_vllm.reply import CONTENT, REASONING, TOOL_CALL, ReplyStream, as_logprobs
from tilert.pd_vllm.request_gate import gate_request
from tilert.pd_vllm.stop_strings import resolve_stop
from tilert.pd_vllm.wire import derive_rid

logger = logging.getLogger("pd_vllm.router")
_HTTP_TIMEOUT_S = float((os.environ.get("TILERT_PD_HTTP_TIMEOUT_S") or "3600").strip() or 3600)
QUEUE_LOG_SECONDS = 0.1
_DECODE_MAX_TOKENS_DEFAULT = 256
_DECODE_BUSY_ATTEMPTS = 2
_DECODE_BUSY_RETRY_S = 0.5
_RETRY_LOG = "decode node %s busy for %s, retrying in %.1fs"


def _log_refusal(verdict: str, node, rid: str, status: int, payload=None) -> None:
    if verdict == BUSY:
        logger.warning(
            "decode node %s still busy for %s after %d attempts",
            node.http_base,
            rid,
            _DECODE_BUSY_ATTEMPTS,
        )
    elif verdict == SERVER_ERROR:
        logger.error(
            "decode node %s returned %d for %s: %s", node.http_base, status, rid, str(payload)[:200]
        )


_DISCONNECT_POLL_S = 0.2
# How long to wait for a cancelled helper task to actually finish. Bounded on
# purpose: `await <cancelled task>` is NOT guaranteed to return (see _watch).
_TASK_SETTLE_S = 5.0


class PrefillClientError(Exception):

    def __init__(self, status: int, payload):
        super().__init__(f"vLLM rejected the request with {status}")
        self.status = status
        self.payload = payload


def first_token_from_logprobs(resp: dict, is_chat: bool) -> int:
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
        f"cannot extract first token id from logprobs ({tok!r}); launch vLLM with --return-tokens-as-token-ids and request logprobs"
    )


def _token_id_of(tok: object) -> int | None:
    if isinstance(tok, str) and tok.startswith("token_id:"):
        try:
            return int(tok.split(":", 1)[1])
        except ValueError:
            return None
    return None


def first_token_logprob_from_prefill(resp: dict, top_n: int):
    choice = (resp.get("choices") or [{}])[0]
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return (None, [])
    entry = content[0]
    lp = entry.get("logprob")
    lp = float(lp) if isinstance(lp, (int, float)) else None
    cands = []
    for alt in (entry.get("top_logprobs") or [])[:top_n]:
        alt_id = _token_id_of(alt.get("token"))
        alt_lp = alt.get("logprob")
        if alt_id is not None and isinstance(alt_lp, (int, float)):
            cands.append((alt_id, float(alt_lp)))
    return (lp, cands)


_PREFILL_DROP_FIELDS = (
    "stream_options",
    "max_completion_tokens",
    "stop",
    "include_stop_str_in_output",
)


def build_prefill_body(
    path: str,
    body: dict,
    node: DecodeNode,
    logprobs_req=None,
    defaults: GenerationDefaults | None = None,
) -> dict:
    prefill_body = dict(body)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    for field in _PREFILL_DROP_FIELDS:
        prefill_body.pop(field, None)
    if path.endswith("chat/completions"):
        prefill_body["logprobs"] = True
        prefill_body["top_logprobs"] = max(1, logprobs_req.top_n if logprobs_req is not None else 0)
    else:
        prefill_body["logprobs"] = 1
    prefill_body.update((defaults or GenerationDefaults()).resolve(body))
    prefill_body["kv_transfer_params"] = {
        "tilert_host": node.host,
        "tilert_ctrl_port": node.ctrl_port,
    }
    return prefill_body


def build_usage(prompt_tokens, completion_tokens: int) -> dict:
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def should_include_usage(body: dict, force: bool = False) -> bool:
    if force:
        return True
    opts = body.get("stream_options")
    return bool(isinstance(opts, dict) and opts.get("include_usage"))


def _json_or_none(response) -> dict | None:
    try:
        return response.json()
    except ValueError:
        return None


class RouterCtx:

    def __init__(
        self,
        vllm_url: str,
        pool: Pool,
        tokenizer,
        parser_name: str,
        force_include_usage: bool = False,
        gen_defaults: GenerationDefaults | None = None,
    ):
        self.vllm_url = vllm_url
        self.pool = pool
        self.tokenizer = tokenizer
        self.parser_name = parser_name
        self.gen_defaults = gen_defaults or GenerationDefaults()
        self.force_include_usage = force_include_usage
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


def _tilert_bind_lease_to_task(lease):
    """Release `lease` when the current ASGI task ends, however it ends.

    Idempotent by way of NodeLease.release, so a handler that releases normally
    is unaffected; this only fires for the exits that run no `finally` of ours
    -- a streaming response cancelled while the generator is suspended at a
    `yield`, which leaves that generator suspended and its `finally` unrun.
    """
    import asyncio
    import logging

    task = asyncio.current_task()
    if task is None:
        return

    def _reap(_t, lease=lease):
        if lease.released:
            return
        logging.getLogger("pd_vllm.pool").warning(
            "request task ended without releasing decode node %s (rid=%s); reclaiming it",
            lease.node.host,
            lease.rid,
        )
        lease.release()

    task.add_done_callback(_reap)


def build_app(ctx: RouterCtx) -> FastAPI:
    app = FastAPI()
    pool = ctx.pool

    def _parser_active(thinking: bool) -> bool:
        return ctx.parser(thinking) is not None

    def _busy_response(waited: float) -> JSONResponse:
        detail = (
            f"no decode node free after waiting {waited:.1f}s"
            if pool.queue_timeout > 0
            else "all decode nodes busy"
        )
        return JSONResponse({"error": detail}, status_code=429)

    @app.get("/health")
    def health():
        return {"status": "ok", "decode_free": sum(1 for n in pool.nodes if not n.busy)}

    @app.get("/pool_status")
    def pool_status():
        return {"nodes": [{"host": n.host, "busy": n.busy} for n in pool.nodes]}

    def _prefill(path, body, node, logprobs_req=None):
        prefill_body = build_prefill_body(path, body, node, logprobs_req, ctx.gen_defaults)
        r = requests.post(f"{ctx.vllm_url}{path}", json=prefill_body, timeout=_HTTP_TIMEOUT_S)
        if 400 <= r.status_code < 500:
            try:
                payload = r.json()
            except ValueError:
                payload = {"error": r.text[:500]}
            raise PrefillClientError(r.status_code, payload)
        r.raise_for_status()
        return r.json()

    def _sampling_of(body):
        forwarded = (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "ignore_eos",
        )
        sampling = {k: body[k] for k in forwarded if k in body}
        sampling.update(ctx.gen_defaults.resolve(body))
        return sampling

    def _decode_body(
        rid, first_token_id, body, grammar_spec, *, stream=False, logprobs_req=None, thinking=True
    ):
        payload = {
            "rid": rid,
            "first_token_id": first_token_id,
            "max_tokens": resolve_max_tokens(body, _DECODE_MAX_TOKENS_DEFAULT),
            "sampling": _sampling_of(body),
        }
        if stream:
            payload["stream"] = True
        if grammar_spec is not None:
            payload["grammar_spec"] = grammar_spec
            payload["enable_thinking"] = thinking
        if logprobs_req is not None:
            payload["top_logprobs"] = logprobs_req.top_n
        return payload

    def _handle(path: str, body: dict):
        try:
            req = gate_request(path, body, tokenizer=ctx.tokenizer, parser_active=_parser_active)
            validate_generation_request(body, pool.capabilities(), ctx.gen_defaults.resolve({}))
        except (CapabilityError, GrammarError, LogprobsUnsupported) as e:
            return JSONResponse(e.to_payload(), status_code=e.http_status)
        is_chat, stop, include_stop = (req.is_chat, req.stop, req.include_stop)
        logprobs_req, grammar_spec = (req.logprobs_req, req.grammar_spec)
        lease, waited = acquire_lease(pool)
        if lease is None:
            return _busy_response(waited)
        node = lease.node
        t0 = time.time()
        created = int(t0)
        reader = None
        try:
            prefill = _prefill(path, body, node, logprobs_req)
            t_prefill = time.time()
            rid = lease.rid = derive_rid(prefill["id"])
            first_token_id = first_token_from_logprobs(prefill, is_chat)
            first_lp = (
                first_token_logprob_from_prefill(prefill, logprobs_req.top_n)
                if logprobs_req is not None
                else None
            )
            parser = ctx.parser(req.thinking) if is_chat else None
            asm = (
                ReplyStream(
                    ctx.tokenizer,
                    stop=stop,
                    include_stop_in_output=include_stop,
                    parser_session=parser.stream() if parser else None,
                    logprobs_req=logprobs_req,
                    first_token_logprob=first_lp,
                )
                if ctx.tokenizer is not None
                else None
            )
            want_stream = bool(stop)
            reader = DecodeReader(stream=asm, logprobs_req=logprobs_req, rid=rid)
            decode_payload = _decode_body(
                rid,
                first_token_id,
                body,
                grammar_spec,
                stream=want_stream,
                logprobs_req=logprobs_req,
                thinking=req.thinking,
            )
            dr = None
            for attempt in range(1, _DECODE_BUSY_ATTEMPTS + 1):
                lease.dispatched = True
                dr = requests.post(
                    f"{node.http_base}/pd/decode",
                    json=decode_payload,
                    timeout=_HTTP_TIMEOUT_S,
                    stream=want_stream,
                )
                if dr.status_code == 200:
                    break
                verdict = classify_decode_status(
                    dr.status_code,
                    _json_or_none(dr),
                    attempts_left=attempt < _DECODE_BUSY_ATTEMPTS,
                    propagated_types=PROPAGATED_ERROR_TYPES,
                )
                if want_stream:
                    dr.close()
                if verdict == RETRY:
                    logger.info(_RETRY_LOG, node.http_base, rid, _DECODE_BUSY_RETRY_S)
                    time.sleep(_DECODE_BUSY_RETRY_S)
                    continue
                _log_refusal(verdict, node, rid, dr.status_code)
                body, status = decode_refusal(verdict, dr.status_code, _json_or_none(dr), rid)
                return JSONResponse(body, status_code=status)
            emissions = []
            try:
                if not want_stream:
                    emissions = reader.feed_blocking(dr.json())
                else:
                    for line in dr.iter_lines(decode_unicode=True):
                        emissions += reader.feed(line)
                        if reader.finished:
                            break
            finally:
                if want_stream and dr is not None:
                    dr.close()
            verdict, payload, status = terminal_verdict(reader)
            if verdict == TRUNCATED:
                logger.warning(
                    "decode stream for %s ended after %d tokens with no done/error message",
                    rid,
                    len(asm.token_ids) if asm else 0,
                )
            timing = reader.timing
            finish = reader.finish_reason
            token_ids = reader.token_ids
            if verdict != OK:
                return JSONResponse(payload, status_code=status)
            if asm is not None:
                emissions += asm.finish()
                token_ids = asm.token_ids
            if asm is None:
                choice, n_completion = textless_choice(
                    is_chat=is_chat, from_node=finish, token_ids=token_ids
                )
            else:
                choice, n_completion = blocking_choice(
                    collect(emissions),
                    is_chat=is_chat,
                    stream=asm,
                    from_node=finish,
                    logprobs_asked=logprobs_req is not None,
                    token_ids=token_ids,
                )
            return JSONResponse(
                blocking_envelope(
                    choice,
                    is_chat=is_chat,
                    prefill=prefill,
                    created=created,
                    usage=build_usage(
                        (prefill.get("usage") or {}).get("prompt_tokens"), n_completion
                    ),
                    timing={"prefill": round(1000 * (t_prefill - t0), 1), **timing},
                )
            )
        except PrefillClientError as e:
            logger.info("vLLM rejected the request (%d): %s", e.status, str(e.payload)[:200])
            return JSONResponse(e.payload, status_code=e.status)
        except Exception as e:
            logger.exception("pd request failed")
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            lease.release(terminated=reader is not None and reader.node_terminated)

    async def _settle(task, what: str) -> None:
        """Wait for a cancelled task to finish, but never indefinitely.

        `await <cancelled task>` reads as a formality and is not one: a task
        can absorb its cancellation and keep running (see _watch), and then
        this await is where the request stops for good. asyncio.wait does not
        re-raise what the task raised, so nothing has to be suppressed here --
        both callers are best effort.
        """
        import asyncio

        done, _pending = await asyncio.wait({task}, timeout=_TASK_SETTLE_S)
        if not done:
            logger.warning(
                "%s did not stop within %.0fs after being cancelled; abandoning it",
                what,
                _TASK_SETTLE_S,
            )

    async def _send_watching_client(client, req, request):
        import asyncio

        send = asyncio.ensure_future(client.send(req, stream=True))
        # Stops the watcher WITHOUT relying on cancellation; see _watch.
        stop = asyncio.Event()

        async def _watch():
            """Poll for a client disconnect until `stop` is set.

            Driven by the event rather than by cancellation, because cancelling
            this task is not reliable. Starlette runs its disconnect poll inside
            an anyio CancelScope it has ALREADY cancelled
            (`cs.cancel(); await self._receive()`), and such a scope absorbs a
            CancelledError delivered from outside it just as readily as its own.
            A watch.cancel() landing in that window is swallowed, the loop goes
            round again, and the `await watch` below never returns, stranding
            the caller and its decode lease.
            """
            while not stop.is_set():
                if await request.is_disconnected():
                    return
                with contextlib.suppress(asyncio.TimeoutError, TimeoutError):
                    await asyncio.wait_for(stop.wait(), _DISCONNECT_POLL_S)

        watch = asyncio.ensure_future(_watch())
        try:
            await asyncio.wait({send, watch}, return_when=asyncio.FIRST_COMPLETED)
            if send.done():
                return send.result()
            send.cancel()
            await _settle(send, "decode send")
            return None
        finally:
            # Order matters: `stop` first, so the watcher ends even if the
            # cancel that follows is swallowed.
            stop.set()
            watch.cancel()
            await _settle(watch, "disconnect watcher")

    async def _handle_stream(path: str, body: dict, request: Request):
        import anyio
        from starlette.concurrency import run_in_threadpool

        try:
            req = gate_request(path, body, tokenizer=ctx.tokenizer, parser_active=_parser_active)
            caps = await run_in_threadpool(pool.capabilities)
            validate_generation_request(body, caps, ctx.gen_defaults.resolve({}))
        except (CapabilityError, GrammarError, LogprobsUnsupported) as e:
            return JSONResponse(e.to_payload(), status_code=e.http_status)
        stop, include_stop = (req.stop, req.include_stop)
        logprobs_req, grammar_spec = (req.logprobs_req, req.grammar_spec)
        lease = None
        try:
            with anyio.CancelScope(shield=True):
                lease, waited = await run_in_threadpool(acquire_lease, pool)
            if lease is None:
                return _busy_response(waited)
            _tilert_bind_lease_to_task(lease)
            node = lease.node
            prefill = await run_in_threadpool(_prefill, path, body, node, logprobs_req)
            rid = lease.rid = derive_rid(prefill["id"])
            first_token_id = first_token_from_logprobs(prefill, True)
            first_lp = (
                first_token_logprob_from_prefill(prefill, logprobs_req.top_n)
                if logprobs_req is not None
                else None
            )
        except PrefillClientError as e:
            if lease is not None:
                lease.release()
            logger.info("vLLM rejected the stream request (%d): %s", e.status, str(e.payload)[:200])
            return JSONResponse(e.payload, status_code=e.status)
        except Exception as e:
            if lease is not None:
                lease.release()
            logger.exception("pd stream request failed before streaming")
            return JSONResponse({"error": str(e)}, status_code=502)
        except BaseException:
            if lease is not None:
                lease.release()
            raise
        import asyncio

        import httpx

        client = httpx.AsyncClient(timeout=httpx.Timeout(_HTTP_TIMEOUT_S, read=_HTTP_TIMEOUT_S))
        decode_resp = None
        try:
            for attempt in range(1, _DECODE_BUSY_ATTEMPTS + 1):
                lease.dispatched = True
                decode_resp = await _send_watching_client(
                    client,
                    client.build_request(
                        "POST",
                        f"{node.http_base}/pd/decode",
                        json=_decode_body(
                            rid,
                            first_token_id,
                            body,
                            grammar_spec,
                            stream=True,
                            logprobs_req=logprobs_req,
                            thinking=req.thinking,
                        ),
                    ),
                    request,
                )
                if decode_resp is None:
                    logger.info(
                        "client disconnected while the decode node was still holding headers for %s",
                        rid,
                    )
                    return JSONResponse(
                        {
                            "error": "client disconnected",
                            "error_type": "request_cancelled",
                            "rid": rid,
                        },
                        status_code=499,
                    )
                if decode_resp.status_code != 429 or attempt == _DECODE_BUSY_ATTEMPTS:
                    break
                await decode_resp.aclose()
                logger.info(_RETRY_LOG, node.http_base, rid, _DECODE_BUSY_RETRY_S)
                await asyncio.sleep(_DECODE_BUSY_RETRY_S)
            if decode_resp.status_code != 200:
                payload = None
                if decode_resp.status_code != 429:
                    try:
                        await decode_resp.aread()
                        payload = decode_resp.json()
                    except Exception:
                        payload = None
                await decode_resp.aclose()
                verdict = classify_decode_status(
                    decode_resp.status_code,
                    payload,
                    attempts_left=False,
                    propagated_types=PROPAGATED_ERROR_TYPES,
                )
                _log_refusal(verdict, node, rid, decode_resp.status_code, payload)
                body, status = decode_refusal(verdict, decode_resp.status_code, payload, rid)
                return JSONResponse(body, status_code=status)
        except Exception as e:
            logger.exception("opening the decode stream failed for %s", rid)
            with contextlib.suppress(Exception):
                if decode_resp is not None:
                    await decode_resp.aclose()
            return JSONResponse({"error": str(e)}, status_code=502)
        finally:
            if decode_resp is None or decode_resp.status_code != 200:
                lease.release()
                with contextlib.suppress(Exception):
                    await client.aclose()
        chunk_id = prefill["id"]
        model = prefill.get("model")
        prompt_tokens = (prefill.get("usage") or {}).get("prompt_tokens")
        parser = ctx.parser(req.thinking)
        created = int(time.time())
        _chunk = functools.partial(sse_chunk, chunk_id=chunk_id, model=model, created=created)
        _usage_chunk = functools.partial(
            usage_chunk, chunk_id=chunk_id, model=model, created=created
        )

        async def _gen():
            import anyio

            asm = ReplyStream(
                ctx.tokenizer,
                stop=stop,
                include_stop_in_output=include_stop,
                parser_session=parser.stream() if parser else None,
                logprobs_req=logprobs_req,
                first_token_logprob=first_lp,
            )
            reader = DecodeReader(stream=asm, logprobs_req=logprobs_req, rid=rid)
            out = SseWriter(asm, _chunk)
            finish_reason = "stop"
            client_gone = False
            decode_done = False
            try:
                async with contextlib.aclosing(decode_resp) as resp:
                    async for line in resp.aiter_lines():
                        if await request.is_disconnected():
                            client_gone = True
                            logger.info("client disconnected, cancelling %s", rid)
                            break
                        for frame in out.frames(reader.feed(line)):
                            yield frame
                        if reader.finished:
                            if reader.stop_hit:
                                logger.info("stop string %r ended %s", asm.stop_reason, rid)
                            break
                decode_done = reader.node_terminated
                finish_reason = reader.finish_reason
                verdict, payload, _status = terminal_verdict(reader, client_gone=client_gone)
                if verdict == REFUSED:
                    logger.warning(
                        "decode node %s sent an unusable logprobs line for %s", node.http_base, rid
                    )
                elif verdict == TRUNCATED:
                    logger.warning(
                        "decode stream for %s ended after %d tokens with no done/error message",
                        rid,
                        len(asm.token_ids) if asm else 0,
                    )
                if verdict in (REFUSED, TRUNCATED, TYPED_ERROR):
                    for _c in out.fail_closed(payload):
                        yield _c
                    return
                if verdict == UNTYPED_ERROR:
                    for _c in out.flush_held():
                        yield _c
                    yield _chunk({"content": f"\n[decode error: {payload['error']}]"})
                    finish_reason = "stop"
                if client_gone:
                    logger.info("client gone mid-stream for %s", rid)
                    return
                for _c in out.flush_held():
                    yield _c
                yield _chunk(
                    {},
                    finish=reply_finish_reason(
                        saw_tool=out.saw_tool, from_node=finish_reason, stream=asm
                    ),
                    stop_reason=asm.stop_reason,
                )
                if should_include_usage(body, ctx.force_include_usage):
                    yield _usage_chunk(build_usage(prompt_tokens, asm.completion_tokens))
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.exception("stream failed mid-flight for %s", rid)
                with contextlib.suppress(Exception):
                    for _c in out.fail_closed(
                        {
                            "error": f"stream failed: {exc}",
                            "error_type": "decode_stream_failed",
                            "rid": rid,
                        }
                    ):
                        yield _c
            finally:
                lease.release(terminated=decode_done)
                with anyio.CancelScope(shield=True):
                    await client.aclose()

        return StreamingResponse(_gen(), media_type="text/event-stream")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        from starlette.concurrency import run_in_threadpool

        body = await request.json()
        if body.get("stream"):
            return await _handle_stream("/v1/chat/completions", body, request)
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
        help="output parser (reasoning + tool calls); anything but 'none' loads vLLM's parser engine and needs vllm importable",
    )
    ap.add_argument("--model", default="", help="model profile")
    ap.add_argument(
        "--generation-config",
        choices=["auto", "vllm"],
        default="auto",
        help="where sampling defaults come from, mirroring vLLM's flag of the same name: 'auto' reads the model's generation_config.json under --model-path, 'vllm' ignores it and uses the neutral defaults. Whichever is chosen, the resolved values are sent explicitly to BOTH legs so they cannot disagree.",
    )
    ap.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="override the resolved temperature default (vLLM: --override-generation-config)",
    )
    ap.add_argument(
        "--default-top-p", type=float, default=None, help="override the resolved top_p default"
    )
    ap.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="override the resolved top_k default; 0 disables the rank cut, as it does in vLLM",
    )
    ap.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="override the resolved repetition_penalty default. Only executable where --model names a family whose decode runtime implements penalties; 1.0 is the runtime no-op.",
    )
    ap.add_argument(
        "--queue-timeout",
        type=float,
        default=0.0,
        help="seconds to wait for a free decode node before answering 429 (0: fail fast)",
    )
    ap.add_argument(
        "--force-include-usage",
        action="store_true",
        help="emit the trailing usage chunk on every stream, even when the client omits stream_options.include_usage (vLLM: enable_force_include_usage). Off by default; note it makes every stream end with a choices:[] chunk, which some clients cannot read.",
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
    try:
        gen_defaults = generation_defaults.load(
            args.model_path,
            args.generation_config,
            model=args.model,
            temperature=args.default_temperature,
            top_p=args.default_top_p,
            top_k=args.default_top_k,
            repetition_penalty=args.default_repetition_penalty,
        )
    except UnsupportedGenerationDefault as e:
        raise SystemExit(f"cannot serve with these sampling defaults:\n{e}")
    ctx = RouterCtx(
        args.vllm_url,
        Pool(nodes, args.queue_timeout),
        tokenizer,
        args.parser,
        force_include_usage=args.force_include_usage,
        gen_defaults=gen_defaults,
    )
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
