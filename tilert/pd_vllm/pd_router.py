"""PD router: client-facing entry that orchestrates vLLM prefill and TileRT decode.

OpenAI-semantics output parsing (reasoning + tool calls), streaming and
non-streaming.

Flow per request (phase-1 hybrid, see design doc):
  1. pick a free decode node (in-memory busy tracking; all busy -> wait up to
     --queue-timeout, then 429)
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
      --vllm-url http://<prefill-node-ip>:8000 \
      --decode <decode-node-ip>:5556:5557 --port 23333 \
      --model-path /path/to/model --model glm5 --parser glm47
"""

import argparse
import contextlib
import functools
import logging
import time

import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from tilert.pd_vllm import generation_defaults
from tilert.pd_vllm.capabilities import CapabilityError, validate_generation_request
from tilert.pd_vllm.decode_pool import DecodeNode, Pool, acquire_lease
from tilert.pd_vllm.decode_response import (
    BUSY,
    OK,
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
from tilert.pd_vllm.generation_defaults import (
    GenerationDefaults,
    UnsupportedGenerationDefault,
)
from tilert.pd_vllm.grammar_spec import GrammarError
from tilert.pd_vllm.logprobs import LogprobsUnsupported
from tilert.pd_vllm.openai_params import resolve_max_tokens
from tilert.pd_vllm.presentation import (
    SseWriter,
    blocking_choice,
    blocking_envelope,
    collect,
)
from tilert.pd_vllm.presentation import finish_reason as reply_finish_reason
from tilert.pd_vllm.presentation import (
    sse_chunk,
    textless_choice,
    usage_chunk,
)
from tilert.pd_vllm.reply import ReplyStream
from tilert.pd_vllm.request_gate import gate_request
from tilert.pd_vllm.wire import derive_rid

logger = logging.getLogger("pd_vllm.router")

QUEUE_LOG_SECONDS = 0.1

# Output length when the client names none; carried over from the original
# import rather than tuned. Decode only -- the prefill request is pinned to
# max_tokens=1 with both client field names stripped (_PREFILL_DROP_FIELDS).
_DECODE_MAX_TOKENS_DEFAULT = 256

# error_types the decode node may return that carry client-meaningful HTTP
# status (400/500/501) and must be propagated verbatim rather than masked as 502.
#
# The rule is "typed and classified by the decode node": those statuses are the
# node's considered answer about THIS request, so flattening them into 502 tells
# the operator to go restart a healthy component. An untyped failure stays 502,
# which is what it is -- a component call that did not work.
# A decode node answers 429 when its single slot is still held. The router's own
# gating should make that rare, but not impossible: it frees its reservation as
# soon as it stops reading, while the node's slot unwinds a little later (a
# cancel handshake, an engine reset), so a request dispatched into that window
# meets the node's admission. One short retry absorbs it; a node still busy after
# that gets an honest 429, which is retryable, rather than a 502, which tells the
# operator to go restart a healthy component.
_DECODE_BUSY_ATTEMPTS = 2
_DECODE_BUSY_RETRY_S = 0.5
_RETRY_LOG = "decode node %s busy for %s, retrying in %.1fs"


def _log_refusal(verdict: str, node, rid: str, status: int, payload=None) -> None:
    """One line per refused dispatch, at the level the verdict deserves."""
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


# How often the streaming preflight checks whether the client is still there
# while the decode node holds its headers. Starlette offers no awaitable for a
# disconnect before the response begins, so this is a poll.
_DISCONNECT_POLL_S = 0.2


# The status each typed error deserves, taken from the exception class that
# raises it on the node so the two protocols cannot disagree.
#
# Needed because they carry the type differently. Over the BLOCKING protocol the
# node answers an HTTP status and the router forwards it. Over the STREAMING one
# the error arrives inside a 200 body -- the status is already spent -- so the
# router has to reconstruct it, and mapping every propagated type to one status
# is how `logprobs_unavailable` came back as 400 on one path and 501 on the
# other for the same inability.
class PrefillClientError(Exception):
    """vLLM rejected the prefill request itself (4xx).

    That is the client's fault, not a component failure, so it must reach the
    client with vLLM's own status and body instead of being flattened into a 502.
    Hit in practice by `response_format` specs vLLM validates before we ever get
    to the decode node (e.g. an uncompilable json_schema).
    """

    def __init__(self, status: int, payload):
        super().__init__(f"vLLM rejected the request with {status}")
        self.status = status
        self.payload = payload


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


def _token_id_of(tok: object) -> int | None:
    """The integer id behind a vLLM ``token_id:N`` string, or None."""
    if isinstance(tok, str) and tok.startswith("token_id:"):
        try:
            return int(tok.split(":", 1)[1])
        except ValueError:
            return None
    return None


def first_token_logprob_from_prefill(resp: dict, top_n: int):
    """Token 1's ``(logprob, candidates)`` out of the prefill response.

    The decode node echoes ``first_token_id`` without sampling it, so the only
    distribution for that position is the prefill instance's. The router already
    reads this entry for the token id; this reads the numbers beside it.

    ``(None, [])`` when there is no usable entry -- the caller surfaces the
    documented sentinel for that one position rather than failing the request.

    The value carries the prefill instance's ``logprobs_mode``, vLLM's default
    being ``raw_logprobs``, so the decode side must be in its raw mode for the
    array to sit on one scale (see the engine adapter's ``supports_logprobs``).
    """
    choice = (resp.get("choices") or [{}])[0]
    content = (choice.get("logprobs") or {}).get("content") or []
    if not content:
        return None, []
    entry = content[0]
    lp = entry.get("logprob")
    lp = float(lp) if isinstance(lp, (int, float)) else None
    cands = []
    for alt in (entry.get("top_logprobs") or [])[:top_n]:
        alt_id = _token_id_of(alt.get("token"))
        alt_lp = alt.get("logprob")
        if alt_id is not None and isinstance(alt_lp, (int, float)):
            cands.append((alt_id, float(alt_lp)))
    return lp, cands


# Client fields that must not survive into the prefill request. The body is
# forwarded verbatim apart from the fields we set, so anything vLLM checks
# against one of those overrides has to go first.
#
#   stream_options        -- we force stream=False, and vLLM rejects the pair
#                            with 400 "Stream options can only be defined when
#                            `stream=True`". Its validator is mode="before", so
#                            the request dies before the model is looked at.
#   max_completion_tokens -- takes precedence over max_tokens in vLLM, so it
#                            would override our max_tokens=1 and make the
#                            prefill instance decode the client's whole output
#                            length, defeating the split.
#   stop, include_stop_str_in_output
#                         -- the router matches these itself, over the whole
#                            reply. Left in, vLLM would match against the one
#                            token it generates and could report
#                            finish_reason="stop" for a prefill that succeeded.
#
# Any streaming client sends the first two, `vllm bench serve
# --backend openai-chat` included -- which made the official benchmark fail
# every request with 502.
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
    """The vLLM request that prefills only and hands the KV state to ``node``.

    Lives outside ``build_app`` so the rewrite can be unit-tested without a
    router process, a vLLM instance or a decode node.

    ``logprobs`` is always requested, because the first token's id is recovered
    from it. When the client asked for logprobs as well, ``top_logprobs`` is
    raised to the count they asked for: token 1's candidate row can only come
    from here, since the decode node never sampled that position.
    """
    prefill_body = dict(body)
    prefill_body["max_tokens"] = 1
    prefill_body["stream"] = False
    for field in _PREFILL_DROP_FIELDS:
        prefill_body.pop(field, None)
    if path.endswith("chat/completions"):
        prefill_body["logprobs"] = True
        # 1 is the floor, not the default: the id extraction needs one entry.
        prefill_body["top_logprobs"] = max(1, logprobs_req.top_n if logprobs_req is not None else 0)
    else:
        prefill_body["logprobs"] = 1
    # Pin temperature/top_p/top_k explicitly on BOTH legs. Left absent, this leg
    # would resolve them through vLLM's own chain (generation_config.json, then
    # the neutral defaults) while the decode leg resolved its own -- the split
    # that had token 1 sampled at temperature 0.6 / top_k 20 and tokens 2..N at
    # 1.0 / uncapped. One resolution, written to both requests, so the
    # prefill instance's own --generation-config can no longer move one leg
    # without the other.
    prefill_body.update((defaults or GenerationDefaults()).resolve(body))
    prefill_body["kv_transfer_params"] = {
        "tilert_host": node.host,
        "tilert_ctrl_port": node.ctrl_port,
    }
    return prefill_body


def build_usage(prompt_tokens, completion_tokens: int) -> dict:
    """``usage``, shaped like vLLM's ``UsageInfo``.

    All three fields are integers there (``prompt_tokens: int = 0``,
    ``total_tokens: int = 0``, ``completion_tokens: int | None = 0``), and
    ``total_tokens`` is always present. This endpoint used to omit it on the
    non-streaming path while sending it on the streaming one, so a client reading
    ``usage.total_tokens`` -- which every OpenAI SDK does -- got a KeyError from
    one shape and a number from the other.

    ``prompt_tokens`` is coerced because it is copied from the prefill response,
    where it can be absent; ``None`` would violate the contract just as surely as
    the missing key did.
    """
    prompt = int(prompt_tokens or 0)
    completion = int(completion_tokens)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def should_include_usage(body: dict, force: bool = False) -> bool:
    """Whether this stream carries usage at all, per the OpenAI contract.

    Absent ``include_usage`` means no usage anywhere in the stream -- not usage
    relocated to another chunk. vLLM (``serve/utils/api_utils.py``) and SGLang
    (``serving_chat.py``) both resolve it through a function of this name and
    shape; this endpoint is judged against them.

    Relocating it, which the router used to do, is worse than non-conformant:
    a client written as ``if chunk.choices: ... elif chunk.usage:`` takes the
    choices branch and never reads usage riding on the same chunk. That is how
    the InferenceX bench reported `Total generated tokens: 0`.

    ``force`` is the deployment escape hatch, mirroring vLLM's
    ``enable_force_include_usage`` and SGLang's
    ``stream_response_default_include_usage``. Off by default in all three.

    Reads the client body only: vLLM never sees ``stream_options`` (stripped,
    since we force stream=False and vLLM 400s the pair) and the decode node is
    sent only sampling params.
    """
    if force:
        return True
    opts = body.get("stream_options")
    return bool(isinstance(opts, dict) and opts.get("include_usage"))


def _json_or_none(response) -> dict | None:
    """The body as JSON, or None.

    A node classifies its failures there; a proxy or a crash answers with something else.
    """
    try:
        return response.json()
    except ValueError:
        return None


class RouterCtx:
    """Immutable per-process context (tokenizer, parser factory, config)."""

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
        # Resolved once at startup, exactly as vLLM resolves its own
        # default_sampling_params. Both legs are handed THESE values, so they
        # cannot disagree whatever they are; see generation_defaults.
        self.gen_defaults = gen_defaults or GenerationDefaults()
        # Defaults off, as it does in vLLM and SGLang: a deployment may opt its
        # whole fleet into usage, but never by omission.
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


def build_app(ctx: RouterCtx) -> FastAPI:
    app = FastAPI()
    pool = ctx.pool

    def _parser_active(thinking: bool) -> bool:
        """Whether an output parser would run, given the thinking flag.

        A router-configuration answer the gate needs but cannot look up.
        """
        return ctx.parser(thinking) is not None

    def _busy_response(waited: float) -> JSONResponse:
        """429 body, telling a full pool apart from an exhausted timeout."""
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

    # ── shared prefill step ──────────────────────────────────────────────
    def _prefill(path, body, node, logprobs_req=None):
        prefill_body = build_prefill_body(path, body, node, logprobs_req, ctx.gen_defaults)
        r = requests.post(f"{ctx.vllm_url}{path}", json=prefill_body, timeout=600)
        if 400 <= r.status_code < 500:
            try:
                payload = r.json()
            except ValueError:
                payload = {"error": r.text[:500]}
            raise PrefillClientError(r.status_code, payload)
        r.raise_for_status()
        return r.json()

    def _sampling_of(body):
        # ignore_eos is the decode node's alone: the prefill request is pinned to
        # max_tokens=1 so it never reaches a stop token; the decode loop owns the
        # stop set and clears it for the flag (adapters declare this via
        # /capabilities, and the router refuses the request if none can).
        #
        # Every OTHER field a client may send is either forwarded here or
        # refused by validate_generation_request. A field that is neither would
        # be applied by the vLLM prefill instance to token 1 and silently
        # dropped for the rest of the reply -- so this whitelist and that gate
        # must be kept in step (test_no_forwarded_field_is_left_ungated).
        forwarded = (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "presence_penalty",
            "ignore_eos",
        )
        sampling = {k: body[k] for k in forwarded if k in body}
        # Resolved, never defaulted downstream: the same call on the same body
        # produced the values already written into the prefill request.
        sampling.update(ctx.gen_defaults.resolve(body))
        return sampling

    def _decode_body(
        rid, first_token_id, body, grammar_spec, *, stream=False, logprobs_req=None, thinking=True
    ):
        payload = {
            "rid": rid,
            "first_token_id": first_token_id,
            # The whole output length is decoded here; the prefill
            # request is pinned to max_tokens=1 and both client field names
            # are stripped
            # from it (see _PREFILL_DROP_FIELDS).
            "max_tokens": resolve_max_tokens(body, _DECODE_MAX_TOKENS_DEFAULT),
            "sampling": _sampling_of(body),
        }
        if stream:
            payload["stream"] = True
        if grammar_spec is not None:
            # Only sent for constrained requests; plain requests stay byte-for-
            # byte on the existing unconstrained path.
            payload["grammar_spec"] = grammar_spec
            payload["enable_thinking"] = thinking
        if logprobs_req is not None:
            # Omitted entirely when not requested, so a decode node that
            # predates the field is unaffected.
            payload["top_logprobs"] = logprobs_req.top_n
        return payload

    # ── non-streaming ────────────────────────────────────────────────────
    def _handle(path: str, body: dict):
        try:
            # Request-only first, then the probe -- `request_gate` says why.
            req = gate_request(path, body, tokenizer=ctx.tokenizer, parser_active=_parser_active)
            validate_generation_request(body, pool.capabilities(), ctx.gen_defaults.resolve({}))
        except (CapabilityError, GrammarError, LogprobsUnsupported) as e:
            return JSONResponse(e.to_payload(), status_code=e.http_status)
        is_chat, stop, include_stop = req.is_chat, req.stop, req.include_stop
        logprobs_req, grammar_spec = req.logprobs_req, req.grammar_spec
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
            # Same object, same arguments as the streaming path; this function
            # only concatenates the emissions instead of framing them.
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

            # Only the router can see a stop, and not before the tokens arrive.
            # Asking for the whole sequence up front would make a stop merely
            # trim the answer: the node would run to max_tokens and hold its slot
            # for all of it. So a request with something to match reads the
            # node's STREAMING protocol even though its own reply is not.
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
            # True from the moment a POST goes out. The node may have admitted
            # the request before the call failed -- a timeout or a reset while
            # waiting for headers -- and it then holds its slot until its own
            # timeout. Cancelling an rid the node never saw is harmless; not
            # cancelling one it did is a slot lost for `timeout_s`.
            dr = None
            for attempt in range(1, _DECODE_BUSY_ATTEMPTS + 1):
                lease.dispatched = True
                dr = requests.post(
                    f"{node.http_base}/pd/decode",
                    json=decode_payload,
                    timeout=600,
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
                    # An unread streamed body holds the connection; a
                    # non-streamed one was already consumed by `post`.
                    dr.close()
                if verdict == RETRY:
                    logger.info(_RETRY_LOG, node.http_base, rid, _DECODE_BUSY_RETRY_S)
                    time.sleep(_DECODE_BUSY_RETRY_S)
                    continue
                _log_refusal(verdict, node, rid, dr.status_code)
                body, status = decode_refusal(verdict, dr.status_code, _json_or_none(dr), rid)
                return JSONResponse(body, status_code=status)
            assert dr is not None  # the loop always posts at least once

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
                    "decode stream for %s ended after %d tokens with " "no done/error message",
                    rid,
                    len(asm.token_ids) if asm else 0,
                )
            timing = reader.timing
            finish = reader.finish_reason
            token_ids = reader.token_ids
            if verdict != OK:
                # Every one of them is a status here; the streaming path has to
                # say the same things inside a spent 200.
                return JSONResponse(payload, status_code=status)
            if asm is not None:
                emissions += asm.finish()
                # One source for the id list, the count and the entries, so
                # they cannot disagree. The node's batching makes them disagree
                # otherwise: a stop lands part-way into a batch, and the tokens
                # behind it are not part of the reply.
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
            # The lease owns both rules; the one fact it cannot know is whether
            # the NODE said it was done.
            lease.release(terminated=reader is not None and reader.node_terminated)

    async def _send_watching_client(client, req, request):
        """Open the decode stream, giving up if the client leaves first.

        Returns the response, or None if the client disconnected while waiting.

        Needed because the decode node holds its response headers through the
        whole KV wire-wait (``/pd/decode`` streams only after the transfer
        lands), so this await can sit for ``timeout_s`` -- 120 s by default. At
        that point ``StreamingResponse`` does not exist yet, so neither the
        generator's ``finally`` nor its ``is_disconnected`` poll is running: a
        client that hangs up here would otherwise hold both the router's
        reservation and the decode slot for the full wait, which is exactly the
        failure this endpoint is meant to have stopped having.

        The disconnect is polled rather than awaited because that is the only
        signal Starlette offers for a request whose response has not begun.
        """
        import asyncio

        send = asyncio.ensure_future(client.send(req, stream=True))

        async def _watch():
            while not await request.is_disconnected():
                await asyncio.sleep(_DISCONNECT_POLL_S)

        watch = asyncio.ensure_future(_watch())
        try:
            await asyncio.wait({send, watch}, return_when=asyncio.FIRST_COMPLETED)
            if send.done():
                return send.result()
            # The watcher won: the client is gone. Abandon the send rather than
            # waiting it out -- the finally below cancels the decode node.
            send.cancel()
            # CancelledError is a BaseException, so suppress(Exception) would
            # let it escape and turn a clean 499 into a 500.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await send
            return None
        finally:
            watch.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await watch

    # ── streaming (chat only) ────────────────────────────────────────────
    async def _handle_stream(path: str, body: dict, request: Request):
        import anyio
        from starlette.concurrency import run_in_threadpool

        try:
            req = gate_request(path, body, tokenizer=ctx.tokenizer, parser_active=_parser_active)
            # capabilities() may probe over HTTP on a cache miss, so it goes
            # off the event loop even though it is usually a dict lookup.
            caps = await run_in_threadpool(pool.capabilities)
            validate_generation_request(body, caps, ctx.gen_defaults.resolve({}))
        except (CapabilityError, GrammarError, LogprobsUnsupported) as e:
            return JSONResponse(e.to_payload(), status_code=e.http_status)
        stop, include_stop = req.stop, req.include_stop
        logprobs_req, grammar_spec = req.logprobs_req, req.grammar_spec
        lease = None  # taken inside the try: every handler below releases it
        try:
            # Shielded: a disconnect must not strand a reservation mid-acquire.
            with anyio.CancelScope(shield=True):
                lease, waited = await run_in_threadpool(acquire_lease, pool)
            if lease is None:
                return _busy_response(waited)
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
            # CancelledError is not an Exception; without this it stays busy.
            if lease is not None:
                lease.release()
            raise

        # Open the decode stream HERE, not inside the generator. Once the
        # generator runs the response has begun and the status is spent, so a
        # busy decode node could only be reported as an SSE error inside a 200 --
        # or, as it was, a stream truncated with no terminator. Sending the
        # request first keeps the status available for exactly the case that
        # needs it.
        import asyncio

        import httpx

        client = httpx.AsyncClient(timeout=httpx.Timeout(600, read=600))
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
                if decode_resp is None:  # client left; see the helper
                    logger.info(
                        "client disconnected while the decode node was "
                        "still holding headers for %s",
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
            assert decode_resp is not None  # the loop always posts at least once

            if decode_resp.status_code != 200:
                payload = None
                if decode_resp.status_code != 429:
                    try:
                        await decode_resp.aread()
                        payload = decode_resp.json()
                    except Exception:
                        payload = None
                await decode_resp.aclose()
                # Same classification and the same answers as the blocking
                # path: only the transport above this line differs.
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
            # Every early return above leaves the request unserved, so the node
            # goes back to the pool, the decode node is told to stop, and the
            # client is closed. The success path hands all of that to the
            # generator instead.
            if decode_resp is None or decode_resp.status_code != 200:
                lease.release()
                with contextlib.suppress(Exception):
                    await client.aclose()

        chunk_id = prefill["id"]
        model = prefill.get("model")
        prompt_tokens = (prefill.get("usage") or {}).get("prompt_tokens")
        parser = ctx.parser(req.thinking)
        # Stamped ONCE for the whole response, as vLLM does (it threads a single
        # `created_time` through every chunk it builds). Re-reading the clock per
        # chunk gave one response several timestamps, which breaks a client that
        # groups or de-duplicates by (id, created).
        created = int(time.time())

        # Every frame carries the same id / model / created, so they are bound
        # once here and the generator passes only what varies.
        _chunk = functools.partial(sse_chunk, chunk_id=chunk_id, model=model, created=created)
        _usage_chunk = functools.partial(
            usage_chunk, chunk_id=chunk_id, model=model, created=created
        )

        async def _gen():
            import anyio

            # Both paths drive this with the same arguments, which is why the
            # two replies agree about text, channels, logprobs and count.
            asm = ReplyStream(
                ctx.tokenizer,
                stop=stop,
                include_stop_in_output=include_stop,
                parser_session=parser.stream() if parser else None,
                logprobs_req=logprobs_req,
                first_token_logprob=first_lp,
            )

            # Shared with the non-streaming path: this loop keeps only the
            # transport (async) and the presentation (SSE).
            reader = DecodeReader(stream=asm, logprobs_req=logprobs_req, rid=rid)
            out = SseWriter(asm, _chunk)
            finish_reason = "stop"
            client_gone = False
            # Whether the decode node has told us it is done with this request.
            # It owns its slot until then, so anything that leaves this loop
            # early -- a client hanging up, a stop string, an exception -- has
            # to cancel; a node that reported `done` must not be.
            decode_done = False
            try:
                async with contextlib.aclosing(decode_resp) as resp:
                    async for line in resp.aiter_lines():
                        # Deterministic client-liveness check: writes to a
                        # dead socket do NOT raise (verified by drill), so
                        # poll the ASGI disconnect state every line.
                        if await request.is_disconnected():
                            client_gone = True
                            logger.info("client disconnected, cancelling %s", rid)
                            break
                        for frame in out.frames(reader.feed(line)):
                            yield frame
                        if reader.finished:
                            if reader.stop_hit:
                                # Complete. The node cannot see text and is
                                # still generating; the finally cancels it.
                                logger.info("stop string %r ended %s", asm.stop_reason, rid)
                            break
                decode_done = reader.node_terminated
                finish_reason = reader.finish_reason
                # Same conditions and the same payloads as the blocking path,
                # from the same function; only how they are SAID differs, and
                # only because this 200 is already spent.
                verdict, payload, _status = terminal_verdict(reader, client_gone=client_gone)
                if verdict == REFUSED:
                    logger.warning(
                        "decode node %s sent an unusable logprobs " "line for %s",
                        node.http_base,
                        rid,
                    )
                elif verdict == TRUNCATED:
                    logger.warning(
                        "decode stream for %s ended after %d tokens " "with no done/error message",
                        rid,
                        len(asm.token_ids) if asm else 0,
                    )
                if verdict in (REFUSED, TRUNCATED, TYPED_ERROR):
                    # A typed error is the node's considered answer about this
                    # request: emitting it as a content marker and finishing
                    # normally would report a successful completion that broke
                    # the contract asked for -- unconstrained output for a
                    # grammar, or a reply without the logprobs requested.
                    for _c in out.fail_closed(payload):
                        yield _c
                    return
                if verdict == UNTYPED_ERROR:
                    # No status left to carry it, so it goes in the text. Flush
                    # first, or the marker lands before text the reply earned:
                    # `prefix[decode error]suffix`.
                    for _c in out.flush_held():
                        yield _c
                    yield _chunk({"content": "\n[decode error: " f"{payload['error']}]"})
                    finish_reason = "stop"
                if client_gone:
                    logger.info("client gone mid-stream for %s", rid)
                    return  # finally fires the cancel
                for _c in out.flush_held():
                    yield _c
                # The same decision the non-streaming path makes, from the
                # same function.
                yield _chunk(
                    {},
                    finish=reply_finish_reason(
                        saw_tool=out.saw_tool, from_node=finish_reason, stream=asm
                    ),
                    stop_reason=asm.stop_reason,
                )
                if should_include_usage(body, ctx.force_include_usage):
                    # From the reply stream, not a counter in this loop.
                    yield _usage_chunk(build_usage(prompt_tokens, asm.completion_tokens))
                yield "data: [DONE]\n\n"
            except Exception as exc:
                # Malformed NDJSON, or a wrong-typed field that makes
                # `reader.feed` raise. The 200 is spent, so exiting here would
                # leave the client with a partial response and no terminator --
                # the same failure the clean-EOF check above refuses, reached by
                # a different route. Best effort: the client may already be gone,
                # in which case yielding raises again and there is nothing left
                # to say.
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
                # Runs under cancellation too (client disconnect cancels this
                # task). Order matters: release first (sync, can't be
                # cancelled), then best-effort cancel via a plain thread
                # (an await here could be cancelled before firing), then a
                # shielded aclose.
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
        help="output parser (reasoning + tool calls); anything but 'none' loads "
        "vLLM's parser engine and needs vllm importable",
    )
    ap.add_argument(
        "--model",
        default="",
        help="model profile the decode nodes serve (glm5 / glm5_2 / "
        "glm5_3 / dsv32); must match their --model. Used to decide "
        "whether a repetition_penalty in the model's generation_config "
        "can be adopted: only a profile whose decode runtime declares "
        "penalties may adopt one (the GLM-5 / GLM-5.2 / DSV3.2 runtimes "
        "do not). Omitted means unknown, and the conservative answer is "
        "taken.",
    )
    ap.add_argument(
        "--generation-config",
        choices=["auto", "vllm"],
        default="auto",
        help="where sampling defaults come from, mirroring vLLM's "
        "flag of the same name: 'auto' reads the model's "
        "generation_config.json under --model-path, 'vllm' "
        "ignores it and uses the neutral defaults. Whichever "
        "is chosen, the resolved values are sent explicitly to "
        "BOTH legs so they cannot disagree.",
    )
    ap.add_argument(
        "--default-temperature",
        type=float,
        default=None,
        help="override the resolved temperature default " "(vLLM: --override-generation-config)",
    )
    ap.add_argument(
        "--default-top-p", type=float, default=None, help="override the resolved top_p default"
    )
    ap.add_argument(
        "--default-top-k",
        type=int,
        default=None,
        help="override the resolved top_k default; 0 disables the " "rank cut, as it does in vLLM",
    )
    ap.add_argument(
        "--default-repetition-penalty",
        type=float,
        default=None,
        help="override the resolved repetition_penalty default. "
        "Only executable where --model names a family whose "
        "decode runtime implements penalties; 1.0 is the "
        "runtime no-op.",
    )
    ap.add_argument(
        "--queue-timeout",
        type=float,
        default=0.0,
        help="seconds to wait for a free decode node before " "answering 429 (0: fail fast)",
    )
    ap.add_argument(
        "--force-include-usage",
        action="store_true",
        help="emit the trailing usage chunk on every stream, even "
        "when the client omits stream_options.include_usage "
        "(vLLM: enable_force_include_usage). Off by default; "
        "note it makes every stream end with a choices:[] "
        "chunk, which some clients cannot read.",
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
        # Startup, not per request: a default the router cannot promise both legs
        # will apply must be settled by the operator before traffic arrives.
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
