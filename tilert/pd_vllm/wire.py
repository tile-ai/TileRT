"""Shared control-plane protocol for vLLM-prefill -> TileRT-decode PD.

Model-agnostic. The per-model wire *layout* (which regions exist, their
sizes and offsets) lives in the model profile (``profiles/``); this module
owns only the framing everything shares:

  - length-prefixed JSON messages (send_msg / recv_msg)
  - the hello envelope (server -> client): common fields + a profile-supplied
    ``layout`` dict of region base addresses
  - request / done messages (client -> server)
  - local_ip / derive_rid helpers

Per-request control flow (one TCP connection per participating rank):
  server -> client : hello   {magic, protocol_version, layout_version,
                              session_id, max_seq_len, busy, **layout}
  client -> server : request {rid, rank, seq_len, last_prompt_token, sampling?,
                              prompt_token_ids?}   (rank 0 only, penalties only)
  server -> client : accept  {accepted: true, rid, rank, generation}
                  or reject  {accepted: false, error, ...}
  client -> server : done    {done, rid, rank, generation}  (after the RDMA write)

The accept step is an ADMISSION step, not an acknowledgement: the receive buffer
holds one request at a time, so a sender that writes before being admitted can
land its KV inside a request the decode node is already serving. Nothing detects
that afterwards -- the victim decodes from a mix of two prompts' state and
returns a confident wrong answer. So the sender must not touch RDMA until it has
an accept whose rid, rank and generation all match what it asked for.
"""

import json
import socket
import struct

MAGIC = "tilert-pd"

# Control-plane version. 1 = the original "write immediately after the request
# message" flow; 2 adds the accept/reject admission step and carries a
# generation on accept/done. Bumped together on both ends: a v1 sender paired
# with a v2 receiver would write without being admitted, which is the exact
# failure this version exists to remove, so the pairing is refused at hello
# rather than tolerated.
PROTOCOL_VERSION = 2

NUM_RANKS = 8
EXPECTED_RANKS = tuple(range(NUM_RANKS))


def local_ip(probe_addr: str | None = None) -> str:
    """Best-effort local IP for the mooncake session identity."""
    import os

    probe = probe_addr or os.environ.get("TILERT_PD_PROBE_ADDR", "8.8.8.8")
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((probe, 1))
        return s.getsockname()[0]
    finally:
        s.close()


def wants_prompt_token_ids(sampling: dict | None) -> bool:
    """Should this request ship its full prompt id list to the decode node?

    Only ``repetition_penalty`` is scoped over prompt UNION output, so only it
    needs the prompt half of the decode-side bitmap. ``presence_penalty`` is
    output-scoped and does not justify the payload on its own; a value of
    exactly 1.0 is the kernel's no-op and needs nothing either. Mirrors vLLM's
    ``needs_prompt_token_ids`` gate in
    ``v1/worker/gpu_input_batch.py::make_sampling_metadata``, which likewise
    skips the copy when no request in the batch has penalties.
    """
    if not sampling:
        return False
    rep = sampling.get("repetition_penalty")
    if rep is None:
        return False
    try:
        return float(rep) != 1.0
    except (TypeError, ValueError):
        return False


def derive_rid(request_id: str) -> str:
    """Map a vLLM request/response id to the client-visible rid.

    Shared by the prefill connector (internal id) and the router (response id)
    so both agree.
    """
    rid = request_id
    for prefix in ("chatcmpl-", "cmpl-"):
        if rid.startswith(prefix):
            rid = rid[len(prefix) :]
            break
    parts = rid.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) <= 8 and all(c in "0123456789abcdef" for c in parts[1]):
        rid = parts[0]
    parts = rid.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) <= 3:
        rid = parts[0]
    return rid


def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode()
    sock.sendall(struct.pack("!I", len(data)) + data)


def recv_msg(sock: socket.socket) -> dict:
    hdr = _recv_exact(sock, 4)
    (n,) = struct.unpack("!I", hdr)
    if n > 16 << 20:
        raise ValueError(f"control message too large: {n}")
    return json.loads(_recv_exact(sock, n).decode())


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("connection closed mid-message")
        buf += chunk
    return buf


def hello_msg(
    transport: str,
    transport_meta: dict,
    max_seq_len: int,
    layout_version: int,
    layout: dict,
    busy: bool,
) -> dict:
    """Common hello envelope.

    ``transport`` names the RDMA backend and ``transport_meta`` carries its connection info
    (mooncake: session_id; nixl: nixl_meta/nixl_dev). ``layout`` carries profile-specific region
    base addresses (e.g. kv_base / pe_base / ki_base).

    ``protocol_version`` is the CONTROL-plane version, independent of
    ``layout_version`` (which versions the buffer geometry). It exists so a
    sender that does not wait for :func:`accept_msg` cannot be paired with a
    receiver that expects it: the mismatch fails at handshake instead of being
    guessed at run time.
    """
    return {
        "magic": MAGIC,
        "protocol_version": PROTOCOL_VERSION,
        "layout_version": layout_version,
        "transport": transport,
        "max_seq_len": max_seq_len,
        "busy": busy,
        **transport_meta,
        **layout,
    }


def accept_msg(rid: str, rank: int, generation: int) -> dict:
    """Receiver -> sender: this rank may now RDMA-write for ``rid``.

    ``generation`` identifies the receive-buffer tenancy the write is authorised
    against. It is echoed in :func:`done_msg` so a ``done`` arriving after the
    buffer has been handed to a later request is recognisable as stale rather
    than counted towards the current one.
    """
    return {"accepted": True, "rid": rid, "rank": rank, "generation": generation}


def reject_msg(reason: str, **extra) -> dict:
    """Receiver -> sender: do NOT write. ``reason`` is for the sender's log.

    The sender must treat anything that is not a matching accept as a rejection,
    so an unrecognised reason is still safe.
    """
    return {"accepted": False, "error": reason, **extra}


def done_msg(rid: str, rank: int, generation: int) -> dict:
    """Sender -> receiver: this rank's RDMA write has completed."""
    return {"done": True, "rid": rid, "rank": rank, "generation": generation}
