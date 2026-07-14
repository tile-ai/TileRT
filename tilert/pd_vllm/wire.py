"""Shared control-plane protocol for vLLM-prefill -> TileRT-decode PD."""

import json
import socket
import struct

MAGIC = "tilert-pd"

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
    """Build the common hello envelope.

    ``transport`` names the RDMA backend and ``transport_meta`` carries its
    connection info (mooncake: session_id; nixl: nixl_meta/nixl_dev).
    ``layout`` carries profile-specific region base addresses (e.g. gdn_base /
    gqa_k_base / kv_base).
    """
    return {
        "magic": MAGIC,
        "layout_version": layout_version,
        "transport": transport,
        "max_seq_len": max_seq_len,
        "busy": busy,
        **transport_meta,
        **layout,
    }
