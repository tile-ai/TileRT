"""Pluggable RDMA transport for the PD data plane: Mooncake (default) or NIXL."""

from __future__ import annotations

import base64
import os


class Transport:
    name = "?"

    def init(self, host: str) -> None: ...
    def register(self, ptr: int, nbytes: int, dev_id: int) -> None: ...
    def local_meta(self) -> dict: ...  # type: ignore[empty-body]
    def write(self, remote_meta: dict, srcs, dsts, lens) -> None: ...


class MooncakeTransport(Transport):
    """serve_sglang precedent: one TransferEngine, P2P handshake, sync write."""

    name = "mooncake"

    def init(self, host: str) -> None:
        from mooncake.engine import TransferEngine

        self.engine = TransferEngine()
        ret = self.engine.initialize(host, "P2PHANDSHAKE", "rdma", "")
        if ret != 0:
            raise RuntimeError(f"Mooncake engine init failed: {ret}")
        self.session_id = f"{host}:{self.engine.get_rpc_port()}"

    def register(self, ptr: int, nbytes: int, dev_id: int) -> None:
        ret = self.engine.batch_register_memory([ptr], [nbytes])
        if ret != 0:
            raise RuntimeError(f"Mooncake register failed: {ret}")

    def local_meta(self) -> dict:
        return {"session_id": self.session_id}

    def write(self, remote_meta: dict, srcs, dsts, lens) -> None:
        ret = self.engine.batch_transfer_sync_write(remote_meta["session_id"], srcs, dsts, lens)
        if ret != 0:
            raise RuntimeError(f"mooncake write failed: {ret}")


class NixlTransport(Transport):
    """NIXL agent over the UCX backend (GPUDirect RDMA).

    Registers VRAM regions with 4-tuple descriptors, exchanges agent metadata
    via the hello, and issues WRITE transfers built from (src,dst,len) triples.
    """

    name = "nixl"
    _MAX_POLL = 2_000_000

    def init(self, host: str) -> None:
        from nixl._api import nixl_agent, nixl_agent_config

        # agent name must be globally unique across the two peers
        self._agent = nixl_agent(f"{host}:{os.getpid()}", nixl_agent_config(backends=["UCX"]))
        self._remotes: dict[bytes, str] = {}  # remote meta -> remote name
        self._dev = 0

    def register(self, ptr: int, nbytes: int, dev_id: int) -> None:
        self._dev = dev_id
        self._agent.register_memory([(ptr, nbytes, dev_id, "")], "VRAM")

    def local_meta(self) -> dict:
        return {
            "nixl_meta": base64.b64encode(self._agent.get_agent_metadata()).decode(),
            "nixl_dev": self._dev,
        }

    def write(self, remote_meta: dict, srcs, dsts, lens) -> None:
        meta_b = base64.b64decode(remote_meta["nixl_meta"])
        rname = self._remotes.get(meta_b)
        if rname is None:
            rname = self._agent.add_remote_agent(meta_b)
            self._remotes[meta_b] = rname
        rdev = int(remote_meta.get("nixl_dev", 0))
        ld = self._agent.get_xfer_descs(
            [(int(s), int(n), self._dev) for s, n in zip(srcs, lens)], "VRAM"
        )
        rd = self._agent.get_xfer_descs(
            [(int(d), int(n), rdev) for d, n in zip(dsts, lens)], "VRAM"
        )
        h = self._agent.initialize_xfer("WRITE", ld, rd, rname)
        try:
            st = self._agent.transfer(h)
            polls = 0
            while st not in ("DONE", "ERR"):
                st = self._agent.check_xfer_state(h)
                polls += 1
                if polls > self._MAX_POLL:
                    raise RuntimeError("nixl xfer timed out")
            if st == "ERR":
                raise RuntimeError("nixl xfer failed")
        finally:
            self._agent.release_xfer_handle(h)


_BACKENDS = {"mooncake": MooncakeTransport, "nixl": NixlTransport}


def make_transport(name: str | None) -> Transport:
    key = (name or "mooncake").lower()
    if key not in _BACKENDS:
        raise ValueError(f"unknown transport {name!r}; " f"choices: {sorted(_BACKENDS)}")
    return _BACKENDS[key]()
