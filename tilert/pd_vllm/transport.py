from __future__ import annotations

import base64
import os


class Transport:
    name = "?"

    def init(self, host: str) -> None: ...

    def register(
        self, ptr: int, nbytes: int, dev_id: int, location: str | None = None, host: bool = False
    ) -> None: ...

    def rails(self) -> int:
        return 0

    def local_meta(self) -> dict: ...  # type: ignore[empty-body]

    def write(self, remote_meta: dict, srcs, dsts, lens) -> None: ...


class MooncakeTransport(Transport):
    name = "mooncake"

    def init(self, host: str) -> None:
        from mooncake.engine import TransferEngine

        self.engine = TransferEngine()
        ret = self.engine.initialize(host, "P2PHANDSHAKE", "rdma", "")
        if ret != 0:
            raise RuntimeError(f"Mooncake engine init failed: {ret}")
        self.session_id = f"{host}:{self.engine.get_rpc_port()}"

    def register(
        self, ptr: int, nbytes: int, dev_id: int, location: str | None = None, host: bool = False
    ) -> None:
        if host:
            location = None
        args = ([ptr], [nbytes]) if location is None else ([ptr], [nbytes], location)
        ret = self.engine.batch_register_memory(*args)
        if ret != 0:
            raise RuntimeError(f"Mooncake register failed: {ret}")

    def rails(self) -> int:
        try:
            import json

            topo = self.engine.get_local_topology()
            topo = json.loads(topo) if isinstance(topo, str) else topo
            return len({h for v in topo.values() for lst in v for h in lst})
        except Exception:
            return 0

    def local_meta(self) -> dict:
        return {"session_id": self.session_id}

    def write(self, remote_meta: dict, srcs, dsts, lens) -> None:
        ret = self.engine.batch_transfer_sync_write(remote_meta["session_id"], srcs, dsts, lens)
        if ret != 0:
            raise RuntimeError(f"mooncake write failed: {ret}")


class NixlTransport(Transport):
    name = "nixl"
    _MAX_POLL = 2000000

    def init(self, host: str) -> None:
        from nixl._api import nixl_agent, nixl_agent_config

        self._agent = nixl_agent(f"{host}:{os.getpid()}", nixl_agent_config(backends=["UCX"]))
        self._remotes: dict[bytes, str] = {}
        self._dev = 0
        self._mem_type = "VRAM"

    def register(
        self, ptr: int, nbytes: int, dev_id: int, location: str | None = None, host: bool = False
    ) -> None:
        self._dev = 0 if host else dev_id
        self._mem_type = "DRAM" if host else "VRAM"
        self._agent.register_memory([(ptr, nbytes, self._dev, "")], self._mem_type)

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
            [(int(s), int(n), self._dev) for s, n in zip(srcs, lens)], self._mem_type
        )
        rd = self._agent.get_xfer_descs(
            [(int(d), int(n), rdev) for d, n in zip(dsts, lens)], self._mem_type
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
        raise ValueError(f"unknown transport {name!r}; choices: {sorted(_BACKENDS)}")
    return _BACKENDS[key]()


def alloc_pinned_huge(total: int):
    import ctypes
    import mmap
    import re

    import torch

    MiB = 1 << 20
    HUGE = 2 * MiB
    if total % HUGE:
        total += HUGE - total % HUGE
    mm = mmap.mmap(-1, total + HUGE, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    raw = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    off = (-raw) % HUGE
    addr = raw + off
    libc = ctypes.CDLL(None, use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    if libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(total), ctypes.c_int(14)) != 0:
        raise RuntimeError(f"madvise(MADV_HUGEPAGE) failed errno={ctypes.get_errno()}")
    if libc.madvise(ctypes.c_void_p(addr), ctypes.c_size_t(total), ctypes.c_int(23)) != 0:
        for o in range(off, off + total, HUGE):
            mm[o] = 0
    huge = 0
    for blk in re.split(r"\n(?=[0-9a-f]+-[0-9a-f]+ )", open("/proc/self/smaps").read()):
        m = re.match(r"([0-9a-f]+)-([0-9a-f]+) ", blk)
        if not m:
            continue
        lo, hi = int(m.group(1), 16), int(m.group(2), 16)
        if hi <= addr or lo >= addr + total:
            continue
        h = re.search(r"AnonHugePages:\s+(\d+) kB", blk)
        huge += int(h.group(1)) * 1024 if h else 0
    if huge < total:
        raise RuntimeError(
            f"PD host buffer is only {huge / 2**30:.2f} of {total / 2**30:.2f} GiB huge-page backed; "
            "the RDMA MR would fall back to 4 KiB pages and exceed the per-HCA entry budget "
            "(check /sys/kernel/mm/transparent_hugepage/{enabled,defrag} and free memory)"
        )
    rc = torch.cuda.cudart().cudaHostRegister(addr, total, 0)
    if int(rc) != 0:
        raise RuntimeError(f"cudaHostRegister/hipHostRegister failed: {rc}")
    buf = torch.frombuffer(mm, dtype=torch.uint8, count=total, offset=off)
    buf._pd_mmap = mm
    return buf
