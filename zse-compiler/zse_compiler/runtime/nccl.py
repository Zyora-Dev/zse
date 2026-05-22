"""ZSE NCCL/RCCL Wrapper — Pure ctypes GPU collective communication.

Zero dependency — loads libnccl.so.2 (NVIDIA) or librccl.so (AMD) directly.
No torch.distributed, no mpi4py, no horovod.

Supports: AllReduce, AllGather, Broadcast, Barrier.

Usage:
    from zse_compiler.runtime.nccl import NcclCommunicator, get_unique_id

    # Rank 0 generates unique ID and shares with all ranks
    uid = get_unique_id()  # bytes, share via IPC/file/pipe

    # Each rank creates communicator
    comm = NcclCommunicator(nranks=2, rank=0, unique_id=uid, backend="cuda")
    comm.all_reduce(sendbuf, recvbuf, count, dtype="float16", op="sum", stream=stream)
    comm.destroy()
"""

import ctypes
import ctypes.util
import os
from typing import Optional


# NCCL constants
NCCL_UNIQUE_ID_BYTES = 128

# ncclDataType_t
NCCL_FLOAT16 = 6
NCCL_FLOAT32 = 7
NCCL_INT32 = 2
NCCL_INT8 = 0
NCCL_UINT8 = 1

# ncclRedOp_t
NCCL_SUM = 0
NCCL_PROD = 1
NCCL_MAX = 2
NCCL_MIN = 3

# Map string names to NCCL enum values
_DTYPE_MAP = {
    "float16": NCCL_FLOAT16,
    "fp16": NCCL_FLOAT16,
    "float32": NCCL_FLOAT32,
    "fp32": NCCL_FLOAT32,
    "int32": NCCL_INT32,
    "int8": NCCL_INT8,
    "uint8": NCCL_UINT8,
}

_OP_MAP = {
    "sum": NCCL_SUM,
    "prod": NCCL_PROD,
    "max": NCCL_MAX,
    "min": NCCL_MIN,
}

_DTYPE_SIZE = {
    NCCL_FLOAT16: 2,
    NCCL_FLOAT32: 4,
    NCCL_INT32: 4,
    NCCL_INT8: 1,
    NCCL_UINT8: 1,
}


class NcclUniqueId(ctypes.Structure):
    """ncclUniqueId — 128 bytes of opaque data for comm init."""
    _fields_ = [("internal", ctypes.c_char * NCCL_UNIQUE_ID_BYTES)]


def _load_nccl(backend: str = "cuda"):
    """Load NCCL (NVIDIA) or RCCL (AMD) shared library."""
    if backend == "rocm":
        names = ["librccl.so", "librccl.so.1", "librccl.so.2"]
        env_key = "RCCL_PATH"
    else:
        names = ["libnccl.so.2", "libnccl.so", "nccl64_2.dll"]
        env_key = "NCCL_PATH"

    # Check env override
    env_path = os.environ.get(env_key)
    if env_path and os.path.exists(env_path):
        try:
            return ctypes.CDLL(env_path)
        except OSError:
            pass

    # Try standard names
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue

    # Try ctypes.util
    lib_name = "rccl" if backend == "rocm" else "nccl"
    path = ctypes.util.find_library(lib_name)
    if path:
        try:
            return ctypes.CDLL(path)
        except OSError:
            pass

    return None


def is_nccl_available(backend: str = "cuda") -> bool:
    """Check if NCCL/RCCL is available."""
    return _load_nccl(backend) is not None


def get_unique_id(backend: str = "cuda") -> bytes:
    """Generate a unique NCCL communicator ID.

    Call this on rank 0, then broadcast the bytes to all ranks.
    """
    lib = _load_nccl(backend)
    if lib is None:
        raise RuntimeError(
            f"{'RCCL' if backend == 'rocm' else 'NCCL'} not found. "
            f"Install {'ROCm' if backend == 'rocm' else 'CUDA'} toolkit with NCCL support."
        )

    uid = NcclUniqueId()
    status = lib.ncclGetUniqueId(ctypes.byref(uid))
    if status != 0:
        raise RuntimeError(f"ncclGetUniqueId failed with status {status}")
    return bytes(uid.internal)


def comm_init_all(ndev: int, backend: str = "cuda"):
    """Initialize all NCCL communicators at once from a single thread.

    Uses ncclCommInitAll — no socket bootstrap needed. Perfect for
    multi-GPU within a single process (threads for actual operations).

    Args:
        ndev: Number of GPUs
        backend: "cuda" or "rocm"

    Returns:
        List of NcclCommunicator objects (one per rank).
    """
    lib = _load_nccl(backend)
    if lib is None:
        raise RuntimeError(f"{'RCCL' if backend == 'rocm' else 'NCCL'} not found.")

    # ncclCommInitAll(ncclComm_t* comms, int ndev, const int* devlist)
    CommArray = ctypes.c_void_p * ndev
    comms = CommArray()
    DevArray = ctypes.c_int * ndev
    devlist = DevArray(*range(ndev))

    status = lib.ncclCommInitAll(comms, ndev, devlist)
    if status != 0:
        raise RuntimeError(f"ncclCommInitAll failed with status {status}")

    result = []
    for rank in range(ndev):
        c = NcclCommunicator.__new__(NcclCommunicator)
        c._nranks = ndev
        c._rank = rank
        c._backend = backend
        c._stream = ctypes.c_void_p(0)
        c._lib = lib
        c._comm = ctypes.c_void_p(comms[rank])
        result.append(c)
    return result


class NcclCommunicator:
    """Zero-dependency NCCL/RCCL communicator via ctypes.

    Each GPU rank creates one communicator. All ranks must call
    operations in the same order (collective semantics).

    Args:
        nranks: Total number of GPUs in the group
        rank: This GPU's index (0 .. nranks-1)
        unique_id: Bytes from get_unique_id() — must be same on all ranks
        backend: "cuda" or "rocm"
        stream: GPU stream handle for async operations (0 = default stream)
    """

    def __init__(
        self,
        nranks: int,
        rank: int,
        unique_id: bytes,
        backend: str = "cuda",
        stream: int = 0,
    ):
        self._nranks = nranks
        self._rank = rank
        self._backend = backend
        self._stream = ctypes.c_void_p(stream)

        self._lib = _load_nccl(backend)
        if self._lib is None:
            raise RuntimeError(
                f"{'RCCL' if backend == 'rocm' else 'NCCL'} not found."
            )

        # Reconstruct NcclUniqueId from bytes
        uid = NcclUniqueId()
        ctypes.memmove(uid.internal, unique_id, NCCL_UNIQUE_ID_BYTES)

        # ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId id, int rank)
        self._comm = ctypes.c_void_p()
        status = self._lib.ncclCommInitRank(
            ctypes.byref(self._comm), nranks, uid, rank
        )
        if status != 0:
            raise RuntimeError(
                f"ncclCommInitRank failed with status {status} "
                f"(rank={rank}, nranks={nranks})"
            )

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def nranks(self) -> int:
        return self._nranks

    def all_reduce(
        self,
        sendbuf: int,
        recvbuf: int,
        count: int,
        dtype: str = "float16",
        op: str = "sum",
        stream: Optional[int] = None,
    ):
        """In-place or out-of-place all-reduce across all ranks.

        Args:
            sendbuf: GPU pointer to send buffer
            recvbuf: GPU pointer to receive buffer (can be same as sendbuf for in-place)
            count: Number of elements (not bytes)
            dtype: "float16", "float32", "int32"
            op: "sum", "prod", "max", "min"
            stream: GPU stream handle (None = use default from init)
        """
        nccl_dtype = _DTYPE_MAP.get(dtype)
        if nccl_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        nccl_op = _OP_MAP.get(op)
        if nccl_op is None:
            raise ValueError(f"Unsupported op: {op}")

        s = ctypes.c_void_p(stream) if stream is not None else self._stream

        status = self._lib.ncclAllReduce(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_size_t(count),
            ctypes.c_int(nccl_dtype),
            ctypes.c_int(nccl_op),
            self._comm,
            s,
        )
        if status != 0:
            raise RuntimeError(f"ncclAllReduce failed with status {status}")

    def all_reduce_inplace(
        self,
        buf: int,
        count: int,
        dtype: str = "float16",
        op: str = "sum",
        stream: Optional[int] = None,
    ):
        """In-place all-reduce (sendbuf == recvbuf)."""
        self.all_reduce(buf, buf, count, dtype, op, stream)

    def all_gather(
        self,
        sendbuf: int,
        recvbuf: int,
        sendcount: int,
        dtype: str = "float16",
        stream: Optional[int] = None,
    ):
        """All-gather: each rank sends sendcount elements, receives nranks * sendcount.

        recvbuf must be nranks * sendcount * dtype_size bytes.
        """
        nccl_dtype = _DTYPE_MAP.get(dtype)
        if nccl_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        s = ctypes.c_void_p(stream) if stream is not None else self._stream

        status = self._lib.ncclAllGather(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_size_t(sendcount),
            ctypes.c_int(nccl_dtype),
            self._comm,
            s,
        )
        if status != 0:
            raise RuntimeError(f"ncclAllGather failed with status {status}")

    def broadcast(
        self,
        buf: int,
        count: int,
        dtype: str = "float16",
        root: int = 0,
        stream: Optional[int] = None,
    ):
        """Broadcast from root rank to all ranks.

        buf is both send (on root) and receive (on all ranks).
        """
        nccl_dtype = _DTYPE_MAP.get(dtype)
        if nccl_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")

        s = ctypes.c_void_p(stream) if stream is not None else self._stream

        status = self._lib.ncclBroadcast(
            ctypes.c_void_p(buf),
            ctypes.c_void_p(buf),
            ctypes.c_size_t(count),
            ctypes.c_int(nccl_dtype),
            ctypes.c_int(root),
            self._comm,
            s,
        )
        if status != 0:
            raise RuntimeError(f"ncclBroadcast failed with status {status}")

    def reduce_scatter(
        self,
        sendbuf: int,
        recvbuf: int,
        recvcount: int,
        dtype: str = "float16",
        op: str = "sum",
        stream: Optional[int] = None,
    ):
        """Reduce-scatter: reduce and scatter result chunks to each rank.

        sendbuf has nranks * recvcount elements.
        recvbuf has recvcount elements (this rank's chunk of the result).
        """
        nccl_dtype = _DTYPE_MAP.get(dtype)
        if nccl_dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype}")
        nccl_op = _OP_MAP.get(op)
        if nccl_op is None:
            raise ValueError(f"Unsupported op: {op}")

        s = ctypes.c_void_p(stream) if stream is not None else self._stream

        status = self._lib.ncclReduceScatter(
            ctypes.c_void_p(sendbuf),
            ctypes.c_void_p(recvbuf),
            ctypes.c_size_t(recvcount),
            ctypes.c_int(nccl_dtype),
            ctypes.c_int(nccl_op),
            self._comm,
            s,
        )
        if status != 0:
            raise RuntimeError(f"ncclReduceScatter failed with status {status}")

    def barrier(self):
        """Synchronize all ranks using a zero-byte all-reduce.

        Note: This is a GPU-side barrier. The host returns immediately
        unless you also synchronize the stream.
        """
        # Use a 0-element all-reduce as barrier
        self.all_reduce(0, 0, 0, "float32", "sum")

    def stream_synchronize(self):
        """Synchronize the NCCL stream on the host side.

        Call after collective operations to ensure they've completed.
        """
        if self._backend == "cuda":
            driver = ctypes.CDLL("libcuda.so.1")
            driver.cuStreamSynchronize(self._stream)
        elif self._backend == "rocm":
            hip = ctypes.CDLL("libamdhip64.so")
            hip.hipStreamSynchronize(self._stream)

    def destroy(self):
        """Destroy the communicator. Must be called by all ranks."""
        if hasattr(self, '_comm') and self._comm:
            self._lib.ncclCommDestroy(self._comm)
            self._comm = None

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass

    def __repr__(self) -> str:
        backend = "RCCL" if self._backend == "rocm" else "NCCL"
        return f"NcclCommunicator({backend}, rank={self._rank}/{self._nranks})"
