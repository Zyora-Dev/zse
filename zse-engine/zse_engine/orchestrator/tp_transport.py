"""Authenticated TP control transport for trusted private GPU networks.

Messages are bounded JSON, never pickle. Authentication does not encrypt traffic;
use only a private network or an encrypted tunnel. Each endpoint owns one GPU.
"""

import base64
import hashlib
import hmac
import json
import multiprocessing
import queue
import secrets
import socket
import struct
import threading
import time
from dataclasses import dataclass, field


MAX_MESSAGE_BYTES = 16 * 1024 * 1024


def _encode_bytes(value):
    if isinstance(value, bytes):
        return {"__bytes__": base64.b64encode(value).decode("ascii")}
    raise TypeError(f"Unsupported TP message type: {type(value).__name__}")


def _decode_bytes(value):
    if set(value) == {"__bytes__"}:
        return base64.b64decode(value["__bytes__"], validate=True)
    return value


class MessageChannel:
    def __init__(self, connection):
        self.connection = connection
        self._write_lock = threading.Lock()

    def put(self, message):
        payload = json.dumps(message, default=_encode_bytes, allow_nan=False).encode("utf-8")
        if len(payload) > MAX_MESSAGE_BYTES:
            raise ValueError("TP message exceeds size limit")
        with self._write_lock:
            self.connection.sendall(struct.pack("!I", len(payload)) + payload)

    def _read_exact(self, length):
        chunks = bytearray()
        while len(chunks) < length:
            chunk = self.connection.recv(length - len(chunks))
            if not chunk:
                raise EOFError("TP peer disconnected")
            chunks.extend(chunk)
        return bytes(chunks)

    def get(self):
        length = struct.unpack("!I", self._read_exact(4))[0]
        if not 0 < length <= MAX_MESSAGE_BYTES:
            raise ValueError("Invalid TP message length")
        return json.loads(self._read_exact(length), object_hook=_decode_bytes)

    def close(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.connection.close()


def _authenticate(channel, authkey, server):
    if len(authkey) < 32:
        raise ValueError("TP authentication key must contain at least 32 bytes")
    if server:
        challenge = secrets.token_bytes(32)
        channel.put(challenge)
        response = channel.get()
        if (not isinstance(response, list) or len(response) != 2
                or not all(isinstance(part, bytes) and len(part) == 32 for part in response)):
            raise PermissionError("Invalid TP authentication response")
        client_nonce, signature = response
        expected = hmac.digest(authkey, b"client" + challenge + client_nonce, hashlib.sha256)
        if not hmac.compare_digest(signature, expected):
            raise PermissionError("TP peer authentication failed")
        channel.put(hmac.digest(authkey, b"server" + challenge + client_nonce, hashlib.sha256))
    else:
        challenge = channel.get()
        if not isinstance(challenge, bytes) or len(challenge) != 32:
            raise PermissionError("Invalid TP authentication challenge")
        client_nonce = secrets.token_bytes(32)
        channel.put([client_nonce, hmac.digest(authkey, b"client" + challenge + client_nonce, hashlib.sha256)])
        response = channel.get()
        expected = hmac.digest(authkey, b"server" + challenge + client_nonce, hashlib.sha256)
        if not isinstance(response, bytes) or not hmac.compare_digest(response, expected):
            raise PermissionError("TP server authentication failed")


@dataclass(frozen=True)
class TPRemoteEndpoint:
    host: str
    port: int
    authkey: bytes = field(repr=False)

    def __post_init__(self):
        if not self.host or not 0 < self.port < 65536 or len(self.authkey) < 32:
            raise ValueError("TP endpoint requires a host, valid port and 32-byte authentication key")


class RemoteTPWorker:
    def __init__(self, endpoint, rank, tp_size, backend, unique_id, result_queue, timeout=60):
        self.rank = rank
        self.pid = None
        self.exitcode = None
        self._stopped = threading.Event()
        self._closed = threading.Event()
        self._result_queue = result_queue
        deadline = time.monotonic() + timeout
        while True:
            try:
                connection = socket.create_connection((endpoint.host, endpoint.port), timeout=5)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"TP rank {rank} connection timed out")
                self._closed.wait(0.1)
        connection.settimeout(30)
        self.channel = MessageChannel(connection)
        try:
            _authenticate(self.channel, endpoint.authkey, server=False)
            self.channel.put({"rank": rank, "tp_size": tp_size, "backend": backend, "unique_id": unique_id})
        except BaseException:
            self.channel.close()
            raise
        connection.settimeout(360)
        self._reader = threading.Thread(target=self._receive, daemon=True)
        self._reader.start()

    def _receive(self):
        try:
            while True:
                message = self.channel.get()
                if not isinstance(message, list) or not message:
                    raise ValueError("Malformed TP worker message")
                if message[0] == "stopped":
                    if len(message) != 3 or message[1] != self.rank:
                        raise ValueError("Invalid TP shutdown acknowledgement")
                    self.exitcode = message[2]
                    self._stopped.set()
                    break
                if message[0] not in {"ready", "released", "error"} or message[1] != self.rank:
                    raise ValueError("Unexpected remote TP message or rank")
                self._result_queue.put(message)
        except Exception as error:
            if not self._closed.is_set():
                self._result_queue.put(("error", self.rank, str(error)))
        finally:
            self.channel.close()

    def put(self, command):
        self.channel.put(command)

    def join(self, timeout=None):
        self._stopped.wait(timeout)

    def is_alive(self):
        return not self._stopped.is_set()

    def terminate(self):
        self._closed.set()
        self.channel.close()


def serve_tp_worker(endpoint, model_path, rank, tp_size, local_rank=0, timeout=840, node_metadata=None):
    """Serve one authenticated leader, then exit. No reconnect or job retries."""
    from zse_engine.orchestrator.tp_engine import _worker_process, CMD_DESTROY

    family = socket.AF_INET6 if ":" in endpoint.host else socket.AF_INET
    worker = None
    channel = None
    commands = None
    results = None
    stopped = threading.Event()
    failed = threading.Event()
    deadline = time.monotonic() + timeout
    try:
        with socket.socket(family, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((endpoint.host, endpoint.port))
            listener.listen(1)
            listener.settimeout(min(timeout, 120))
            connection, _ = listener.accept()
        connection.settimeout(30)
        channel = MessageChannel(connection)
        _authenticate(channel, endpoint.authkey, server=True)
        config = channel.get()
        if (not isinstance(config, dict) or config.get("rank") != rank
                or config.get("tp_size") != tp_size or rank not in range(1, tp_size)
                or config.get("backend") not in {"cuda", "rocm"}
                or not isinstance(config.get("unique_id"), bytes)
                or len(config["unique_id"]) != 128):
            raise ValueError("Invalid TP rendezvous configuration")
        context = multiprocessing.get_context("spawn")
        commands, results = context.Queue(), context.Queue()
        worker = context.Process(target=_worker_process, args=(
            rank, tp_size, model_path, config["backend"], config["unique_id"],
            commands, results, True, local_rank,
        ))
        worker.start()

        def forward_results():
            try:
                while not stopped.is_set():
                    try:
                        message = results.get(timeout=0.2)
                    except queue.Empty:
                        if not worker.is_alive():
                            channel.put(("error", rank, "Remote GPU worker exited unexpectedly"))
                            channel.close()
                            return
                        continue
                    if message[0] == "ready" and node_metadata is not None:
                        message[2]["node"] = node_metadata
                    channel.put(message)
                    if message[0] == "error":
                        failed.set()
            except Exception:
                channel.close()

        forwarder = threading.Thread(target=forward_results, daemon=True)
        forwarder.start()
        while True:
            connection.settimeout(max(0.1, deadline - time.monotonic()))
            command = channel.get()
            if (not isinstance(command, list) or not command
                    or command[0] not in {1, 2, 3, CMD_DESTROY}):
                raise ValueError("Invalid TP worker command")
            if command[0] == CMD_DESTROY:
                stopped.set()
                forwarder.join(timeout=1)
                commands.put(command)
                worker.join(timeout=10)
                if worker.is_alive():
                    raise TimeoutError("Remote GPU worker did not stop cleanly")
                exitcode = worker.exitcode or int(failed.is_set())
                channel.put(("stopped", rank, exitcode))
                if exitcode != 0:
                    raise RuntimeError(f"Remote GPU worker exited with {worker.exitcode}")
                return {"rank": rank, "local_rank": local_rank, "workers_stopped": True}
            commands.put(command)
    finally:
        stopped.set()
        if worker is not None and worker.is_alive():
            worker.terminate()
            worker.join(timeout=5)
        if channel is not None:
            channel.close()
        for worker_queue in (commands, results):
            if worker_queue is not None:
                worker_queue.cancel_join_thread()
                worker_queue.close()