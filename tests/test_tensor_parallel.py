"""Tests for tensor parallelism — weight splitting, dimension validation, TP group."""

import struct
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "zse-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "zse-compiler"))

import pytest


def test_tp_transport_round_trip_and_size_limit():
    import socket
    from zse_engine.orchestrator.tp_transport import MessageChannel, MAX_MESSAGE_BYTES

    sender_socket, receiver_socket = socket.socketpair()
    sender, receiver = MessageChannel(sender_socket), MessageChannel(receiver_socket)
    try:
        sender.put(("logits", bytes(range(256))))
        assert receiver.get() == ["logits", bytes(range(256))]
        sender_socket.sendall(struct.pack("!I", MAX_MESSAGE_BYTES + 1))
        with pytest.raises(ValueError, match="length"):
            receiver.get()
    finally:
        sender.close()
        receiver.close()


@pytest.mark.parametrize("matching", [True, False])
def test_tp_transport_authentication(matching):
    import socket
    from concurrent.futures import ThreadPoolExecutor
    from zse_engine.orchestrator.tp_transport import MessageChannel, _authenticate

    server_socket, client_socket = socket.socketpair()
    server_socket.settimeout(1)
    client_socket.settimeout(1)
    server, client = MessageChannel(server_socket), MessageChannel(client_socket)
    key = b"a" * 32

    def authenticate_server():
        try:
            _authenticate(server, key, server=True)
        finally:
            server.close()

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(authenticate_server)
            if matching:
                _authenticate(client, key, server=False)
                future.result(timeout=2)
            else:
                with pytest.raises(EOFError):
                    _authenticate(client, b"b" * 32, server=False)
                with pytest.raises(PermissionError):
                    future.result(timeout=2)
    finally:
        client.close()


@pytest.mark.parametrize("worker_error", [False, True])
def test_tp_remote_worker_command_lifecycle(monkeypatch, worker_error):
    import queue
    import socket
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace
    from zse_engine.orchestrator import tp_engine, tp_transport

    observed = []

    class TestQueue(queue.Queue):
        def cancel_join_thread(self):
            pass

        def close(self):
            pass

    class TestProcess:
        def __init__(self, target, args):
            self.thread = threading.Thread(target=target, args=args, daemon=True)
            self.exitcode = 0

        def start(self):
            self.thread.start()

        def join(self, timeout=None):
            self.thread.join(timeout)

        def is_alive(self):
            return self.thread.is_alive()

        def terminate(self):
            raise AssertionError("Healthy remote worker must stop via command")

    def fake_worker(rank, tp_size, path, backend, uid, commands, results, quiet, local_rank):
        observed.append((rank, tp_size, local_rank, uid))
        results.put(("ready", rank, {"local_rank": local_rank}))
        while True:
            command = commands.get(timeout=3)
            observed.append(command)
            if command[0] == tp_engine.CMD_DESTROY:
                return
            if command[0] == tp_engine.CMD_DECODE and worker_error:
                results.put(("error", rank, "injected decode failure"))
            if command[0] == tp_engine.CMD_STOP:
                results.put(("released", rank, command[1], {"num_sequences": 0}))

    monkeypatch.setattr(tp_engine, "_worker_process", fake_worker)
    monkeypatch.setattr(tp_transport.multiprocessing, "get_context", lambda kind: SimpleNamespace(
        Queue=TestQueue, Process=TestProcess,
    ))
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    endpoint = tp_transport.TPRemoteEndpoint("127.0.0.1", port, b"x" * 32)
    results = queue.Queue()
    with ThreadPoolExecutor(max_workers=1) as executor:
        server = executor.submit(tp_transport.serve_tp_worker, endpoint, "model.zse", 1, 2, timeout=5,
                                 node_metadata={"hostname": "follower"})
        remote = tp_transport.RemoteTPWorker(endpoint, 1, 2, "cuda", bytes(128), results, timeout=3)
        try:
            ready = results.get(timeout=3)
            assert ready == ["ready", 1, {"local_rank": 0, "node": {"hostname": "follower"}}]
            remote.put((tp_engine.CMD_PREFILL, [1, 2], 9))
            remote.put((tp_engine.CMD_DECODE, 3, 9, 2, True))
            if worker_error:
                assert results.get(timeout=3) == ["error", 1, "injected decode failure"]
            remote.put((tp_engine.CMD_STOP, 9))
            assert results.get(timeout=3) == ["released", 1, 9, {"num_sequences": 0}]
            remote.put((tp_engine.CMD_DESTROY,))
            remote.join(timeout=3)
            assert not remote.is_alive()
            assert remote.exitcode == int(worker_error)
            if worker_error:
                with pytest.raises(RuntimeError, match="Remote GPU worker exited"):
                    server.result(timeout=3)
            else:
                assert server.result(timeout=3)["workers_stopped"]
            assert observed[0] == (1, 2, 0, bytes(128))
            assert observed[1:] == [[1, [1, 2], 9], [2, 3, 9, 2, True], [3, 9], [4]]
        finally:
            remote.terminate()


def test_tp_cluster_harness_allocation_and_constructor_contract():
    import ast
    import inspect
    from pathlib import Path
    from zse_engine.orchestrator.tp_engine import TPEngine

    tree = ast.parse(Path(__file__).with_name("test_modal_tp.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "test_multi_node")
    allocation, cluster = function.decorator_list
    options = {keyword.arg: keyword.value for keyword in allocation.keywords}
    assert ast.literal_eval(options["gpu"]) == "A100-80GB:8"
    assert ast.literal_eval(options["timeout"]) == 900
    assert ast.literal_eval(options["retries"]) == 0
    assert ast.literal_eval(cluster.keywords[0].value) == 2
    call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "TPEngine")
    inspect.signature(TPEngine).bind("model.zse", **{keyword.arg: None for keyword in call.keywords})


@pytest.mark.parametrize("local_rank,expected_device", [(None, 1), (0, 0)])
def test_tp_worker_separates_global_and_local_rank(monkeypatch, local_rank, expected_device):
    from unittest.mock import Mock
    from zse_engine.orchestrator import tp_engine

    memory = Mock(side_effect=RuntimeError("stop before GPU initialization"))
    monkeypatch.setattr(tp_engine, "GPUMemory", memory)
    monkeypatch.setattr("signal.signal", Mock())
    results = Mock()
    tp_engine._worker_process(
        1, 2, "model.zse", "cuda", bytes(128), Mock(), results, True,
        local_rank=local_rank,
    )
    memory.assert_called_once_with(backend="cuda", device_index=expected_device)
    results.put.assert_called_once_with(("error", 1, "stop before GPU initialization"))


@pytest.mark.parametrize("cross_host", [False, True])
def test_tp_engine_uses_spawn_context(monkeypatch, cross_host):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from zse_engine.orchestrator import tp_engine

    device = SimpleNamespace(name="test GPU", total_memory=8 * 1024**3)
    config = SimpleNamespace(num_heads=4, num_kv_heads=2, intermediate_size=256)
    context = Mock()
    context.Queue.return_value.get.side_effect = [
        ("ready", rank, {"weight_time": 0.0, "kernel_time": 0.0})
        for rank in range(2)
    ]
    get_context = Mock(return_value=context)
    monkeypatch.setattr(tp_engine.multiprocessing, "get_context", get_context)
    monkeypatch.setattr(tp_engine, "detect_backend", lambda: "cuda")
    monkeypatch.setattr(tp_engine, "get_devices", lambda backend: [device, device])
    monkeypatch.setattr(tp_engine, "is_nccl_available", lambda backend: True)
    monkeypatch.setattr(tp_engine, "get_unique_id", lambda backend: bytes(128))
    monkeypatch.setattr(tp_engine, "ZSELoader", lambda path: SimpleNamespace(
        config=config, tokenizer=None
    ))
    endpoints = None
    if cross_host:
        from zse_engine.orchestrator import tp_transport
        endpoints = [tp_transport.TPRemoteEndpoint("127.0.0.1", 23456, b"a" * 32)]
        remote = Mock()
        monkeypatch.setattr(tp_transport, "RemoteTPWorker", remote)
        monkeypatch.setattr(tp_engine, "get_devices", lambda backend: [device])
    engine = tp_engine.TPEngine("model.zse", quiet=True, remote_endpoints=endpoints)
    try:
        get_context.assert_called_once_with("spawn")
        assert context.Queue.call_count == (2 if cross_host else 3)
        assert context.Process.call_count == (1 if cross_host else 2)
        assert context.Process.return_value.start.call_count == (1 if cross_host else 2)
        if cross_host:
            assert remote.call_args.args[:4] == (endpoints[0], 1, 2, "cuda")
            assert engine._cmd_queues[1] is remote.return_value
            engine._broadcast_cmd((tp_engine.CMD_STOP, 12))
            remote.return_value.put.assert_called_once_with((tp_engine.CMD_STOP, 12))
    finally:
        engine.destroy()


# =============================================================================
# NCCL wrapper tests (unit — no GPU needed)
# =============================================================================

def test_tp_prefill_attention_buffers_and_collectives():
    from unittest.mock import Mock
    from zse_engine.orchestrator.model_runner import TPModelRunner

    runner = Mock()
    runner._hidden_size = 16
    runner._num_heads = 2
    runner._num_kv_heads = 1
    runner._head_dim = 4
    runner._num_layers = 1
    runner._intermediate_size = 32
    runner._scale = 0.5
    runner._rope_theta = 10000.0
    runner._kv_cache.block_size = 8
    metadata = Mock(max_blocks_per_seq=1)
    TPModelRunner._transformer_block_prefill(
        runner, 0, 1, 3, Mock(), Mock(), Mock(), metadata,
    )
    attention_calls = [
        call for call in runner._kernels.launch.call_args_list
        if call.args[0] == "prefill_attention"
    ]
    assert len(attention_calls) == 1
    attention = attention_calls[0]
    assert attention.args[3] is runner._scratch.norm_out
    assert attention.args[5] is runner._scratch.attn_out
    assert attention.kwargs == {"shared_mem_bytes": 12}
    projection = runner._launch_matmul.call_args_list[3]
    assert projection.args[1] is runner._scratch.norm_out
    assert runner._tp_group.all_reduce_inplace.call_count == 2
    for collective in runner._tp_group.all_reduce_inplace.call_args_list:
        assert collective.args == (runner._scratch.hidden.data_ptr, 48)
        assert collective.kwargs == {"dtype": "float16"}


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("stop_id", [151643, 151645])
def test_tp_generation_stops_at_tokenizer_eos(streaming, stop_id):
    from unittest.mock import Mock
    from zse_engine.orchestrator.tp_engine import TPEngine

    engine = TPEngine.__new__(TPEngine)
    engine._tp_size = 2
    engine._seq_counter = 0
    engine._total_tokens = 0
    engine._total_gen_time = 0.0
    engine._config = Mock(vocab_size=151936)
    engine._tokenizer = Mock()
    engine._tokenizer.special_tokens.eos_id = 151643
    marker_ids = {"<|im_end|>": [151645], "<|im_start|>": [151644]}
    engine._tokenizer.encode.side_effect = lambda text, **kwargs: marker_ids.get(text, [10, 11])
    engine._tokenizer.decode.side_effect = lambda tokens: "answer" if tokens == [42] else "STOP"
    engine._sampler = Mock()
    engine._sampler.sample.side_effect = [42, stop_id]
    engine._result_queue = Mock()
    engine._result_queue.get.side_effect = [
        ("logits", b""), ("token", stop_id),
        ("released", 0, 0, {}), ("released", 1, 0, {}),
    ]
    engine._broadcast_cmd = Mock()
    engine._workers = []
    engine._cmd_queues = []
    if streaming:
        assert list(engine.stream_generate("question", max_tokens=32)) == ["answer"]
    else:
        assert engine.generate("question", max_tokens=32, temperature=0.0) == "answer"
    from zse_engine.orchestrator.tp_engine import CMD_STOP
    assert engine._broadcast_cmd.call_count == 3
    assert engine._broadcast_cmd.call_args.args == ((CMD_STOP, 0),)


def test_tp_prefill_projects_only_last_row():
    from unittest.mock import Mock
    from zse_engine.orchestrator.model_runner import ModelRunner

    runner = Mock()
    runner._hidden_size = 16
    runner._num_layers = 0
    runner._is_gemma4 = False
    runner._config.vocab_size = 128
    runner._scratch.norm_out.data_ptr = 4096
    ModelRunner.prefill(runner, list(range(129)), 0)
    runner._make_tensor_from_ptr.assert_called_with(4096 + 128 * 16 * 2, (1, 16))
    runner._launch_matmul.assert_called_once_with(
        runner._scratch.logits, runner._make_tensor_from_ptr.return_value,
        runner._weights.get.return_value, 1, 128, 16,
    )
    runner._download_fp16.assert_called_once_with(runner._scratch.logits, 128)


@pytest.mark.parametrize("outcome", ["success", "error", "close"])
def test_tp_request_lifecycle_releases_sequence(outcome):
    from unittest.mock import Mock
    from zse_engine.orchestrator.tp_engine import TPEngine

    engine = TPEngine.__new__(TPEngine)
    engine._seq_counter = 7
    engine._release_sequence = Mock()
    engine._workers = []
    engine._cmd_queues = []

    def request():
        with engine._request_sequence() as seq_id:
            assert seq_id == 7
            yield "token"
            if outcome == "error":
                raise ValueError("request failed")

    stream = request()
    assert next(stream) == "token"
    if outcome == "error":
        with pytest.raises(ValueError, match="request failed"):
            next(stream)
    elif outcome == "close":
        stream.close()
    else:
        assert list(stream) == []
    engine._release_sequence.assert_called_once_with(7)
    assert engine._seq_counter == 8


@pytest.mark.parametrize("message", [
    ("released", 0, 2, {}),
    ("released", 4, 1, {}),
    ("error", 0, "failed"),
    ("released", 0, 1, {}),
])
def test_tp_release_rejects_invalid_acknowledgement(message):
    from unittest.mock import Mock
    from zse_engine.orchestrator.tp_engine import TPEngine

    engine = TPEngine.__new__(TPEngine)
    engine._tp_size = 2
    engine._broadcast_cmd = Mock()
    engine._result_queue = Mock()
    engine._result_queue.get.side_effect = [("released", 0, 1, {}), message]
    engine._workers = []
    engine._cmd_queues = []
    with pytest.raises(RuntimeError, match="Sequence release failed"):
        engine._release_sequence(1)


class TestNcclModule:
    """Test NCCL module imports and constants."""

    def test_binary_unique_id_roundtrip(self, monkeypatch):
        import ctypes
        from unittest.mock import Mock
        from zse_compiler.runtime import nccl

        payload = bytes(range(nccl.NCCL_UNIQUE_ID_BYTES))
        library = Mock()

        def get_id(destination):
            ctypes.memmove(destination, payload, len(payload))
            return 0

        def init_rank(destination, nranks, uid, rank):
            assert bytes(uid) == payload
            assert nranks == 2
            assert rank == 1
            return 0

        library.ncclGetUniqueId.side_effect = get_id
        library.ncclCommInitRank.side_effect = init_rank
        monkeypatch.setattr(nccl, "_load_nccl", lambda backend: library)
        unique_id = nccl.get_unique_id()
        assert unique_id == payload
        communicator = nccl.NcclCommunicator(2, 1, unique_id)
        assert communicator.rank == 1
        library.ncclCommInitRank.assert_called_once()

    @pytest.mark.parametrize("size", [0, 127, 129])
    def test_invalid_unique_id_size(self, monkeypatch, size):
        from unittest.mock import Mock
        from zse_compiler.runtime import nccl

        library = Mock()
        monkeypatch.setattr(nccl, "_load_nccl", lambda backend: library)
        with pytest.raises(ValueError, match="128 bytes"):
            nccl.NcclCommunicator(2, 0, bytes(size))
        library.ncclCommInitRank.assert_not_called()

    def test_import(self):
        from zse_compiler.runtime.nccl import (
            NcclCommunicator, get_unique_id, is_nccl_available,
            NCCL_UNIQUE_ID_BYTES, NCCL_FLOAT16, NCCL_SUM,
        )
        assert NCCL_UNIQUE_ID_BYTES == 128
        assert NCCL_FLOAT16 == 6
        assert NCCL_SUM == 0

    def test_dtype_map(self):
        from zse_compiler.runtime.nccl import _DTYPE_MAP, _OP_MAP
        assert _DTYPE_MAP["float16"] == 6
        assert _DTYPE_MAP["float32"] == 7
        assert _DTYPE_MAP["fp16"] == 6
        assert _OP_MAP["sum"] == 0
        assert _OP_MAP["max"] == 2

    def test_is_nccl_available_returns_bool(self):
        from zse_compiler.runtime.nccl import is_nccl_available
        result = is_nccl_available("cuda")
        assert isinstance(result, bool)
        result_rocm = is_nccl_available("rocm")
        assert isinstance(result_rocm, bool)


# =============================================================================
# TensorParallelGroup tests (no GPU needed)
# =============================================================================

class TestTPConfig:
    """Test TP configuration validation."""

    def test_single_gpu_always_valid(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=1)
        assert not cfg.is_enabled()
        cfg.validate(32, 8, 11008)  # Any dims OK for tp_size=1

    def test_valid_tp2(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=2)
        assert cfg.is_enabled()
        cfg.validate(32, 8, 11008)  # All divisible by 2

    def test_invalid_heads_not_divisible(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=4)
        with pytest.raises(ValueError, match="num_heads"):
            cfg.validate(30, 8, 11008)  # 30 not divisible by 4

    def test_invalid_kv_heads_not_divisible(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=4)
        with pytest.raises(ValueError, match="num_kv_heads"):
            cfg.validate(32, 6, 11008)  # 6 not divisible by 4

    def test_invalid_intermediate_not_divisible(self):
        from zse_engine.orchestrator.tensor_parallel import TPConfig
        cfg = TPConfig(tp_size=4)
        with pytest.raises(ValueError, match="intermediate_size"):
            cfg.validate(32, 8, 11007)  # 11007 not divisible by 4


class TestTensorParallelGroup:
    """Test TP group weight splitting logic (no NCCL needed)."""

    def _make_tp(self, tp_size, rank):
        """Create a TP group without NCCL (for unit testing split logic)."""
        from zse_engine.orchestrator.tensor_parallel import TensorParallelGroup
        tp = TensorParallelGroup.__new__(TensorParallelGroup)
        tp.tp_size = tp_size
        tp.rank = rank
        tp.backend = "cuda"
        tp._stream = 0
        tp._comm = None
        return tp

    def test_split_strategy_qkv(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.self_attn.q_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.5.self_attn.k_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.31.self_attn.v_proj.weight") == COLUMN_PARALLEL

    def test_split_strategy_o_proj(self):
        from zse_engine.orchestrator.tensor_parallel import ROW_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.self_attn.o_proj.weight") == ROW_PARALLEL

    def test_split_strategy_mlp(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL, ROW_PARALLEL
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.mlp.gate_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.0.mlp.up_proj.weight") == COLUMN_PARALLEL
        assert tp.get_split_strategy("model.layers.0.mlp.down_proj.weight") == ROW_PARALLEL

    def test_split_strategy_norms_replicated(self):
        from zse_engine.orchestrator.tensor_parallel import REPLICATED
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("model.layers.0.input_layernorm.weight") == REPLICATED
        assert tp.get_split_strategy("model.norm.weight") == REPLICATED
        assert tp.get_split_strategy("embed_tokens.weight") == REPLICATED

    def test_split_strategy_lm_head(self):
        from zse_engine.orchestrator.tensor_parallel import REPLICATED
        tp = self._make_tp(2, 0)
        assert tp.get_split_strategy("lm_head.weight") == REPLICATED

    def test_shard_range_column_parallel(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp0 = self._make_tp(2, 0)
        tp1 = self._make_tp(2, 1)
        assert tp0.compute_shard_range(4096, COLUMN_PARALLEL) == (0, 2048)
        assert tp1.compute_shard_range(4096, COLUMN_PARALLEL) == (2048, 4096)

    def test_shard_range_row_parallel(self):
        from zse_engine.orchestrator.tensor_parallel import ROW_PARALLEL
        tp0 = self._make_tp(4, 0)
        tp1 = self._make_tp(4, 1)
        tp3 = self._make_tp(4, 3)
        assert tp0.compute_shard_range(4096, ROW_PARALLEL) == (0, 1024)
        assert tp1.compute_shard_range(4096, ROW_PARALLEL) == (1024, 2048)
        assert tp3.compute_shard_range(4096, ROW_PARALLEL) == (3072, 4096)

    def test_shard_range_replicated(self):
        from zse_engine.orchestrator.tensor_parallel import REPLICATED
        tp = self._make_tp(4, 2)
        assert tp.compute_shard_range(4096, REPLICATED) == (0, 4096)

    def test_shard_size(self):
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL, REPLICATED
        tp = self._make_tp(4, 0)
        assert tp.shard_size(4096, COLUMN_PARALLEL) == 1024
        assert tp.shard_size(4096, REPLICATED) == 4096

    def test_tp1_noop(self):
        """TP size 1 should return full ranges."""
        from zse_engine.orchestrator.tensor_parallel import COLUMN_PARALLEL
        tp = self._make_tp(1, 0)
        assert tp.compute_shard_range(4096, COLUMN_PARALLEL) == (0, 4096)
        assert tp.shard_size(4096, COLUMN_PARALLEL) == 4096


# =============================================================================
# TPWeightLoader shard computation tests
# =============================================================================

class TestTPWeightLoaderSharding:
    """Test weight shard dimension calculations."""

    def _make_entry(self, name, shape, dtype="int4", data_nbytes=0,
                    scale_nbytes=0, zeros_nbytes=0, group_size=128):
        """Create a mock WeightEntry."""
        from zse_engine.format.weight_index import WeightEntry
        N = shape[0]
        K = shape[1] if len(shape) > 1 else 0

        if data_nbytes == 0:
            if dtype == "int4":
                data_nbytes = N * (K // 2) if K > 0 else N
            elif dtype == "int8":
                data_nbytes = N * K if K > 0 else N
            else:
                num_el = 1
                for s in shape:
                    num_el *= s
                data_nbytes = num_el * 2  # fp16

        if scale_nbytes == 0 and dtype in ("int4", "int8") and K > 0:
            num_groups = K // group_size
            scale_nbytes = N * num_groups * 2  # fp16

        if zeros_nbytes == 0 and dtype == "int4" and K > 0:
            num_groups = K // group_size
            zeros_nbytes = N * num_groups * 2

        return WeightEntry(
            name=name,
            shape=shape,
            dtype=dtype,
            group_size=group_size,
            data_nbytes=data_nbytes,
            scale_nbytes=scale_nbytes,
            zeros_nbytes=zeros_nbytes,
        )

    def _make_tp_loader(self, tp_size, rank):
        """Create TPWeightLoader with mock TP group."""
        from zse_engine.orchestrator.tp_weight_loader import TPWeightLoader
        from zse_engine.orchestrator.tensor_parallel import TensorParallelGroup

        tp = TensorParallelGroup.__new__(TensorParallelGroup)
        tp.tp_size = tp_size
        tp.rank = rank
        tp.backend = "cuda"
        tp._stream = 0
        tp._comm = None

        loader = TPWeightLoader.__new__(TPWeightLoader)
        loader._tp = tp
        loader._loader = None
        loader._gpu_mem = None
        return loader

    def test_column_parallel_int4_shard_dims(self):
        """Q projection: [4096, 4096] INT4 split column-wise with tp=2."""
        loader = self._make_tp_loader(2, 0)
        entry = self._make_entry("model.layers.0.self_attn.q_proj.weight",
                                  (4096, 4096), "int4")
        shard = loader._compute_shard_info(entry, "column")
        assert shard["shape"] == (2048, 4096)
        assert shard["num_elements"] == 2048 * 4096
        assert shard["row_start"] == 0
        assert shard["row_end"] == 2048
        # INT4: data = N * K/2
        assert shard["data_nbytes"] == 2048 * 2048  # 2048 * (4096/2)

    def test_column_parallel_rank1(self):
        """Second rank gets second half of rows."""
        loader = self._make_tp_loader(2, 1)
        entry = self._make_entry("model.layers.0.self_attn.q_proj.weight",
                                  (4096, 4096), "int4")
        shard = loader._compute_shard_info(entry, "column")
        assert shard["row_start"] == 2048
        assert shard["row_end"] == 4096

    def test_row_parallel_int4_shard_dims(self):
        """O projection: [4096, 4096] INT4 split row-wise (K dim) with tp=2."""
        loader = self._make_tp_loader(2, 0)
        entry = self._make_entry("model.layers.0.self_attn.o_proj.weight",
                                  (4096, 4096), "int4")
        shard = loader._compute_shard_info(entry, "row")
        assert shard["shape"] == (4096, 2048)
        assert shard["col_start"] == 0
        assert shard["col_end"] == 2048
        # INT4: data = N * shard_K/2
        assert shard["data_nbytes"] == 4096 * 1024  # 4096 * (2048/2)

    def test_replicated_full_copy(self):
        """Norm weight should be full copy."""
        loader = self._make_tp_loader(4, 2)
        entry = self._make_entry("model.layers.0.input_layernorm.weight",
                                  (4096,), "float16",
                                  data_nbytes=4096*2, scale_nbytes=0, zeros_nbytes=0)
        shard = loader._compute_shard_info(entry, "replicated")
        assert shard["shape"] == (4096,)
        assert shard["data_nbytes"] == 4096 * 2

    def test_tp4_column_parallel(self):
        """4-way split of gate_proj [11008, 4096]."""
        loader = self._make_tp_loader(4, 2)
        entry = self._make_entry("model.layers.0.mlp.gate_proj.weight",
                                  (11008, 4096), "int4")
        shard = loader._compute_shard_info(entry, "column")
        assert shard["shape"] == (2752, 4096)  # 11008/4
        assert shard["row_start"] == 2752 * 2
        assert shard["row_end"] == 2752 * 3


# =============================================================================
# GPU Memory device_index test
# =============================================================================

class TestGPUMemoryDeviceIndex:
    """Test that GPUMemory accepts device_index."""

    def test_default_device_index(self):
        """GPUMemory should accept device_index=0 without error."""
        from zse_compiler.runtime.memory import GPUMemory
        # Don't actually create — just verify the signature accepts it
        # (Creating requires a GPU)
        import inspect
        sig = inspect.signature(GPUMemory.__init__)
        params = list(sig.parameters.keys())
        assert "device_index" in params

    def test_ensure_context_exists(self):
        """GPUMemory should have ensure_context method."""
        from zse_compiler.runtime.memory import GPUMemory
        assert hasattr(GPUMemory, 'ensure_context')


# =============================================================================
# TPModelRunner import test
# =============================================================================

class TestTPModelRunnerImport:
    """Test that TPModelRunner can be imported."""

    def test_import(self):
        from zse_engine.orchestrator.model_runner import TPModelRunner
        assert TPModelRunner is not None

    def test_is_subclass(self):
        from zse_engine.orchestrator.model_runner import ModelRunner, TPModelRunner
        assert issubclass(TPModelRunner, ModelRunner)


# =============================================================================
# TPEngine import test
# =============================================================================

class TestTPEngineImport:
    """Test TPEngine can be imported."""

    def test_import(self):
        from zse_engine.orchestrator.tp_engine import TPEngine
        assert TPEngine is not None

    def test_cmd_constants(self):
        from zse_engine.orchestrator.tp_engine import CMD_PREFILL, CMD_DECODE, CMD_STOP, CMD_DESTROY
        assert CMD_PREFILL == 1
        assert CMD_DECODE == 2
        assert CMD_STOP == 3
        assert CMD_DESTROY == 4


# =============================================================================
# CLI --tp flag test
# =============================================================================

class TestCLITPFlag:
    """Test that CLI accepts --tp flag."""

    def test_serve_parser_has_tp(self):
        """The serve subcommand should accept --tp."""
        import argparse
        from zse_engine.cli import main
        # Parse args directly
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        # Import cli and check the serve parser accepts --tp
        import zse_engine.cli as cli_mod
        # Re-run the parser setup logic by calling main with --help-like args
        # Simpler: just check the source
        import inspect
        source = inspect.getsource(cli_mod.main)
        assert "--tp" in source or "tensor-parallel" in source or "tp_size" in source


# =============================================================================
# Server tp_size test
# =============================================================================

class TestServerTPSize:
    """Test server accepts tp_size parameter."""

    def test_server_init_signature(self):
        """ZSEServer should accept tp_size parameter."""
        from zse_engine.server.app import ZSEServer
        import inspect
        sig = inspect.signature(ZSEServer.__init__)
        assert "tp_size" in sig.parameters
