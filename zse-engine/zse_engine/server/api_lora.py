"""ZSE Server LoRA API — LoRA adapter management endpoints.

Endpoints:
    GET  /v1/lora/list     — List loaded adapters
    POST /v1/lora/load     — Load adapter from file
    POST /v1/lora/unload   — Unload adapter
    GET  /v1/lora/stats    — LoRA memory usage
"""

from zse_engine.server.router import Request, Response
from zse_engine.server.auth import AuthManager


class LoRAAPI:
    """LoRA adapter management endpoint handlers.

    Args:
        auth: AuthManager instance
        engine: ZStreamerEngine instance
    """

    def __init__(self, auth: AuthManager, engine=None):
        self._auth = auth
        self._engine = engine

    def register(self, router):
        """Register LoRA routes."""
        router.get("/v1/lora/list", self.handle_list)
        router.post("/v1/lora/load", self.handle_load)
        router.post("/v1/lora/unload", self.handle_unload)
        router.get("/v1/lora/stats", self.handle_stats)

    def _get_lora_manager(self):
        """Get LoRA manager from engine."""
        if self._engine and hasattr(self._engine, 'lora_manager'):
            return self._engine.lora_manager
        return None

    async def handle_list(self, request: Request) -> Response:
        """List loaded LoRA adapters."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        manager = self._get_lora_manager()
        if manager is None:
            return Response.json({"adapters": [], "message": "LoRA not available"})

        adapters = []
        for adapter_id in manager.list_adapters():
            adapter = manager.get_adapter(adapter_id)
            if adapter:
                adapters.append({
                    "id": adapter.adapter_id,
                    "rank": adapter.rank,
                    "alpha": adapter.alpha,
                    "num_layers": adapter.num_layers,
                    "target_modules": adapter.target_modules,
                    "total_gpu_bytes": adapter.total_gpu_bytes,
                })

        return Response.json({"adapters": adapters})

    async def handle_load(self, request: Request) -> Response:
        """Load a LoRA adapter from file."""
        # Admin only
        if not self._auth.is_admin(request.authorization):
            return Response.error("Admin access required", 403)

        body = request.json
        adapter_id = body.get("adapter_id")
        path = body.get("path")

        if not adapter_id:
            return Response.error("'adapter_id' field required")
        if not path:
            return Response.error("'path' field required")

        manager = self._get_lora_manager()
        if manager is None:
            return Response.error("LoRA not available", 503)

        if manager.has_adapter(adapter_id):
            return Response.error(f"Adapter '{adapter_id}' already loaded", 409)

        try:
            # Get weight shapes from model config
            config = self._engine._config if self._engine else None
            if config is None:
                return Response.error("Model config not available", 503)

            weight_shapes = {
                "q_proj": (config.hidden_size, config.hidden_size),
                "k_proj": (config.hidden_size, config.num_kv_heads * config.head_dim),
                "v_proj": (config.hidden_size, config.num_kv_heads * config.head_dim),
                "o_proj": (config.hidden_size, config.hidden_size),
            }
            if config.intermediate_size > 0:
                weight_shapes.update({
                    "gate_proj": (config.hidden_size, config.intermediate_size),
                    "up_proj": (config.hidden_size, config.intermediate_size),
                    "down_proj": (config.intermediate_size, config.hidden_size),
                })

            adapter = manager.load_adapter_from_file(adapter_id, path, weight_shapes)
            return Response.json({
                "status": "loaded",
                "adapter_id": adapter.adapter_id,
                "rank": adapter.rank,
                "total_gpu_bytes": adapter.total_gpu_bytes,
            }, status=201)
        except Exception as e:
            return Response.error(f"Failed to load adapter: {str(e)}", 500)

    async def handle_unload(self, request: Request) -> Response:
        """Unload a LoRA adapter."""
        if not self._auth.is_admin(request.authorization):
            return Response.error("Admin access required", 403)

        body = request.json
        adapter_id = body.get("adapter_id")
        if not adapter_id:
            return Response.error("'adapter_id' field required")

        manager = self._get_lora_manager()
        if manager is None:
            return Response.error("LoRA not available", 503)

        success = manager.unload_adapter(adapter_id)
        if success:
            return Response.json({"status": "unloaded", "adapter_id": adapter_id})
        return Response.error(f"Adapter '{adapter_id}' not found", 404)

    async def handle_stats(self, request: Request) -> Response:
        """LoRA memory usage statistics."""
        auth = self._auth.authenticate(request.authorization)
        if not auth.authenticated:
            return Response.error(auth.error, auth.status_code)

        manager = self._get_lora_manager()
        if manager is None:
            return Response.json({"status": "unavailable"})

        return Response.json(manager.stats())
