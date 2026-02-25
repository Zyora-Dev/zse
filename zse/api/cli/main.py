"""
ZSE CLI - Command Line Interface

Main entry point for the ZSE inference engine.
Interactive and powerful CLI for LLM inference.
"""

from __future__ import annotations

import sys
import time
import asyncio
import signal
import os
import warnings
from enum import Enum

# Suppress pynvml deprecation warning
warnings.filterwarnings("ignore", message=".*pynvml package is deprecated.*")
from pathlib import Path
from typing import Annotated, Optional, Dict, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt, Confirm
from rich.text import Text
from rich.align import Align
from rich.live import Live
from rich.markdown import Markdown
from rich import box

from zse.version import __version__

# Initialize Typer app
app = typer.Typer(
    name="zse",
    help="ZSE - Z Server Engine: Ultra memory-efficient LLM inference",
    add_completion=True,
    no_args_is_help=False,
    rich_markup_mode="rich",
)

# Rich console for pretty output
console = Console()


# ASCII Art Logo
ZSE_LOGO = """
[bold blue]
 ███████╗███████╗███████╗
 ╚══███╔╝██╔════╝██╔════╝
   ███╔╝ ███████╗█████╗  
  ███╔╝  ╚════██║██╔══╝  
 ███████╗███████║███████╗
 ╚══════╝╚══════╝╚══════╝
[/bold blue]
"""

ZSE_TAGLINE = "[bold white]Z Server Engine[/bold white] - [dim]Ultra Memory-Efficient LLM Inference[/dim]"


class EfficiencyMode(str, Enum):
    """Memory efficiency modes."""
    speed = "speed"
    balanced = "balanced"
    memory = "memory"
    ultra = "ultra"


class OutputFormat(str, Enum):
    """Output format options."""
    text = "text"
    json = "json"


def show_banner() -> None:
    """Display the ZSE banner."""
    console.print(ZSE_LOGO)
    console.print(Align.center(ZSE_TAGLINE))
    console.print(Align.center(f"[dim]Version {__version__}[/dim]\n"))


def show_memory_targets() -> None:
    """Display memory efficiency targets."""
    table = Table(
        title="[bold cyan]Memory Efficiency Targets[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Model Size", style="cyan", justify="center")
    table.add_column("Standard FP16", style="red", justify="center")
    table.add_column("ZSE Target", style="green", justify="center")
    table.add_column("Savings", style="yellow", justify="center")
    
    table.add_row("7B", "14+ GB", "3 - 3.5 GB", "~75%")
    table.add_row("14B", "28+ GB", "6 GB", "~78%")
    table.add_row("32B", "64+ GB", "16 - 20 GB", "~70%")
    table.add_row("70B", "140+ GB", "24 - 32 GB", "~77%")
    
    console.print(Align.center(table))


def show_quick_commands() -> None:
    """Display quick command reference."""
    table = Table(
        title="[bold cyan]Quick Commands[/bold cyan]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Command", style="green")
    table.add_column("Description", style="white")
    
    table.add_row("zse serve <model>", "Start inference server")
    table.add_row("zse chat <model>", "Interactive chat session")
    table.add_row("zse convert <model> -o <file>", "Convert to .zse format")
    table.add_row("zse info <model>", "Show model information")
    table.add_row("zse hardware", "Display hardware info")
    table.add_row("zse benchmark <model>", "Run benchmarks")
    
    console.print(Align.center(table))


def show_features() -> None:
    """Display ZSE features."""
    features = [
        ("🧠 zAttention", "Custom CUDA kernels for paged, flash, sparse attention"),
        ("🗜️ zQuantize", "Per-tensor INT2-8 mixed precision quantization"),
        ("💾 zKV", "Quantized KV cache with 4x memory savings"),
        ("🌊 zStream", "Layer streaming - run 70B on 24GB GPU"),
        ("🎯 zOrchestrator", "Smart recommendations based on FREE memory"),
        ("⚡ Efficiency", "speed | balanced | memory | ultra modes"),
    ]
    
    table = Table(
        title="[bold cyan]Key Features[/bold cyan]",
        box=box.ROUNDED,
        show_header=False,
    )
    table.add_column("Feature", style="green", width=15)
    table.add_column("Description", style="white")
    
    for name, desc in features:
        table.add_row(name, desc)
    
    console.print(Align.center(table))


def _get_hardware_info() -> Dict[str, Any]:
    """Get hardware info from orchestrator or pynvml."""
    try:
        from zse.engine.orchestrator.core import IntelligenceOrchestrator
        return IntelligenceOrchestrator.get_gpu_info()
    except Exception:
        pass
    
    # Fallback to pynvml
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_count = pynvml.nvmlDeviceGetCount()
        gpus = []
        total_vram = 0
        
        for i in range(gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024**3)
            total_gb = mem_info.total / (1024**3)
            gpus.append({
                "id": i,
                "name": name,
                "total_memory_gb": round(total_gb, 2),
                "free_memory_gb": round(free_gb, 2),
            })
            total_vram += total_gb
        
        pynvml.nvmlShutdown()
        return {"available": True, "count": gpu_count, "total_vram_gb": total_vram, "gpus": gpus}
    except Exception:
        return {"available": False, "count": 0, "gpus": []}


def _efficiency_to_quantization(efficiency: EfficiencyMode) -> str:
    """Map efficiency mode to quantization type."""
    mapping = {
        EfficiencyMode.speed: "fp16",
        EfficiencyMode.balanced: "int8",
        EfficiencyMode.memory: "int4",
        EfficiencyMode.ultra: "int4",
    }
    return mapping.get(efficiency, "auto")


def _parse_memory_str(memory_str: Optional[str]) -> Optional[float]:
    """Parse memory string like '16GB' to float."""
    if memory_str is None:
        return None
    memory_str = memory_str.strip().upper()
    if memory_str.endswith("GB"):
        return float(memory_str[:-2])
    elif memory_str.endswith("G"):
        return float(memory_str[:-1])
    elif memory_str.endswith("MB"):
        return float(memory_str[:-2]) / 1024
    elif memory_str.endswith("M"):
        return float(memory_str[:-1]) / 1024
    return float(memory_str)


def interactive_mode() -> None:
    """Run interactive mode when no command is given."""
    show_banner()
    show_features()
    console.print()
    show_memory_targets()
    console.print()
    show_quick_commands()
    console.print()
    
    _show_hardware_summary()
    
    console.print()
    console.print(Panel.fit(
        "[bold green]Ready to serve LLMs with ultra memory efficiency![/bold green]\n\n"
        "Get started:\n"
        "  [cyan]zse serve meta-llama/Llama-3-8B[/cyan]\n"
        "  [cyan]zse serve model.gguf --max-memory 8GB[/cyan]\n"
        "  [cyan]zse chat mistral-7b --efficiency ultra[/cyan]\n\n"
        "[dim]Run 'zse --help' for all commands[/dim]",
        title="[bold blue]🚀 Get Started[/bold blue]",
        border_style="blue",
    ))
    
    console.print()
    if Confirm.ask("[bold cyan]Would you like to start an interactive session?[/bold cyan]", default=False):
        _interactive_session()


def _interactive_session() -> None:
    """Run an interactive command session."""
    console.print("\n[bold green]Interactive Mode[/bold green]")
    console.print("[dim]Commands: help, load <model>, chat <model>, info <model>, hardware, exit[/dim]")
    console.print("[dim]To chat: first 'load <model>', then type messages directly[/dim]\n")
    
    # State for loaded model
    loaded_orchestrator = None
    loaded_model_name = None
    
    while True:
        try:
            command = Prompt.ask("[bold blue]zse[/bold blue]")
            
            if not command.strip():
                continue
            
            cmd_lower = command.lower().strip()
            
            if cmd_lower in ("exit", "quit", "q"):
                if loaded_orchestrator:
                    loaded_orchestrator.unload()
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            
            if cmd_lower == "help":
                _show_interactive_help()
                continue
            
            if cmd_lower == "hardware":
                _show_hardware_info()
                continue
            
            if cmd_lower == "clear":
                console.clear()
                show_banner()
                continue
            
            if cmd_lower.startswith("load "):
                model = command[5:].strip()
                console.print(f"[cyan]Loading {model}...[/cyan]")
                try:
                    from zse.engine.orchestrator.core import IntelligenceOrchestrator
                    if loaded_orchestrator:
                        loaded_orchestrator.unload()
                    loaded_orchestrator = IntelligenceOrchestrator.auto(model).load()
                    loaded_model_name = model
                    console.print(f"[green]✅ Model loaded: {model}[/green]")
                except Exception as e:
                    console.print(f"[red]❌ Failed to load: {e}[/red]")
                continue
            
            if cmd_lower == "unload":
                if loaded_orchestrator:
                    loaded_orchestrator.unload()
                    loaded_orchestrator = None
                    loaded_model_name = None
                    console.print("[green]Model unloaded[/green]")
                else:
                    console.print("[yellow]No model loaded[/yellow]")
                continue
            
            if cmd_lower.startswith("serve "):
                model = command[6:].strip()
                _run_server(model, "127.0.0.1", 8000, EfficiencyMode.balanced, None)
                continue
            
            if cmd_lower.startswith("chat "):
                model = command[5:].strip()
                _run_chat_session(model, EfficiencyMode.balanced, None, None)
                continue
            
            if cmd_lower.startswith("info "):
                model = command[5:].strip()
                _show_model_info_real(model)
                continue
            
            # If model is loaded, treat input as chat prompt
            if loaded_orchestrator and not cmd_lower.startswith(("serve", "chat", "info", "benchmark")):
                console.print()
                try:
                    for chunk in loaded_orchestrator.generate(command, max_tokens=512, stream=True):
                        console.print(chunk, end="")
                    console.print("\n")
                except Exception as e:
                    console.print(f"[red]Generation error: {e}[/red]")
                continue
            
            # No model loaded - check if it looks like chat input
            if not loaded_orchestrator and not cmd_lower.startswith(("serve", "chat", "info", "benchmark", "load")):
                console.print(f"[yellow]No model loaded.[/yellow] To chat, first load a model:")
                console.print("  [cyan]load Qwen/Qwen2.5-0.5B-Instruct[/cyan]  [dim](small, fast)[/dim]")
                console.print("  [cyan]load Qwen/Qwen2.5-7B-Instruct[/cyan]   [dim](requires GPU)[/dim]")
                console.print("[dim]Or use 'chat <model>' for dedicated session[/dim]")
                continue
            
            console.print(f"[red]Unknown command: {command}[/red]")
            console.print("[dim]Type 'help' for available commands[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Use 'exit' to quit[/yellow]")
        except EOFError:
            if loaded_orchestrator:
                loaded_orchestrator.unload()
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break


def _show_interactive_help() -> None:
    """Show help for interactive mode."""
    table = Table(box=box.SIMPLE)
    table.add_column("Command", style="cyan")
    table.add_column("Description", style="white")
    
    table.add_row("load <model>", "Load a model for chat")
    table.add_row("unload", "Unload current model")
    table.add_row("serve <model>", "Start inference server")
    table.add_row("chat <model>", "Start dedicated chat session")
    table.add_row("info <model>", "Show model info")
    table.add_row("hardware", "Show hardware info")
    table.add_row("clear", "Clear screen")
    table.add_row("help", "Show this help")
    table.add_row("exit", "Exit interactive mode")
    table.add_row("[dim]<text>[/dim]", "[dim]Generate if model loaded[/dim]")
    
    console.print(table)


def _show_model_info_real(model: str) -> None:
    """Show real model information."""
    console.print(f"\n[bold]Model:[/bold] [green]{model}[/green]")
    
    try:
        from zse.engine.orchestrator.core import estimate_requirements
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
            task = progress.add_task("[cyan]Analyzing model...", total=None)
            req = estimate_requirements(model)
            progress.update(task, description="[green]Analysis complete")
        
        table = Table(box=box.ROUNDED, show_header=False)
        table.add_column("Property", style="cyan", width=25)
        table.add_column("Value", style="white")
        
        table.add_row("Model", model)
        table.add_row("Estimated Parameters", f"{req['estimated_params_b']:.1f}B")
        table.add_row("", "")
        table.add_row("[bold]Memory Requirements:[/bold]", "")
        table.add_row("  FP16 (fastest)", f"{req['requirements']['fp16']['vram_gb']:.1f} GB")
        table.add_row("  INT8 (balanced)", f"{req['requirements']['int8']['vram_gb']:.1f} GB")
        table.add_row("  INT4 (minimum)", f"{req['requirements']['int4']['vram_gb']:.1f} GB")
        table.add_row("", "")
        table.add_row("[bold]Recommendations:[/bold]", "")
        table.add_row("  4GB GPU", req['recommendations']['4gb_gpu'].upper())
        table.add_row("  8GB GPU", req['recommendations']['8gb_gpu'].upper())
        table.add_row("  16GB GPU", req['recommendations']['16gb_gpu'].upper())
        table.add_row("  24GB GPU", req['recommendations']['24gb_gpu'].upper())
        
        console.print(table)
        
        # Try to get HuggingFace config for more details
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            
            detail_table = Table(title="[bold cyan]Model Architecture[/bold cyan]", box=box.ROUNDED)
            detail_table.add_column("Property", style="cyan")
            detail_table.add_column("Value", style="white")
            
            if hasattr(config, 'architectures'):
                detail_table.add_row("Architecture", ", ".join(config.architectures))
            if hasattr(config, 'hidden_size'):
                detail_table.add_row("Hidden Size", str(config.hidden_size))
            if hasattr(config, 'num_hidden_layers'):
                detail_table.add_row("Layers", str(config.num_hidden_layers))
            if hasattr(config, 'num_attention_heads'):
                detail_table.add_row("Attention Heads", str(config.num_attention_heads))
            if hasattr(config, 'intermediate_size'):
                detail_table.add_row("FFN Size", str(config.intermediate_size))
            if hasattr(config, 'vocab_size'):
                detail_table.add_row("Vocab Size", f"{config.vocab_size:,}")
            if hasattr(config, 'max_position_embeddings'):
                detail_table.add_row("Max Seq Length", f"{config.max_position_embeddings:,}")
            
            console.print()
            console.print(detail_table)
        except Exception:
            pass  # HF config not available
        
    except Exception as e:
        console.print(f"[red]Error analyzing model: {e}[/red]")


def _show_hardware_summary() -> None:
    """Show a brief hardware summary."""
    import psutil
    
    ram = psutil.virtual_memory()
    ram_free = ram.available / (1024**3)
    ram_total = ram.total / (1024**3)
    
    gpu_info = _get_hardware_info()
    
    if gpu_info["available"] and gpu_info["gpus"]:
        gpu = gpu_info["gpus"][0]
        gpu_str = f"{gpu['name']} ({gpu['free_memory_gb']:.1f}GB free / {gpu['total_memory_gb']:.1f}GB)"
        if gpu_info["count"] > 1:
            gpu_str += f" + {gpu_info['count'] - 1} more"
    else:
        gpu_str = "[dim]No GPU detected[/dim]"
    
    panel = Panel(
        f"[bold]RAM:[/bold] {ram_free:.1f}GB free / {ram_total:.1f}GB total\n"
        f"[bold]GPU:[/bold] {gpu_str}",
        title="[bold cyan]💻 System[/bold cyan]",
        border_style="cyan",
        width=60,
    )
    console.print(Align.center(panel))


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        show_banner()
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: Annotated[
        Optional[bool],
        typer.Option(
            "--version", "-v",
            help="Show version and exit.",
            callback=version_callback,
            is_eager=True,
        ),
    ] = None,
) -> None:
    """
    ZSE - Z Server Engine
    
    Ultra memory-efficient LLM inference engine.
    
    Memory Targets:
    - 7B model in 3-3.5GB VRAM
    - 32B model in 16-20GB VRAM
    - 70B model in 24-32GB VRAM
    """
    if ctx.invoked_subcommand is None:
        interactive_mode()


def _run_server(
    model: Optional[str],
    host: str,
    port: int,
    efficiency: EfficiencyMode,
    max_memory: Optional[str],
    device: str = "auto",
) -> None:
    """Actually run the server."""
    try:
        import uvicorn
        from zse.api.server.app import create_app
        from zse.api.server.state import server_state
        
        console.print("[cyan]Initializing ZSE engine...[/cyan]")
        
        # Pre-load model if specified
        if model:
            from zse.engine.orchestrator.core import IntelligenceOrchestrator
            quant = _efficiency_to_quantization(efficiency)
            target_vram = _parse_memory_str(max_memory)
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("[cyan]Loading model...", total=None)
                
                try:
                    if target_vram:
                        orchestrator = IntelligenceOrchestrator.for_vram(target_vram, model, device=device)
                    elif quant != "auto":
                        orchestrator = IntelligenceOrchestrator(model, quantization=quant, device=device)
                    else:
                        orchestrator = IntelligenceOrchestrator.auto(model, device=device)
                    
                    orchestrator.load(verbose=False)
                    progress.update(task, description="[green]Model loaded!")
                    
                    # Register with server state
                    import psutil
                    process = psutil.Process()
                    memory_used = process.memory_info().rss / (1024**3)  # RSS in GB
                    model_id = server_state.generate_model_id(model)
                    server_state.add_model(
                        model_id=model_id,
                        model_name=model,
                        quantization=orchestrator.quantization,
                        vram_used_gb=memory_used,
                        orchestrator=orchestrator
                    )
                    
                except Exception as e:
                    progress.update(task, description=f"[yellow]Model load deferred: {e}")
        
        # Create app
        app = create_app()
        
        # Check auth status
        from zse.api.server.auth import get_key_manager
        key_manager = get_key_manager()
        auth_status = "[green]Enabled[/green]" if key_manager.is_enabled() else "[yellow]Disabled[/yellow]"
        
        model_info = f"[cyan]{model}[/cyan]" if model else "[dim]None (load via dashboard)[/dim]"
        
        console.print(Panel.fit(
            f"[bold green]✅ ZSE Server Ready![/bold green]\n\n"
            f"[bold]Model:[/bold] {model_info}\n"
            f"[bold]Base URL:[/bold] [blue]http://{host}:{port}[/blue]\n"
            f"[bold]API Docs:[/bold] [blue]http://{host}:{port}/docs[/blue]\n"
            f"[bold]Dashboard:[/bold] [blue]http://{host}:{port}/dashboard[/blue]\n"
            f"[bold]Auth:[/bold] {auth_status}\n\n"
            f"[dim]OpenAI-compatible endpoints:[/dim]\n"
            f"  POST /v1/chat/completions\n"
            f"  POST /v1/completions\n"
            f"  GET  /v1/models\n\n"
            f"[dim]Press Ctrl+C to stop[/dim]",
            title="[bold blue]🚀 ZSE Server[/bold blue]",
            border_style="green",
        ))
        
        # Run server
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except ImportError as e:
        console.print(f"[red]Missing dependency: {e}[/red]")
        console.print("[yellow]Install with: pip install uvicorn[/yellow]")
    except Exception as e:
        console.print(f"[red]Server error: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def serve(
    model: Annotated[
        Optional[str],
        typer.Argument(
            help="Model to serve (HuggingFace ID, local path, or .zse file). Optional - can load via dashboard.",
        ),
    ] = None,
    host: Annotated[
        str,
        typer.Option("--host", "-h", help="Host to bind to"),
    ] = "127.0.0.1",
    port: Annotated[
        int,
        typer.Option("--port", "-p", help="Port to bind to"),
    ] = 8000,
    efficiency: Annotated[
        EfficiencyMode,
        typer.Option("--efficiency", "-e", help="Efficiency mode"),
    ] = EfficiencyMode.balanced,
    max_memory: Annotated[
        Optional[str],
        typer.Option("--max-memory", "-m", help="Maximum memory to use (e.g., '16GB', '24GB')"),
    ] = None,
    quantization: Annotated[
        Optional[str],
        typer.Option("--quantization", "-q", help="Quantization method (fp16, int8, int4)"),
    ] = None,
    tensor_parallel: Annotated[
        int,
        typer.Option("--tensor-parallel", "-tp", help="Number of GPUs for tensor parallelism"),
    ] = 1,
    cpu_offload: Annotated[
        bool,
        typer.Option("--cpu-offload", help="Enable CPU offloading for large models"),
    ] = False,
    device: Annotated[
        str,
        typer.Option("--device", "-d", help="Device to use: 'auto' (detect), 'cuda', 'cpu', or 'cuda:N'"),
    ] = "auto",
    recommend: Annotated[
        bool,
        typer.Option("--recommend", "-r", help="Show recommendations before loading"),
    ] = False,
    mode: Annotated[
        str,
        typer.Option("--mode", help="Deployment mode (dev, enterprise)"),
    ] = "dev",
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
) -> None:
    """
    Start the ZSE inference server.
    
    Examples:
        zse serve                              # Start server, load model via dashboard
        zse serve meta-llama/Llama-3-8B        # Start with model pre-loaded
        zse serve ./model.zse --max-memory 16GB
        zse serve mistral-7b --efficiency ultra --recommend
        zse serve qwen-0.5b --device cpu   # Run on CPU only
    """
    show_banner()
    
    if model:
        console.print(Panel.fit(
            f"[bold]Model:[/bold] [green]{model}[/green]\n"
            f"[bold]Efficiency:[/bold] [yellow]{efficiency.value}[/yellow]\n"
            f"[bold]Address:[/bold] [blue]http://{host}:{port}[/blue]",
            title="[bold blue]🚀 Starting ZSE Server[/bold blue]",
            border_style="blue",
        ))
        
        if recommend:
            _show_recommendations(model, max_memory, efficiency)
            if not Confirm.ask("\n[cyan]Proceed with these settings?[/cyan]", default=True):
                raise typer.Exit()
    else:
        console.print(Panel.fit(
            f"[bold]Mode:[/bold] [yellow]No model pre-loaded[/yellow]\n"
            f"[bold]Address:[/bold] [blue]http://{host}:{port}[/blue]\n"
            f"[bold]Dashboard:[/bold] [blue]http://{host}:{port}/dashboard[/blue]\n\n"
            f"[dim]Load models via dashboard or API[/dim]",
            title="[bold blue]🚀 Starting ZSE Server[/bold blue]",
            border_style="blue",
        ))
    
    _run_server(model, host, port, efficiency, max_memory, device)


def _run_chat_session(
    model: str,
    efficiency: EfficiencyMode,
    max_memory: Optional[str],
    system_prompt: Optional[str],
) -> None:
    """Run an actual chat session with the model."""
    from zse.engine.orchestrator.core import IntelligenceOrchestrator
    
    quant = _efficiency_to_quantization(efficiency)
    target_vram = _parse_memory_str(max_memory)
    
    # Load model
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading model...", total=100)
        
        try:
            progress.update(task, advance=20, description="[cyan]Initializing orchestrator...")
            
            if target_vram:
                orchestrator = IntelligenceOrchestrator.for_vram(target_vram, model)
            elif quant != "auto":
                orchestrator = IntelligenceOrchestrator(model, quantization=quant)
            else:
                orchestrator = IntelligenceOrchestrator.auto(model)
            
            progress.update(task, advance=30, description="[cyan]Loading model weights...")
            orchestrator.load(verbose=False)
            
            progress.update(task, completed=100, description="[green]Model loaded!")
            
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            import traceback
            console.print(f"[dim]{traceback.format_exc()}[/dim]")
            return
    
    config = orchestrator.get_config()
    console.print(Panel.fit(
        f"[bold green]✅ Model Ready![/bold green]\n\n"
        f"[bold]Model:[/bold] {model}\n"
        f"[bold]Quantization:[/bold] {config.quantization.upper()}\n"
        f"[bold]VRAM Used:[/bold] {config.estimated_vram_gb:.2f} GB\n"
        f"[bold]Expected Speed:[/bold] ~{config.expected_tokens_per_sec:.0f} tok/s\n\n"
        f"[dim]Type your message and press Enter. Type 'exit' to quit.[/dim]",
        title="[bold blue]💬 ZSE Chat[/bold blue]",
        border_style="green",
    ))
    
    # Build conversation history
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    
    while True:
        try:
            console.print()
            user_input = Prompt.ask("[bold cyan]You[/bold cyan]")
            
            if not user_input.strip():
                continue
            
            if user_input.lower().strip() in ("exit", "quit", "q"):
                break
            
            if user_input.lower().strip() == "clear":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                console.clear()
                show_banner()
                console.print("[green]Conversation cleared.[/green]")
                continue
            
            if user_input.lower().strip() == "stats":
                _show_chat_stats(orchestrator)
                continue
            
            messages.append({"role": "user", "content": user_input})
            
            # Format prompt for the model
            try:
                prompt = orchestrator.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception:
                # Fallback for models without chat template
                prompt = "\n".join([
                    f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
                    for m in messages
                ])
                prompt += "\nAssistant:"
            
            # Stream response
            console.print()
            console.print("[bold green]Assistant[/bold green]: ", end="")
            
            response_text = []
            start_time = time.perf_counter()
            token_count = 0
            
            for chunk in orchestrator.generate(prompt, max_tokens=1024, stream=True):
                console.print(chunk, end="")
                response_text.append(chunk)
                token_count += 1
            
            elapsed = time.perf_counter() - start_time
            console.print()
            console.print(f"[dim]({token_count} tokens, {token_count/elapsed:.1f} tok/s)[/dim]")
            
            # Add to history
            full_response = "".join(response_text)
            messages.append({"role": "assistant", "content": full_response})
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Generation interrupted. Type 'exit' to quit.[/yellow]")
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
    
    # Cleanup
    console.print("\n[yellow]Unloading model...[/yellow]")
    orchestrator.unload()
    console.print("[green]Goodbye! 👋[/green]")


def _show_chat_stats(orchestrator) -> None:
    """Show current chat session stats."""
    import torch
    
    table = Table(title="[bold cyan]Session Statistics[/bold cyan]", box=box.ROUNDED)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    
    config = orchestrator.get_config()
    table.add_row("Model", config.model_name)
    table.add_row("Quantization", config.quantization.upper())
    
    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated() / (1024**3)
        vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
        table.add_row("VRAM Used", f"{vram_used:.2f} GB")
        table.add_row("VRAM Peak", f"{vram_peak:.2f} GB")
    
    console.print(table)


@app.command()
def chat(
    model: Annotated[
        str,
        typer.Argument(
            help="Model for chat (HuggingFace ID, local path, or .zse file)",
        ),
    ],
    efficiency: Annotated[
        EfficiencyMode,
        typer.Option("--efficiency", "-e", help="Efficiency mode"),
    ] = EfficiencyMode.balanced,
    max_memory: Annotated[
        Optional[str],
        typer.Option("--max-memory", "-m", help="Maximum memory to use"),
    ] = None,
    system_prompt: Annotated[
        Optional[str],
        typer.Option("--system", "-s", help="System prompt"),
    ] = None,
) -> None:
    """
    Start an interactive chat session.
    
    Examples:
        zse chat meta-llama/Llama-3-8B
        zse chat ./model.zse --system "You are a helpful assistant"
    """
    show_banner()
    
    console.print(Panel.fit(
        f"[bold]Model:[/bold] [green]{model}[/green]\n"
        f"[bold]Efficiency:[/bold] [yellow]{efficiency.value}[/yellow]\n"
        f"[dim]Type 'exit' to quit, 'clear' to reset, 'stats' for info[/dim]",
        title="[bold blue]💬 ZSE Chat[/bold blue]",
        border_style="blue",
    ))
    
    _run_chat_session(model, efficiency, max_memory, system_prompt)


@app.command()
def convert(
    source: Annotated[
        str,
        typer.Argument(
            help="Source model (HuggingFace ID or local path)",
        ),
    ],
    output: Annotated[
        Path,
        typer.Option("--output", "-o", help="Output .zse file path"),
    ],
    quantization: Annotated[
        str,
        typer.Option("--quantization", "-q", help="Quantization method (int4, int8, fp16)"),
    ] = "int4",
    target_memory: Annotated[
        Optional[str],
        typer.Option("--target-memory", "-t", help="Target memory for auto-quantization"),
    ] = None,
    calibration_dataset: Annotated[
        Optional[str],
        typer.Option("--calibration", help="Calibration dataset for quantization"),
    ] = None,
) -> None:
    """
    Convert model to ZSE native format (.zse) with quantization.
    
    Examples:
        zse convert meta-llama/Llama-3-70B -o llama-70b.zse
        zse convert ./model -o model.zse --target-memory 24GB
    """
    show_banner()
    
    console.print(Panel.fit(
        f"[bold]Source:[/bold] [green]{source}[/green]\n"
        f"[bold]Output:[/bold] [green]{output}[/green]\n"
        f"[bold]Quantization:[/bold] [yellow]{quantization}[/yellow]",
        title="[bold blue]🔄 ZSE Convert[/bold blue]",
        border_style="blue",
    ))
    
    if target_memory:
        console.print(f"[bold]Target Memory:[/bold] [cyan]{target_memory}[/cyan]")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
        import torch
        import json
        
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Converting model...", total=100)
            
            # Load config
            progress.update(task, advance=10, description="[cyan]Loading model config...")
            config = AutoConfig.from_pretrained(source, trust_remote_code=True)
            
            # Load tokenizer
            progress.update(task, advance=10, description="[cyan]Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            
            # Determine target dtype
            progress.update(task, advance=10, description="[cyan]Loading model...")
            
            if quantization == "fp16":
                dtype = torch.float16
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    torch_dtype=dtype,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            elif quantization in ("int8", "int4"):
                # Use bitsandbytes for quantization
                from transformers import BitsAndBytesConfig
                
                if quantization == "int8":
                    quant_config = BitsAndBytesConfig(load_in_8bit=True)
                else:
                    quant_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    quantization_config=quant_config,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            else:
                # Default FP16
                model = AutoModelForCausalLM.from_pretrained(
                    source,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    trust_remote_code=True,
                )
            
            progress.update(task, advance=30, description="[cyan]Saving model...")
            
            # Save in ZSE format (directory structure)
            zse_dir = output_path.with_suffix("")
            zse_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            model.save_pretrained(zse_dir / "model", safe_serialization=True)
            
            # Save tokenizer
            tokenizer.save_pretrained(zse_dir / "tokenizer")
            
            # Save ZSE metadata
            zse_metadata = {
                "version": __version__,
                "source": source,
                "quantization": quantization,
                "target_memory": target_memory,
                "model_config": {
                    "architecture": config.architectures[0] if hasattr(config, 'architectures') else "unknown",
                    "hidden_size": getattr(config, 'hidden_size', None),
                    "num_layers": getattr(config, 'num_hidden_layers', None),
                    "num_heads": getattr(config, 'num_attention_heads', None),
                    "vocab_size": getattr(config, 'vocab_size', None),
                },
            }
            
            with open(zse_dir / "zse_config.json", "w") as f:
                json.dump(zse_metadata, f, indent=2)
            
            progress.update(task, completed=100, description="[green]Conversion complete!")
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in zse_dir.rglob("*") if f.is_file())
        size_gb = total_size / (1024**3)
        
        console.print(Panel.fit(
            f"[bold green]✅ Conversion Complete![/bold green]\n\n"
            f"[bold]Output:[/bold] {zse_dir}\n"
            f"[bold]Size:[/bold] {size_gb:.2f} GB\n"
            f"[bold]Quantization:[/bold] {quantization.upper()}\n\n"
            f"[dim]Load with: zse serve {zse_dir}[/dim]",
            title="[bold blue]🎉 Success[/bold blue]",
            border_style="green",
        ))
        
    except Exception as e:
        console.print(f"[red]❌ Conversion failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def info(
    model: Annotated[
        str,
        typer.Argument(
            help="Model to inspect",
        ),
    ],
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", "-f", help="Output format"),
    ] = OutputFormat.text,
) -> None:
    """
    Show model information and memory estimates.
    
    Examples:
        zse info meta-llama/Llama-3-70B
        zse info ./model.zse --format json
    """
    show_banner()
    
    console.print(Panel.fit(
        f"[bold]Model:[/bold] [green]{model}[/green]",
        title="[bold blue]ℹ️  Model Info[/bold blue]",
        border_style="blue",
    ))
    
    if output_format == OutputFormat.json:
        _show_model_info_json(model)
    else:
        _show_model_info_real(model)


def _show_model_info_json(model: str) -> None:
    """Show model info in JSON format."""
    import json
    
    try:
        from zse.engine.orchestrator.core import estimate_requirements
        
        result = estimate_requirements(model)
        
        # Try to add HF config
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model, trust_remote_code=True)
            result["architecture"] = {
                "type": config.architectures[0] if hasattr(config, 'architectures') else "unknown",
                "hidden_size": getattr(config, 'hidden_size', None),
                "num_layers": getattr(config, 'num_hidden_layers', None),
                "num_heads": getattr(config, 'num_attention_heads', None),
                "vocab_size": getattr(config, 'vocab_size', None),
                "max_seq_length": getattr(config, 'max_position_embeddings', None),
            }
        except Exception:
            pass
        
        console.print_json(json.dumps(result, indent=2))
        
    except Exception as e:
        console.print(f'{{"error": "{e}"}}')


@app.command()
def benchmark(
    model: Annotated[
        str,
        typer.Argument(
            help="Model to benchmark",
        ),
    ],
    prompt_length: Annotated[
        int,
        typer.Option("--prompt-length", "-p", help="Input prompt length in tokens"),
    ] = 128,
    output_length: Annotated[
        int,
        typer.Option("--output-length", "-o", help="Output length in tokens"),
    ] = 128,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", "-b", help="Batch size"),
    ] = 1,
    iterations: Annotated[
        int,
        typer.Option("--iterations", "-i", help="Number of iterations"),
    ] = 5,
    efficiency: Annotated[
        EfficiencyMode,
        typer.Option("--efficiency", "-e", help="Efficiency mode"),
    ] = EfficiencyMode.balanced,
    warmup: Annotated[
        int,
        typer.Option("--warmup", "-w", help="Warmup iterations"),
    ] = 2,
) -> None:
    """
    Run inference benchmarks.
    
    Examples:
        zse benchmark meta-llama/Llama-3-8B
        zse benchmark ./model.zse --iterations 20 --efficiency speed
    """
    show_banner()
    
    console.print(Panel.fit(
        f"[bold]Model:[/bold] [green]{model}[/green]\n"
        f"[bold]Prompt:[/bold] {prompt_length} tokens\n"
        f"[bold]Output:[/bold] {output_length} tokens\n"
        f"[bold]Batch Size:[/bold] {batch_size}\n"
        f"[bold]Iterations:[/bold] {iterations}\n"
        f"[bold]Warmup:[/bold] {warmup}",
        title="[bold blue]⚡ ZSE Benchmark[/bold blue]",
        border_style="blue",
    ))
    
    try:
        from zse.engine.orchestrator.core import IntelligenceOrchestrator
        import torch
        
        quant = _efficiency_to_quantization(efficiency)
        
        # Load model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Loading model...", total=None)
            
            if quant != "auto":
                orchestrator = IntelligenceOrchestrator(model, quantization=quant)
            else:
                orchestrator = IntelligenceOrchestrator.auto(model)
            
            orchestrator.load(verbose=False)
            progress.update(task, description="[green]Model loaded!")
        
        config = orchestrator.get_config()
        console.print(f"\n[bold]Loaded:[/bold] {config.quantization.upper()}, {config.estimated_vram_gb:.2f} GB VRAM")
        
        # Create test prompt
        test_prompt = "Write a detailed explanation of how transformers work in deep learning. Include the attention mechanism, positional encoding, and the encoder-decoder architecture."
        
        # Warmup
        console.print(f"\n[cyan]Running {warmup} warmup iterations...[/cyan]")
        for i in range(warmup):
            _ = list(orchestrator.generate(test_prompt, max_tokens=output_length, stream=True))
        
        # Benchmark
        console.print(f"[cyan]Running {iterations} benchmark iterations...[/cyan]")
        
        results = []
        torch.cuda.reset_peak_memory_stats()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Benchmarking...", total=iterations)
            
            for i in range(iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                token_count = 0
                for chunk in orchestrator.generate(test_prompt, max_tokens=output_length, stream=True):
                    token_count += 1
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                results.append({
                    "tokens": token_count,
                    "time": elapsed,
                    "tps": token_count / elapsed,
                })
                
                progress.update(task, advance=1, description=f"[cyan]Iteration {i+1}: {token_count/elapsed:.1f} tok/s")
        
        # Calculate stats
        tps_values = [r["tps"] for r in results]
        avg_tps = sum(tps_values) / len(tps_values)
        min_tps = min(tps_values)
        max_tps = max(tps_values)
        
        peak_memory = torch.cuda.max_memory_allocated() / (1024**3)
        
        # Results table
        console.print()
        results_table = Table(title="[bold cyan]Benchmark Results[/bold cyan]", box=box.ROUNDED)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="green")
        
        results_table.add_row("Model", model)
        results_table.add_row("Quantization", config.quantization.upper())
        results_table.add_row("", "")
        results_table.add_row("[bold]Throughput[/bold]", "")
        results_table.add_row("  Average", f"{avg_tps:.2f} tok/s")
        results_table.add_row("  Min", f"{min_tps:.2f} tok/s")
        results_table.add_row("  Max", f"{max_tps:.2f} tok/s")
        results_table.add_row("", "")
        results_table.add_row("[bold]Memory[/bold]", "")
        results_table.add_row("  Allocated", f"{config.estimated_vram_gb:.2f} GB")
        results_table.add_row("  Peak", f"{peak_memory:.2f} GB")
        results_table.add_row("", "")
        results_table.add_row("[bold]Latency[/bold]", "")
        avg_latency = (sum(r["time"] for r in results) / len(results)) * 1000
        results_table.add_row("  Avg per generation", f"{avg_latency:.1f} ms")
        results_table.add_row("  Avg per token", f"{1000/avg_tps:.2f} ms")
        
        console.print(results_table)
        
        # Comparison to FP16 baseline
        if config.quantization != "fp16":
            memory_savings = (config.estimated_vram_gb * 2) / config.estimated_vram_gb if config.quantization == "int8" else (config.estimated_vram_gb * 4) / config.estimated_vram_gb
            console.print(f"\n[dim]Memory vs FP16: {1/memory_savings:.1f}x reduction[/dim]")
        
        # Cleanup
        orchestrator.unload()
        
    except Exception as e:
        console.print(f"[red]❌ Benchmark failed: {e}[/red]")
        import traceback
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def hardware() -> None:
    """
    Show detected hardware and memory information.
    """
    show_banner()
    
    console.print(Panel.fit(
        "[bold]Detecting system hardware...[/bold]",
        title="[bold blue]💻 Hardware Detection[/bold blue]",
        border_style="blue",
    ))
    
    _show_hardware_info()


def _show_hardware_info() -> None:
    """Display detailed hardware information."""
    import psutil
    
    console.print()
    
    # System Info Table
    sys_table = Table(
        title="[bold cyan]System Information[/bold cyan]",
        box=box.ROUNDED,
    )
    sys_table.add_column("Component", style="cyan", width=20)
    sys_table.add_column("Details", style="green")
    
    # CPU
    cpu_count = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    sys_table.add_row("CPU Cores", f"{cpu_count} physical, {cpu_count_logical} logical")
    
    # RAM
    ram = psutil.virtual_memory()
    ram_total = ram.total / (1024**3)
    ram_available = ram.available / (1024**3)
    ram_percent = ram.percent
    sys_table.add_row(
        "RAM", 
        f"[green]{ram_available:.1f} GB free[/green] / {ram_total:.1f} GB total ({ram_percent}% used)"
    )
    
    console.print(sys_table)
    console.print()
    
    # GPU Info Table
    gpu_table = Table(
        title="[bold cyan]GPU Information[/bold cyan]",
        box=box.ROUNDED,
    )
    gpu_table.add_column("GPU", style="cyan", width=10)
    gpu_table.add_column("Name", style="white", width=30)
    gpu_table.add_column("Memory", style="green")
    gpu_table.add_column("Utilization", style="yellow")
    
    gpu_info = _get_hardware_info()
    
    if gpu_info["available"]:
        try:
            import pynvml
            pynvml.nvmlInit()
            
            for gpu in gpu_info["gpus"]:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu["id"])
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_str = f"{util.gpu}%"
                except Exception:
                    util_str = "N/A"
                
                mem_used_percent = ((gpu["total_memory_gb"] - gpu["free_memory_gb"]) / gpu["total_memory_gb"]) * 100
                
                gpu_table.add_row(
                    f"GPU {gpu['id']}",
                    gpu["name"],
                    f"[green]{gpu['free_memory_gb']:.1f} GB free[/green] / {gpu['total_memory_gb']:.1f} GB ({mem_used_percent:.0f}% used)",
                    util_str
                )
            
            pynvml.nvmlShutdown()
        except Exception:
            for gpu in gpu_info["gpus"]:
                gpu_table.add_row(f"GPU {gpu['id']}", gpu["name"], f"{gpu['free_memory_gb']:.1f} / {gpu['total_memory_gb']:.1f} GB", "N/A")
        
        console.print(gpu_table)
    else:
        console.print("[yellow]⚠️  No GPU detected[/yellow]")
    
    console.print()
    _show_fitting_recommendations()


def _show_fitting_recommendations() -> None:
    """Show what models can fit based on current hardware."""
    import psutil
    
    gpu_info = _get_hardware_info()
    gpu_mem = gpu_info["gpus"][0]["free_memory_gb"] if gpu_info["available"] and gpu_info["gpus"] else 0.0
    
    ram = psutil.virtual_memory()
    ram_free = ram.available / (1024**3)
    
    table = Table(
        title="[bold cyan]What Can You Run? (ZSE Ultra Mode)[/bold cyan]",
        box=box.ROUNDED,
    )
    table.add_column("Model", style="white")
    table.add_column("GPU Only", style="green")
    table.add_column("GPU + CPU Hybrid", style="yellow")
    
    models = [
        ("7B", 3.5, 5.0),
        ("14B", 6.0, 10.0),
        ("32B", 18.0, 25.0),
        ("70B", 30.0, 50.0),
    ]
    
    for model, gpu_req, hybrid_req in models:
        gpu_status = "✅" if gpu_mem >= gpu_req else "❌"
        hybrid_status = "✅" if (gpu_mem + ram_free * 0.5) >= hybrid_req else "❌"
        table.add_row(model, gpu_status, hybrid_status)
    
    console.print(table)
    
    if gpu_mem > 0:
        console.print(f"\n[dim]Based on {gpu_mem:.1f} GB free GPU memory and {ram_free:.1f} GB free RAM[/dim]")


def _show_recommendations(
    model: str,
    max_memory: Optional[str],
    efficiency: EfficiencyMode,
) -> None:
    """Show memory recommendations for the model."""
    console.print("\n[bold]🔍 Analyzing system and model...[/bold]\n")
    
    _show_hardware_summary()
    
    try:
        from zse.engine.orchestrator.core import estimate_requirements
        
        req = estimate_requirements(model)
        
        console.print()
        
        table = Table(
            title="[bold cyan]📋 Loading Recommendations[/bold cyan]",
            box=box.ROUNDED,
        )
        table.add_column("#", style="cyan", width=3)
        table.add_column("Precision", style="green", width=12)
        table.add_column("GPU Memory", style="yellow", width=20)
        table.add_column("Strategy", style="blue", width=15)
        table.add_column("Est. Speed", style="magenta", width=12)
        
        params_b = req["estimated_params_b"]
        
        # Calculate recommendations
        int4_mem = params_b * 0.5
        int8_mem = params_b * 1.0
        fp16_mem = params_b * 2.0
        
        # Speed estimates (relative)
        table.add_row("1", "INT4-NF4", f"~{int4_mem:.1f} GB", "GPU only", "~15 tok/s")
        table.add_row("2", "INT8", f"~{int8_mem:.1f} GB", "GPU only", "~25 tok/s")
        table.add_row("3", "FP16", f"~{fp16_mem:.1f} GB", "GPU only", "~50 tok/s")
        
        gpu_info = _get_hardware_info()
        if gpu_info["available"] and gpu_info["gpus"]:
            gpu_mem = gpu_info["gpus"][0]["free_memory_gb"]
            
            # Mark recommended
            if gpu_mem >= fp16_mem * 1.1:
                recommended = "FP16"
            elif gpu_mem >= int8_mem * 1.1:
                recommended = "INT8"
            else:
                recommended = "INT4-NF4"
            
            console.print(table)
            console.print(f"\n[green]★ Recommended:[/green] [bold]{recommended}[/bold] based on {gpu_mem:.1f} GB free GPU memory")
        else:
            console.print(table)
    
    except Exception as e:
        console.print(f"[yellow]Could not analyze model: {e}[/yellow]")


# =============================================================================
# API Key Management Commands
# =============================================================================

@app.command(name="api-key")
def api_key_command(
    action: Annotated[str, typer.Argument(help="Action: create, list, delete, enable, disable, status")],
    name: Annotated[Optional[str], typer.Argument(help="Key name (for create/delete/status)")] = None,
    rate_limit: Annotated[Optional[int], typer.Option("--rate-limit", "-r", help="Requests per minute limit")] = None,
):
    """
    Manage API keys for authentication.
    
    Examples:
      zse api-key create my-app                 Create a new API key
      zse api-key create my-app --rate-limit 60 Create key with 60 req/min limit
      zse api-key list                          List all API keys
      zse api-key delete my-app                 Delete the 'my-app' key
      zse api-key status my-app                 Show rate limit status  
      zse api-key enable                        Enable API key authentication
      zse api-key disable                       Disable authentication (allow all)
    """
    from zse.api.server.auth import get_key_manager, get_rate_limit_status, reset_rate_limit
    
    manager = get_key_manager()
    
    if action == "create":
        if not name:
            console.print("[red]Error: Key name required for 'create'[/red]")
            console.print("Usage: zse api-key create <name> [--rate-limit N]")
            raise typer.Exit(1)
        
        key = manager.create_key(name, rate_limit=rate_limit)
        rate_info = f"\n[bold]Rate Limit:[/bold] {rate_limit}/min" if rate_limit else ""
        console.print(Panel(
            f"[bold green]API Key Created[/bold green]\n\n"
            f"[bold]Name:[/bold] {name}\n"
            f"[bold]Key:[/bold] [cyan]{key}[/cyan]{rate_info}\n\n"
            f"[yellow]⚠️  Save this key now - it won't be shown again![/yellow]\n\n"
            f"[dim]Usage:[/dim]\n"
            f"  curl -H 'X-API-Key: {key}' http://localhost:8000/v1/chat/completions\n"
            f"  curl -H 'Authorization: Bearer {key}' http://localhost:8000/v1/chat/completions",
            title="🔑 New API Key",
            border_style="green"
        ))
        
    elif action == "list":
        keys = manager.list_keys()
        if not keys:
            console.print("[yellow]No API keys configured.[/yellow]")
            console.print("Create one with: [cyan]zse api-key create <name>[/cyan]")
            return
        
        table = Table(title="[bold cyan]API Keys[/bold cyan]", box=box.ROUNDED)
        table.add_column("Name", style="cyan")
        table.add_column("Created", style="white")
        table.add_column("Last Used", style="white")
        table.add_column("Requests", style="green")
        table.add_column("Rate Limit", style="yellow")
        
        for k in keys:
            last_used = k["last_used"][:10] if k["last_used"] else "Never"
            rate_limit = f"{k['rate_limit']}/min" if k["rate_limit"] else "Unlimited"
            table.add_row(
                k["name"],
                k["created_at"][:10],
                last_used,
                str(k["request_count"]),
                rate_limit
            )
        
        console.print(table)
        
        status = "[green]Enabled[/green]" if manager.is_enabled() else "[yellow]Disabled[/yellow]"
        console.print(f"\nAuthentication: {status}")
        
    elif action == "delete":
        if not name:
            console.print("[red]Error: Key name required for 'delete'[/red]")
            raise typer.Exit(1)
        
        if manager.delete_key(name):
            console.print(f"[green]✅ Deleted API key: {name}[/green]")
        else:
            console.print(f"[red]❌ API key not found: {name}[/red]")
            
    elif action == "enable":
        manager.enable()
        console.print("[green]✅ API key authentication enabled[/green]")
        console.print("[dim]All requests now require a valid API key[/dim]")
        
    elif action == "disable":
        manager.disable()
        console.print("[yellow]⚠️  API key authentication disabled[/yellow]")
        console.print("[dim]All requests will be allowed without authentication[/dim]")
    
    elif action == "status":
        if not name:
            console.print("[red]Error: Key name required for 'status'[/red]")
            console.print("Usage: zse api-key status <name>")
            raise typer.Exit(1)
        
        # Find the key
        keys = manager.list_keys()
        key_info = next((k for k in keys if k["name"] == name), None)
        
        if not key_info:
            console.print(f"[red]❌ API key not found: {name}[/red]")
            raise typer.Exit(1)
        
        # Get rate limit status
        if key_info["rate_limit"]:
            # Need to find the actual APIKey object to get hash
            for key_hash, api_key in manager.keys.items():
                if api_key.name == name:
                    status = get_rate_limit_status(api_key)
                    console.print(Panel(
                        f"[bold]Key:[/bold] {name}\n"
                        f"[bold]Rate Limit:[/bold] {status['limit']}/min\n"
                        f"[bold]Current Usage:[/bold] {status['current']}/{status['limit']}\n"
                        f"[bold]Remaining:[/bold] {status['remaining']}\n"
                        f"[bold]Window:[/bold] {status['window_seconds']}s",
                        title="📊 Rate Limit Status",
                        border_style="cyan"
                    ))
                    break
        else:
            console.print(f"[yellow]Key '{name}' has no rate limit configured.[/yellow]")
    
    elif action == "reset":
        if not name:
            console.print("[red]Error: Key name required for 'reset'[/red]")
            raise typer.Exit(1)
        
        if reset_rate_limit(name):
            console.print(f"[green]✅ Rate limit reset for: {name}[/green]")
        else:
            console.print(f"[red]❌ API key not found: {name}[/red]")
        
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: create, list, delete, enable, disable, status, reset")
        raise typer.Exit(1)


# =============================================================================
# Model Registry & Discovery Commands
# =============================================================================

@app.command(name="models")
def models_command(
    action: Annotated[str, typer.Argument(help="Action: list, search, check, info")] = "list",
    query: Annotated[Optional[str], typer.Argument(help="Search query or model ID")] = None,
    category: Annotated[Optional[str], typer.Option("--category", "-c", help="Filter by: chat, instruct, code, reasoning")] = None,
    size: Annotated[Optional[str], typer.Option("--size", "-s", help="Filter by: tiny, small, medium, large, xlarge, xxl")] = None,
    max_vram: Annotated[Optional[float], typer.Option("--max-vram", "-v", help="Max VRAM in GB")] = None,
    recommended: Annotated[bool, typer.Option("--recommended", "-r", help="Show only recommended models")] = False,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results for search")] = 15,
):
    """
    Browse and discover models compatible with ZSE.
    
    Examples:
      zse models                      List all registered models
      zse models list -r              List recommended models
      zse models list -c code         List code-focused models
      zse models list -v 8            Models fitting in 8GB VRAM
      zse models search llama         Search HuggingFace for llama models
      zse models check meta-llama/Llama-3.1-8B   Check compatibility
      zse models info Qwen/Qwen2.5-7B-Instruct   Get detailed info
    """
    from zse.models.registry import get_registry, ModelCategory, ModelSize
    
    if action == "list":
        registry = get_registry()
        
        # Apply filters
        if recommended:
            models = registry.get_recommended(max_vram_gb=max_vram)
            title = "Recommended Models"
        elif max_vram:
            models = registry.filter_by_vram(max_vram, quantization="int8")
            title = f"Models fitting in {max_vram}GB VRAM (INT8)"
        elif category:
            try:
                cat = ModelCategory(category.lower())
                models = registry.filter_by_category(cat)
                title = f"{category.title()} Models"
            except ValueError:
                console.print(f"[red]Invalid category: {category}[/red]")
                console.print(f"Valid: {[c.value for c in ModelCategory]}")
                raise typer.Exit(1)
        elif size:
            try:
                sz = ModelSize(size.lower())
                models = registry.filter_by_size(sz)
                title = f"{size.title()} Models"
            except ValueError:
                console.print(f"[red]Invalid size: {size}[/red]")
                console.print(f"Valid: {[s.value for s in ModelSize]}")
                raise typer.Exit(1)
        else:
            models = registry.list_all()
            title = "ZSE Model Registry"
        
        if not models:
            console.print("[yellow]No models match the criteria.[/yellow]")
            return
        
        # Create table
        table = Table(title=f"[bold cyan]{title}[/bold cyan]", box=box.ROUNDED, show_lines=True)
        table.add_column("Model", style="cyan", max_width=40)
        table.add_column("Params", justify="right", style="green")
        table.add_column("VRAM (INT8)", justify="right", style="yellow")
        table.add_column("Categories", style="white", max_width=25)
        table.add_column("Tags", style="dim", max_width=20)
        
        for m in models:
            tags = ", ".join(m.tags[:3])
            cats = ", ".join(c.value for c in m.categories)
            rec = "★ " if "recommended" in m.tags else ""
            table.add_row(
                f"{rec}{m.model_id}",
                m.parameters,
                f"{m.vram_int8_gb:.1f} GB",
                cats,
                tags,
            )
        
        console.print(table)
        console.print(f"\n[dim]Total: {len(models)} models[/dim]")
        console.print("\n[dim]Download: huggingface-cli download <model_id>[/dim]")
        console.print("[dim]Convert:  zse convert <model_path> -f zse[/dim]")
        console.print("[dim]Serve:    zse serve <model_id_or_path>[/dim]")
        
    elif action == "search":
        if not query:
            console.print("[red]Search requires a query[/red]")
            console.print("Usage: zse models search <query>")
            raise typer.Exit(1)
        
        try:
            from zse.models.discovery import get_discovery
            
            console.print(f"[dim]Searching HuggingFace for '{query}'...[/dim]")
            discovery = get_discovery()
            models = discovery.search(query=query, limit=limit, only_compatible=True)
            
            if not models:
                console.print("[yellow]No compatible models found.[/yellow]")
                return
            
            table = Table(title=f"[bold cyan]HuggingFace: '{query}'[/bold cyan]", box=box.ROUNDED)
            table.add_column("Model ID", style="cyan", max_width=45)
            table.add_column("Downloads", justify="right", style="green")
            table.add_column("Likes", justify="right", style="yellow")
            table.add_column("Arch", style="white")
            table.add_column("Compatible", justify="center")
            
            for m in models:
                compat = "[green]✓[/green]" if m.is_compatible else "[yellow]?[/yellow]"
                arch = m.architecture[:15] + "..." if m.architecture and len(m.architecture) > 15 else (m.architecture or "?")
                table.add_row(
                    m.model_id,
                    f"{m.downloads:,}",
                    f"{m.likes:,}",
                    arch,
                    compat,
                )
            
            console.print(table)
            console.print(f"\n[dim]Found {len(models)} compatible models. Use 'zse models check <id>' for details.[/dim]")
            
        except ImportError:
            console.print("[red]Search requires httpx: pip install httpx[/red]")
            raise typer.Exit(1)
            
    elif action == "check":
        if not query:
            console.print("[red]Check requires a model ID[/red]")
            console.print("Usage: zse models check <model_id>")
            raise typer.Exit(1)
        
        try:
            from zse.models.discovery import get_discovery
            
            console.print(f"[dim]Checking compatibility for '{query}'...[/dim]")
            discovery = get_discovery()
            result = discovery.check_compatibility(query)
            
            # Display results
            status = "[green]✓ Compatible[/green]" if result["compatible"] else "[red]✗ Not Compatible[/red]"
            
            panel_content = f"[bold]Model:[/bold] {result['model_id']}\n"
            panel_content += f"[bold]Architecture:[/bold] {result.get('architecture', 'Unknown')}\n"
            panel_content += f"[bold]Status:[/bold] {status}\n"
            
            if result.get("size_info"):
                info = result["size_info"]
                panel_content += f"\n[bold]Size Estimates:[/bold]\n"
                panel_content += f"  Parameters: ~{info['estimated_params_b']:.1f}B\n"
                panel_content += f"  VRAM FP16:  {info['estimated_vram_fp16_gb']:.1f} GB\n"
                panel_content += f"  VRAM INT8:  {info['estimated_vram_int8_gb']:.1f} GB\n"
                panel_content += f"  VRAM INT4:  {info['estimated_vram_int4_gb']:.1f} GB\n"
                panel_content += f"  Safetensors: {'Yes' if info['has_safetensors'] else 'No'}\n"
            
            if result.get("issues"):
                panel_content += f"\n[yellow]Issues:[/yellow]\n"
                for issue in result["issues"]:
                    panel_content += f"  • {issue}\n"
            
            if result.get("recommendations"):
                panel_content += f"\n[cyan]Recommendations:[/cyan]\n"
                for rec in result["recommendations"]:
                    panel_content += f"  • {rec}\n"
            
            console.print(Panel(panel_content, title="Model Compatibility Check", border_style="cyan"))
            
            if result["compatible"]:
                console.print("\n[bold green]Ready to use![/bold green]")
                console.print(f"  [dim]Download:[/dim] huggingface-cli download {query}")
                console.print(f"  [dim]Serve:[/dim]    zse serve {query}")
                console.print(f"  [dim]Convert:[/dim]  zse convert <path> -f zse")
            
        except ImportError:
            console.print("[red]Check requires httpx: pip install httpx[/red]")
            raise typer.Exit(1)
            
    elif action == "info":
        if not query:
            console.print("[red]Info requires a model ID[/red]")
            console.print("Usage: zse models info <model_id>")
            raise typer.Exit(1)
        
        # First check registry
        registry = get_registry()
        model = registry.get(query)
        
        if model:
            panel_content = f"[bold]Model ID:[/bold] {model.model_id}\n"
            panel_content += f"[bold]Name:[/bold] {model.name}\n"
            panel_content += f"[bold]Description:[/bold] {model.description}\n"
            panel_content += f"[bold]Provider:[/bold] {model.provider}\n"
            panel_content += f"[bold]License:[/bold] {model.license}\n"
            panel_content += f"\n[bold]Specifications:[/bold]\n"
            panel_content += f"  Parameters: {model.parameters}\n"
            panel_content += f"  Architecture: {model.architecture}\n"
            panel_content += f"  Context Length: {model.context_length:,}\n"
            panel_content += f"\n[bold]VRAM Requirements:[/bold]\n"
            panel_content += f"  FP16: {model.vram_fp16_gb:.1f} GB\n"
            panel_content += f"  INT8: {model.vram_int8_gb:.1f} GB [recommended: {model.recommended_quant}]\n"
            panel_content += f"  INT4: {model.vram_int4_gb:.1f} GB\n"
            panel_content += f"\n[bold]Categories:[/bold] {', '.join(c.value for c in model.categories)}\n"
            panel_content += f"[bold]Tags:[/bold] {', '.join(model.tags)}\n"
            
            console.print(Panel(panel_content, title=f"[cyan]{model.name}[/cyan]", border_style="cyan"))
            
            console.print("\n[bold]Quick Start:[/bold]")
            console.print(f"  [dim]1. Download:[/dim] huggingface-cli download {model.model_id}")
            console.print(f"  [dim]2. Serve:[/dim]    zse serve {model.model_id}")
            console.print(f"  [dim]3. Convert:[/dim]  zse convert ~/.cache/huggingface/hub/models--{model.model_id.replace('/', '--')}/snapshots/<hash> -f zse -o {model.model_id.split('/')[-1]}.zse")
        else:
            # Try discovery
            console.print("[yellow]Not in registry, checking HuggingFace...[/yellow]")
            try:
                from zse.models.discovery import get_discovery
                discovery = get_discovery()
                result = discovery.check_compatibility(query)
                
                if result:
                    # Show basic info from discovery
                    console.print(f"\n[bold]Model:[/bold] {result['model_id']}")
                    console.print(f"[bold]Architecture:[/bold] {result.get('architecture', 'Unknown')}")
                    console.print(f"[bold]Compatible:[/bold] {'Yes' if result['compatible'] else 'No'}")
                    if result.get("size_info"):
                        console.print(f"[bold]Est. Params:[/bold] ~{result['size_info']['estimated_params_b']:.1f}B")
            except ImportError:
                console.print("[red]Not found in registry. Install httpx for HuggingFace lookup.[/red]")
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: list, search, check, info")
        raise typer.Exit(1)


# =============================================================================
# Audit Log Commands
# =============================================================================

@app.command(name="audit")
def audit_command(
    action: Annotated[str, typer.Argument(help="Action: summary, recent, query, export, clear")] = "summary",
    hours: Annotated[Optional[int], typer.Option("--hours", "-h", help="Hours to look back (default: 24)")] = None,
    api_key: Annotated[Optional[str], typer.Option("--key", "-k", help="Filter by API key name")] = None,
    path: Annotated[Optional[str], typer.Option("--path", "-p", help="Filter by path prefix")] = None,
    status: Annotated[Optional[int], typer.Option("--status", "-s", help="Filter by status code")] = None,
    limit: Annotated[int, typer.Option("--limit", "-l", help="Max results")] = 50,
    output: Annotated[Optional[str], typer.Option("--output", "-o", help="Export output file")] = None,
    format: Annotated[str, typer.Option("--format", "-f", help="Export format: jsonl, csv")] = "jsonl",
    all_logs: Annotated[bool, typer.Option("--all", help="Include/clear all rotated logs")] = False,
):
    """
    View and manage API request audit logs.
    
    Examples:
      zse audit                       Show 24-hour summary
      zse audit summary -h 1          Show last hour summary
      zse audit recent                Show recent requests
      zse audit recent -l 20          Show last 20 requests  
      zse audit query -k my-app       Filter by API key
      zse audit query -s 429          Show rate-limited requests
      zse audit query -p /v1/chat     Filter by path
      zse audit export -o logs.jsonl  Export to file
      zse audit export -f csv -o logs.csv   Export as CSV
      zse audit clear                 Clear current log
      zse audit clear --all           Clear all logs including rotated
    """
    from zse.api.server.audit import (
        get_audit_summary, 
        query_audit_logs, 
        get_recent_requests,
        export_audit_logs,
        clear_audit_logs,
        get_audit_logger,
    )
    
    if action == "summary":
        h = hours or 24
        summary = get_audit_summary(h)
        
        console.print(Panel(
            f"[bold]Period:[/bold] Last {summary['period_hours']} hours\n"
            f"[bold]Total Requests:[/bold] {summary['total_requests']}\n"
            f"[bold]Unique API Keys:[/bold] {summary['unique_keys']}\n"
            f"[bold]Avg Latency:[/bold] {summary['avg_latency_ms']} ms\n"
            f"[bold]Total Tokens:[/bold] {summary['total_tokens']}\n"
            f"[bold]Errors:[/bold] {summary['errors']}",
            title="📊 Audit Summary",
            border_style="cyan"
        ))
        
        if summary.get("endpoints"):
            table = Table(title="Top Endpoints", box=box.ROUNDED)
            table.add_column("Path", style="cyan")
            table.add_column("Requests", style="green", justify="right")
            
            for path, count in list(summary["endpoints"].items())[:10]:
                table.add_row(path, str(count))
            console.print(table)
        
        if summary.get("status_codes"):
            table = Table(title="Status Codes", box=box.ROUNDED)
            table.add_column("Code", style="yellow")
            table.add_column("Count", style="green", justify="right")
            
            for code, count in sorted(summary["status_codes"].items()):
                style = "green" if code < 400 else "yellow" if code < 500 else "red"
                table.add_row(f"[{style}]{code}[/{style}]", str(count))
            console.print(table)
    
    elif action == "recent":
        entries = get_recent_requests(limit)
        
        if not entries:
            console.print("[yellow]No recent audit entries in memory buffer.[/yellow]")
            console.print("[dim]Entries are kept in memory during server runtime.[/dim]")
            return
        
        table = Table(title=f"Recent Requests (last {len(entries)})", box=box.ROUNDED)
        table.add_column("Time", style="dim", width=12)
        table.add_column("Method", style="cyan", width=6)
        table.add_column("Path", style="white")
        table.add_column("Status", width=6)
        table.add_column("Latency", style="magenta", width=10, justify="right")
        table.add_column("API Key", style="yellow", width=12)
        
        for entry in entries[-limit:]:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")[11:]  # HH:MM:SS
            method = entry.get("method", "")
            path = entry.get("path", "")[:40]
            status = entry.get("status_code", 0)
            latency = f"{entry.get('latency_ms', 0):.0f}ms"
            key_name = entry.get("api_key_name", "-") or "-"
            
            status_style = "green" if status < 400 else "yellow" if status < 500 else "red"
            
            table.add_row(ts, method, path, f"[{status_style}]{status}[/{status_style}]", latency, key_name[:12])
        
        console.print(table)
    
    elif action == "query":
        entries = query_audit_logs(
            hours=hours,
            api_key=api_key,
            path=path,
            status=status,
            limit=limit,
        )
        
        if not entries:
            console.print("[yellow]No matching entries found.[/yellow]")
            return
        
        table = Table(title=f"Audit Logs ({len(entries)} entries)", box=box.ROUNDED)
        table.add_column("Timestamp", style="dim", width=20)
        table.add_column("Method", style="cyan", width=6)
        table.add_column("Path", style="white")
        table.add_column("Status", width=6)
        table.add_column("Latency", style="magenta", width=10, justify="right")
        table.add_column("API Key", style="yellow", width=12)
        table.add_column("Client IP", style="dim", width=15)
        
        for entry in entries[:limit]:
            ts = entry.get("timestamp", "")[:19].replace("T", " ")
            method = entry.get("method", "")
            path_str = entry.get("path", "")[:35]
            status_code = entry.get("status_code", 0)
            latency = f"{entry.get('latency_ms', 0):.0f}ms"
            key_name = entry.get("api_key_name", "-") or "-"
            client_ip = entry.get("client_ip", "-")
            
            status_style = "green" if status_code < 400 else "yellow" if status_code < 500 else "red"
            
            table.add_row(ts, method, path_str, f"[{status_style}]{status_code}[/{status_style}]", latency, key_name[:12], client_ip)
        
        console.print(table)
    
    elif action == "export":
        if not output:
            console.print("[red]Error: --output/-o required for export[/red]")
            console.print("Usage: zse audit export -o logs.jsonl")
            raise typer.Exit(1)
        
        count = export_audit_logs(output, format=format, hours=hours)
        
        if count > 0:
            console.print(f"[green]✅ Exported {count} entries to {output}[/green]")
        else:
            console.print("[yellow]No entries to export.[/yellow]")
    
    elif action == "clear":
        if not all_logs:
            console.print("[yellow]⚠️  This will clear the current audit log.[/yellow]")
        else:
            console.print("[yellow]⚠️  This will clear ALL audit logs including rotated files.[/yellow]")
        
        from rich.prompt import Confirm
        if Confirm.ask("Are you sure?", default=False):
            clear_audit_logs(all_logs=all_logs)
            console.print("[green]✅ Audit logs cleared.[/green]")
        else:
            console.print("[dim]Cancelled.[/dim]")
    
    elif action == "stats":
        logger = get_audit_logger()
        stats = logger.get_stats()
        
        console.print(Panel(
            f"[bold]Log File:[/bold] {stats['log_file']}\n"
            f"[bold]Log File Size:[/bold] {stats['log_file_size_mb']:.2f} MB\n"
            f"[bold]Rotated Files:[/bold] {stats['rotated_files']}\n"
            f"[bold]Buffer Size:[/bold] {stats['buffer_size']}\n"
            f"[bold]Total Logged:[/bold] {stats['total_logged']}\n"
            f"[bold]Errors:[/bold] {stats['errors']}\n"
            f"[bold]Rotations:[/bold] {stats['rotations']}",
            title="📁 Audit System Stats",
            border_style="cyan"
        ))
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Valid actions: summary, recent, query, export, clear, stats")
        raise typer.Exit(1)
