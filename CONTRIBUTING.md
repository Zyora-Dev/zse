# Contributing to ZSE

Thanks for your interest in contributing to ZSE — Zyora Server Engine! 🚀

## Getting Started
1. Fork the repository
2. Clone your fork: `git clone https://github.com/<your-username>/zse`
3. Install development dependencies:
   ```bash
   # Basic installation (CPU-only fallback if CUDA not available)
   pip install -e .
   
   # With all optional dependencies (recommended for development)
   pip install -e .[dev,cuda,training,enterprise,gguf]
   
   # Or install specific optional dependencies as needed:
   # pip install -e .[dev]        # Development tools (testing, linting, formatting)
   # pip install -e .[cuda]       # CUDA extension compilation tools
   # pip install -e .[training]   # Model training dependencies
   # pip install -e .[enterprise] # Enterprise features (Redis, databases, etc.)
   # pip install -e .[gguf]       # GGUF format support
   ```

## Setup Verification
After installation, verify your setup:
```bash
# Check if ZSE CLI is working
zse --help

# Verify installation by checking version
zse hardware  # Shows GPU/VRAM information if available

# Run basic tests
pytest tests/ -v  # If you have tests configured
```

## Development Environment
- **Python**: 3.11+ required
- **CUDA**: Recommended for GPU acceleration (not required for CPU-only development)
- **Git**: For version control
- **Editor**: Any modern code editor (VSCode, PyCharm, etc.)

## Submitting a PR
- Branch name: `feature/your-feature` or `fix/your-fix`
- Keep commits small and focused
- Write a clear PR description explaining what and why
- Ensure code follows existing style (we use Ruff for linting, Black for formatting)
- Add tests for new functionality when possible
- Update documentation if your changes affect usage

## Contact
Questions? Mail us at zse@zyoralabs.com

More sections coming soon — contributions to this doc are welcome!
