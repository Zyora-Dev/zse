"""
ZSE Test Configuration

Pytest fixtures and configuration for ZSE tests.
"""

import os
import sys
from pathlib import Path
from typing import Generator

import pytest


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest."""
    # Register custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "cuda: marks tests requiring CUDA GPU"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def has_cuda() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.fixture(scope="session")
def cuda_device_count() -> int:
    """Return number of CUDA devices."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.device_count()
    except ImportError:
        pass
    return 0


@pytest.fixture(scope="session")
def gpu_memory_gb() -> float:
    """Return total GPU memory in GB (first GPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024**3)
    except ImportError:
        pass
    return 0.0


@pytest.fixture
def temp_model_dir(tmp_path: Path) -> Path:
    """Create a temporary directory for model files."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    return model_dir


@pytest.fixture
def temp_config_file(tmp_path: Path) -> Path:
    """Create a temporary config file."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("""
mode: dev
server:
  host: 127.0.0.1
  port: 8000
""")
    return config_path


def skip_if_no_cuda(func):
    """Decorator to skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not _cuda_available(),
        reason="CUDA not available"
    )(func)


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False
