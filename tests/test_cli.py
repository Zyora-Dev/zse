"""
ZSE CLI Tests

Tests for the command-line interface.
"""

import pytest
from typer.testing import CliRunner

from zse.api.cli.main import app
from zse.version import __version__


runner = CliRunner()


class TestCLIBasics:
    """Test basic CLI functionality."""
    
    def test_version_flag(self) -> None:
        """Test --version flag shows version."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.stdout
    
    def test_version_short_flag(self) -> None:
        """Test -v flag shows version."""
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert __version__ in result.stdout
    
    def test_help(self) -> None:
        """Test --help shows help text."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "ZSE" in result.stdout
        assert "serve" in result.stdout
        assert "chat" in result.stdout
    
    def test_no_args_shows_interactive_banner(self) -> None:
        """Test no arguments shows interactive banner."""
        result = runner.invoke(app, [], input="n\n")  # Answer 'n' to interactive prompt
        # Banner should show even if exit code varies
        assert "ZSE" in result.stdout or "Z Server Engine" in result.stdout


class TestServeCommand:
    """Test the serve command."""
    
    def test_serve_help(self) -> None:
        """Test serve --help."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "serve" in result.stdout.lower()
        assert "--host" in result.stdout
        assert "--port" in result.stdout
        assert "--efficiency" in result.stdout
    
    def test_serve_requires_model(self) -> None:
        """Test serve requires model argument."""
        result = runner.invoke(app, ["serve"])
        assert result.exit_code != 0
        assert "Missing argument" in result.stdout or "MODEL" in result.stdout
    
    def test_serve_with_model(self) -> None:
        """Test serve with model argument."""
        result = runner.invoke(app, ["serve", "test-model"])
        # Should not error (just show pending message for now)
        assert result.exit_code == 0
        assert "test-model" in result.stdout
    
    def test_serve_efficiency_modes(self) -> None:
        """Test serve with different efficiency modes."""
        for mode in ["speed", "balanced", "memory", "ultra"]:
            result = runner.invoke(app, ["serve", "test-model", "--efficiency", mode])
            assert result.exit_code == 0
            assert mode in result.stdout


class TestChatCommand:
    """Test the chat command."""
    
    def test_chat_help(self) -> None:
        """Test chat --help."""
        result = runner.invoke(app, ["chat", "--help"])
        assert result.exit_code == 0
        assert "chat" in result.stdout.lower()
    
    def test_chat_requires_model(self) -> None:
        """Test chat requires model argument."""
        result = runner.invoke(app, ["chat"])
        assert result.exit_code != 0


class TestConvertCommand:
    """Test the convert command."""
    
    def test_convert_help(self) -> None:
        """Test convert --help."""
        result = runner.invoke(app, ["convert", "--help"])
        assert result.exit_code == 0
        assert "convert" in result.stdout.lower()
        assert "--output" in result.stdout
        assert "--quantization" in result.stdout
    
    def test_convert_requires_output(self) -> None:
        """Test convert requires --output."""
        result = runner.invoke(app, ["convert", "test-model"])
        assert result.exit_code != 0


class TestInfoCommand:
    """Test the info command."""
    
    def test_info_help(self) -> None:
        """Test info --help."""
        result = runner.invoke(app, ["info", "--help"])
        assert result.exit_code == 0
        assert "info" in result.stdout.lower()
    
    def test_info_with_model(self) -> None:
        """Test info with model argument."""
        result = runner.invoke(app, ["info", "test-model"])
        assert result.exit_code == 0


class TestBenchmarkCommand:
    """Test the benchmark command."""
    
    def test_benchmark_help(self) -> None:
        """Test benchmark --help."""
        result = runner.invoke(app, ["benchmark", "--help"])
        assert result.exit_code == 0
        assert "benchmark" in result.stdout.lower()
        assert "--batch-size" in result.stdout
        assert "--iterations" in result.stdout


class TestHardwareCommand:
    """Test the hardware command."""
    
    def test_hardware_shows_info(self) -> None:
        """Test hardware command shows system info."""
        result = runner.invoke(app, ["hardware"])
        assert result.exit_code == 0
        assert "CPU" in result.stdout or "RAM" in result.stdout
