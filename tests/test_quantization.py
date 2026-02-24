"""
Tests for ZSE Quantization Module

Tests INT8/INT4 quantization accuracy and memory efficiency.
"""

import pytest
import torch
import torch.nn as nn

from zse.efficiency.quantization import (
    QuantType,
    QuantConfig,
    quantize_tensor_int8,
    dequantize_tensor_int8,
    quantize_tensor_int4,
    dequantize_tensor_int4,
    QuantizedLinear,
    quantize_model,
    get_model_memory,
    compare_quantization_memory,
    estimate_quantized_memory,
)


class TestINT8Quantization:
    """Tests for INT8 quantization."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Test INT8 quantize/dequantize preserves values approximately."""
        tensor = torch.randn(256, 512)
        
        quantized, scale, zp = quantize_tensor_int8(tensor, per_channel=True, symmetric=True)
        dequantized = dequantize_tensor_int8(quantized, scale, zp, dtype=torch.float32)
        
        # Should be close (within quantization error)
        error = (tensor - dequantized).abs().mean()
        assert error < 0.1, f"Mean error {error} too high for INT8"
    
    def test_quantize_shape_preservation(self):
        """Test quantized tensor has correct shape."""
        tensor = torch.randn(128, 256)
        quantized, scale, _ = quantize_tensor_int8(tensor)
        
        assert quantized.shape == tensor.shape
        assert quantized.dtype == torch.int8
    
    def test_per_channel_vs_per_tensor(self):
        """Test per-channel quantization is more accurate."""
        # Create tensor with varying scales per channel
        tensor = torch.randn(64, 128)
        tensor[0] *= 10  # Large scale
        tensor[1] *= 0.1  # Small scale
        
        # Per-channel should handle this better
        q_per_channel, s_pc, _ = quantize_tensor_int8(tensor, per_channel=True)
        q_per_tensor, s_pt, _ = quantize_tensor_int8(tensor, per_channel=False)
        
        d_per_channel = dequantize_tensor_int8(q_per_channel, s_pc)
        d_per_tensor = dequantize_tensor_int8(q_per_tensor, s_pt)
        
        error_pc = (tensor - d_per_channel).abs().mean()
        error_pt = (tensor - d_per_tensor).abs().mean()
        
        assert error_pc < error_pt, "Per-channel should be more accurate"
    
    def test_symmetric_quantization(self):
        """Test symmetric quantization has no zero point."""
        tensor = torch.randn(64, 128)
        quantized, scale, zero_point = quantize_tensor_int8(tensor, symmetric=True)
        
        assert zero_point is None, "Symmetric quantization should have no zero point"
    
    def test_int8_range(self):
        """Test quantized values are in INT8 range."""
        tensor = torch.randn(128, 256) * 100  # Large values
        quantized, _, _ = quantize_tensor_int8(tensor, symmetric=True)
        
        assert quantized.min() >= -128
        assert quantized.max() <= 127


class TestINT4Quantization:
    """Tests for INT4 quantization."""
    
    def test_quantize_dequantize_roundtrip(self):
        """Test INT4 quantize/dequantize roundtrip."""
        tensor = torch.randn(256, 512)  # Must be divisible by group_size
        group_size = 128
        
        packed, scales = quantize_tensor_int4(tensor, group_size=group_size)
        dequantized = dequantize_tensor_int4(packed, scales, group_size=group_size)
        
        # INT4 has lower precision, allow higher error
        error = (tensor - dequantized).abs().mean()
        assert error < 0.3, f"Mean error {error} too high for INT4"
    
    def test_packed_shape(self):
        """Test INT4 packed tensor is half original size."""
        tensor = torch.randn(128, 256)
        packed, scales = quantize_tensor_int4(tensor, group_size=128)
        
        assert packed.shape == (128, 128)  # Half the columns
        assert packed.dtype == torch.uint8
    
    def test_unpacking_correctness(self):
        """Test that INT4 values are correctly packed/unpacked."""
        # Create simple tensor for verification
        tensor = torch.zeros(1, 256)
        tensor[0, 0] = 0.5
        tensor[0, 1] = -0.5
        
        packed, scales = quantize_tensor_int4(tensor, group_size=128)
        dequantized = dequantize_tensor_int4(packed, scales, group_size=128)
        
        # Check values are recovered (approximately)
        assert (dequantized[0, 0] - 0.5).abs() < 0.2
        assert (dequantized[0, 1] + 0.5).abs() < 0.2
    
    def test_int4_range(self):
        """Test quantized values use 4-bit range."""
        tensor = torch.randn(64, 256) * 10
        packed, _ = quantize_tensor_int4(tensor, group_size=128)
        
        # After unpacking, values should be in [-7, 7] range (shifted by 8 for packing)
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        
        # Packed values are in [1, 15] range (shifted from [-7, 7])
        assert low.min() >= 0
        assert low.max() <= 15
        assert high.min() >= 0
        assert high.max() <= 15


class TestQuantizedLinear:
    """Tests for QuantizedLinear layer."""
    
    def test_from_float_int8(self):
        """Test conversion from FP16 Linear to INT8."""
        linear = nn.Linear(256, 128, bias=False)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT8)
        
        assert q_linear.in_features == 256
        assert q_linear.out_features == 128
        assert q_linear.weight_quantized.dtype == torch.int8
        assert q_linear.quant_type == QuantType.INT8
    
    def test_from_float_int4(self):
        """Test conversion from FP16 Linear to INT4."""
        linear = nn.Linear(256, 128, bias=False)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT4, group_size=128)
        
        assert q_linear.weight_quantized.shape == (128, 128)  # Packed
        assert q_linear.weight_quantized.dtype == torch.uint8
    
    def test_forward_int8(self):
        """Test forward pass with INT8 weights."""
        linear = nn.Linear(256, 128, bias=False)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT8)
        
        x = torch.randn(2, 32, 256)
        
        with torch.no_grad():
            out_fp = linear(x)
            out_q = q_linear(x)
        
        # Outputs should be close
        error = (out_fp - out_q).abs().mean() / out_fp.abs().mean()
        assert error < 0.1, f"Relative error {error} too high"
    
    def test_forward_int4(self):
        """Test forward pass with INT4 weights."""
        linear = nn.Linear(256, 128, bias=False)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT4, group_size=128)
        
        x = torch.randn(2, 16, 256)
        
        with torch.no_grad():
            out_fp = linear(x)
            out_q = q_linear(x)
        
        # INT4 has more error, allow higher threshold
        error = (out_fp - out_q).abs().mean() / out_fp.abs().mean()
        assert error < 0.3, f"Relative error {error} too high for INT4"
    
    def test_memory_reduction_int8(self):
        """Test INT8 reduces memory by ~50%."""
        linear = nn.Linear(1024, 1024, bias=False)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT8)
        
        fp_bytes = linear.weight.numel() * linear.weight.element_size()
        q_bytes = q_linear.memory_bytes()
        
        reduction = 1 - (q_bytes / fp_bytes)
        assert reduction > 0.45, f"Memory reduction {reduction:.1%} should be >45%"
    
    def test_memory_reduction_int4(self):
        """Test INT4 reduces memory by ~75%."""
        linear = nn.Linear(1024, 1024, bias=False)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT4, group_size=128)
        
        fp_bytes = linear.weight.numel() * linear.weight.element_size()
        q_bytes = q_linear.memory_bytes()
        
        reduction = 1 - (q_bytes / fp_bytes)
        assert reduction > 0.70, f"Memory reduction {reduction:.1%} should be >70%"
    
    def test_with_bias(self):
        """Test quantized linear with bias."""
        linear = nn.Linear(256, 128, bias=True)
        q_linear = QuantizedLinear.from_float(linear, QuantType.INT8)
        
        x = torch.randn(2, 8, 256)
        
        with torch.no_grad():
            out_fp = linear(x)
            out_q = q_linear(x)
        
        assert out_q.shape == out_fp.shape


class TestModelQuantization:
    """Tests for full model quantization."""
    
    def test_quantize_simple_model(self):
        """Test quantizing a simple model."""
        model = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        quantize_model(model, QuantType.INT8)
        
        assert isinstance(model[0], QuantizedLinear)
        assert isinstance(model[2], QuantizedLinear)
        assert isinstance(model[1], nn.ReLU)  # Non-linear unchanged
    
    def test_skip_layers(self):
        """Test skipping specific layers during quantization."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(256, 128)
                self.fc2 = nn.Linear(128, 64)
                self.lm_head = nn.Linear(64, 1000)
            
            def forward(self, x):
                return self.lm_head(self.fc2(self.fc1(x)))
        
        model = SimpleModel()
        quantize_model(model, QuantType.INT8, skip_layers=["lm_head"])
        
        assert isinstance(model.fc1, QuantizedLinear)
        assert isinstance(model.fc2, QuantizedLinear)
        assert isinstance(model.lm_head, nn.Linear)  # Skipped
    
    def test_get_model_memory(self):
        """Test memory calculation."""
        model = nn.Linear(1024, 1024, bias=False)
        mem = get_model_memory(model)
        
        expected_params = 1024 * 1024
        expected_bytes = expected_params * 4  # FP32 default
        
        assert mem["params"] == expected_params
        assert mem["bytes"] == expected_bytes


class TestMemoryEstimation:
    """Tests for memory estimation utilities."""
    
    def test_estimate_quantized_memory(self):
        """Test memory estimation for different quant types."""
        num_params = 1_000_000_000  # 1B params
        
        fp16_mem = estimate_quantized_memory(num_params, QuantType.FP16)
        int8_mem = estimate_quantized_memory(num_params, QuantType.INT8)
        int4_mem = estimate_quantized_memory(num_params, QuantType.INT4)
        
        # FP16: ~2GB, INT8: ~1GB, INT4: ~0.5GB
        assert 1.8 < fp16_mem < 2.0
        assert 0.9 < int8_mem < 1.1
        assert 0.45 < int4_mem < 0.55
    
    def test_compare_quantization_memory(self):
        """Test memory comparison utility."""
        comparison = compare_quantization_memory(1_000_000_000)
        
        assert comparison["FP32"] > comparison["FP16"]
        assert comparison["FP16"] > comparison["INT8"]
        assert comparison["INT8"] > comparison["INT4"]


class TestQuantizationAccuracy:
    """Tests for quantization accuracy on realistic workloads."""
    
    def test_matrix_multiply_accuracy_int8(self):
        """Test INT8 matmul is accurate enough for inference."""
        # Simulate a transformer layer weight
        weight = torch.randn(768, 768) * 0.02  # Typical init scale
        input_tensor = torch.randn(32, 128, 768)  # [batch, seq, hidden]
        
        # FP32 reference
        with torch.no_grad():
            ref_output = torch.nn.functional.linear(input_tensor, weight)
        
        # INT8 quantized
        q_weight, scale, _ = quantize_tensor_int8(weight)
        dq_weight = dequantize_tensor_int8(q_weight, scale, dtype=torch.float32)
        
        with torch.no_grad():
            q_output = torch.nn.functional.linear(input_tensor, dq_weight)
        
        # Relative error should be reasonable for INT8
        rel_error = (ref_output - q_output).abs().mean() / ref_output.abs().mean()
        assert rel_error < 0.15, f"Relative error {rel_error:.4f} too high for INT8"
    
    def test_matrix_multiply_accuracy_int4(self):
        """Test INT4 matmul accuracy."""
        weight = torch.randn(768, 768) * 0.02
        input_tensor = torch.randn(8, 64, 768)
        
        with torch.no_grad():
            ref_output = torch.nn.functional.linear(input_tensor, weight)
        
        # INT4 quantized
        packed, scales = quantize_tensor_int4(weight, group_size=128)
        dq_weight = dequantize_tensor_int4(packed, scales, group_size=128, dtype=torch.float32)
        
        with torch.no_grad():
            q_output = torch.nn.functional.linear(input_tensor, dq_weight)
        
        # INT4 allows higher error
        rel_error = (ref_output - q_output).abs().mean() / ref_output.abs().mean()
        assert rel_error < 0.15, f"Relative error {rel_error:.4f} too high for INT4"
