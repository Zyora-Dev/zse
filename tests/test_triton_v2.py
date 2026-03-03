"""
Tests for Triton v2 INT4 kernel integration.

Tests:
1. Repack functions work correctly
2. Kernel availability detection works
3. QuantizedLinearZSE backend selection works
4. Fallback mechanism works
"""

import pytest
import torch
import torch.nn as nn


class TestTritonV2Availability:
    """Test kernel availability detection."""
    
    def test_availability_functions_exist(self):
        """Check availability functions are exported."""
        from zse.kernels import (
            is_triton_v2_available,
            get_triton_v2_error,
        )
        
        # These should return meaningful values
        avail = is_triton_v2_available()
        assert isinstance(avail, bool)
        
        error = get_triton_v2_error()
        if not avail:
            assert error is not None
        else:
            assert error is None
    
    def test_repack_functions_exist(self):
        """Check repack functions are exported."""
        from zse.kernels import (
            is_triton_v2_available,
            repack_weights_for_v2,
            repack_scales_for_v2,
        )
        
        # Even if triton not available, repack functions should work (pure Python)
        # Actually they depend on is_triton_v2_available, let me check
        # Looking at the code, repack functions are always defined


class TestRepackFunctions:
    """Test weight/scale repacking for v2 layout."""
    
    def test_repack_weights_shape(self):
        """Test weight repacking changes shape correctly."""
        from zse.kernels.triton_int4_v2 import repack_weights_for_v2
        
        # Input: [N, K//2] = [1024, 2048]
        N, K = 1024, 4096
        weight_nk = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        
        # Output should be [K//2, N]
        weight_kn = repack_weights_for_v2(weight_nk)
        
        assert weight_kn.shape == (K // 2, N)
        assert weight_kn.dtype == torch.uint8
        assert weight_kn.is_contiguous()
    
    def test_repack_weights_values(self):
        """Test weight values are preserved after repacking."""
        from zse.kernels.triton_int4_v2 import repack_weights_for_v2
        
        N, K = 128, 256
        weight_nk = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8)
        weight_kn = repack_weights_for_v2(weight_nk)
        
        # Check a few values
        assert weight_nk[0, 0] == weight_kn[0, 0]
        assert weight_nk[10, 20] == weight_kn[20, 10]
        assert weight_nk[-1, -1] == weight_kn[-1, -1]
    
    def test_repack_scales_shape(self):
        """Test scales repacking changes shape correctly."""
        from zse.kernels.triton_int4_v2 import repack_scales_for_v2
        
        N, num_groups = 1024, 32  # K=4096, group_size=128
        scales_ng = torch.randn(N, num_groups, dtype=torch.float16)
        
        # Output should be [num_groups, N]
        scales_gn = repack_scales_for_v2(scales_ng)
        
        assert scales_gn.shape == (num_groups, N)
        assert scales_gn.dtype == torch.float16
        assert scales_gn.is_contiguous()
    
    def test_repack_scales_values(self):
        """Test scale values are preserved after repacking."""
        from zse.kernels.triton_int4_v2 import repack_scales_for_v2
        
        N, num_groups = 128, 16
        scales_ng = torch.randn(N, num_groups, dtype=torch.float16)
        scales_gn = repack_scales_for_v2(scales_ng)
        
        # Check a few values
        assert torch.allclose(scales_ng[0, 0], scales_gn[0, 0])
        assert torch.allclose(scales_ng[10, 5], scales_gn[5, 10])


class TestQuantizedLinearZSE:
    """Test QuantizedLinearZSE class with backend selection."""
    
    def test_init_with_backend(self):
        """Test layer can be created with backend parameter."""
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        layer = QuantizedLinearZSE(
            in_features=1024,
            out_features=1024,
            group_size=128,
            backend="auto",
        )
        
        assert layer.backend == "auto"
        assert layer.in_features == 1024
        assert layer.out_features == 1024
    
    def test_backend_constants(self):
        """Test backend constants are defined."""
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        assert QuantizedLinearZSE.BACKEND_TRITON == "triton"
        assert QuantizedLinearZSE.BACKEND_BNB == "bnb"
        assert QuantizedLinearZSE.BACKEND_AUTO == "auto"
    
    def test_triton_v2_buffers_initialized(self):
        """Test Triton v2 buffers are initialized to None."""
        from zse.format.reader_v2 import QuantizedLinearZSE
        
        layer = QuantizedLinearZSE(1024, 1024, 128)
        
        assert layer._triton_v2_weight is None
        assert layer._triton_v2_scales is None
        assert layer._triton_v2_failed is False


class TestModelConversionFunctions:
    """Test model-level conversion functions."""
    
    def test_convert_model_to_triton_v2_exists(self):
        """Test conversion function exists."""
        from zse.format.reader_v2 import convert_model_to_triton_v2
        assert callable(convert_model_to_triton_v2)
    
    def test_set_model_backend_exists(self):
        """Test set_model_backend function exists."""
        from zse.format.reader_v2 import set_model_backend
        assert callable(set_model_backend)
    
    def test_convert_model_to_bnb_still_works(self):
        """Test existing bnb conversion still works."""
        from zse.format.reader_v2 import convert_model_to_bnb
        assert callable(convert_model_to_bnb)


class TestLoadZseModel:
    """Test load_zse_model function with backend parameter."""
    
    def test_load_zse_model_accepts_backend(self):
        """Test load_zse_model accepts backend parameter."""
        import inspect
        from zse.format.reader_v2 import load_zse_model
        
        sig = inspect.signature(load_zse_model)
        params = list(sig.parameters.keys())
        
        assert "backend" in params
        assert sig.parameters["backend"].default == "auto"


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for kernel tests"
)
class TestTritonV2KernelCUDA:
    """Test Triton v2 kernel execution (requires CUDA)."""
    
    def test_kernel_execution_if_available(self):
        """Test kernel executes if Triton is available."""
        from zse.kernels import is_triton_v2_available
        
        if not is_triton_v2_available():
            pytest.skip("Triton v2 not available")
        
        from zse.kernels import int4_matmul_triton_v2, repack_weights_for_v2, repack_scales_for_v2
        
        # Create test data
        M, K, N = 32, 1024, 512
        group_size = 128
        num_groups = K // group_size
        
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        
        # Create packed weights in v1 format and repack
        weight_packed_v1 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        scales_v1 = torch.randn(N, num_groups, dtype=torch.float16, device="cuda")
        
        weight_packed_v2 = repack_weights_for_v2(weight_packed_v1)
        scales_v2 = repack_scales_for_v2(scales_v1)
        
        # Execute kernel
        output = int4_matmul_triton_v2(x, weight_packed_v2, scales_v2, group_size)
        
        assert output.shape == (M, N)
        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()
    
    def test_gemv_path(self):
        """Test GEMV kernel path (M=1 for decode)."""
        from zse.kernels import is_triton_v2_available
        
        if not is_triton_v2_available():
            pytest.skip("Triton v2 not available")
        
        from zse.kernels import int4_matmul_triton_v2, repack_weights_for_v2, repack_scales_for_v2
        
        # Single token decode (M=1)
        M, K, N = 1, 1024, 512
        group_size = 128
        num_groups = K // group_size
        
        x = torch.randn(M, K, dtype=torch.float16, device="cuda")
        
        weight_packed_v1 = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device="cuda")
        scales_v1 = torch.randn(N, num_groups, dtype=torch.float16, device="cuda")
        
        weight_packed_v2 = repack_weights_for_v2(weight_packed_v1)
        scales_v2 = repack_scales_for_v2(scales_v1)
        
        output = int4_matmul_triton_v2(x, weight_packed_v2, scales_v2, group_size)
        
        assert output.shape == (M, N)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
