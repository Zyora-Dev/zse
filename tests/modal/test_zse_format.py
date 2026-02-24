"""
Test ZSE format conversion on Modal GPU.
"""
import modal
import time
import os

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = os.path.dirname(os.path.dirname(TESTS_DIR))

app = modal.App("zse-format-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "safetensors",
        "sentencepiece",
        "accelerate",
    )
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)


@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
)
def test_zse_format():
    """Test ZSE format conversion and loading."""
    import sys
    sys.path.insert(0, "/root/zse")
    
    import os
    import torch
    import tempfile
    from pathlib import Path
    
    results = []
    
    print("=" * 60)
    print("ZSE FORMAT CONVERSION TEST")
    print("=" * 60)
    
    # Test 1: Format imports
    print("\n📦 Test 1: Format module imports...")
    try:
        from zse.format import (
            ZSEHeader, ZSEReader, ZSEWriter,
            TensorInfo, LayerGroup, TensorDType,
            encode_header, decode_header, load_zse
        )
        print("  ✅ All format imports successful")
        results.append(("Format Imports", True, ""))
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        results.append(("Format Imports", False, str(e)))
        return results
    
    # Test 2: Header encode/decode
    print("\n📋 Test 2: Header encoding/decoding...")
    try:
        header = ZSEHeader(
            architecture='LlamaForCausalLM',
            model_type='llama',
            hidden_size=4096,
            num_hidden_layers=32,
            num_attention_heads=32,
            vocab_size=32000,
        )
        header.tensors.append(TensorInfo(
            name='model.embed_tokens.weight',
            shape=(32000, 4096),
            dtype=TensorDType.FLOAT16,
            offset=1000,
            size=262144000,
        ))
        
        encoded = encode_header(header)
        decoded, size = decode_header(encoded)
        
        assert decoded.architecture == 'LlamaForCausalLM'
        assert decoded.num_hidden_layers == 32
        assert len(decoded.tensors) == 1
        assert decoded.tensors[0].name == 'model.embed_tokens.weight'
        
        print(f"  ✅ Header encode/decode works")
        print(f"     Encoded size: {len(encoded)} bytes")
        results.append(("Header Encode/Decode", True, ""))
    except Exception as e:
        print(f"  ❌ Header test failed: {e}")
        results.append(("Header Encode/Decode", False, str(e)))
    
    # Test 3: Convert small model
    print("\n🔄 Test 3: Converting Qwen2.5-0.5B to .zse format...")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        
        # Load model and tokenizer
        print(f"  Loading {model_name}...")
        start = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map="cpu"  # Keep on CPU for conversion
        )
        load_time = time.time() - start
        print(f"  Loaded in {load_time:.2f}s")
        
        # Get model config
        config = model.config
        print(f"  Model: {config.architectures[0] if hasattr(config, 'architectures') else 'unknown'}")
        print(f"  Hidden size: {config.hidden_size}")
        print(f"  Layers: {config.num_hidden_layers}")
        print(f"  Vocab size: {config.vocab_size}")
        
        # Create .zse file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "model.zse"
            
            print(f"\n  Converting to .zse format...")
            start = time.time()
            
            writer = ZSEWriter(output_path)
            # Use convert_from_state_dict with the already-loaded model
            state_dict = model.state_dict()
            writer.convert_from_state_dict(state_dict, config, tokenizer)
            
            convert_time = time.time() - start
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            
            print(f"  ✅ Conversion complete!")
            print(f"     Time: {convert_time:.2f}s")
            print(f"     File size: {file_size:.2f} MB")
            
            results.append(("Model Conversion", True, f"{file_size:.1f}MB in {convert_time:.1f}s"))
            
            # Test 4: Read back the .zse file
            print("\n📖 Test 4: Reading .zse file...")
            try:
                reader = ZSEReader(output_path)
                
                # Check header
                assert reader.header.hidden_size == config.hidden_size
                assert reader.header.num_hidden_layers == config.num_hidden_layers
                assert reader.header.vocab_size == config.vocab_size
                
                print(f"  ✅ Header validated")
                print(f"     Architecture: {reader.header.architecture}")
                print(f"     Layers: {reader.header.num_hidden_layers}")
                print(f"     Tensors: {len(reader.header.tensors)}")
                
                # Test memory-mapped loading
                print("\n  Testing memory-mapped tensor loading...")
                state_dict = reader.load_state_dict()
                
                tensor_count = len(state_dict)
                total_params = sum(t.numel() for t in state_dict.values())
                
                print(f"  ✅ Loaded {tensor_count} tensors")
                print(f"     Total parameters: {total_params:,}")
                
                # Validate a few tensors
                sample_keys = list(state_dict.keys())[:3]
                for key in sample_keys:
                    tensor = state_dict[key]
                    print(f"     {key}: {tuple(tensor.shape)}")
                
                reader.close()
                results.append(("File Reading", True, f"{tensor_count} tensors, {total_params:,} params"))
                
            except Exception as e:
                import traceback
                print(f"  ❌ Read failed: {e}")
                traceback.print_exc()
                results.append(("File Reading", False, str(e)))
            
            # Test 5: Layer streaming
            print("\n🌊 Test 5: Layer streaming...")
            try:
                reader = ZSEReader(output_path)
                
                layer_count = 0
                for layer_idx, layer_tensors in reader.iter_layers():
                    layer_count += 1
                    if layer_idx <= 2:  # Print first few
                        print(f"     Layer {layer_idx}: {len(layer_tensors)} tensors")
                
                print(f"  ✅ Streamed {layer_count} layers")
                reader.close()
                results.append(("Layer Streaming", True, f"{layer_count} layers"))
                
            except Exception as e:
                print(f"  ❌ Streaming failed: {e}")
                results.append(("Layer Streaming", False, str(e)))
            
            # Test 6: Tokenizer preservation  
            print("\n🔤 Test 6: Tokenizer preservation...")
            try:
                reader = ZSEReader(output_path)
                
                restored_tokenizer = reader.load_tokenizer()
                
                # Test encoding
                test_text = "Hello, this is a test."
                original_tokens = tokenizer.encode(test_text)
                restored_tokens = restored_tokenizer.encode(test_text)
                
                assert original_tokens == restored_tokens, "Token mismatch!"
                
                print(f"  ✅ Tokenizer restored successfully")
                print(f"     Test text: '{test_text}'")
                print(f"     Tokens: {original_tokens}")
                
                reader.close()
                results.append(("Tokenizer Preservation", True, ""))
                
            except Exception as e:
                print(f"  ❌ Tokenizer test failed: {e}")
                results.append(("Tokenizer Preservation", False, str(e)))
            
            # Test 7: GPU loading
            print("\n🎮 Test 7: GPU tensor loading...")
            try:
                reader = ZSEReader(output_path)
                state_dict = reader.load_state_dict(device="cuda")
                
                # Check tensors are on GPU
                sample_tensor = list(state_dict.values())[0]
                assert sample_tensor.device.type == "cuda", "Tensor not on GPU!"
                
                print(f"  ✅ Loaded to GPU successfully")
                print(f"     Device: {sample_tensor.device}")
                print(f"     GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
                
                reader.close()
                results.append(("GPU Loading", True, f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"))
                
                # Cleanup
                del state_dict
                torch.cuda.empty_cache()
                
            except Exception as e:
                import traceback
                print(f"  ❌ GPU loading failed: {e}")
                traceback.print_exc()
                results.append(("GPU Loading", False, str(e)))
        
    except Exception as e:
        import traceback
        print(f"  ❌ Conversion failed: {e}")
        traceback.print_exc()
        results.append(("Model Conversion", False, str(e)))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, detail in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        detail_str = f" ({detail})" if detail else ""
        print(f"  {name}: {status}{detail_str}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    
    return results


@app.local_entrypoint()
def main():
    """Run ZSE format tests."""
    print("Starting ZSE format tests on Modal GPU...")
    results = test_zse_format.remote()
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n{'='*60}")
    print(f"Final Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ZSE format working correctly!")
    else:
        print("⚠️  Some tests failed - check output above")
