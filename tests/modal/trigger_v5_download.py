"""
Trigger v6 benchmark: Download models (H100 for fast network)
"""

import modal

def main():
    print("=" * 60)
    print("70B BENCHMARK V6 - Step 1: Download Models (H100)")
    print("=" * 60)
    
    try:
        download_fn = modal.Function.from_name("zse-70b-benchmark-v6", "download_models")
        
        print("Starting download on H100 (fast network)...")
        
        call = download_fn.spawn()
        
        print(f"\n✅ Download started!")
        print(f"   Call ID: {call.object_id}")
        print(f"\nView logs:")
        print(f"   modal app logs zse-70b-benchmark-v6")
        print(f"\nOnce download completes, run step 2:")
        print(f"   python tests/modal/trigger_v5_benchmark.py")
        
    except modal.exception.NotFoundError:
        print("❌ App not deployed. Run: modal deploy tests/modal/test_70b_v5.py")


if __name__ == "__main__":
    main()
