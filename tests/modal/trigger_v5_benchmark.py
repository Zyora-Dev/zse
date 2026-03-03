"""
Trigger v6 benchmark Step 2: Run Benchmark (H100)
"""

import modal

def main():
    print("=" * 60)
    print("70B BENCHMARK V6 - Step 2: Run Benchmark (H100)")
    print("=" * 60)
    
    try:
        benchmark_fn = modal.Function.from_name("zse-70b-benchmark-v6", "run_benchmark")
        
        print("Starting benchmark on H100...")
        
        call = benchmark_fn.spawn()
        
        print(f"\n✅ Benchmark started!")
        print(f"   Call ID: {call.object_id}")
        print(f"\nView logs:")
        print(f"   modal app logs zse-70b-benchmark-v6")
        
    except modal.exception.NotFoundError:
        print("❌ App not deployed. Run: modal deploy tests/modal/test_70b_v5.py")


if __name__ == "__main__":
    main()
