"""
Trigger 70B benchmark in detached mode.

Usage:
1. First deploy: modal deploy tests/modal/test_70b_benchmark.py
2. Then trigger: python tests/modal/trigger_70b_benchmark.py

Check results at Modal dashboard or: modal app logs zse-70b-benchmark
"""

import modal

def main():
    print("=" * 60)
    print("70B BENCHMARK TRIGGER (Detached Mode)")
    print("=" * 60)
    
    print("\nLooking up deployed benchmark function...")
    
    try:
        # Get the deployed function
        benchmark_fn = modal.Function.from_name("zse-70b-benchmark-v4", "benchmark_70b")
        
        print("Spawning benchmark in detached mode...")
        print("(This returns immediately, benchmark runs server-side)")
        
        # Spawn detached - returns immediately
        call = benchmark_fn.spawn()
        
        print(f"\n✅ Benchmark triggered successfully!")
        print(f"   Function call ID: {call.object_id}")
        print(f"\n" + "=" * 60)
        print("HOW TO VIEW RESULTS:")
        print("=" * 60)
        print(f"\n1. Modal Dashboard:")
        print(f"   https://modal.com/apps")
        print(f"\n2. Command line logs:")
        print(f"   modal app logs zse-70b-benchmark-v4")
        print(f"\n3. Wait ~15-30 minutes for 70B model benchmark to complete")
        print("=" * 60)
        
    except modal.exception.NotFoundError:
        print("\n❌ Function not deployed yet!")
        print("\nFirst deploy the benchmark:")
        print("   modal deploy tests/modal/test_70b_benchmark.py")
        print("\nThen run this trigger again.")


if __name__ == "__main__":
    main()
