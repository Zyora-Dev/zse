"""
Trigger the deployed benchmark function.
"""

import modal

app = modal.App("benchmark-trigger")

@app.local_entrypoint()
def main():
    print("Looking up deployed benchmark function...")
    
    # Get the deployed function
    benchmark_fn = modal.Function.from_name("zse-vs-llamacpp-benchmark", "test_llamacpp_vs_zse")
    
    print("Triggering benchmark (runs server-side)...")
    print("This will run completely on Modal's servers.")
    
    # Spawn detached - returns immediately 
    call = benchmark_fn.spawn()
    
    print(f"\n✅ Benchmark triggered!")
    print(f"   Function call ID: {call.object_id}")
    print(f"\nView results at:")
    print(f"   https://modal.com/apps/zyoralabsai/main/deployed/zse-vs-llamacpp-benchmark")
    print(f"\nOr run: modal app logs zse-vs-llamacpp-benchmark")
