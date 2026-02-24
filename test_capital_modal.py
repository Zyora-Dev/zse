"""Quick Modal test: What is the capital of France?"""

import modal
import os

# Get ZSE root for mounting
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ZSE_ROOT = DEPLOY_DIR

app = modal.App("zse-capital-test")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "accelerate>=0.25.0",
    )
    .add_local_dir(ZSE_ROOT, remote_path="/root/zse")
)

@app.function(image=image, gpu="T4", timeout=300)
def ask_capital():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import sys
    sys.path.insert(0, "/root/zse")
    
    from zse.engine.generation import TextGenerator, SamplingParams
    
    print("Loading TinyLlama-1.1B...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    
    print(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
    
    # Create ZSE generator
    generator = TextGenerator(model, tokenizer, device="cuda")
    
    # Format as chat
    prompt = """<|system|>
You are a helpful assistant.</s>
<|user|>
What is the capital of France?</s>
<|assistant|>
"""
    
    params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_new_tokens=50,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    print("\n🔮 Question: What is the capital of France?")
    print("-" * 40)
    print("Answer: ", end="", flush=True)
    
    # Stream the output using ZSE
    answer_tokens = []
    for chunk in generator.generate_stream(prompt, params):
        print(chunk.text, end="", flush=True)
        answer_tokens.append(chunk.token_id)
    
    print("\n" + "-" * 40)
    
    return {"answer": "".join(tokenizer.decode(answer_tokens))}

@app.local_entrypoint()
def main():
    result = ask_capital.remote()
    print(f"\nResult: {result}")
