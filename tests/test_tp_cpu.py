"""Quick CPU test for tensor parallelism weight splitting."""

import torch
import torch.nn as nn
from zse.core.zdistributed.tensor_parallel import (
    ColumnParallelLinear, RowParallelLinear, TensorParallel, _replace_module
)

# Test ColumnParallelLinear
print('=== Column Parallel ===')
linear = nn.Linear(256, 512, bias=True)
linear.weight.data = torch.randn(512, 256, dtype=torch.float16)
linear.bias.data = torch.randn(512, dtype=torch.float16)

col0 = ColumnParallelLinear.from_linear(linear, tp_size=2, tp_rank=0)
col1 = ColumnParallelLinear.from_linear(linear, tp_size=2, tp_rank=1)
print(f'  Full weight: {linear.weight.shape}')
print(f'  Rank 0 shard: {col0.weight.shape}')
print(f'  Rank 1 shard: {col1.weight.shape}')
assert col0.weight.shape == (256, 256)
assert col1.weight.shape == (256, 256)
reconstructed = torch.cat([col0.weight.data, col1.weight.data], dim=0)
assert torch.allclose(reconstructed, linear.weight.data)
print('  Weight split: CORRECT')

# Forward pass
x = torch.randn(1, 10, 256, dtype=torch.float16)
full_out = linear(x)
shard0 = col0(x)
shard1 = col1(x)
reconstructed_out = torch.cat([shard0, shard1], dim=-1)
assert torch.allclose(full_out, reconstructed_out, atol=1e-3)
print('  Forward: CORRECT')

# Test RowParallelLinear
print('=== Row Parallel ===')
linear2 = nn.Linear(512, 256, bias=False)
linear2.weight.data = torch.randn(256, 512, dtype=torch.float16)

row0 = RowParallelLinear.from_linear(linear2, tp_size=2, tp_rank=0, reduce_output=False)
row1 = RowParallelLinear.from_linear(linear2, tp_size=2, tp_rank=1, reduce_output=False)
print(f'  Full weight: {linear2.weight.shape}')
print(f'  Rank 0 shard: {row0.weight.shape}')
print(f'  Rank 1 shard: {row1.weight.shape}')
assert row0.weight.shape == (256, 256)
assert row1.weight.shape == (256, 256)
reconstructed2 = torch.cat([row0.weight.data, row1.weight.data], dim=1)
assert torch.allclose(reconstructed2, linear2.weight.data)
print('  Weight split: CORRECT')

# Row parallel forward
x2 = torch.randn(1, 10, 512, dtype=torch.float16)
full_out2 = linear2(x2)
x2_0 = x2[..., :256]
x2_1 = x2[..., 256:]
p0 = row0(x2_0)
p1 = row1(x2_1)
summed = p0 + p1
assert torch.allclose(full_out2, summed, atol=0.1), f'Max diff: {(full_out2 - summed).abs().max()}'
print('  Forward: CORRECT')

# Test TensorParallel on a simple model
print('=== TensorParallel.apply ===')

class FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict({
            'embed_tokens': nn.Embedding(1000, 128),
            'layers': nn.ModuleList([
                nn.ModuleDict({
                    'self_attn': nn.ModuleDict({
                        'q_proj': nn.Linear(128, 128),
                        'k_proj': nn.Linear(128, 128),
                        'v_proj': nn.Linear(128, 128),
                        'o_proj': nn.Linear(128, 128),
                    }),
                    'mlp': nn.ModuleDict({
                        'gate_proj': nn.Linear(128, 256),
                        'up_proj': nn.Linear(128, 256),
                        'down_proj': nn.Linear(256, 128),
                    }),
                })
            ]),
        })
        self.lm_head = nn.Linear(128, 1000)
        # Fake config
        self.config = type('Config', (), {
            'num_attention_heads': 4,
            'num_key_value_heads': 4,
        })()

fake = FakeTransformer()
tp = TensorParallel(tp_size=2, tp_rank=0, tp_group=None)
tp.apply(fake)

# Check that layers were replaced
attn = fake.model['layers'][0]['self_attn']
mlp = fake.model['layers'][0]['mlp']
print(f'  q_proj type: {type(attn["q_proj"]).__name__}')
print(f'  o_proj type: {type(attn["o_proj"]).__name__}')
print(f'  gate_proj type: {type(mlp["gate_proj"]).__name__}')
print(f'  down_proj type: {type(mlp["down_proj"]).__name__}')
print(f'  lm_head type: {type(fake.lm_head).__name__}')
print(f'  num_attention_heads: {fake.config.num_attention_heads}')
assert isinstance(attn['q_proj'], ColumnParallelLinear)
assert isinstance(attn['o_proj'], RowParallelLinear)
assert isinstance(mlp['gate_proj'], ColumnParallelLinear)
assert isinstance(mlp['down_proj'], RowParallelLinear)
assert isinstance(fake.lm_head, ColumnParallelLinear)
assert fake.config.num_attention_heads == 2  # 4 / tp_size=2
print('  Layer replacement: CORRECT')

print()
print('ALL TESTS PASSED')
