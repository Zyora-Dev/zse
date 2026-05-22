"""ZSE data types — maps Python types to backend-specific GPU types."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DType:
    name: str
    size_bits: int
    cuda_type: str
    hip_type: str
    metal_type: str

    def size_bytes(self) -> int:
        return (self.size_bits + 7) // 8

    def __repr__(self) -> str:
        return f"zse.{self.name}"


# Core types
float32 = DType(name="float32", size_bits=32, cuda_type="float", hip_type="float", metal_type="float")
float16 = DType(name="float16", size_bits=16, cuda_type="half", hip_type="half", metal_type="half")
bfloat16 = DType(name="bfloat16", size_bits=16, cuda_type="__nv_bfloat16", hip_type="hip_bfloat16", metal_type="bfloat")
int32 = DType(name="int32", size_bits=32, cuda_type="int", hip_type="int", metal_type="int")
int16 = DType(name="int16", size_bits=16, cuda_type="short", hip_type="short", metal_type="short")
int8 = DType(name="int8", size_bits=8, cuda_type="int8_t", hip_type="int8_t", metal_type="char")
uint32 = DType(name="uint32", size_bits=32, cuda_type="unsigned int", hip_type="unsigned int", metal_type="uint")
uint16 = DType(name="uint16", size_bits=16, cuda_type="unsigned short", hip_type="unsigned short", metal_type="ushort")
uint8 = DType(name="uint8", size_bits=8, cuda_type="uint8_t", hip_type="uint8_t", metal_type="uchar")

# Quantized types — packed representations
int4 = DType(name="int4", size_bits=4, cuda_type="int8_t", hip_type="int8_t", metal_type="char")  # 2 values per byte
uint4 = DType(name="uint4", size_bits=4, cuda_type="uint8_t", hip_type="uint8_t", metal_type="uchar")  # 2 values per byte


DTYPE_MAP = {
    "float32": float32,
    "float16": float16,
    "bfloat16": bfloat16,
    "int32": int32,
    "int16": int16,
    "int8": int8,
    "uint32": uint32,
    "uint16": uint16,
    "uint8": uint8,
    "int4": int4,
    "uint4": uint4,
    # Aliases
    "fp32": float32,
    "fp16": float16,
    "bf16": bfloat16,
}


def from_string(name: str) -> DType:
    if name not in DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {name}. Available: {list(DTYPE_MAP.keys())}")
    return DTYPE_MAP[name]
