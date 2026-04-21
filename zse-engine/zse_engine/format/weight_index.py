"""ZSE Weight Index — Metadata for each tensor in the weight data section."""

from dataclasses import dataclass
from typing import List, Tuple, Optional

from zse_engine.format import serializer
from zse_engine.format.spec import QuantMethod


@dataclass
class WeightEntry:
    """Metadata for one tensor in the .zse file."""
    name: str                        # e.g. "layers.0.self_attn.q_proj.weight"
    shape: Tuple[int, ...]           # Original shape
    dtype: str                       # "int4" or "float16"
    quant_method: int = QuantMethod.NONE
    group_size: int = 128

    # Offsets relative to start of WEIGHT_DATA section
    data_offset: int = 0
    data_nbytes: int = 0
    scale_offset: int = 0
    scale_nbytes: int = 0
    zeros_offset: int = 0
    zeros_nbytes: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "quant_method": self.quant_method,
            "group_size": self.group_size,
            "data_offset": self.data_offset,
            "data_nbytes": self.data_nbytes,
            "scale_offset": self.scale_offset,
            "scale_nbytes": self.scale_nbytes,
            "zeros_offset": self.zeros_offset,
            "zeros_nbytes": self.zeros_nbytes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'WeightEntry':
        d = dict(d)
        d["shape"] = tuple(d["shape"])
        return cls(**d)

    @property
    def num_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


class WeightIndex:
    """Collection of all weight entries in the model."""

    def __init__(self):
        self.entries: List[WeightEntry] = []

    def add(self, entry: WeightEntry):
        self.entries.append(entry)

    def find(self, name: str) -> Optional[WeightEntry]:
        for e in self.entries:
            if e.name == name:
                return e
        return None

    def serialize(self) -> bytes:
        data = [e.to_dict() for e in self.entries]
        return serializer.encode(data)

    @classmethod
    def deserialize(cls, data: bytes) -> 'WeightIndex':
        entries_data = serializer.decode(data)
        index = cls()
        for d in entries_data:
            index.add(WeightEntry.from_dict(d))
        return index

    @property
    def total_data_bytes(self) -> int:
        if not self.entries:
            return 0
        last = self.entries[-1]
        # Find the max end offset across all entries
        max_end = 0
        for e in self.entries:
            ends = [
                e.data_offset + e.data_nbytes,
                e.scale_offset + e.scale_nbytes if e.scale_nbytes > 0 else 0,
                e.zeros_offset + e.zeros_nbytes if e.zeros_nbytes > 0 else 0,
            ]
            for end in ends:
                if end > max_end:
                    max_end = end
        return max_end

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self):
        return iter(self.entries)
