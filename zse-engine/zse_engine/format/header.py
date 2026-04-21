"""ZSE File Header — Pack/unpack the 64-byte header and section table."""

import struct
import zlib
from dataclasses import dataclass
from typing import List

from zse_engine.format.spec import (
    MAGIC, VERSION, HEADER_SIZE, SECTION_ENTRY_SIZE, SectionType, Flags,
)


@dataclass
class SectionEntry:
    """One entry in the section table."""
    type: SectionType
    offset: int = 0
    size: int = 0
    crc32: int = 0

    def pack(self) -> bytes:
        """Pack to 32 bytes."""
        return struct.pack('<I Q Q I 8x',
                           self.type, self.offset, self.size, self.crc32)

    @classmethod
    def unpack(cls, data: bytes) -> 'SectionEntry':
        type_val, offset, size, crc = struct.unpack_from('<I Q Q I', data)
        return cls(type=SectionType(type_val), offset=offset, size=size, crc32=crc)


@dataclass
class FileHeader:
    """The 64-byte fixed header."""
    version: int = VERSION
    total_size: int = 0
    flags: int = 0
    num_sections: int = 0

    def pack(self) -> bytes:
        """Pack to exactly 64 bytes."""
        header = struct.pack('<4s I Q I I',
                             MAGIC, self.version, self.total_size,
                             self.flags, self.num_sections)
        # Pad to 64 bytes
        return header + b'\x00' * (HEADER_SIZE - len(header))

    @classmethod
    def unpack(cls, data: bytes) -> 'FileHeader':
        magic, version, total_size, flags, num_sections = struct.unpack_from('<4s I Q I I', data)
        if magic != MAGIC:
            raise ValueError(f"Not a .zse file: magic={magic!r}, expected {MAGIC!r}")
        if version > VERSION:
            raise ValueError(f"Unsupported .zse version {version}, max supported={VERSION}")
        return cls(version=version, total_size=total_size,
                   flags=flags, num_sections=num_sections)


def read_header_and_sections(data: bytes) -> tuple:
    """Read header + section table from file data.

    Returns: (FileHeader, List[SectionEntry])
    """
    header = FileHeader.unpack(data[:HEADER_SIZE])

    sections = []
    offset = HEADER_SIZE
    for _ in range(header.num_sections):
        entry = SectionEntry.unpack(data[offset:offset + SECTION_ENTRY_SIZE])
        sections.append(entry)
        offset += SECTION_ENTRY_SIZE

    return header, sections


def compute_crc32(data: bytes) -> int:
    """Compute CRC32 checksum."""
    return zlib.crc32(data) & 0xFFFFFFFF
