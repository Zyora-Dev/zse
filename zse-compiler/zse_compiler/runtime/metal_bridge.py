"""ZSE Metal Runtime Bridge — Launch Metal kernels on Apple GPUs.

Uses ctypes + objc_msgSend to interact with Metal framework directly.
No PyObjC dependency, no Swift — pure ctypes to Objective-C runtime.

The approach:
1. Load Metal framework via ctypes
2. Get default MTLDevice
3. Create MTLLibrary from compiled .metallib
4. Create MTLComputePipelineState
5. Create MTLCommandQueue → MTLCommandBuffer → MTLComputeCommandEncoder
6. Dispatch threadgroups
"""

import ctypes
import ctypes.util
import platform
import os
from typing import Tuple, Optional
from dataclasses import dataclass


def _load_objc_runtime():
    """Load the Objective-C runtime."""
    path = ctypes.util.find_library("objc")
    if path is None:
        return None
    return ctypes.CDLL(path)


def _load_metal_framework():
    """Load the Metal framework."""
    try:
        return ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
    except OSError:
        return None


# ObjC runtime types
_objc = _load_objc_runtime()

if _objc:
    # Setup objc_msgSend
    _objc.objc_getClass.restype = ctypes.c_void_p
    _objc.objc_getClass.argtypes = [ctypes.c_char_p]
    _objc.sel_registerName.restype = ctypes.c_void_p
    _objc.sel_registerName.argtypes = [ctypes.c_char_p]

    # objc_msgSend — the universal ObjC message dispatch
    _msg = _objc.objc_msgSend
    _msg.restype = ctypes.c_void_p
    _msg.argtypes = [ctypes.c_void_p, ctypes.c_void_p]


def _sel(name: str) -> ctypes.c_void_p:
    """Create an ObjC selector."""
    return _objc.sel_registerName(name.encode())


def _cls(name: str) -> ctypes.c_void_p:
    """Get an ObjC class."""
    return _objc.objc_getClass(name.encode())


class MetalDevice:
    """Wraps a Metal GPU device via ObjC runtime."""

    def __init__(self):
        if not _objc or platform.system() != "Darwin":
            raise RuntimeError("Metal is only available on macOS")

        _load_metal_framework()

        # MTLCreateSystemDefaultDevice()
        metal = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        self._device = metal.MTLCreateSystemDefaultDevice()
        if not self._device:
            raise RuntimeError("No Metal device found")

        # Get device name
        name_obj = ctypes.c_void_p(_msg(self._device, _sel("name")))
        if name_obj:
            _objc.objc_msgSend.restype = ctypes.c_char_p
            utf8_sel = _sel("UTF8String")
            name_bytes = _msg(name_obj, utf8_sel)
            _objc.objc_msgSend.restype = ctypes.c_void_p
            self.name = ctypes.string_at(name_bytes).decode() if name_bytes else "Apple GPU"
        else:
            self.name = "Apple GPU"

        # Create command queue
        self._queue = ctypes.c_void_p(_msg(self._device, _sel("newCommandQueue")))

    def allocate_buffer(self, nbytes: int) -> ctypes.c_void_p:
        """Allocate a Metal buffer (MTLBuffer)."""
        # [device newBufferWithLength:nbytes options:MTLResourceStorageModeShared]
        # Need a typed objc_msgSend with correct argument types
        _send = _objc.objc_msgSend
        _send.restype = ctypes.c_void_p
        _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_uint64, ctypes.c_uint64]
        # MTLResourceStorageModeShared = 0
        buf = _send(self._device, _sel("newBufferWithLength:options:"), nbytes, 0)
        # Reset argtypes for other calls
        _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        return ctypes.c_void_p(buf)

    def buffer_contents(self, buffer: ctypes.c_void_p) -> ctypes.c_void_p:
        """Get CPU-accessible pointer to buffer contents."""
        return ctypes.c_void_p(_msg(buffer, _sel("contents")))

    def launch_kernel(
        self,
        metallib_path: str,
        kernel_name: str,
        buffers: list,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
    ):
        """Load a compiled .metallib and launch a kernel."""
        # Load library from file
        ns_string_cls = _cls("NSString")
        path_str = ctypes.c_void_p(_msg(
            ctypes.c_void_p(_msg(ns_string_cls, _sel("alloc"))),
            _sel("initWithUTF8String:"),
            metallib_path.encode()
        ))

        ns_url_cls = _cls("NSURL")
        url = ctypes.c_void_p(_msg(
            ns_url_cls, _sel("fileURLWithPath:"), path_str
        ))

        # newLibraryWithURL:error:
        error = ctypes.c_void_p(0)
        library = ctypes.c_void_p(_msg(
            self._device, _sel("newLibraryWithURL:error:"), url, ctypes.byref(error)
        ))
        if not library:
            raise RuntimeError(f"Failed to load Metal library: {metallib_path}")

        # Get function from library
        func_name = ctypes.c_void_p(_msg(
            ctypes.c_void_p(_msg(ns_string_cls, _sel("alloc"))),
            _sel("initWithUTF8String:"),
            kernel_name.encode()
        ))
        function = ctypes.c_void_p(_msg(
            library, _sel("newFunctionWithName:"), func_name
        ))
        if not function:
            raise RuntimeError(f"Kernel function '{kernel_name}' not found in library")

        # Create compute pipeline state
        pipeline = ctypes.c_void_p(_msg(
            self._device, _sel("newComputePipelineStateWithFunction:error:"),
            function, ctypes.byref(error)
        ))

        # Create command buffer + encoder
        cmd_buffer = ctypes.c_void_p(_msg(self._queue, _sel("commandBuffer")))
        encoder = ctypes.c_void_p(_msg(cmd_buffer, _sel("computeCommandEncoder")))

        # Set pipeline state
        _msg(encoder, _sel("setComputePipelineState:"), pipeline)

        # Set buffers
        for i, buf in enumerate(buffers):
            # [encoder setBuffer:buf offset:0 atIndex:i]
            _send = _objc.objc_msgSend
            _send.restype = None
            _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64]
            _send(encoder, _sel("setBuffer:offset:atIndex:"), buf, 0, i)
            _send.restype = ctypes.c_void_p
            _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        # Dispatch
        # MTLSize is a struct {uint64, uint64, uint64}
        class MTLSize(ctypes.Structure):
            _fields_ = [("width", ctypes.c_uint64),
                        ("height", ctypes.c_uint64),
                        ("depth", ctypes.c_uint64)]

        grid_size = MTLSize(*grid)
        block_size = MTLSize(*block)

        _send = _objc.objc_msgSend
        _send.restype = None
        _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                          MTLSize, MTLSize]
        _send(
            encoder,
            _sel("dispatchThreadgroups:threadsPerThreadgroup:"),
            grid_size, block_size
        )
        _send.restype = ctypes.c_void_p
        _send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]

        # End encoding and commit
        _msg(encoder, _sel("endEncoding"))
        _msg(cmd_buffer, _sel("commit"))
        _msg(cmd_buffer, _sel("waitUntilCompleted"))

    def __repr__(self) -> str:
        return f"MetalDevice('{self.name}')"
