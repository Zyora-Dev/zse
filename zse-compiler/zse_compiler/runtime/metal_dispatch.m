// zse_compiler/runtime/metal_dispatch.m
// Minimal C bridge for Metal GPU dispatch on Apple Silicon.
// Compiled with: clang -O2 -shared -o metal_dispatch.dylib metal_dispatch.m
//                -framework Metal -framework Foundation
// Zero Python dependencies — loaded via ctypes.

#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

// --- Device Init ---

void* zse_metal_init(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    return (__bridge_retained void*)device;
}

void* zse_metal_create_queue(void* device_ptr) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    id<MTLCommandQueue> queue = [device newCommandQueue];
    return (__bridge_retained void*)queue;
}

// --- Device Info ---

uint64_t zse_metal_device_memory(void* device_ptr) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    return [device recommendedMaxWorkingSetSize];
}

static char _name_buf[256];
const char* zse_metal_device_name(void* device_ptr) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    NSString* name = [device name];
    [name getCString:_name_buf maxLength:256 encoding:NSUTF8StringEncoding];
    return _name_buf;
}

// --- Runtime MSL Compile (NO xcrun needed) ---

void* zse_metal_compile(void* device_ptr, const char* source, const char* kernel_name) {
    @autoreleasepool {
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
        NSString* src = [NSString stringWithUTF8String:source];
        NSString* name = [NSString stringWithUTF8String:kernel_name];
        NSError* error = nil;

        // Compile MSL source at runtime on the GPU driver
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        id<MTLLibrary> library = [device newLibraryWithSource:src
                                                      options:options
                                                        error:&error];
        if (library == nil) {
            NSLog(@"ZSE Metal compile error: %@", error);
            return NULL;
        }

        // Get kernel function
        id<MTLFunction> func = [library newFunctionWithName:name];
        if (func == nil) {
            NSLog(@"ZSE Metal: kernel '%s' not found in compiled library", kernel_name);
            return NULL;
        }

        // Create compute pipeline
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:func error:&error];
        if (pipeline == nil) {
            NSLog(@"ZSE Metal pipeline error: %@", error);
            return NULL;
        }

        return (__bridge_retained void*)pipeline;
    }
}

// --- Buffer Management ---

void* zse_metal_alloc_buffer(void* device_ptr, uint64_t nbytes) {
    id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
    // StorageModeShared = unified memory, CPU+GPU accessible
    id<MTLBuffer> buffer = [device newBufferWithLength:nbytes
                                              options:MTLResourceStorageModeShared];
    if (buffer == nil) return NULL;
    return (__bridge_retained void*)buffer;
}

void* zse_metal_buffer_contents(void* buffer_ptr) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
    return [buffer contents];
}

uint64_t zse_metal_buffer_length(void* buffer_ptr) {
    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)buffer_ptr;
    return [buffer length];
}

void zse_metal_free_buffer(void* buffer_ptr) {
    if (buffer_ptr) {
        CFRelease(buffer_ptr);
    }
}

// --- Kernel Dispatch ---

double zse_metal_dispatch(void* queue_ptr, void* pipeline_ptr,
                          void** buffer_ptrs, int num_buffers,
                          int gx, int gy, int gz,
                          int tx, int ty, int tz) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
        id<MTLComputePipelineState> pipeline =
            (__bridge id<MTLComputePipelineState>)pipeline_ptr;

        id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];

        for (int i = 0; i < num_buffers; i++) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
            [encoder setBuffer:buf offset:0 atIndex:i];
        }

        MTLSize grid = MTLSizeMake(gx, gy, gz);
        MTLSize tgSize = MTLSizeMake(tx, ty, tz);
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
        [encoder endEncoding];

        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        return (cmdBuf.GPUEndTime - cmdBuf.GPUStartTime) * 1000.0;
    }
}

// --- Async Dispatch ---

static id<MTLCommandBuffer> _pending_cmd_buf = nil;

void zse_metal_dispatch_async(void* queue_ptr, void* pipeline_ptr,
                              void** buffer_ptrs, int num_buffers,
                              int gx, int gy, int gz,
                              int tx, int ty, int tz) {
    // Wait for previous pending work
    if (_pending_cmd_buf != nil) {
        [_pending_cmd_buf waitUntilCompleted];
        _pending_cmd_buf = nil;
    }

    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)queue_ptr;
    id<MTLComputePipelineState> pipeline =
        (__bridge id<MTLComputePipelineState>)pipeline_ptr;

    id<MTLCommandBuffer> cmdBuf = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [cmdBuf computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];

    for (int i = 0; i < num_buffers; i++) {
        id<MTLBuffer> buf = (__bridge id<MTLBuffer>)buffer_ptrs[i];
        [encoder setBuffer:buf offset:0 atIndex:i];
    }

    MTLSize grid = MTLSizeMake(gx, gy, gz);
    MTLSize tgSize = MTLSizeMake(tx, ty, tz);
    [encoder dispatchThreadgroups:grid threadsPerThreadgroup:tgSize];
    [encoder endEncoding];

    [cmdBuf commit];
    _pending_cmd_buf = cmdBuf;
}

double zse_metal_sync(void) {
    if (_pending_cmd_buf == nil) return 0.0;
    [_pending_cmd_buf waitUntilCompleted];
    double gpu_ms = (_pending_cmd_buf.GPUEndTime - _pending_cmd_buf.GPUStartTime) * 1000.0;
    _pending_cmd_buf = nil;
    return gpu_ms;
}

// --- Compile Error Message ---

const char* zse_metal_compile_error(void* device_ptr, const char* source) {
    @autoreleasepool {
        static char _err_buf[4096];
        id<MTLDevice> device = (__bridge id<MTLDevice>)device_ptr;
        NSString* src = [NSString stringWithUTF8String:source];
        NSError* error = nil;
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            [[error localizedDescription] getCString:_err_buf
                                           maxLength:4096
                                            encoding:NSUTF8StringEncoding];
            return _err_buf;
        }
        return NULL;
    }
}
