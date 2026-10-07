// Minimal Metal surface for off-macOS syntax checking only.
// Declares just the members csrc/**/*.mm touches, so clang can parse the
// Objective-C++ on Linux. Never compiled or linked on macOS -- the real SDK
// headers win there. Add members as the sources need them.
#pragma once
#import <Foundation/Foundation.h>
typedef struct {
    NSUInteger width, height, depth;
} MTLSize;
static inline MTLSize MTLSizeMake(NSUInteger w, NSUInteger h, NSUInteger d) {
    MTLSize s;
    s.width = w;
    s.height = h;
    s.depth = d;
    return s;
}
typedef NSInteger MTLResourceOptions;
static const MTLResourceOptions MTLResourceStorageModeShared = 0;
typedef NSInteger MTLCommandBufferStatus;
static const MTLCommandBufferStatus MTLCommandBufferStatusError = 5;
typedef NSInteger MTLCommandBufferError;
static const MTLCommandBufferError MTLCommandBufferErrorOutOfMemory = 8;
static const MTLCommandBufferError MTLCommandBufferErrorDeviceRemoved = 11;
extern NSString* const MTLCommandBufferErrorDomain;
@protocol MTLBuffer
@property(readonly) void* contents;
@property(readonly) NSUInteger length;
@end
@protocol MTLFunction
@end
@protocol MTLLibrary
- (id<MTLFunction>)newFunctionWithName:(NSString*)name;
@end
@protocol MTLComputePipelineState
@property(readonly) NSUInteger maxTotalThreadsPerThreadgroup;
@end
@protocol MTLComputeCommandEncoder
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state;
- (void)setBuffer:(id<MTLBuffer>)b offset:(NSUInteger)o atIndex:(NSUInteger)i;
- (void)setBytes:(const void*)p length:(NSUInteger)l atIndex:(NSUInteger)i;
- (void)dispatchThreads:(MTLSize)g threadsPerThreadgroup:(MTLSize)t;
- (void)endEncoding;
@end
@protocol MTLCommandBuffer
@property(readonly) MTLCommandBufferStatus status;
@property(readonly) NSError* error;
- (id<MTLComputeCommandEncoder>)computeCommandEncoder;
- (void)commit;
- (void)waitUntilCompleted;
@end
@protocol MTLCommandQueue
- (id<MTLCommandBuffer>)commandBuffer;
@end
@protocol MTLDevice
@property(readonly) NSString* name;
@property(readonly) BOOL hasUnifiedMemory;
@property(readonly) NSUInteger registryID;
@property(readonly) NSUInteger recommendedMaxWorkingSetSize;
@property(readonly) NSUInteger currentAllocatedSize;
- (id<MTLCommandQueue>)newCommandQueue;
- (id<MTLBuffer>)newBufferWithLength:(NSUInteger)l options:(MTLResourceOptions)o;
- (id<MTLLibrary>)newLibraryWithURL:(id)url error:(NSError**)e;
- (id<MTLComputePipelineState>)newComputePipelineStateWithFunction:(id<MTLFunction>)f
                                                             error:(NSError**)e;
@end
id<MTLDevice> MTLCreateSystemDefaultDevice(void);
