#include <jni.h>
#include <stdlib.h>
#include <string.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static id<MTLDevice> gDevice = nil;
static id<MTLCommandQueue> gQueue = nil;
static id<MTLLibrary> gLibrary = nil;
static id<MTLComputePipelineState> gMatVecPso = nil;
static id<MTLComputePipelineState> gOuterPso = nil;
static id<MTLComputePipelineState> gLinearActPso = nil;
static id<MTLComputePipelineState> gMatMulPso = nil;
static id<MTLComputePipelineState> gAddRowBiasPso = nil;
static id<MTLComputePipelineState> gLinearBatchPso = nil;
static id<MTLComputePipelineState> gSoftmaxBatchPso = nil;
static id<MTLComputePipelineState> gCrossEntropyBatchPso = nil;

// Lightweight reusable buffer caches keyed by required byte capacity.
static id<MTLBuffer> gMatVecMatrixBuf = nil;
static id<MTLBuffer> gMatVecVecBuf = nil;
static id<MTLBuffer> gMatVecOutBuf = nil;
static id<MTLBuffer> gMatVecColsBuf = nil;
static NSUInteger gMatVecMatrixCap = 0;
static NSUInteger gMatVecVecCap = 0;
static NSUInteger gMatVecOutCap = 0;
static NSUInteger gMatVecColsCap = 0;

static id<MTLBuffer> gOuterABuf = nil;
static id<MTLBuffer> gOuterBBuf = nil;
static id<MTLBuffer> gOuterOutBuf = nil;
static id<MTLBuffer> gOuterColsBuf = nil;
static NSUInteger gOuterACap = 0;
static NSUInteger gOuterBCap = 0;
static NSUInteger gOuterOutCap = 0;
static NSUInteger gOuterColsCap = 0;

static id<MTLBuffer> gLinearMatrixBuf = nil;
static id<MTLBuffer> gLinearXBuf = nil;
static id<MTLBuffer> gLinearBiasBuf = nil;
static id<MTLBuffer> gLinearZBuf = nil;
static id<MTLBuffer> gLinearABuf = nil;
static id<MTLBuffer> gLinearColsBuf = nil;
static id<MTLBuffer> gLinearActBuf = nil;
static NSUInteger gLinearMatrixCap = 0;
static NSUInteger gLinearXCap = 0;
static NSUInteger gLinearBiasCap = 0;
static NSUInteger gLinearZCap = 0;
static NSUInteger gLinearACap = 0;
static NSUInteger gLinearColsCap = 0;
static NSUInteger gLinearActCap = 0;

static id<MTLBuffer> gMatMulABuf = nil;
static id<MTLBuffer> gMatMulBBuf = nil;
static id<MTLBuffer> gMatMulOutBuf = nil;
static id<MTLBuffer> gMatMulKBuf = nil;
static id<MTLBuffer> gMatMulNBuf = nil;
static NSUInteger gMatMulACap = 0;
static NSUInteger gMatMulBCap = 0;
static NSUInteger gMatMulOutCap = 0;
static NSUInteger gMatMulKCap = 0;
static NSUInteger gMatMulNCap = 0;

static id<MTLBuffer> gLinearBatchXBuf = nil;
static id<MTLBuffer> gLinearBatchWBuf = nil;
static id<MTLBuffer> gLinearBatchBiasBuf = nil;
static id<MTLBuffer> gLinearBatchZBuf = nil;
static id<MTLBuffer> gLinearBatchABuf = nil;
static id<MTLBuffer> gLinearBatchInColsBuf = nil;
static id<MTLBuffer> gLinearBatchOutColsBuf = nil;
static id<MTLBuffer> gLinearBatchActBuf = nil;
static NSUInteger gLinearBatchXCap = 0;
static NSUInteger gLinearBatchWCap = 0;
static NSUInteger gLinearBatchBiasCap = 0;
static NSUInteger gLinearBatchZCap = 0;
static NSUInteger gLinearBatchACap = 0;
static NSUInteger gLinearBatchInColsCap = 0;
static NSUInteger gLinearBatchOutColsCap = 0;
static NSUInteger gLinearBatchActCap = 0;

static id<MTLBuffer> gSoftmaxInBuf = nil;
static id<MTLBuffer> gSoftmaxOutBuf = nil;
static id<MTLBuffer> gSoftmaxColsBuf = nil;
static NSUInteger gSoftmaxInCap = 0;
static NSUInteger gSoftmaxOutCap = 0;
static NSUInteger gSoftmaxColsCap = 0;

static id<MTLBuffer> gCeProbBuf = nil;
static id<MTLBuffer> gCeTargetsBuf = nil;
static id<MTLBuffer> gCeOutBuf = nil;
static id<MTLBuffer> gCeColsBuf = nil;
static NSUInteger gCeProbCap = 0;
static NSUInteger gCeTargetsCap = 0;
static NSUInteger gCeOutCap = 0;
static NSUInteger gCeColsCap = 0;

static BOOL ensureBuffer(id<MTLBuffer> __strong *buffer, NSUInteger *capacity, NSUInteger requiredBytes) {
    if (*buffer == nil || *capacity < requiredBytes) {
        *buffer = [gDevice newBufferWithLength:requiredBytes options:MTLResourceStorageModeShared];
        *capacity = (*buffer != nil) ? requiredBytes : 0;
    }
    return *buffer != nil;
}

static BOOL ensureMetalInitialized(void) {
    if (gDevice == nil) {
        gDevice = MTLCreateSystemDefaultDevice();
    }
    if (gDevice == nil) return NO;
    if (gQueue == nil) {
        gQueue = [gDevice newCommandQueue];
    }
    return gQueue != nil;
}

static BOOL ensurePipelines(NSError **outError) {
    if (!ensureMetalInitialized()) return NO;
    if (gMatVecPso != nil && gOuterPso != nil && gLinearActPso != nil &&
        gMatMulPso != nil && gAddRowBiasPso != nil && gLinearBatchPso != nil &&
        gSoftmaxBatchPso != nil && gCrossEntropyBatchPso != nil) return YES;

    NSString *src =
        @"#include <metal_stdlib>\n"
         "using namespace metal;\n"
         "kernel void matVecMulKernel(\n"
         "  const device float* matrix [[buffer(0)]],\n"
         "  const device float* vec [[buffer(1)]],\n"
         "  device float* out [[buffer(2)]],\n"
         "  constant uint& cols [[buffer(3)]],\n"
         "  uint gid [[thread_position_in_grid]]) {\n"
         "  float sum = 0.0f;\n"
         "  uint base = gid * cols;\n"
         "  for (uint c = 0; c < cols; ++c) {\n"
         "    sum += matrix[base + c] * vec[c];\n"
         "  }\n"
         "  out[gid] = sum;\n"
         "}\n"
         "kernel void outerKernel(\n"
         "  const device float* a [[buffer(0)]],\n"
         "  const device float* b [[buffer(1)]],\n"
         "  device float* out [[buffer(2)]],\n"
         "  constant uint& colsB [[buffer(3)]],\n"
         "  uint2 gid [[thread_position_in_grid]]) {\n"
         "  uint r = gid.y;\n"
         "  uint c = gid.x;\n"
         "  out[r * colsB + c] = a[r] * b[c];\n"
         "}\n"
         "kernel void linearActivationKernel(\n"
         "  const device float* matrix [[buffer(0)]],\n"
         "  const device float* x [[buffer(1)]],\n"
         "  const device float* bias [[buffer(2)]],\n"
         "  device float* zOut [[buffer(3)]],\n"
         "  device float* aOut [[buffer(4)]],\n"
         "  constant uint& cols [[buffer(5)]],\n"
         "  constant uint& activation [[buffer(6)]],\n"
         "  uint gid [[thread_position_in_grid]]) {\n"
         "  float sum = bias[gid];\n"
         "  uint base = gid * cols;\n"
         "  for (uint c = 0; c < cols; ++c) {\n"
         "    sum += matrix[base + c] * x[c];\n"
         "  }\n"
         "  zOut[gid] = sum;\n"
         "  float a = (activation == 1) ? fmax(sum, 0.0f) : tanh(sum);\n"
         "  aOut[gid] = a;\n"
         "}\n"
         "kernel void matMulKernel(\n"
         "  const device float* a [[buffer(0)]],\n"
         "  const device float* b [[buffer(1)]],\n"
         "  device float* out [[buffer(2)]],\n"
         "  constant uint& kDim [[buffer(3)]],\n"
         "  constant uint& nDim [[buffer(4)]],\n"
         "  uint2 gid [[thread_position_in_grid]]) {\n"
         "  uint col = gid.x;\n"
         "  uint row = gid.y;\n"
         "  float sum = 0.0f;\n"
         "  uint aBase = row * kDim;\n"
         "  for (uint k = 0; k < kDim; ++k) {\n"
         "    sum += a[aBase + k] * b[k * nDim + col];\n"
         "  }\n"
         "  out[row * nDim + col] = sum;\n"
         "}\n"
         "kernel void addRowBiasKernel(\n"
         "  const device float* in [[buffer(0)]],\n"
         "  const device float* bias [[buffer(1)]],\n"
         "  device float* out [[buffer(2)]],\n"
         "  constant uint& cols [[buffer(3)]],\n"
         "  uint2 gid [[thread_position_in_grid]]) {\n"
         "  uint col = gid.x;\n"
         "  uint row = gid.y;\n"
         "  out[row * cols + col] = in[row * cols + col] + bias[col];\n"
         "}\n"
         "kernel void linearActivationBatchKernel(\n"
         "  const device float* x [[buffer(0)]],\n"
         "  const device float* wT [[buffer(1)]],\n"
         "  const device float* bias [[buffer(2)]],\n"
         "  device float* zOut [[buffer(3)]],\n"
         "  device float* aOut [[buffer(4)]],\n"
         "  constant uint& inCols [[buffer(5)]],\n"
         "  constant uint& outCols [[buffer(6)]],\n"
         "  constant uint& activation [[buffer(7)]],\n"
         "  uint2 gid [[thread_position_in_grid]]) {\n"
         "  uint col = gid.x;\n"
         "  uint row = gid.y;\n"
         "  float sum = bias[col];\n"
         "  uint xBase = row * inCols;\n"
         "  for (uint k = 0; k < inCols; ++k) {\n"
         "    sum += x[xBase + k] * wT[k * outCols + col];\n"
         "  }\n"
         "  zOut[row * outCols + col] = sum;\n"
         "  float a = (activation == 1) ? fmax(sum, 0.0f) : tanh(sum);\n"
         "  aOut[row * outCols + col] = a;\n"
         "}\n"
         "kernel void softmaxRowwiseKernel(\n"
         "  const device float* logits [[buffer(0)]],\n"
         "  device float* probs [[buffer(1)]],\n"
         "  constant uint& cols [[buffer(2)]],\n"
         "  uint row [[thread_position_in_grid]]) {\n"
         "  uint base = row * cols;\n"
         "  float maxV = logits[base];\n"
         "  for (uint c = 1; c < cols; ++c) {\n"
         "    float v = logits[base + c];\n"
         "    if (v > maxV) maxV = v;\n"
         "  }\n"
         "  float denom = 0.0f;\n"
         "  for (uint c = 0; c < cols; ++c) {\n"
         "    float e = exp(logits[base + c] - maxV);\n"
         "    probs[base + c] = e;\n"
         "    denom += e;\n"
         "  }\n"
         "  float inv = 1.0f / denom;\n"
         "  for (uint c = 0; c < cols; ++c) {\n"
         "    probs[base + c] *= inv;\n"
         "  }\n"
         "}\n"
         "kernel void crossEntropyBatchKernel(\n"
         "  const device float* probs [[buffer(0)]],\n"
         "  const device uint* targets [[buffer(1)]],\n"
         "  device float* outLoss [[buffer(2)]],\n"
         "  constant uint& cols [[buffer(3)]],\n"
         "  uint row [[thread_position_in_grid]]) {\n"
         "  uint t = targets[row];\n"
         "  float p = probs[row * cols + t];\n"
         "  if (p < 1e-12f) p = 1e-12f;\n"
         "  outLoss[row] = -log(p);\n"
         "}\n";

    NSError *err = nil;
    gLibrary = [gDevice newLibraryWithSource:src options:nil error:&err];
    if (gLibrary == nil) {
        if (outError) *outError = err;
        return NO;
    }

    id<MTLFunction> fnMatVec = [gLibrary newFunctionWithName:@"matVecMulKernel"];
    id<MTLFunction> fnOuter = [gLibrary newFunctionWithName:@"outerKernel"];
    id<MTLFunction> fnLinear = [gLibrary newFunctionWithName:@"linearActivationKernel"];
    id<MTLFunction> fnMatMul = [gLibrary newFunctionWithName:@"matMulKernel"];
    id<MTLFunction> fnAddRowBias = [gLibrary newFunctionWithName:@"addRowBiasKernel"];
    id<MTLFunction> fnLinearBatch = [gLibrary newFunctionWithName:@"linearActivationBatchKernel"];
    id<MTLFunction> fnSoftmaxBatch = [gLibrary newFunctionWithName:@"softmaxRowwiseKernel"];
    id<MTLFunction> fnCrossEntropyBatch = [gLibrary newFunctionWithName:@"crossEntropyBatchKernel"];
    if (fnMatVec == nil || fnOuter == nil || fnLinear == nil ||
        fnMatMul == nil || fnAddRowBias == nil || fnLinearBatch == nil ||
        fnSoftmaxBatch == nil || fnCrossEntropyBatch == nil) {
        if (outError) {
            *outError = [NSError errorWithDomain:@"metal-jni"
                                            code:1001
                                        userInfo:@{NSLocalizedDescriptionKey:@"Missing required Metal function(s)"}];
        }
        return NO;
    }

    if (gMatVecPso == nil) {
        gMatVecPso = [gDevice newComputePipelineStateWithFunction:fnMatVec error:&err];
        if (gMatVecPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gOuterPso == nil) {
        gOuterPso = [gDevice newComputePipelineStateWithFunction:fnOuter error:&err];
        if (gOuterPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gLinearActPso == nil) {
        gLinearActPso = [gDevice newComputePipelineStateWithFunction:fnLinear error:&err];
        if (gLinearActPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gMatMulPso == nil) {
        gMatMulPso = [gDevice newComputePipelineStateWithFunction:fnMatMul error:&err];
        if (gMatMulPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gAddRowBiasPso == nil) {
        gAddRowBiasPso = [gDevice newComputePipelineStateWithFunction:fnAddRowBias error:&err];
        if (gAddRowBiasPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gLinearBatchPso == nil) {
        gLinearBatchPso = [gDevice newComputePipelineStateWithFunction:fnLinearBatch error:&err];
        if (gLinearBatchPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gSoftmaxBatchPso == nil) {
        gSoftmaxBatchPso = [gDevice newComputePipelineStateWithFunction:fnSoftmaxBatch error:&err];
        if (gSoftmaxBatchPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }
    if (gCrossEntropyBatchPso == nil) {
        gCrossEntropyBatchPso = [gDevice newComputePipelineStateWithFunction:fnCrossEntropyBatch error:&err];
        if (gCrossEntropyBatchPso == nil) {
            if (outError) *outError = err;
            return NO;
        }
    }

    return YES;
}

JNIEXPORT jboolean JNICALL Java_compute_MetalNativeBridge_00024_initNative(JNIEnv *env, jobject obj) {
    @autoreleasepool {
        return ensureMetalInitialized() ? JNI_TRUE : JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL Java_compute_MetalNativeBridge_00024_isAvailableNative(JNIEnv *env, jobject obj) {
    @autoreleasepool {
        return ensureMetalInitialized() ? JNI_TRUE : JNI_FALSE;
    }
}

JNIEXPORT jstring JNICALL Java_compute_MetalNativeBridge_00024_deviceNameNative(JNIEnv *env, jobject obj) {
    @autoreleasepool {
        ensureMetalInitialized();
        NSString *name = gDevice != nil ? [gDevice name] : @"No Metal Device";
        return (*env)->NewStringUTF(env, [name UTF8String]);
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_matVecMulNative(
    JNIEnv *env, jobject obj, jdoubleArray matrixArr, jint rows, jint cols, jdoubleArray vecArr) {
    @autoreleasepool {
        if (rows <= 0 || cols <= 0) {
            return (*env)->NewDoubleArray(env, 0);
        }

        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "Metal device unavailable");
            return NULL;
        }

        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }

        jsize matrixLen = (*env)->GetArrayLength(env, matrixArr);
        jsize vecLen = (*env)->GetArrayLength(env, vecArr);
        if (matrixLen != rows * cols || vecLen != cols) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:matVecMulNative shape mismatch");
            return NULL;
        }

        jboolean isCopyA = JNI_FALSE;
        jboolean isCopyV = JNI_FALSE;
        jdouble *matrixD = (*env)->GetDoubleArrayElements(env, matrixArr, &isCopyA);
        jdouble *vecD = (*env)->GetDoubleArrayElements(env, vecArr, &isCopyV);

        NSUInteger matrixCount = (NSUInteger)(rows * cols);
        NSUInteger vecCount = (NSUInteger)cols;
        NSUInteger outCount = (NSUInteger)rows;

        float *matrixF = (float *)malloc(sizeof(float) * matrixCount);
        float *vecF = (float *)malloc(sizeof(float) * vecCount);
        if (matrixF == NULL || vecF == NULL) {
            free(matrixF);
            free(vecF);
            (*env)->ReleaseDoubleArrayElements(env, matrixArr, matrixD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, vecArr, vecD, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "Failed to allocate float staging buffers");
            return NULL;
        }

        for (NSUInteger i = 0; i < matrixCount; ++i) matrixF[i] = (float)matrixD[i];
        for (NSUInteger i = 0; i < vecCount; ++i) vecF[i] = (float)vecD[i];

        (*env)->ReleaseDoubleArrayElements(env, matrixArr, matrixD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, vecArr, vecD, JNI_ABORT);

        NSUInteger matrixBytes = sizeof(float) * matrixCount;
        NSUInteger vecBytes = sizeof(float) * vecCount;
        NSUInteger outBytes = sizeof(float) * outCount;
        NSUInteger colsBytes = sizeof(uint32_t);
        BOOL ok = ensureBuffer(&gMatVecMatrixBuf, &gMatVecMatrixCap, matrixBytes) &&
                  ensureBuffer(&gMatVecVecBuf, &gMatVecVecCap, vecBytes) &&
                  ensureBuffer(&gMatVecOutBuf, &gMatVecOutCap, outBytes) &&
                  ensureBuffer(&gMatVecColsBuf, &gMatVecColsCap, colsBytes);
        if (!ok) {
            free(matrixF);
            free(vecF);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate Metal buffers");
            return NULL;
        }
        memcpy([gMatVecMatrixBuf contents], matrixF, matrixBytes);
        memcpy([gMatVecVecBuf contents], vecF, vecBytes);

        free(matrixF);
        free(vecF);

        uint32_t colsValue = (uint32_t)cols;
        memcpy([gMatVecColsBuf contents], &colsValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gMatVecPso];
        [enc setBuffer:gMatVecMatrixBuf offset:0 atIndex:0];
        [enc setBuffer:gMatVecVecBuf offset:0 atIndex:1];
        [enc setBuffer:gMatVecOutBuf offset:0 atIndex:2];
        [enc setBuffer:gMatVecColsBuf offset:0 atIndex:3];

        NSUInteger threadsPerGrid = outCount;
        NSUInteger w = gMatVecPso.threadExecutionWidth;
        if (w == 0) w = 1;
        NSUInteger threadsPerThreadgroup = w;
        if (threadsPerThreadgroup > threadsPerGrid) threadsPerThreadgroup = threadsPerGrid;

        MTLSize gridSize = MTLSizeMake(threadsPerGrid, 1, 1);
        MTLSize tgSize = MTLSizeMake(threadsPerThreadgroup, 1, 1);
        [enc dispatchThreads:gridSize threadsPerThreadgroup:tgSize];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *outF = (float *)[gMatVecOutBuf contents];
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)outCount);
        if (outArr == NULL) return NULL;

        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * outCount);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "Failed to allocate output staging buffer");
            return NULL;
        }
        for (NSUInteger i = 0; i < outCount; ++i) tmp[i] = (jdouble)outF[i];
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)outCount, tmp);
        free(tmp);
        return outArr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_outerNative(
    JNIEnv *env, jobject obj, jdoubleArray aArr, jdoubleArray bArr) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "E_DEVICE:Metal device unavailable");
            return NULL;
        }
        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }

        jsize rows = (*env)->GetArrayLength(env, aArr);
        jsize cols = (*env)->GetArrayLength(env, bArr);
        if (rows <= 0 || cols <= 0) {
            return (*env)->NewDoubleArray(env, 0);
        }

        jboolean copyA = JNI_FALSE;
        jboolean copyB = JNI_FALSE;
        jdouble *aD = (*env)->GetDoubleArrayElements(env, aArr, &copyA);
        jdouble *bD = (*env)->GetDoubleArrayElements(env, bArr, &copyB);

        NSUInteger rowsN = (NSUInteger)rows;
        NSUInteger colsN = (NSUInteger)cols;
        NSUInteger outCount = rowsN * colsN;
        NSUInteger aBytes = sizeof(float) * rowsN;
        NSUInteger bBytes = sizeof(float) * colsN;
        NSUInteger outBytes = sizeof(float) * outCount;
        NSUInteger colsBytes = sizeof(uint32_t);

        float *aF = (float *)malloc(aBytes);
        float *bF = (float *)malloc(bBytes);
        if (aF == NULL || bF == NULL) {
            free(aF); free(bF);
            (*env)->ReleaseDoubleArrayElements(env, aArr, aD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, bArr, bD, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate outer staging buffers");
            return NULL;
        }
        for (NSUInteger i = 0; i < rowsN; ++i) aF[i] = (float)aD[i];
        for (NSUInteger i = 0; i < colsN; ++i) bF[i] = (float)bD[i];
        (*env)->ReleaseDoubleArrayElements(env, aArr, aD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, bArr, bD, JNI_ABORT);

        BOOL ok = ensureBuffer(&gOuterABuf, &gOuterACap, aBytes) &&
                  ensureBuffer(&gOuterBBuf, &gOuterBCap, bBytes) &&
                  ensureBuffer(&gOuterOutBuf, &gOuterOutCap, outBytes) &&
                  ensureBuffer(&gOuterColsBuf, &gOuterColsCap, colsBytes);
        if (!ok) {
            free(aF); free(bF);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate outer Metal buffers");
            return NULL;
        }
        memcpy([gOuterABuf contents], aF, aBytes);
        memcpy([gOuterBBuf contents], bF, bBytes);
        free(aF); free(bF);
        uint32_t colsValue = (uint32_t)cols;
        memcpy([gOuterColsBuf contents], &colsValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gOuterPso];
        [enc setBuffer:gOuterABuf offset:0 atIndex:0];
        [enc setBuffer:gOuterBBuf offset:0 atIndex:1];
        [enc setBuffer:gOuterOutBuf offset:0 atIndex:2];
        [enc setBuffer:gOuterColsBuf offset:0 atIndex:3];

        MTLSize grid = MTLSizeMake((NSUInteger)cols, (NSUInteger)rows, 1);
        NSUInteger w = gOuterPso.threadExecutionWidth;
        NSUInteger h = gOuterPso.maxTotalThreadsPerThreadgroup / w;
        if (h == 0) h = 1;
        MTLSize tg = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *outF = (float *)[gOuterOutBuf contents];
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)outCount);
        if (outArr == NULL) return NULL;
        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * outCount);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate outer output staging buffer");
            return NULL;
        }
        for (NSUInteger i = 0; i < outCount; ++i) tmp[i] = (jdouble)outF[i];
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)outCount, tmp);
        free(tmp);
        return outArr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_linearActivationNative(
    JNIEnv *env, jobject obj, jdoubleArray matrixArr, jint rows, jint cols, jdoubleArray xArr, jdoubleArray biasArr, jint activationCode) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "E_DEVICE:Metal device unavailable");
            return NULL;
        }
        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }

        if (rows <= 0 || cols <= 0) return (*env)->NewDoubleArray(env, 0);

        jsize matrixLen = (*env)->GetArrayLength(env, matrixArr);
        jsize xLen = (*env)->GetArrayLength(env, xArr);
        jsize biasLen = (*env)->GetArrayLength(env, biasArr);
        if (matrixLen != rows * cols || xLen != cols || biasLen != rows) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:linearActivationNative shape mismatch");
            return NULL;
        }

        jdouble *matrixD = (*env)->GetDoubleArrayElements(env, matrixArr, NULL);
        jdouble *xD = (*env)->GetDoubleArrayElements(env, xArr, NULL);
        jdouble *biasD = (*env)->GetDoubleArrayElements(env, biasArr, NULL);

        NSUInteger matrixCount = (NSUInteger)(rows * cols);
        NSUInteger xCount = (NSUInteger)cols;
        NSUInteger outCount = (NSUInteger)rows;

        NSUInteger matrixBytes = sizeof(float) * matrixCount;
        NSUInteger xBytes = sizeof(float) * xCount;
        NSUInteger biasBytes = sizeof(float) * outCount;
        NSUInteger outBytes = sizeof(float) * outCount;

        float *matrixF = (float *)malloc(matrixBytes);
        float *xF = (float *)malloc(xBytes);
        float *biasF = (float *)malloc(biasBytes);
        if (matrixF == NULL || xF == NULL || biasF == NULL) {
            free(matrixF); free(xF); free(biasF);
            (*env)->ReleaseDoubleArrayElements(env, matrixArr, matrixD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, xArr, xD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, biasArr, biasD, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate linearActivation staging buffers");
            return NULL;
        }
        for (NSUInteger i = 0; i < matrixCount; ++i) matrixF[i] = (float)matrixD[i];
        for (NSUInteger i = 0; i < xCount; ++i) xF[i] = (float)xD[i];
        for (NSUInteger i = 0; i < outCount; ++i) biasF[i] = (float)biasD[i];
        (*env)->ReleaseDoubleArrayElements(env, matrixArr, matrixD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, xArr, xD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, biasArr, biasD, JNI_ABORT);

        BOOL ok = ensureBuffer(&gLinearMatrixBuf, &gLinearMatrixCap, matrixBytes) &&
                  ensureBuffer(&gLinearXBuf, &gLinearXCap, xBytes) &&
                  ensureBuffer(&gLinearBiasBuf, &gLinearBiasCap, biasBytes) &&
                  ensureBuffer(&gLinearZBuf, &gLinearZCap, outBytes) &&
                  ensureBuffer(&gLinearABuf, &gLinearACap, outBytes) &&
                  ensureBuffer(&gLinearColsBuf, &gLinearColsCap, sizeof(uint32_t)) &&
                  ensureBuffer(&gLinearActBuf, &gLinearActCap, sizeof(uint32_t));
        if (!ok) {
            free(matrixF); free(xF); free(biasF);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate linearActivation Metal buffers");
            return NULL;
        }
        memcpy([gLinearMatrixBuf contents], matrixF, matrixBytes);
        memcpy([gLinearXBuf contents], xF, xBytes);
        memcpy([gLinearBiasBuf contents], biasF, biasBytes);
        free(matrixF); free(xF); free(biasF);
        uint32_t colsValue = (uint32_t)cols;
        uint32_t actValue = (uint32_t)activationCode;
        memcpy([gLinearColsBuf contents], &colsValue, sizeof(uint32_t));
        memcpy([gLinearActBuf contents], &actValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gLinearActPso];
        [enc setBuffer:gLinearMatrixBuf offset:0 atIndex:0];
        [enc setBuffer:gLinearXBuf offset:0 atIndex:1];
        [enc setBuffer:gLinearBiasBuf offset:0 atIndex:2];
        [enc setBuffer:gLinearZBuf offset:0 atIndex:3];
        [enc setBuffer:gLinearABuf offset:0 atIndex:4];
        [enc setBuffer:gLinearColsBuf offset:0 atIndex:5];
        [enc setBuffer:gLinearActBuf offset:0 atIndex:6];

        NSUInteger threadsPerGrid = outCount;
        NSUInteger w = gLinearActPso.threadExecutionWidth;
        if (w == 0) w = 1;
        NSUInteger threadsPerThreadgroup = w > threadsPerGrid ? threadsPerGrid : w;
        [enc dispatchThreads:MTLSizeMake(threadsPerGrid, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *zF = (float *)[gLinearZBuf contents];
        float *aF = (float *)[gLinearABuf contents];
        NSUInteger packedCount = outCount * 2;
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)packedCount);
        if (outArr == NULL) return NULL;
        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * packedCount);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:Failed to allocate packed output staging buffer");
            return NULL;
        }
        for (NSUInteger i = 0; i < outCount; ++i) {
            tmp[i] = (jdouble)zF[i];
            tmp[outCount + i] = (jdouble)aF[i];
        }
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)packedCount, tmp);
        free(tmp);
        return outArr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_matMulNative(
    JNIEnv *env, jobject obj, jdoubleArray aArr, jint aRows, jint aCols, jdoubleArray bArr, jint bRows, jint bCols) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "E_DEVICE:Metal device unavailable");
            return NULL;
        }
        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }
        if (aRows <= 0 || aCols <= 0 || bRows <= 0 || bCols <= 0 || aCols != bRows) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:matMulNative shape mismatch");
            return NULL;
        }
        jsize aLen = (*env)->GetArrayLength(env, aArr);
        jsize bLen = (*env)->GetArrayLength(env, bArr);
        if (aLen != aRows * aCols || bLen != bRows * bCols) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:matMulNative array length mismatch");
            return NULL;
        }

        jdouble *aD = (*env)->GetDoubleArrayElements(env, aArr, NULL);
        jdouble *bD = (*env)->GetDoubleArrayElements(env, bArr, NULL);
        NSUInteger aCount = (NSUInteger)(aRows * aCols);
        NSUInteger bCount = (NSUInteger)(bRows * bCols);
        NSUInteger outCount = (NSUInteger)(aRows * bCols);
        float *aF = (float *)malloc(sizeof(float) * aCount);
        float *bF = (float *)malloc(sizeof(float) * bCount);
        if (aF == NULL || bF == NULL) {
            free(aF); free(bF);
            (*env)->ReleaseDoubleArrayElements(env, aArr, aD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, bArr, bD, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:matMul staging allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < aCount; ++i) aF[i] = (float)aD[i];
        for (NSUInteger i = 0; i < bCount; ++i) bF[i] = (float)bD[i];
        (*env)->ReleaseDoubleArrayElements(env, aArr, aD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, bArr, bD, JNI_ABORT);

        NSUInteger aBytes = sizeof(float) * aCount;
        NSUInteger bBytes = sizeof(float) * bCount;
        NSUInteger outBytes = sizeof(float) * outCount;
        BOOL ok = ensureBuffer(&gMatMulABuf, &gMatMulACap, aBytes) &&
                  ensureBuffer(&gMatMulBBuf, &gMatMulBCap, bBytes) &&
                  ensureBuffer(&gMatMulOutBuf, &gMatMulOutCap, outBytes) &&
                  ensureBuffer(&gMatMulKBuf, &gMatMulKCap, sizeof(uint32_t)) &&
                  ensureBuffer(&gMatMulNBuf, &gMatMulNCap, sizeof(uint32_t));
        if (!ok) {
            free(aF); free(bF);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:matMul Metal buffer allocation failed");
            return NULL;
        }
        memcpy([gMatMulABuf contents], aF, aBytes);
        memcpy([gMatMulBBuf contents], bF, bBytes);
        free(aF); free(bF);
        uint32_t kValue = (uint32_t)aCols;
        uint32_t nValue = (uint32_t)bCols;
        memcpy([gMatMulKBuf contents], &kValue, sizeof(uint32_t));
        memcpy([gMatMulNBuf contents], &nValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gMatMulPso];
        [enc setBuffer:gMatMulABuf offset:0 atIndex:0];
        [enc setBuffer:gMatMulBBuf offset:0 atIndex:1];
        [enc setBuffer:gMatMulOutBuf offset:0 atIndex:2];
        [enc setBuffer:gMatMulKBuf offset:0 atIndex:3];
        [enc setBuffer:gMatMulNBuf offset:0 atIndex:4];
        MTLSize grid = MTLSizeMake((NSUInteger)bCols, (NSUInteger)aRows, 1);
        NSUInteger w = gMatMulPso.threadExecutionWidth;
        NSUInteger h = gMatMulPso.maxTotalThreadsPerThreadgroup / w;
        if (h == 0) h = 1;
        MTLSize tg = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *outF = (float *)[gMatMulOutBuf contents];
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)outCount);
        if (outArr == NULL) return NULL;
        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * outCount);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:matMul output allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < outCount; ++i) tmp[i] = (jdouble)outF[i];
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)outCount, tmp);
        free(tmp);
        return outArr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_linearActivationBatchNative(
    JNIEnv *env, jobject obj, jdoubleArray xArr, jint rows, jint inCols,
    jdoubleArray wTArr, jint wRows, jint wCols, jdoubleArray biasArr, jint activationCode) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "E_DEVICE:Metal device unavailable");
            return NULL;
        }
        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }
        if (rows <= 0 || inCols <= 0 || wRows <= 0 || wCols <= 0 || inCols != wRows) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:linearActivationBatchNative shape mismatch");
            return NULL;
        }
        jsize xLen = (*env)->GetArrayLength(env, xArr);
        jsize wLen = (*env)->GetArrayLength(env, wTArr);
        jsize bLen = (*env)->GetArrayLength(env, biasArr);
        if (xLen != rows * inCols || wLen != wRows * wCols || bLen != wCols) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:linearActivationBatchNative array length mismatch");
            return NULL;
        }

        jdouble *xD = (*env)->GetDoubleArrayElements(env, xArr, NULL);
        jdouble *wD = (*env)->GetDoubleArrayElements(env, wTArr, NULL);
        jdouble *bD = (*env)->GetDoubleArrayElements(env, biasArr, NULL);
        NSUInteger xCount = (NSUInteger)(rows * inCols);
        NSUInteger wCount = (NSUInteger)(wRows * wCols);
        NSUInteger outCount = (NSUInteger)(rows * wCols);
        float *xF = (float *)malloc(sizeof(float) * xCount);
        float *wF = (float *)malloc(sizeof(float) * wCount);
        float *bF = (float *)malloc(sizeof(float) * (NSUInteger)wCols);
        if (xF == NULL || wF == NULL || bF == NULL) {
            free(xF); free(wF); free(bF);
            (*env)->ReleaseDoubleArrayElements(env, xArr, xD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, wTArr, wD, JNI_ABORT);
            (*env)->ReleaseDoubleArrayElements(env, biasArr, bD, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:linearActivationBatch staging allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < xCount; ++i) xF[i] = (float)xD[i];
        for (NSUInteger i = 0; i < wCount; ++i) wF[i] = (float)wD[i];
        for (NSUInteger i = 0; i < (NSUInteger)wCols; ++i) bF[i] = (float)bD[i];
        (*env)->ReleaseDoubleArrayElements(env, xArr, xD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, wTArr, wD, JNI_ABORT);
        (*env)->ReleaseDoubleArrayElements(env, biasArr, bD, JNI_ABORT);

        BOOL ok = ensureBuffer(&gLinearBatchXBuf, &gLinearBatchXCap, sizeof(float) * xCount) &&
                  ensureBuffer(&gLinearBatchWBuf, &gLinearBatchWCap, sizeof(float) * wCount) &&
                  ensureBuffer(&gLinearBatchBiasBuf, &gLinearBatchBiasCap, sizeof(float) * (NSUInteger)wCols) &&
                  ensureBuffer(&gLinearBatchZBuf, &gLinearBatchZCap, sizeof(float) * outCount) &&
                  ensureBuffer(&gLinearBatchABuf, &gLinearBatchACap, sizeof(float) * outCount) &&
                  ensureBuffer(&gLinearBatchInColsBuf, &gLinearBatchInColsCap, sizeof(uint32_t)) &&
                  ensureBuffer(&gLinearBatchOutColsBuf, &gLinearBatchOutColsCap, sizeof(uint32_t)) &&
                  ensureBuffer(&gLinearBatchActBuf, &gLinearBatchActCap, sizeof(uint32_t));
        if (!ok) {
            free(xF); free(wF); free(bF);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:linearActivationBatch Metal buffer allocation failed");
            return NULL;
        }
        memcpy([gLinearBatchXBuf contents], xF, sizeof(float) * xCount);
        memcpy([gLinearBatchWBuf contents], wF, sizeof(float) * wCount);
        memcpy([gLinearBatchBiasBuf contents], bF, sizeof(float) * (NSUInteger)wCols);
        free(xF); free(wF); free(bF);
        uint32_t inColsValue = (uint32_t)inCols;
        uint32_t outColsValue = (uint32_t)wCols;
        uint32_t actValue = (uint32_t)activationCode;
        memcpy([gLinearBatchInColsBuf contents], &inColsValue, sizeof(uint32_t));
        memcpy([gLinearBatchOutColsBuf contents], &outColsValue, sizeof(uint32_t));
        memcpy([gLinearBatchActBuf contents], &actValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gLinearBatchPso];
        [enc setBuffer:gLinearBatchXBuf offset:0 atIndex:0];
        [enc setBuffer:gLinearBatchWBuf offset:0 atIndex:1];
        [enc setBuffer:gLinearBatchBiasBuf offset:0 atIndex:2];
        [enc setBuffer:gLinearBatchZBuf offset:0 atIndex:3];
        [enc setBuffer:gLinearBatchABuf offset:0 atIndex:4];
        [enc setBuffer:gLinearBatchInColsBuf offset:0 atIndex:5];
        [enc setBuffer:gLinearBatchOutColsBuf offset:0 atIndex:6];
        [enc setBuffer:gLinearBatchActBuf offset:0 atIndex:7];
        MTLSize grid = MTLSizeMake((NSUInteger)wCols, (NSUInteger)rows, 1);
        NSUInteger w = gLinearBatchPso.threadExecutionWidth;
        NSUInteger h = gLinearBatchPso.maxTotalThreadsPerThreadgroup / w;
        if (h == 0) h = 1;
        MTLSize tg = MTLSizeMake(w, h, 1);
        [enc dispatchThreads:grid threadsPerThreadgroup:tg];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        NSUInteger packedCount = outCount * 2;
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)packedCount);
        if (outArr == NULL) return NULL;
        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * packedCount);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:linearActivationBatch output allocation failed");
            return NULL;
        }
        float *zF = (float *)[gLinearBatchZBuf contents];
        float *aF = (float *)[gLinearBatchABuf contents];
        for (NSUInteger i = 0; i < outCount; ++i) {
            tmp[i] = (jdouble)zF[i];
            tmp[outCount + i] = (jdouble)aF[i];
        }
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)packedCount, tmp);
        free(tmp);
        return outArr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_softmaxBatchNative(
    JNIEnv *env, jobject obj, jdoubleArray logitsArr, jint rows, jint cols) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "E_DEVICE:Metal device unavailable");
            return NULL;
        }
        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }
        if (rows <= 0 || cols <= 0) return (*env)->NewDoubleArray(env, 0);
        jsize len = (*env)->GetArrayLength(env, logitsArr);
        if (len != rows * cols) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:softmaxBatchNative shape mismatch");
            return NULL;
        }
        jdouble *logitsD = (*env)->GetDoubleArrayElements(env, logitsArr, NULL);
        NSUInteger count = (NSUInteger)(rows * cols);
        float *logitsF = (float *)malloc(sizeof(float) * count);
        if (logitsF == NULL) {
            (*env)->ReleaseDoubleArrayElements(env, logitsArr, logitsD, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:softmaxBatch staging allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < count; ++i) logitsF[i] = (float)logitsD[i];
        (*env)->ReleaseDoubleArrayElements(env, logitsArr, logitsD, JNI_ABORT);
        BOOL ok = ensureBuffer(&gSoftmaxInBuf, &gSoftmaxInCap, sizeof(float) * count) &&
                  ensureBuffer(&gSoftmaxOutBuf, &gSoftmaxOutCap, sizeof(float) * count) &&
                  ensureBuffer(&gSoftmaxColsBuf, &gSoftmaxColsCap, sizeof(uint32_t));
        if (!ok) {
            free(logitsF);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:softmaxBatch Metal buffer allocation failed");
            return NULL;
        }
        memcpy([gSoftmaxInBuf contents], logitsF, sizeof(float) * count);
        free(logitsF);
        uint32_t colsValue = (uint32_t)cols;
        memcpy([gSoftmaxColsBuf contents], &colsValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gSoftmaxBatchPso];
        [enc setBuffer:gSoftmaxInBuf offset:0 atIndex:0];
        [enc setBuffer:gSoftmaxOutBuf offset:0 atIndex:1];
        [enc setBuffer:gSoftmaxColsBuf offset:0 atIndex:2];
        NSUInteger threadsPerGrid = (NSUInteger)rows;
        NSUInteger w = gSoftmaxBatchPso.threadExecutionWidth;
        if (w == 0) w = 1;
        NSUInteger tgThreads = w > threadsPerGrid ? threadsPerGrid : w;
        [enc dispatchThreads:MTLSizeMake(threadsPerGrid, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tgThreads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *outF = (float *)[gSoftmaxOutBuf contents];
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)count);
        if (outArr == NULL) return NULL;
        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * count);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:softmaxBatch output allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < count; ++i) tmp[i] = (jdouble)outF[i];
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)count, tmp);
        free(tmp);
        return outArr;
    }
}

JNIEXPORT jdoubleArray JNICALL Java_compute_MetalNativeBridge_00024_crossEntropyBatchNative(
    JNIEnv *env, jobject obj, jdoubleArray probsArr, jint rows, jint cols, jintArray targetsArr) {
    @autoreleasepool {
        if (!ensureMetalInitialized()) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalStateException");
            (*env)->ThrowNew(env, exCls, "E_DEVICE:Metal device unavailable");
            return NULL;
        }
        NSError *psoErr = nil;
        if (!ensurePipelines(&psoErr)) {
            NSString *msg = psoErr != nil ? [psoErr localizedDescription] : @"Failed to build Metal pipeline";
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            NSString *final = [NSString stringWithFormat:@"E_PIPELINE:%@", msg];
            (*env)->ThrowNew(env, exCls, [final UTF8String]);
            return NULL;
        }
        if (rows <= 0 || cols <= 0) return (*env)->NewDoubleArray(env, 0);
        jsize probsLen = (*env)->GetArrayLength(env, probsArr);
        jsize targetsLen = (*env)->GetArrayLength(env, targetsArr);
        if (probsLen != rows * cols || targetsLen != rows) {
            jclass exCls = (*env)->FindClass(env, "java/lang/IllegalArgumentException");
            (*env)->ThrowNew(env, exCls, "E_SHAPE:crossEntropyBatchNative shape mismatch");
            return NULL;
        }
        jdouble *probsD = (*env)->GetDoubleArrayElements(env, probsArr, NULL);
        jint *targetsI = (*env)->GetIntArrayElements(env, targetsArr, NULL);
        NSUInteger probCount = (NSUInteger)(rows * cols);
        NSUInteger outCount = (NSUInteger)rows;
        float *probsF = (float *)malloc(sizeof(float) * probCount);
        uint32_t *targetsU = (uint32_t *)malloc(sizeof(uint32_t) * outCount);
        if (probsF == NULL || targetsU == NULL) {
            free(probsF); free(targetsU);
            (*env)->ReleaseDoubleArrayElements(env, probsArr, probsD, JNI_ABORT);
            (*env)->ReleaseIntArrayElements(env, targetsArr, targetsI, JNI_ABORT);
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:crossEntropyBatch staging allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < probCount; ++i) probsF[i] = (float)probsD[i];
        for (NSUInteger i = 0; i < outCount; ++i) targetsU[i] = (uint32_t)targetsI[i];
        (*env)->ReleaseDoubleArrayElements(env, probsArr, probsD, JNI_ABORT);
        (*env)->ReleaseIntArrayElements(env, targetsArr, targetsI, JNI_ABORT);

        BOOL ok = ensureBuffer(&gCeProbBuf, &gCeProbCap, sizeof(float) * probCount) &&
                  ensureBuffer(&gCeTargetsBuf, &gCeTargetsCap, sizeof(uint32_t) * outCount) &&
                  ensureBuffer(&gCeOutBuf, &gCeOutCap, sizeof(float) * outCount) &&
                  ensureBuffer(&gCeColsBuf, &gCeColsCap, sizeof(uint32_t));
        if (!ok) {
            free(probsF); free(targetsU);
            jclass exCls = (*env)->FindClass(env, "java/lang/RuntimeException");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:crossEntropyBatch Metal buffer allocation failed");
            return NULL;
        }
        memcpy([gCeProbBuf contents], probsF, sizeof(float) * probCount);
        memcpy([gCeTargetsBuf contents], targetsU, sizeof(uint32_t) * outCount);
        free(probsF); free(targetsU);
        uint32_t colsValue = (uint32_t)cols;
        memcpy([gCeColsBuf contents], &colsValue, sizeof(uint32_t));

        id<MTLCommandBuffer> cb = [gQueue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        [enc setComputePipelineState:gCrossEntropyBatchPso];
        [enc setBuffer:gCeProbBuf offset:0 atIndex:0];
        [enc setBuffer:gCeTargetsBuf offset:0 atIndex:1];
        [enc setBuffer:gCeOutBuf offset:0 atIndex:2];
        [enc setBuffer:gCeColsBuf offset:0 atIndex:3];
        NSUInteger threadsPerGrid = outCount;
        NSUInteger w = gCrossEntropyBatchPso.threadExecutionWidth;
        if (w == 0) w = 1;
        NSUInteger tgThreads = w > threadsPerGrid ? threadsPerGrid : w;
        [enc dispatchThreads:MTLSizeMake(threadsPerGrid, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(tgThreads, 1, 1)];
        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        float *outF = (float *)[gCeOutBuf contents];
        jdoubleArray outArr = (*env)->NewDoubleArray(env, (jsize)outCount);
        if (outArr == NULL) return NULL;
        jdouble *tmp = (jdouble *)malloc(sizeof(jdouble) * outCount);
        if (tmp == NULL) {
            jclass exCls = (*env)->FindClass(env, "java/lang/OutOfMemoryError");
            (*env)->ThrowNew(env, exCls, "E_ALLOC:crossEntropyBatch output allocation failed");
            return NULL;
        }
        for (NSUInteger i = 0; i < outCount; ++i) tmp[i] = (jdouble)outF[i];
        (*env)->SetDoubleArrayRegion(env, outArr, 0, (jsize)outCount, tmp);
        free(tmp);
        return outArr;
    }
}
