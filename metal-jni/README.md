# Metal JNI Bridge

This module provides a minimal JNI bridge for Metal device probing on macOS.

## Build

```bash
metal-jni/scripts/build-metal-jni.sh
```

## Runtime loading

The Scala runtime tries to load JNI in this order:

1. `METAL_JNI_LIB` env var path
2. `metal-jni/build/libmetal_jni.dylib`
3. `System.loadLibrary("metal_jni")`

## Current status

- Device init/probe supported
- Kernel execution plumbing is scaffolded through backend architecture
- Compute ops currently use CPU fallback in `MetalBackend` while preserving API parity
