#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT_DIR/src/main/objectivec/MetalBridge.m"
OUT_DIR="$ROOT_DIR/build"
OUT_LIB="$OUT_DIR/libmetal_jni.dylib"

JAVA_HOME="${JAVA_HOME:-$(/usr/libexec/java_home)}"
JNI_INCLUDE="$JAVA_HOME/include"
JNI_INCLUDE_DARWIN="$JNI_INCLUDE/darwin"

mkdir -p "$OUT_DIR"

clang -fobjc-arc -dynamiclib \
  -I"$JNI_INCLUDE" \
  -I"$JNI_INCLUDE_DARWIN" \
  -framework Foundation \
  -framework Metal \
  "$SRC" \
  -o "$OUT_LIB"

echo "Built: $OUT_LIB"
