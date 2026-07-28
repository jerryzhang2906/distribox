#!/bin/bash
# scripts/build_android.sh — Build DistriBox Worker APK
#
# Prerequisites:
#   - Android NDK installed (set ANDROID_NDK_HOME)
#   - Android SDK installed (set ANDROID_HOME)
#   - Go 1.21+ with gomobile: go install golang.org/x/mobile/cmd/gomobile@latest
#   - CMake + Ninja for C cross-compilation
#
# Usage: ./scripts/build_android.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"

echo "=== DistriBox Android APK Builder ==="

# ── Check prerequisites ─────────────────────────────────
if [ -z "$ANDROID_NDK_HOME" ]; then
    echo "Error: ANDROID_NDK_HOME not set"
    echo "  Example: export ANDROID_NDK_HOME=~/Android/Sdk/ndk/26.1.10909125"
    exit 1
fi

# ── Step 1: Build C engine for Android ARM64 ────────────
echo ""
echo "[1/4] Building C engine for Android ARM64..."
cmake -B "$BUILD_DIR/engine-android" -S "$PROJECT_DIR/engine" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_HOME/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-26 \
    -DANDROID_STL=c++_shared
cmake --build "$BUILD_DIR/engine-android" --config Release

mkdir -p "$PROJECT_DIR/android/app/src/main/jniLibs/arm64-v8a"
cp "$BUILD_DIR/engine-android/libdistribox_engine.so" \
   "$PROJECT_DIR/android/app/src/main/jniLibs/arm64-v8a/"
echo "  Native engine library ready"

# ── Step 2: Build Go bridge via gomobile ────────────────
echo ""
echo "[2/4] Building Go bridge (gomobile)..."
cd "$PROJECT_DIR/cmd/worker/gobridge"

# Initialize gomobile if needed
gomobile init 2>/dev/null || true

# Build AAR
gomobile bind -target=android -androidapi=26 \
    -o "$BUILD_DIR/android/distribox.aar" \
    .

mkdir -p "$PROJECT_DIR/android/app/libs"
cp "$BUILD_DIR/android/distribox.aar" "$PROJECT_DIR/android/app/libs/"
echo "  AAR library ready"

# ── Step 3: Build APK via Gradle ────────────────────────
echo ""
echo "[3/4] Building APK with Gradle..."
cd "$PROJECT_DIR/android"
./gradlew assembleRelease
echo "  APK built"

# ── Step 4: Copy APK to build directory ─────────────────
echo ""
echo "[4/4] Copying APK to build..."
mkdir -p "$BUILD_DIR"
cp app/build/outputs/apk/release/app-release.apk "$BUILD_DIR/distribox-worker.apk"

echo ""
echo "=== Build Complete ==="
echo "APK: $BUILD_DIR/distribox-worker.apk"
ls -lh "$BUILD_DIR/distribox-worker.apk"
