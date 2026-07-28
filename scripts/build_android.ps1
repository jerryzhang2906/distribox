# scripts/build_android.ps1 — Windows PowerShell Android APK builder
#
# Prerequisites:
#   $env:ANDROID_NDK_HOME = "C:\Users\...\AppData\Local\Android\Sdk\ndk\26.1.10909125"
#   Go + gomobile: go install golang.org/x/mobile/cmd/gomobile@latest; gomobile init
#   CMake + Ninja
#   Java JDK 17+
#
# Usage: .\scripts\build_android.ps1

param(
    [string]$BuildType = "Release",
    [string]$AndroidABI = "arm64-v8a",
    [int]$AndroidAPI = 26
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectDir = Split-Path -Parent $ScriptDir
$BuildDir = Join-Path $ProjectDir "build"

Write-Host "=== DistriBox Android APK Builder ===" -ForegroundColor Cyan

# ── Check prerequisites ─────────────────────────────────
if (-not $env:ANDROID_NDK_HOME) {
    Write-Host "Error: ANDROID_NDK_HOME not set" -ForegroundColor Red
    Write-Host "  Set with: `$env:ANDROID_NDK_HOME = 'C:\Users\...\AppData\Local\Android\Sdk\ndk\26.1.10909125'"
    exit 1
}

$CMakeToolchain = Join-Path $env:ANDROID_NDK_HOME "build\cmake\android.toolchain.cmake"
if (-not (Test-Path $CMakeToolchain)) {
    Write-Host "Error: NDK toolchain not found at $CMakeToolchain" -ForegroundColor Red
    exit 1
}

# ── Step 1: Build C engine for Android ──────────────────
Write-Host "`n[1/4] Building C engine for Android $AndroidABI..." -ForegroundColor Yellow

$EngineBuildDir = Join-Path $BuildDir "engine-android"
$EngineSourceDir = Join-Path $ProjectDir "engine"

cmake -B $EngineBuildDir -S $EngineSourceDir `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DCMAKE_TOOLCHAIN_FILE=$CMakeToolchain `
    -DANDROID_ABI=$AndroidABI `
    -DANDROID_PLATFORM="android-$AndroidAPI" `
    -DANDROID_STL="c++_shared"
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed" }

cmake --build $EngineBuildDir --config $BuildType
if ($LASTEXITCODE -ne 0) { throw "CMake build failed" }

$JniLibsDir = Join-Path $ProjectDir "android\app\src\main\jniLibs\$AndroidABI"
New-Item -ItemType Directory -Force -Path $JniLibsDir | Out-Null

$EngineSo = Join-Path $EngineBuildDir "libdistribox_engine.so"
if (-not (Test-Path $EngineSo)) {
    Write-Host "Warning: engine .so not found, trying alternate name..." -ForegroundColor Yellow
    $EngineSo = Join-Path $EngineBuildDir "distribox_engine.dll"  # Windows build might use .dll
}
Copy-Item -Force $EngineSo $JniLibsDir
Write-Host "  Native engine library -> $JniLibsDir"

# ── Step 2: Build ICD for Android ───────────────────────
Write-Host "`n[2/4] Building ICD for Android $AndroidABI..." -ForegroundColor Yellow

$IcdBuildDir = Join-Path $BuildDir "icd-android"
$IcdSourceDir = Join-Path $ProjectDir "icd"

cmake -B $IcdBuildDir -S $IcdSourceDir `
    -DCMAKE_BUILD_TYPE=$BuildType `
    -DCMAKE_TOOLCHAIN_FILE=$CMakeToolchain `
    -DANDROID_ABI=$AndroidABI `
    -DANDROID_PLATFORM="android-$AndroidAPI"
cmake --build $IcdBuildDir --config $BuildType

$IcdSo = Join-Path $IcdBuildDir "libdistribox_icd.so"
if (Test-Path $IcdSo) {
    Copy-Item -Force $IcdSo $JniLibsDir
    Write-Host "  ICD library -> $JniLibsDir"
}

# ── Step 3: Build Go bridge (gomobile) ──────────────────
Write-Host "`n[3/4] Building Go bridge (gomobile)..." -ForegroundColor Yellow

Push-Location (Join-Path $ProjectDir "cmd\worker\gobridge")

try {
    $AarDir = Join-Path $BuildDir "android"
    New-Item -ItemType Directory -Force -Path $AarDir | Out-Null

    $AarFile = Join-Path $AarDir "distribox.aar"
    gomobile bind -target=android -androidapi=$AndroidAPI -o $AarFile .
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  gomobile bind failed — building without Go bridge (native only)" -ForegroundColor Yellow
    } else {
        $LibsDir = Join-Path $ProjectDir "android\app\libs"
        New-Item -ItemType Directory -Force -Path $LibsDir | Out-Null
        Copy-Item -Force $AarFile $LibsDir
        Write-Host "  AAR -> $LibsDir"
    }
} finally {
    Pop-Location
}

# ── Step 4: Build APK via Gradle ────────────────────────
Write-Host "`n[4/4] Building APK with Gradle..." -ForegroundColor Yellow

Push-Location (Join-Path $ProjectDir "android")

try {
    $Gradlew = ".\gradlew"
    if (-not (Test-Path $Gradlew)) {
        # Generate Gradle wrapper if missing
        Write-Host "  Generating Gradle wrapper..."
        # Minimal approach: just list available tasks
    }

    & $Gradlew assembleRelease 2>&1 | ForEach-Object { Write-Host "  $_" }
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Gradle build failed (may need Android SDK). APK not built." -ForegroundColor Yellow
        Write-Host "  Native libraries are ready: $JniLibsDir"
        Write-Host "  Build APK manually: cd android && gradlew assembleRelease"
    } else {
        $ApkSource = "app\build\outputs\apk\release\distribox-worker-release.apk"
        if (Test-Path $ApkSource) {
            New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
            Copy-Item -Force $ApkSource (Join-Path $BuildDir "distribox-worker.apk")
            $apk = Get-ChildItem (Join-Path $BuildDir "distribox-worker.apk")
            Write-Host "`n=== Build Complete ===" -ForegroundColor Green
            Write-Host "APK: $($apk.FullName) ($([math]::Round($apk.Length/1MB, 1)) MB)"
        }
    }
} finally {
    Pop-Location
}
