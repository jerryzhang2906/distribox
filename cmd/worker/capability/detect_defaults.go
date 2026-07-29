/*
 * cmd/worker/capability/detect_defaults.go — Default hardware detection stubs
 *
 * Used on non-Windows platforms (Linux, macOS, Android).
 * These provide conservative defaults; platform-specific files can override.
 */

//go:build !windows

package capability

import (
	"fmt"
	"runtime"
)

func detectCPUPlatform(info *HardwareInfo) {
	switch runtime.GOOS {
	case "linux":
		info.CPUModel = "Generic x86_64 CPU"
		info.CPUFreqMHz = 2400
	case "darwin":
		info.CPUModel = "Apple Silicon"
		info.CPUFreqMHz = 3200
	default:
		info.CPUModel = fmt.Sprintf("Unknown %s CPU", runtime.GOARCH)
		info.CPUFreqMHz = 2000
	}

	// ISA features
	if runtime.GOARCH == "amd64" {
		info.ISAFatures = []string{"SSE", "SSE2", "SSE3", "SSE4", "AVX", "AVX2", "FMA"}
	} else if runtime.GOARCH == "arm64" {
		info.ISAFatures = []string{"NEON", "FP16"}
	}
}

func detectMemoryPlatform(info *HardwareInfo) {
	switch runtime.GOOS {
	case "android":
		info.TotalRAMMB = 4096
	default:
		info.TotalRAMMB = 8192
	}
	info.AvailableRAMMB = int(float64(info.TotalRAMMB) * 0.7)
	info.MemoryBwGBps = 25.0
}

func detectGPUPlatform(info *HardwareInfo) {
	if runtime.GOOS == "android" {
		info.HasGPU = true
		info.GPUVendor = "ARM"
		info.GPUModel = "Mali GPU"
		info.GPUVramMB = 2048
		info.GPUComputeUnits = 8
		info.GPUFreqMHz = 800
		info.OpenCLVersion = "OpenCL 2.0"
	} else {
		info.HasGPU = false
	}
}
