/*
 * cmd/worker/capability/detect.go — Hardware capability detection
 *
 * Detects CPU, GPU, memory, and network specs of the local device.
 * On desktop: uses OS APIs. On Android: uses gomobile bridge.
 */

package capability

import (
	"fmt"
	"runtime"
)

// HardwareInfo represents the detected capabilities of the device
type HardwareInfo struct {
	// CPU
	CPUModel    string
	CPUCores    int
	CPULogical  int
	CPUFreqMHz  int
	ISAFatures  []string
	BenchGFLOPS float64 // Measured via micro-benchmark

	// GPU
	HasGPU       bool
	GPUVendor    string
	GPUModel     string
	GPUVramMB    int
	GPUComputeUnits int
	GPUFreqMHz   int
	OpenCLVersion string
	GPUBenchGFLOPS float64

	// Memory
	TotalRAMMB     int
	AvailableRAMMB int
	MemoryBwGBps   float64 // Estimated

	// OS
	OS   string
	Arch string
}

// Detector handles capability detection for current platform
type Detector struct{}

func NewDetector() *Detector {
	return &Detector{}
}

// Detect returns full hardware capabilities
func (d *Detector) Detect() *HardwareInfo {
	info := &HardwareInfo{
		OS:   runtime.GOOS,
		Arch: runtime.GOARCH,
	}

	d.detectCPU(info)
	d.detectMemory(info)
	d.detectGPU(info)

	return info
}

// ── CPU detection ─────────────────────────────────────

func (d *Detector) detectCPU(info *HardwareInfo) {
	info.CPUCores = runtime.NumCPU()
	info.CPULogical = runtime.NumCPU()

	// Detect CPU model and features
	// This is OS-specific. For MVP, we use runtime info + guesses.
	switch runtime.GOOS {
	case "linux":
		d.detectCPULinux(info)
	case "windows":
		d.detectCPUWindows(info)
	case "darwin":
		d.detectCPUDarwin(info)
	default:
		info.CPUModel = fmt.Sprintf("Unknown %s CPU", runtime.GOARCH)
	}

	// Detect ISA features
	if runtime.GOARCH == "amd64" {
		info.ISAFatures = []string{"SSE", "SSE2", "SSE3", "SSE4", "AVX", "AVX2", "FMA"}
	} else if runtime.GOARCH == "arm64" {
		info.ISAFatures = []string{"NEON", "FP16"}
	}

	// Estimate GFLOPS from core count * frequency * ISA factor
	// Rough: 4 FLOPS/cycle/core for AVX2, 8 for FMA
	info.BenchGFLOPS = float64(info.CPUCores) * float64(info.CPUFreqMHz) * 4.0 / 1000.0
}

func (d *Detector) detectCPULinux(info *HardwareInfo) {
	// Read /proc/cpuinfo
	info.CPUModel = "Generic x86_64 CPU"
	info.CPUFreqMHz = 2400
}

func (d *Detector) detectCPUWindows(info *HardwareInfo) {
	info.CPUModel = "Generic x64 CPU"
	info.CPUFreqMHz = 2400
}

func (d *Detector) detectCPUDarwin(info *HardwareInfo) {
	info.CPUModel = "Apple Silicon"
	info.CPUFreqMHz = 3200
}

// ── Memory detection ──────────────────────────────────

func (d *Detector) detectMemory(info *HardwareInfo) {
	// Platform-specific memory query
	// For MVP: use reasonable defaults
	switch runtime.GOOS {
	case "linux":
		info.TotalRAMMB = 8192
	case "windows":
		info.TotalRAMMB = 8192
	case "darwin":
		info.TotalRAMMB = 8192
	case "android":
		info.TotalRAMMB = 4096
	default:
		info.TotalRAMMB = 4096
	}

	// Available is 70% of total by default (user can change)
	info.AvailableRAMMB = int(float64(info.TotalRAMMB) * 0.7)
	info.MemoryBwGBps = 25.0 // DDR4 dual-channel estimate
}

// ── GPU detection ─────────────────────────────────────

func (d *Detector) detectGPU(info *HardwareInfo) {
	// Try to detect GPU via OpenCL or system APIs
	// For MVP: check for common GPUs
	if runtime.GOOS == "android" {
		// Android typically has ARM Mali or Adreno
		info.HasGPU = true
		info.GPUVendor = "ARM"
		info.GPUModel = "Mali GPU"
		info.GPUVramMB = 2048
		info.GPUComputeUnits = 8
		info.GPUFreqMHz = 800
		info.OpenCLVersion = "OpenCL 2.0"
	} else {
		// Desktop: check for NVIDIA/AMD via driver presence
		// For MVP: mark as "GPU present, specs TBD by engine"
		info.HasGPU = false // Will be updated by engine_opencl.c at startup
	}
}

// ── Benchmark (runs on first launch) ──────────────────

// RunMicroBenchmark executes a quick (~5 second) compute benchmark
// to get empirical GFLOPS measurements for the local device.
func (d *Detector) RunMicroBenchmark() {
	// TODO: run small matrix multiply via C engine
	// Measure GFLOPS = 2 * M*N*K * iterations / time
	// This provides runtime-specific scores that account for:
	//   - Actual CPU/GPU frequency under load
	//   - Thermal throttling
	//   - Background process interference
}
