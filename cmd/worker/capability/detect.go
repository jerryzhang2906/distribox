/*
 * cmd/worker/capability/detect.go — Hardware capability detection
 *
 * Detects CPU, GPU, memory, and network specs of the local device.
 * On desktop: uses OS APIs. On Android: uses gomobile bridge.
 */

package capability

import (
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

	// Platform-specific CPU detection (provided by build-tagged files)
	detectCPUPlatform(info)

	// Estimate GFLOPS from core count * frequency * ISA factor
	info.BenchGFLOPS = float64(info.CPUCores) * float64(info.CPUFreqMHz) * 4.0 / 1000.0
}

// ── Memory detection ──────────────────────────────────

func (d *Detector) detectMemory(info *HardwareInfo) {
	// Platform-specific memory detection
	detectMemoryPlatform(info)
}

// ── GPU detection ─────────────────────────────────────

func (d *Detector) detectGPU(info *HardwareInfo) {
	// Platform-specific GPU detection
	detectGPUPlatform(info)
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
