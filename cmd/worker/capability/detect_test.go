/*
 * cmd/worker/capability/detect_test.go — Hardware detection tests
 *
 * Tests that capability detection returns real hardware data
 * without relying on hardcoded defaults. Cross-platform.
 */

package capability

import (
	"runtime"
	"testing"
)

func TestDetect_ReturnsNonNil(t *testing.T) {
	d := NewDetector()
	info := d.Detect()
	if info == nil {
		t.Fatal("Detect() returned nil")
	}
}

func TestDetect_HasArch(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if info.Arch == "" {
		t.Error("Arch is empty — should be amd64 or arm64")
	}
	if info.Arch != runtime.GOARCH {
		t.Errorf("Arch mismatch: detected=%s runtime=%s", info.Arch, runtime.GOARCH)
	}
}

func TestDetect_HasOS(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if info.OS == "" {
		t.Error("OS is empty — should be windows, linux, or android")
	}
	if info.OS != runtime.GOOS {
		t.Errorf("OS mismatch: detected=%s runtime=%s", info.OS, runtime.GOOS)
	}
}

func TestDetect_CPUInfoNotEmpty(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if info.CPUModel == "" {
		t.Error("CPUModel is empty")
	}
	if info.CPUCores <= 0 {
		t.Errorf("CPUCores is %d, expected > 0", info.CPUCores)
	}
	if info.CPULogical <= 0 {
		t.Errorf("CPULogical is %d, expected > 0", info.CPULogical)
	}
	if info.CPULogical < info.CPUCores {
		t.Errorf("CPULogical (%d) < CPUCores (%d) — logical must be >= physical",
			info.CPULogical, info.CPUCores)
	}
	if info.CPUFreqMHz <= 0 {
		t.Errorf("CPUFreqMHz is %d, expected > 0", info.CPUFreqMHz)
	}
	// Verify no hardcoded fake values
	if runtime.GOOS == "windows" && info.CPUModel == "Unknown x64 CPU" {
		t.Error("CPUModel is hardcoded fallback 'Unknown x64 CPU' — detection failed on Windows")
	}
	if runtime.GOOS == "android" && info.CPUModel == "Unknown arm64 CPU" {
		t.Error("CPUModel is hardcoded fallback — real detection should work on Android")
	}
}

func TestDetect_MemoryInfoNotEmpty(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if info.TotalRAMMB <= 0 {
		t.Errorf("TotalRAMMB is %d, expected > 0", info.TotalRAMMB)
	}
	if info.AvailableRAMMB <= 0 {
		t.Errorf("AvailableRAMMB is %d, expected > 0", info.AvailableRAMMB)
	}
	if info.AvailableRAMMB > info.TotalRAMMB {
		t.Errorf("AvailableRAMMB (%d) > TotalRAMMB (%d)", info.AvailableRAMMB, info.TotalRAMMB)
	}
	// Verify no hardcoded fake values matching exact default values
	// Old hardcoded default was 4096*0.7 ≈ 2867
	if runtime.GOOS == "android" && info.TotalRAMMB == 4096 && info.AvailableRAMMB == 2867 {
		t.Log("WARNING: memory values match old hardcoded defaults. This may be real, but verify.")
	}
}

func TestDetect_GPUInfo_DetectsOrNotDetectsCorrectly(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if !info.HasGPU {
		// Acceptable: some platforms genuinely don't have GPU
		t.Log("No GPU detected — verify this matches expected hardware")
		return
	}

	// GPU detected — verify fields are populated
	if info.GPUModel == "" {
		t.Error("HasGPU=true but GPUModel is empty")
	}
	if info.GPUVendor == "" {
		t.Error("HasGPU=true but GPUVendor is empty")
	}

	// Verify no hardcoded fake data on Android
	if runtime.GOOS == "android" {
		if info.GPUModel == "Mali GPU" {
			t.Error("GPUModel is generic 'Mali GPU' — real detection should return specific model")
		}
		if info.GPUVendor != "" && info.GPUVendor != "ARM" && info.GPUVendor != "Qualcomm" {
			// Unexpected vendor — could be real but verify
			t.Logf("Unexpected GPU vendor on Android: %s", info.GPUVendor)
		}
		// The old code had GPUVramMB=2048 as hardcoded. Verify different value.
		if info.GPUVramMB == 2048 && info.GPUComputeUnits == 8 && info.GPUFreqMHz == 800 {
			t.Log("WARNING: GPU values match old hardcoded defaults exactly. May be coincidence, verify.")
		}
	}
}

func TestDetect_HasISAFatures(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if len(info.ISAFatures) == 0 {
		t.Error("ISA features list is empty")
	}

	if info.Arch == "arm64" {
		hasNeon := false
		for _, f := range info.ISAFatures {
			if f == "NEON" {
				hasNeon = true
				break
			}
		}
		if !hasNeon {
			t.Error("ARM64 CPU missing NEON ISA feature")
		}
	}
}

func TestDetect_BenchGFLOPS_Estimate(t *testing.T) {
	d := NewDetector()
	info := d.Detect()

	if info.BenchGFLOPS <= 0 {
		t.Errorf("BenchGFLOPS is %f, expected > 0", info.BenchGFLOPS)
	}
	// GFLOPS estimate should be reasonable: cores * freq(MHz) * ops_per_cycle / 1000
	expectedMin := float64(info.CPUCores) * float64(info.CPUFreqMHz) * 0.5 / 1000.0
	if info.BenchGFLOPS < expectedMin {
		t.Errorf("BenchGFLOPS (%f) is too low for %d cores at %d MHz (min expected: %f)",
			info.BenchGFLOPS, info.CPUCores, info.CPUFreqMHz, expectedMin)
	}
}

func TestDetect_Consistency(t *testing.T) {
	// Run detect twice and verify results are consistent
	d := NewDetector()
	info1 := d.Detect()
	info2 := d.Detect()

	if info1.CPUCores != info2.CPUCores {
		t.Error("CPUCores changed between detections")
	}
	if info1.TotalRAMMB != info2.TotalRAMMB {
		t.Error("TotalRAMMB changed between detections")
	}
	if info1.HasGPU != info2.HasGPU {
		t.Error("HasGPU changed between detections")
	}
}

func TestNewDetector(t *testing.T) {
	d := NewDetector()
	if d == nil {
		t.Fatal("NewDetector() returned nil")
	}
}
