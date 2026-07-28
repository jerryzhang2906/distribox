/*
 * vgpu/calibrate/calibrate.go — Auto-match virtual GPU to NVIDIA model
 *
 * Collects all worker capabilities, computes aggregate cluster performance,
 * and maps to the closest NVIDIA GPU model. Updates virtual device specs
 * and ICD-reported device info dynamically as workers join/leave.
 */

package calibrate

import (
	"fmt"
	"math"
	"sort"
	"sync"

	"github.com/distribox/vgpu/scheduler"
)

// ── NVIDIA GPU Database ────────────────────────────────

type GPUModel struct {
	Name        string  // GPU model name
	VRAMMB      uint64  // Video RAM in MB
	CUDAcores   uint32  // CUDA cores (or equivalent)
	ComputeUnits uint32 // OpenCL compute units
	TFLOPS      float64 // FP32 TFLOPS
	ClockMHz    uint32  // Base/boost clock
	Arch        string  // Architecture
	Tier        string  // "entry", "mid", "high", "enthusiast"
}

var nvidiaDB = []GPUModel{
	{Name: "NVIDIA GeForce GT 1030",       VRAMMB: 2048,  CUDAcores: 384,   ComputeUnits: 3,   TFLOPS: 1.1,  ClockMHz: 1468, Arch: "Pascal",     Tier: "entry"},
	{Name: "NVIDIA GeForce GTX 1050",      VRAMMB: 2048,  CUDAcores: 640,   ComputeUnits: 5,   TFLOPS: 1.8,  ClockMHz: 1455, Arch: "Pascal",     Tier: "entry"},
	{Name: "NVIDIA GeForce GTX 1050 Ti",   VRAMMB: 4096,  CUDAcores: 768,   ComputeUnits: 6,   TFLOPS: 2.1,  ClockMHz: 1392, Arch: "Pascal",     Tier: "entry"},
	{Name: "NVIDIA GeForce GTX 1060 6GB",  VRAMMB: 6144,  CUDAcores: 1280,  ComputeUnits: 10,  TFLOPS: 4.4,  ClockMHz: 1708, Arch: "Pascal",     Tier: "mid"},
	{Name: "NVIDIA GeForce GTX 1070",      VRAMMB: 8192,  CUDAcores: 1920,  ComputeUnits: 15,  TFLOPS: 6.5,  ClockMHz: 1683, Arch: "Pascal",     Tier: "mid"},
	{Name: "NVIDIA GeForce GTX 1080",      VRAMMB: 8192,  CUDAcores: 2560,  ComputeUnits: 20,  TFLOPS: 8.9,  ClockMHz: 1733, Arch: "Pascal",     Tier: "high"},
	{Name: "NVIDIA GeForce GTX 1080 Ti",   VRAMMB: 11264, CUDAcores: 3584,  ComputeUnits: 28,  TFLOPS: 11.3, ClockMHz: 1582, Arch: "Pascal",     Tier: "high"},
	{Name: "NVIDIA GeForce RTX 2060",      VRAMMB: 6144,  CUDAcores: 1920,  ComputeUnits: 30,  TFLOPS: 6.5,  ClockMHz: 1680, Arch: "Turing",     Tier: "mid"},
	{Name: "NVIDIA GeForce RTX 2070",      VRAMMB: 8192,  CUDAcores: 2304,  ComputeUnits: 36,  TFLOPS: 7.5,  ClockMHz: 1620, Arch: "Turing",     Tier: "mid"},
	{Name: "NVIDIA GeForce RTX 2080",      VRAMMB: 8192,  CUDAcores: 2944,  ComputeUnits: 46,  TFLOPS: 10.1, ClockMHz: 1710, Arch: "Turing",     Tier: "high"},
	{Name: "NVIDIA GeForce RTX 3060",      VRAMMB: 12288, CUDAcores: 3584,  ComputeUnits: 28,  TFLOPS: 12.7, ClockMHz: 1777, Arch: "Ampere",     Tier: "mid"},
	{Name: "NVIDIA GeForce RTX 3070",      VRAMMB: 8192,  CUDAcores: 5888,  ComputeUnits: 46,  TFLOPS: 20.3, ClockMHz: 1725, Arch: "Ampere",     Tier: "high"},
	{Name: "NVIDIA GeForce RTX 3080",      VRAMMB: 10240, CUDAcores: 8704,  ComputeUnits: 68,  TFLOPS: 29.8, ClockMHz: 1710, Arch: "Ampere",     Tier: "enthusiast"},
	{Name: "NVIDIA GeForce RTX 3090",      VRAMMB: 24576, CUDAcores: 10496, ComputeUnits: 82,  TFLOPS: 35.6, ClockMHz: 1695, Arch: "Ampere",     Tier: "enthusiast"},
	{Name: "NVIDIA GeForce RTX 4090",      VRAMMB: 24576, CUDAcores: 16384, ComputeUnits: 128, TFLOPS: 82.6, ClockMHz: 2520, Arch: "Ada Lovelace", Tier: "enthusiast"},
}

// ── Worker Performance Model ────────────────────────────

type WorkerPerf struct {
	WorkerID      string
	IsGPU         bool
	VRAMMB        uint64
	ComputeUnits  uint32
	EstTFLOPS     float64
	Cores         int
	CPUFreqMHz    float64
	RAMMB         uint64
}

// ── Aggregate Cluster Profile ──────────────────────────

type ClusterProfile struct {
	TotalVRAMMB      uint64
	TotalTFLOPS      float64
	TotalComputeUnits uint32
	TotalCores       int
	WorkerCount      int
	GPUWorkerCount   int
	MatchedGPU       GPUModel
	Confidence       float64 // 0-1 how well the profile fits
}

// ── Calibration Engine ─────────────────────────────────

type Engine struct {
	mu        sync.RWMutex
	sched     *scheduler.Scheduler
	profile   ClusterProfile
	onUpdate  func(ClusterProfile)
}

func NewEngine(sched *scheduler.Scheduler) *Engine {
	return &Engine{sched: sched}
}

// OnUpdate registers a callback when the cluster profile changes
func (e *Engine) OnUpdate(fn func(ClusterProfile)) {
	e.onUpdate = fn
}

// Profile returns current calibrated cluster profile
func (e *Engine) Profile() ClusterProfile {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.profile
}

// Recalibrate recalculates the cluster profile and finds best GPU match
func (e *Engine) Recalibrate() ClusterProfile {
	workers := e.sched.GetActiveWorkers()
	if len(workers) == 0 {
		e.mu.Lock()
		e.profile = ClusterProfile{MatchedGPU: nvidiaDB[0]}
		e.mu.Unlock()
		return e.profile
	}

	// Estimate per-worker performance
	var perfs []WorkerPerf
	for _, w := range workers {
		pf := estimateWorker(w)
		perfs = append(perfs, pf)
	}

	// Aggregate
	profile := aggregate(perfs)

	// Match to closest NVIDIA GPU
	profile.MatchedGPU = findBestMatch(profile)

	e.mu.Lock()
	e.profile = profile
	e.mu.Unlock()

	if e.onUpdate != nil {
		e.onUpdate(profile)
	}

	return profile
}

// ── Estimation ─────────────────────────────────────────

func estimateWorker(w *scheduler.WorkerInfo) WorkerPerf {
	pf := WorkerPerf{WorkerID: w.ID, IsGPU: w.HasGPU}
	ramMB := w.AvailableRAM / (1024 * 1024)

	if w.HasGPU {
		// GPU worker: ARM Mali / mobile GPU
		// VRAM from available memory
		pf.VRAMMB = ramMB / 2
		if pf.VRAMMB < 512 { pf.VRAMMB = 512 }
		if pf.VRAMMB > 8192 { pf.VRAMMB = 8192 }
		// Mali GPU compute units: ~8 CUs for mid-range
		pf.ComputeUnits = 8
		if w.CapabilityScore > 1.5 { pf.ComputeUnits = 16 }
		// ~0.4 TFLOPS per CU (Mali mid-range)
		pf.EstTFLOPS = float64(pf.ComputeUnits) * 0.35
		if pf.EstTFLOPS < 0.5 { pf.EstTFLOPS = 0.5 }
		// ARM cores for the device
		pf.Cores = 6
		pf.CPUFreqMHz = 2000
		pf.RAMMB = ramMB
	} else {
		// CPU worker: x86 desktop
		pf.RAMMB = ramMB
		if pf.RAMMB < 1024 { pf.RAMMB = 1024 }
		// Estimate cores from RAM: ~2GB per core typical
		pf.Cores = int(pf.RAMMB / 2048)
		if pf.Cores < 2 { pf.Cores = 2 }
		if pf.Cores > 32 { pf.Cores = 32 }
		// x86 CPU: ~0.1 TFLOPS per core (modern cores with AVX2)
		pf.EstTFLOPS = float64(pf.Cores) * 0.1
		if pf.EstTFLOPS < 0.15 { pf.EstTFLOPS = 0.15 }
		pf.CPUFreqMHz = 3300
	}

	return pf
}

func aggregate(perfs []WorkerPerf) ClusterProfile {
	cp := ClusterProfile{WorkerCount: len(perfs)}
	for _, pf := range perfs {
		cp.TotalTFLOPS += pf.EstTFLOPS
		cp.TotalCores += pf.Cores
		if pf.IsGPU {
			cp.TotalComputeUnits += pf.ComputeUnits
			cp.TotalVRAMMB += pf.VRAMMB
			cp.GPUWorkerCount++
		} else {
			// CPU cores contribute to "compute units" (4 cores ≈ 1 CU)
			cp.TotalComputeUnits += uint32(pf.Cores / 4)
			if cp.TotalComputeUnits < 1 { cp.TotalComputeUnits = 1 }
			// 25% of CPU RAM available as virtual VRAM
			cp.TotalVRAMMB += pf.RAMMB / 4
		}
	}
	// Ensure minimums
	if cp.TotalComputeUnits < 2 { cp.TotalComputeUnits = 2 }
	if cp.TotalVRAMMB < 512 { cp.TotalVRAMMB = 512 }
	return cp
}

// ── GPU Matching ───────────────────────────────────────

func findBestMatch(profile ClusterProfile) GPUModel {
	if profile.WorkerCount == 0 {
		return nvidiaDB[0] // GT 1030 as minimum
	}

	// Sort database by TFLOPS
	sorted := make([]GPUModel, len(nvidiaDB))
	copy(sorted, nvidiaDB)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].TFLOPS < sorted[j].TFLOPS
	})

	// Score each model based on multi-dimensional fit
	type scored struct {
		model GPUModel
		score float64
	}
	var candidates []scored

	for _, gpu := range sorted {
		// TFLOPS similarity (closest is best)
		tflopsDiff := math.Abs(gpu.TFLOPS-profile.TotalTFLOPS) / math.Max(gpu.TFLOPS, profile.TotalTFLOPS)

		// VRAM similarity
		vramDiff := math.Abs(float64(gpu.VRAMMB)-float64(profile.TotalVRAMMB)) /
			math.Max(float64(gpu.VRAMMB), float64(profile.TotalVRAMMB))

		// CU similarity
		cuDiff := math.Abs(float64(gpu.ComputeUnits)-float64(profile.TotalComputeUnits)) /
			math.Max(float64(gpu.ComputeUnits), float64(profile.TotalComputeUnits))

		// Combined score (lower is better)
		score := tflopsDiff*0.5 + vramDiff*0.3 + cuDiff*0.2

		candidates = append(candidates, scored{gpu, score})
	}

	// Find best match
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].score < candidates[j].score
	})

	best := candidates[0]

	// If we significantly UNDERperform the match, step down one tier
	if profile.TotalTFLOPS < best.model.TFLOPS*0.6 {
		for _, c := range candidates {
			if c.model.TFLOPS <= profile.TotalTFLOPS*1.5 &&
				c.model.TFLOPS >= profile.TotalTFLOPS*0.5 {
				best = c
				break
			}
		}
	}

	// If we significantly OVERperform the match, step up
	if profile.TotalTFLOPS > best.model.TFLOPS*1.8 && best.model.Tier != "enthusiast" {
		for _, c := range candidates {
			if c.model.TFLOPS > best.model.TFLOPS && c.model.TFLOPS <= profile.TotalTFLOPS*1.5 {
				best = c
				break
			}
		}
	}

	best.model.VRAMMB = profile.TotalVRAMMB // Override VRAM with actual
	best.model.ComputeUnits = profile.TotalComputeUnits

	return best.model
}

// ── Format helpers ─────────────────────────────────────

func FormatProfile(profile ClusterProfile) string {
	return fmt.Sprintf(
		"%s | %d MB VRAM | %.1f TFLOPS | %d CUs | %d workers (%d GPU)",
		profile.MatchedGPU.Name,
		profile.TotalVRAMMB,
		profile.TotalTFLOPS,
		profile.TotalComputeUnits,
		profile.WorkerCount,
		profile.GPUWorkerCount,
	)
}

func (g GPUModel) Summary() string {
	return fmt.Sprintf("%s (%s, %.1f TFLOPS, %d MB, %d CUs)",
		g.Name, g.Arch, g.TFLOPS, g.VRAMMB, g.ComputeUnits)
}
