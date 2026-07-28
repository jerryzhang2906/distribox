/*
 * cmd/worker/monitor/resources.go — Real-time resource monitoring
 *
 * Periodically samples CPU, GPU, memory, battery, and thermal state.
 * Used for heartbeat reports and to decide if worker can accept tasks.
 */

package monitor

import (
	"math"
	"runtime"
	"sync"
	"time"
)

// ResourceSnapshot captures a point-in-time resource state
type ResourceSnapshot struct {
	CPUPct           float64
	GPUPct           float64
	MemoryPct        float64
	MemoryUsedMB     int64
	MemoryAvailableMB int64
	BatteryPct       float64
	Charging         bool
	ThermalThrottled bool
	TemperatureC     float64
	Timestamp        time.Time
}

// ResourceMonitor periodically samples system resources
type ResourceMonitor struct {
	latest ResourceSnapshot
	mu     sync.RWMutex
}

func NewResourceMonitor() *ResourceMonitor {
	return &ResourceMonitor{
		latest: ResourceSnapshot{
			BatteryPct: 100,
			Charging:   true,
			Timestamp:  time.Now(),
		},
	}
}

// Run starts periodic sampling (call as goroutine)
func (m *ResourceMonitor) Run() {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		m.sample()
	}
}

func (m *ResourceMonitor) sample() {
	snap := ResourceSnapshot{
		Timestamp: time.Now(),
	}

	// ── CPU ─────────────────────────────────────────
	// Use runtime metrics for CPU count
	numCPU := runtime.NumCPU()
	_ = numCPU
	// In a full implementation: read /proc/stat (Linux) or Performance Counter (Windows)
	snap.CPUPct = estimateCPUUsage()

	// ── Memory ──────────────────────────────────────
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	snap.MemoryUsedMB = int64(memStats.Alloc / (1024 * 1024))
	// Total system memory would come from OS APIs
	snap.MemoryAvailableMB = 4096 // Placeholder
	if snap.MemoryAvailableMB > 0 {
		snap.MemoryPct = float64(snap.MemoryUsedMB) / float64(snap.MemoryAvailableMB) * 100
	}

	// ── GPU ─────────────────────────────────────────
	// Platform-specific GPU utilization query
	snap.GPUPct = 0.0

	// ── Battery ─────────────────────────────────────
	// Platform-specific battery query
	// Android: BatteryManager API (via gomobile bridge)
	// Linux: /sys/class/power_supply/
	// Windows: GetSystemPowerStatus
	// macOS: IOKit
	snap.BatteryPct = 100.0
	snap.Charging = true

	// ── Thermal ─────────────────────────────────────
	snap.ThermalThrottled = false
	snap.TemperatureC = 45.0

	m.mu.Lock()
	m.latest = snap
	m.mu.Unlock()
}

// Snapshot returns the latest resource snapshot
func (m *ResourceMonitor) Snapshot() ResourceSnapshot {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.latest
}

// estimateCPUUsage provides a rough CPU usage estimate
// In production, this would use OS-specific APIs
func estimateCPUUsage() float64 {
	// Use a simple heuristic based on goroutine count
	numGoroutines := runtime.NumGoroutine()
	// Rough: more goroutines = more CPU usage (very approximate)
	return math.Min(float64(numGoroutines)/float64(runtime.NumCPU())*5.0, 100.0)
}
