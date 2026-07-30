/*
 * cmd/worker/monitor/resources.go — Real-time resource monitoring
 *
 * Periodically samples CPU, GPU, memory, battery, and thermal state.
 * Used for heartbeat reports and to decide if worker can accept tasks.
 */

package monitor

import (
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
	snap.CPUPct = sampleCPU()

	// ── Memory ──────────────────────────────────────
	snap.MemoryUsedMB, snap.MemoryAvailableMB = sampleMemory()
	if snap.MemoryAvailableMB > 0 {
		snap.MemoryPct = float64(snap.MemoryUsedMB) / float64(snap.MemoryAvailableMB) * 100
	}

	// ── GPU ─────────────────────────────────────────
	snap.GPUPct = sampleGPU()

	// ── Battery ─────────────────────────────────────
	snap.BatteryPct, snap.Charging = sampleBattery()

	// ── Thermal ─────────────────────────────────────
	snap.TemperatureC, snap.ThermalThrottled = sampleThermal()

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

