/*
 * cmd/worker/agent/heartbeat.go — Heartbeat message construction
 */

package agent

import (
	"time"
)

// HeartbeatData represents the information sent in each heartbeat
type HeartbeatData struct {
	TimestampUnixMs   int64
	WorkerID          string
	CPUUtilization    float64
	GPUUtilization    float64
	MemoryUsedPct     float64
	BatteryPct        float64
	Charging          bool
	ThermalThrottled  bool
	TemperatureC      float64
	NetworkBandwidthMbps uint32
}

// BuildHeartbeat constructs a heartbeat from current resource state
func (w *Worker) BuildHeartbeat() *HeartbeatData {
	snapshot := w.config.ResourceMon.Snapshot()

	return &HeartbeatData{
		TimestampUnixMs:   time.Now().UnixMilli(),
		WorkerID:          w.workerID,
		CPUUtilization:    snapshot.CPUPct,
		GPUUtilization:    snapshot.GPUPct,
		MemoryUsedPct:     snapshot.MemoryPct,
		BatteryPct:        snapshot.BatteryPct,
		Charging:          snapshot.Charging,
		ThermalThrottled:  snapshot.ThermalThrottled,
		TemperatureC:      snapshot.TemperatureC,
	}
}

// IsAvailable checks whether the worker can accept new tasks
func (w *Worker) IsAvailable() bool {
	if !w.connected {
		return false
	}

	snapshot := w.config.ResourceMon.Snapshot()
	policy := w.config.Policy

	// Check charging requirement
	if policy.OnlyWhenCharging && !snapshot.Charging {
		return false
	}

	// Check thermal throttle
	if snapshot.ThermalThrottled {
		return false
	}

	// Check battery threshold
	if snapshot.BatteryPct < 5 && !snapshot.Charging {
		return false
	}

	return true
}
