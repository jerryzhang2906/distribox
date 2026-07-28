/*
 * cmd/worker/agent/register.go — Worker registration and connection lifecycle
 */

package agent

import (
	"context"
	"fmt"
	"log"
	"runtime"
	"time"

	"github.com/distribox/cmd/worker/capability"
	"github.com/distribox/cmd/worker/monitor"
)

// ── Worker configuration ──────────────────────────────

type UserPolicy struct {
	Intensity        float64 // 0.0 - 1.0
	MaxRAMMB         int64   // 0 = auto
	MaxCPUCores      int     // 0 = auto
	OnlyWhenCharging bool
	OnlyOnWiFi       bool
}

type WorkerConfig struct {
	Name             string
	OrchestratorAddr string
	ClusterToken     string
	Capabilities     *capability.HardwareInfo
	Policy           UserPolicy
	ResourceMon      *monitor.ResourceMonitor
}

// ── Worker ────────────────────────────────────────────

type Worker struct {
	config   WorkerConfig
	workerID string
	sessionToken string
	connected    bool
	stopCh       chan struct{}
}

func NewWorker(config WorkerConfig) *Worker {
	return &Worker{
		config: config,
		stopCh: make(chan struct{}),
	}
}

// Connect establishes gRPC connection and registers with orchestrator
func (w *Worker) Connect() error {
	// TODO: actual gRPC dial + Register RPC
	// For MVP skeleton: simulate registration
	w.workerID = fmt.Sprintf("worker-%s-%x", runtime.GOOS, time.Now().Unix())
	w.sessionToken = fmt.Sprintf("sess-%x", time.Now().UnixNano())
	w.connected = true

	log.Printf("Registered as %s (name: %s)", w.workerID, w.config.Name)

	// Start heartbeat loop
	go w.heartbeatLoop()

	// Send initial capability report
	w.reportCapability()

	return nil
}

func (w *Worker) Disconnect() {
	close(w.stopCh)
	w.connected = false
	// TODO: gRPC graceful disconnect
}

// ── Heartbeat ─────────────────────────────────────────

func (w *Worker) heartbeatLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-w.stopCh:
			return
		case <-ticker.C:
			if !w.connected {
				return
			}
			// TODO: send gRPC heartbeat with current metrics
			dynamicMetrics := w.config.ResourceMon.Snapshot()
			_ = dynamicMetrics

			if dynamicMetrics.BatteryPct < 15 && w.config.Policy.OnlyWhenCharging && !dynamicMetrics.Charging {
				log.Printf("Battery low (%.0f%%) and not charging — pausing work", dynamicMetrics.BatteryPct)
				continue
			}
		}
	}
}

// ── Capability reporting ──────────────────────────────

func (w *Worker) reportCapability() {
	caps := w.config.Capabilities

	log.Printf("Capability report: CPU=%s, %d cores, %d MB RAM, GPU=%v",
		caps.CPUModel, caps.CPUCores, caps.TotalRAMMB, caps.HasGPU)
	if caps.HasGPU {
		log.Printf("  GPU: %s %s, %d MB VRAM, %d CUs, OpenCL=%s",
			caps.GPUVendor, caps.GPUModel, caps.GPUVramMB,
			caps.GPUComputeUnits, caps.OpenCLVersion)
	}
	log.Printf("  Policy: intensity=%.2f, max_cores=%d, max_ram=%d MB, only_charging=%v",
		w.config.Policy.Intensity, w.config.Policy.MaxCPUCores,
		w.config.Policy.MaxRAMMB, w.config.Policy.OnlyWhenCharging)

	// TODO: send gRPC CapabilityReport message
}

// ── Worker ID ─────────────────────────────────────────

func (w *Worker) WorkerID() string {
	return w.workerID
}

// Connected returns whether the worker is connected to orchestrator
func (w *Worker) Connected() bool {
	return w.connected
}

// ── Task execution context ────────────────────────────

// ExecuteTask prepares for and triggers local task execution
// In full implementation, this is called when a gRPC TaskAssignment arrives
func (w *Worker) ExecuteTask(ctx context.Context, taskID string, taskData []byte) error {
	log.Printf("Task %s received (%d bytes)", taskID, len(taskData))
	// TODO: call engine C library to execute the task
	return nil
}
