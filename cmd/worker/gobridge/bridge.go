/*
 * Go bridge for Android — exports Go functions to Java via gomobile.
 *
 * This package is compiled with `gomobile bind` to produce an AAR
 * that the Android app loads to run the Worker agent natively.
 *
 * Build: gomobile bind -target=android -androidapi=26 -o distribox.aar .
 */
package gobridge

import (
	"fmt"
	"time"

	"github.com/distribox/cmd/worker/agent"
	"github.com/distribox/cmd/worker/capability"
	"github.com/distribox/cmd/worker/monitor"
	"github.com/distribox/pkg/discovery"
)

var (
	workerClient *agent.GRPCWorkerClient
	resourceMon  *monitor.ResourceMonitor
)

// StartWorker starts the worker agent. Called from Java/Kotlin on service start.
// orchestratorAddr: "host:port" of the VGPU Core (empty for mDNS auto-discovery)
// token: cluster token (empty for mDNS auto-discovery)
// name: device name for display
// Returns: error message or empty string on success
func StartWorker(orchestratorAddr, token, name string) string {
	// Auto-discover orchestrator if not provided
	if orchestratorAddr == "" {
		discovered := discoverOrchestrator()
		if discovered == "" {
			// Will retry in background
			go autoConnectLoop(name)
			return "discovering"
		}
		orchestratorAddr = discovered
	}

	if token == "" {
		token = "insecure-lan-mode"
	}

	// Detect hardware capabilities
	detector := capability.NewDetector()
	caps := detector.Detect()

	// Start resource monitor
	resourceMon = monitor.NewResourceMonitor()
	go resourceMon.Run()

	// Create and connect worker
	workerClient = agent.NewGRPCWorkerClient(agent.WorkerConfig{
		Name:             name,
		OrchestratorAddr: orchestratorAddr,
		ClusterToken:     token,
		Capabilities:     caps,
		Policy:           agent.UserPolicy{Intensity: 0.8, OnlyWhenCharging: true},
		ResourceMon:      resourceMon,
	})

	err := workerClient.Connect(nil)
	if err != nil {
		return err.Error()
	}

	// Start task receive loop in background
	go func() {
		workerClient.Run()
	}()

	return ""
}

// StopWorker stops the worker agent.
func StopWorker() {
	if workerClient != nil {
		workerClient.Disconnect()
		workerClient = nil
	}
}

// WorkerStatus returns the current status of the worker.
func WorkerStatus() string {
	if workerClient == nil {
		return "stopped"
	}
	return "running"
}

// Discover searches for orchestrators on the LAN via mDNS.
// Returns the orchestrator address "host:port" or empty string.
func Discover() string {
	return discoverOrchestrator()
}

// DeviceInfo returns a summary of the device's hardware.
func DeviceInfo() string {
	detector := capability.NewDetector()
	caps := detector.Detect()
	if caps.HasGPU {
		return caps.GPUModel + " (" + caps.GPUVendor + ", " + caps.OpenCLVersion + ")"
	}
	return caps.CPUModel + " (" + caps.Arch + ")"
}

func discoverOrchestrator() string {
	mdnsDisc := discovery.New("worker", discovery.DeviceInfo{
		Name: "android-worker",
		Arch: "arm64",
		OS:   "android",
	})
	ch, err := mdnsDisc.Browse()
	if err != nil {
		return ""
	}
	defer mdnsDisc.StopAdvertising()

	select {
	case device := <-ch:
		if device.Role == "orchestrator" {
			return fmt.Sprintf("%s:%d", device.Host, device.Port)
		}
	default:
	}
	return ""
}

func autoConnectLoop(name string) {
	for {
		addr := discoverOrchestrator()
		if addr != "" {
			StartWorker(addr, "", name)
			return
		}
		time.Sleep(15 * time.Second)
	}
}
