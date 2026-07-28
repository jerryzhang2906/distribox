/*
 * cmd/worker/main.go — Worker Agent entry point
 *
 * The Worker Agent runs on each device in the cluster.
 * It connects to the Virtual GPU Core (Orchestrator) via gRPC,
 * reports capabilities, receives compute tasks, and executes them
 * using the local C compute engine.
 *
 * Usage:
 *   distribox-worker --orchestrator <host:port> --token <cluster-token>
 */

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"runtime"
	"syscall"
	"time"

	"github.com/distribox/cmd/worker/agent"
	"github.com/distribox/cmd/worker/capability"
	"github.com/distribox/cmd/worker/monitor"
	"github.com/distribox/pkg/discovery"
)

var (
	orchestratorAddr = flag.String("orchestrator", "localhost:13800", "Orchestrator gRPC address")
	clusterToken     = flag.String("token", "", "Cluster authentication token")
	workerName       = flag.String("name", "", "Worker display name (default: hostname)")
	intensity        = flag.Float64("intensity", 1.0, "Compute intensity (0.0-1.0)")
	maxRAMMB         = flag.Int64("max-ram", 0, "Max RAM to use in MB (0=auto)")
	maxCPUCores      = flag.Int("max-cores", 0, "Max CPU cores (0=auto)")
	onlyWhenCharging = flag.Bool("only-charging", false, "Only compute when charging")
	onlyOnWiFi       = flag.Bool("only-wifi", false, "Only compute on WiFi")
)

func main() {
	flag.Parse()

	// ── mDNS auto-discovery (when no orchestrator specified) ──
	if *orchestratorAddr == "localhost:13800" && *clusterToken == "" {
		log.Println("No orchestrator specified — searching LAN via mDNS...")
		hostname, _ := os.Hostname()
		localIP, _ := discovery.GetLocalIP()

		mdnsDisc := discovery.New("worker", discovery.DeviceInfo{
			Name: hostname,
			Arch: runtime.GOARCH,
			OS:   runtime.GOOS,
		})

		ch, err := mdnsDisc.Browse()
		if err == nil {
			log.Println("mDNS browse started — waiting for orchestrator...")
			select {
			case device := <-ch:
				if device.Role == "orchestrator" {
					*orchestratorAddr = fmt.Sprintf("%s:%d", device.Host, device.Port)
					if device.ClusterToken != "" {
						*clusterToken = device.ClusterToken
					}
					log.Printf("Discovered orchestrator: %s (token=%s...)", *orchestratorAddr, device.ClusterToken[:8])
				}
			case <-time.After(10 * time.Second):
				log.Println("mDNS: no orchestrator found within 10s")
			}
			mdnsDisc.StopAdvertising()
		}
		_ = localIP
	}

	if *clusterToken == "" {
		log.Println("Warning: no cluster token — using insecure mode (LAN-only)")
		*clusterToken = "insecure-lan-mode"
	}

	if *workerName == "" {
		hostname, _ := os.Hostname()
		*workerName = hostname
	}

	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Printf("DistriBox Worker Agent starting on %s/%s...", runtime.GOOS, runtime.GOARCH)

	// ── Detect local hardware capabilities ──────────────
	detector := capability.NewDetector()
	caps := detector.Detect()
	log.Printf("Hardware: CPU=%s (%d cores), RAM=%d MB, GPU=%v",
		caps.CPUModel, caps.CPUCores, caps.TotalRAMMB, caps.HasGPU)
	if caps.HasGPU {
		log.Printf("GPU: %s %s (%d MB VRAM)", caps.GPUVendor, caps.GPUModel, caps.GPUVramMB)
	}

	// ── Start resource monitor ──────────────────────────
	resMon := monitor.NewResourceMonitor()
	go resMon.Run()

	// ── Apply user policy ───────────────────────────────
	policy := agent.UserPolicy{
		Intensity:       *intensity,
		MaxRAMMB:        *maxRAMMB,
		MaxCPUCores:     *maxCPUCores,
		OnlyWhenCharging: *onlyWhenCharging,
		OnlyOnWiFi:      *onlyOnWiFi,
	}

	// ── Connect to Orchestrator via gRPC ──────────────────
	worker := agent.NewGRPCWorkerClient(agent.WorkerConfig{
		Name:            *workerName,
		OrchestratorAddr: *orchestratorAddr,
		ClusterToken:    *clusterToken,
		Capabilities:    caps,
		Policy:          policy,
		ResourceMon:     resMon,
	})

	ctx := context.Background()
	if err := worker.Connect(ctx); err != nil {
		log.Fatalf("Failed to connect to orchestrator: %v", err)
	}
	defer worker.Disconnect()

	// Start the task receive loop in background
	go func() {
		if err := worker.Run(); err != nil {
			log.Printf("Worker run error: %v", err)
		}
	}()

	log.Printf("Connected to orchestrator at %s", *orchestratorAddr)
	log.Println("Worker ready — waiting for tasks...")

	// ── Wait for shutdown ───────────────────────────────
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Worker shutting down...")
}
