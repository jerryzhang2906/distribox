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
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net"
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
		localIP, err := discovery.GetLocalIP()
		if err != nil {
			localIP = "127.0.0.1"
		}

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
				if device.Role == "orchestrator" && device.Host != "" && device.Host != "unknown" {
					*orchestratorAddr = fmt.Sprintf("%s:%d", device.Host, device.Port)
					if device.ClusterToken != "" {
						*clusterToken = device.ClusterToken
					}
					log.Printf("Discovered orchestrator: %s (token=%s...)", *orchestratorAddr, device.ClusterToken[:8])
				}
			case <-time.After(15 * time.Second):
				log.Println("mDNS: no orchestrator found within 15s")
			}
			mdnsDisc.StopAdvertising()
		}

		// UDP broadcast fallback when mDNS failed
		if *orchestratorAddr == "localhost:13800" {
			log.Println("Trying UDP broadcast fallback...")
			discovered := udpBroadcastDiscover()
			if discovered != "" {
				*orchestratorAddr = discovered
				log.Printf("Discovered orchestrator via broadcast: %s", discovered)
			} else {
				log.Printf("No orchestrator found. Using %s (will retry in background)", *orchestratorAddr)
				go broadcastRetryLoop(hostname)
			}
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

// ── UDP broadcast discovery ────────────────────────────────

func udpBroadcastDiscover() string {
	conn, err := net.ListenUDP("udp4", &net.UDPAddr{Port: 13801})
	if err != nil {
		return ""
	}
	defer conn.Close()
	conn.SetDeadline(time.Now().Add(3 * time.Second))

	// Send discovery probe
	msg := []byte(`{"type":"discover","role":"worker","version":"1.0"}`)
	bcast := &net.UDPAddr{IP: net.IPv4bcast, Port: 13800}
	conn.WriteTo(msg, bcast)

	// Listen for response
	buf := make([]byte, 512)
	for {
		n, addr, err := conn.ReadFromUDP(buf)
		if err != nil {
			break
		}
		var resp struct {
			Type  string `json:"type"`
			Role  string `json:"role"`
			Port  int    `json:"port"`
			Token string `json:"token"`
		}
		if json.Unmarshal(buf[:n], &resp) == nil && resp.Type == "orchestrator_hello" {
			return fmt.Sprintf("%s:%d", addr.IP.String(), resp.Port)
		}
	}
	return ""
}

func broadcastRetryLoop(name string) {
	for {
		time.Sleep(30 * time.Second)
		addr := udpBroadcastDiscover()
		if addr != "" {
			log.Printf("Broadcast discovered orchestrator at %s", addr)
			return
		}
	}
}
