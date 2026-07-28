/*
 * cmd/distribox/main.go — DistriBox Unified Launcher
 *
 * Single executable that runs the complete DistriBox stack:
 *   - Virtual GPU Core (ICD server + gRPC orchestrator + HTTP dashboard)
 *   - Local Worker (CPU/GPU compute provider)
 *   - mDNS auto-discovery
 *
 * Usage:
 *   distribox.exe                           # Full stack (VGPU + Worker)
 *   distribox.exe --mode vgpu               # VGPU Core only
 *   distribox.exe --mode worker             # Worker only
 *   distribox.exe --mode both               # Full stack (default)
 *   distribox.exe --no-worker               # VGPU Core only (same as --mode vgpu)
 *   distribox.exe install                   # Install ICD + register services
 *   distribox.exe status                    # Show cluster status
 */

package main

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/pkg/discovery"
	"github.com/distribox/vgpu/calibrate"
	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/monitor"
	"github.com/distribox/vgpu/queue"
	"github.com/distribox/vgpu/scheduler"
	"github.com/distribox/vgpu/server"

	wagent "github.com/distribox/cmd/worker/agent"
	wcap "github.com/distribox/cmd/worker/capability"
	wmon "github.com/distribox/cmd/worker/monitor"
)

var (
	mode       = flag.String("mode", "both", "Run mode: vgpu, worker, both")
	noWorker   = flag.Bool("no-worker", false, "Disable local worker (same as --mode vgpu)")
	ipcAddr    = flag.String("ipc-addr", "127.0.0.1:9876", "IPC listen address")
	grpcPort   = flag.Int("grpc-port", 13800, "gRPC port")
	httpPort   = flag.Int("http-port", 13801, "HTTP dashboard port")
	workerName = flag.String("name", "", "Worker display name")
	intensity  = flag.Float64("intensity", 0.8, "Compute intensity (0.0-1.0)")
)

func main() {
	flag.Parse()

	if *noWorker {
		*mode = "vgpu"
	}

	// Handle subcommands
	args := flag.Args()
	if len(args) > 0 {
		switch args[0] {
		case "install":
			runInstall()
			return
		case "status":
			runStatus()
			return
		case "version":
			fmt.Println("DistriBox v0.2.0 — Distributed Virtual GPU")
			return
		case "help":
			flag.Usage()
			return
		}
	}

	log.SetFlags(log.LstdFlags | log.Lshortfile)
	printBanner()

	switch strings.ToLower(*mode) {
	case "vgpu":
		runVGPUOnly()
	case "worker":
		runWorkerOnly()
	default:
		runFullStack()
	}
}

func printBanner() {
	log.Println("╔══════════════════════════════════════════════╗")
	log.Println("║       DistriBox — Distributed Virtual GPU    ║")
	log.Println("║       v0.2.0  |  Unified Launcher            ║")
	log.Println("╚══════════════════════════════════════════════╝")
}

// ── Full Stack (VGPU Core + Local Worker) ──────────────

func runFullStack() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	log.Println("[MODE] Full stack — VGPU Core + Local Worker")

	// ── Generate cluster token ────────────────────────────
	clusterToken := generateToken()
	log.Printf("Cluster token: %s", clusterToken)

	// ── Start VGPU Core ───────────────────────────────────
	vgpuReady := make(chan struct{})
	go func() {
		if err := startVGPU(ctx, clusterToken, vgpuReady); err != nil {
			log.Fatalf("VGPU Core failed: %v", err)
		}
	}()

	// Wait for VGPU to be ready before starting worker
	select {
	case <-vgpuReady:
		log.Println("VGPU Core ready — starting local worker...")
	case <-time.After(5 * time.Second):
		log.Fatal("VGPU Core failed to start within 5s")
	}

	// ── Start Local Worker ────────────────────────────────
	go func() {
		if err := startWorker(ctx, fmt.Sprintf("localhost:%d", *grpcPort), clusterToken); err != nil {
			log.Printf("Local worker error: %v", err)
		}
	}()

	// ── Wait for shutdown ─────────────────────────────────
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")
	cancel()
	time.Sleep(500 * time.Millisecond)
	log.Println("DistriBox stopped.")
}

// ── VGPU Core Only ──────────────────────────────────────

func runVGPUOnly() {
	log.Println("[MODE] VGPU Core only")
	clusterToken := generateToken()
	log.Printf("Cluster token: %s", clusterToken)

	ctx := context.Background()
	vgpuReady := make(chan struct{})
	if err := startVGPU(ctx, clusterToken, vgpuReady); err != nil {
		log.Fatalf("VGPU Core failed: %v", err)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	log.Println("VGPU Core stopped.")
}

// ── Worker Only ─────────────────────────────────────────

func runWorkerOnly() {
	log.Println("[MODE] Worker only")

	ctx := context.Background()

	// Auto-discover orchestrator if not specified
	orchAddr := fmt.Sprintf("localhost:%d", *grpcPort)
	if flag.Lookup("orchestrator") == nil {
		mdnsDisc := discovery.New("worker", discovery.DeviceInfo{
			Name: getDefaultName(),
			Arch: runtime.GOARCH,
			OS:   runtime.GOOS,
		})
		ch, _ := mdnsDisc.Browse()
		select {
		case dev := <-ch:
			if dev.Role == "orchestrator" && dev.ClusterToken != "" {
				orchAddr = fmt.Sprintf("%s:%d", dev.Host, dev.Port)
				log.Printf("Discovered orchestrator: %s", orchAddr)
			}
		case <-time.After(10 * time.Second):
			log.Printf("No orchestrator found, using %s", orchAddr)
		}
		mdnsDisc.StopAdvertising()
	}

	if err := startWorker(ctx, orchAddr, ""); err != nil {
		log.Fatalf("Worker failed: %v", err)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh
	log.Println("Worker stopped.")
}

// ── VGPU Core startup ───────────────────────────────────

func startVGPU(ctx context.Context, clusterToken string, ready chan<- struct{}) error {
	vram := mem.NewVRAMManager()
	sched := scheduler.NewScheduler()
	workerMon := monitor.NewWorkerMonitor()
	cmdQueue := queue.NewCommandQueueManager()
	calEngine := calibrate.NewEngine(sched)

	orchestratorSvc := server.NewOrchestratorService(sched)
	orchestratorSvc.SetWorkerMonitor(workerMon)

	workerMon.OnWorkerLost = func(id string) { sched.RemoveWorker(id) }
	workerMon.OnWorkerReturn = func(id string) {
		sched.RegisterWorker(&scheduler.WorkerInfo{ID: id, Status: "idle"})
	}

	// IPC server
	ipcSrv, err := server.NewIPCServer(*ipcAddr, vram, cmdQueue, sched)
	if err != nil {
		return fmt.Errorf("IPC: %w", err)
	}
	ipcSrv.SetOrchestrator(orchestratorSvc)
	go func() {
		log.Printf("IPC: %s", *ipcAddr)
		ipcSrv.Serve()
	}()

	// gRPC server
	grpcListener, err := net.Listen("tcp", fmt.Sprintf(":%d", *grpcPort))
	if err != nil {
		return fmt.Errorf("gRPC: %w", err)
	}
	grpcSrv := grpc.NewServer()
	distriv1.RegisterOrchestratorServer(grpcSrv, orchestratorSvc)
	go func() {
		log.Printf("gRPC: :%d", *grpcPort)
		grpcSrv.Serve(grpcListener)
	}()

	// HTTP API
	apiHandler := server.NewAPIHandler(vram, sched, workerMon)
	http.HandleFunc("/api/v1/status", apiHandler.HandleStatus)
	http.HandleFunc("/api/v1/workers", apiHandler.HandleWorkers)
	http.HandleFunc("/api/v1/device", apiHandler.HandleDevice)
	http.HandleFunc("/", server.DashboardHandler)
	http.HandleFunc("/sse", server.SSEHandler)
	http.HandleFunc("/api/v1/icd/status", server.HandleICDStatus)
	http.HandleFunc("/api/v1/icd/install", server.HandleICDInstall)
	http.HandleFunc("/api/v1/icd/uninstall", server.HandleICDUninstall)
	go func() {
		log.Printf("Dashboard: http://localhost:%d", *httpPort)
		http.ListenAndServe(fmt.Sprintf(":%d", *httpPort), nil)
	}()

	server.StartDashboardCollector(sched, vram, workerMon)

	// Auto-calibration
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		time.Sleep(2 * time.Second)
		for range ticker.C {
			profile := calEngine.Recalibrate()
			if profile.WorkerCount > 0 {
				vram.UpdateSpec(mem.VirtualDeviceSpec{
					Name:             profile.MatchedGPU.Name,
					VRAMTotal:        profile.TotalVRAMMB * 1024 * 1024,
					ComputeUnits:     profile.MatchedGPU.ComputeUnits,
					MaxClockMHz:      profile.MatchedGPU.ClockMHz,
					MaxWorkGroupSize: 1024,
					MaxWorkItemSizes: [3]uint64{1024, 1024, 64},
				})
			}
		}
	}()

	go workerMon.Run()

	// mDNS
	hostname, _ := os.Hostname()
	mdnsDisc := discovery.New("orchestrator", discovery.DeviceInfo{
		Name:         hostname,
		Arch:         runtime.GOARCH,
		OS:           runtime.GOOS,
		ClusterToken: clusterToken,
	})
	if err := mdnsDisc.Advertise(*grpcPort); err != nil {
		log.Printf("mDNS: %v (non-fatal)", err)
	}

	ready <- struct{}{}
	return nil
}

// ── Worker startup ──────────────────────────────────────

func startWorker(ctx context.Context, orchAddr, token string) error {
	detector := wcap.NewDetector()
	caps := detector.Detect()

	log.Printf("Hardware: CPU=%s (%d cores), RAM=%d MB, GPU=%v",
		caps.CPUModel, caps.CPUCores, caps.TotalRAMMB, caps.HasGPU)
	if caps.HasGPU {
		log.Printf("  GPU: %s %s (%d MB VRAM, OpenCL %s)",
			caps.GPUVendor, caps.GPUModel, caps.GPUVramMB, caps.OpenCLVersion)
	}

	resMon := wmon.NewResourceMonitor()
	go resMon.Run()

	if token == "" {
		token = "insecure-lan-mode"
	}

	name := *workerName
	if name == "" {
		name = getDefaultName()
	}

	worker := wagent.NewGRPCWorkerClient(wagent.WorkerConfig{
		Name:             name,
		OrchestratorAddr: orchAddr,
		ClusterToken:     token,
		Capabilities:     caps,
		Policy:           wagent.UserPolicy{Intensity: *intensity},
		ResourceMon:      resMon,
	})

	if err := worker.Connect(ctx); err != nil {
		return fmt.Errorf("connect: %w", err)
	}

	go func() {
		if err := worker.Run(); err != nil {
			log.Printf("Worker run: %v", err)
		}
	}()

	log.Printf("Worker ready — connected to %s", orchAddr)
	return nil
}

// ── Subcommands ─────────────────────────────────────────

func runInstall() {
	fmt.Println("DistriBox Installation")
	fmt.Println()
	fmt.Println("To install the OpenCL ICD proxy (requires Administrator):")
	fmt.Println("  1. Copy build\\icd\\distribox_icd.dll to C:\\Windows\\System32\\")
	fmt.Println("  2. Register via registry:")
	fmt.Println("     reg add \"HKLM\\SOFTWARE\\Khronos\\OpenCL\\Vendors\" /v distribox_icd.dll /t REG_DWORD /d 0 /f")
	fmt.Println()
	fmt.Println("Or start VGPU Core and visit http://localhost:13801 to install from the dashboard.")
	fmt.Println()
	fmt.Println("To install Vulkan layer:")
	fmt.Println("  reg add \"HKLM\\SOFTWARE\\Khronos\\Vulkan\\ImplicitLayers\" /v distribox /d \"C:\\Windows\\System32\\distribox_vk_layer.json\" /t REG_SZ /f")
	fmt.Println()
	fmt.Println("To install CUDA proxy:")
	fmt.Println("  ren C:\\Windows\\System32\\nvcuda.dll nvcuda_orig.dll")
	fmt.Println("  copy build\\cuda\\nvcuda.dll C:\\Windows\\System32\\")
}

func runStatus() {
	fmt.Println("DistriBox Cluster Status")
	fmt.Println("────────────────────────")
	resp, err := httpGet(fmt.Sprintf("http://localhost:%d/api/v1/status", *httpPort))
	if err != nil {
		fmt.Println("VGPU Core not running. Start with: distribox.exe")
		return
	}
	fmt.Println(string(resp))
}

// ── Helpers ─────────────────────────────────────────────

func generateToken() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

func getDefaultName() string {
	hostname, _ := os.Hostname()
	if hostname != "" {
		return hostname
	}
	return fmt.Sprintf("PC-%s", runtime.GOARCH)
}

func httpGet(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	buf := make([]byte, 4096)
	n, _ := resp.Body.Read(buf)
	return buf[:n], nil
}
