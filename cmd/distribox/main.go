/*
 * cmd/distribox/main.go — DistriBox Unified Launcher v0.3.0
 *
 * Single executable that runs the complete DistriBox stack:
 *   - Virtual GPU Core (IPC server + gRPC orchestrator + HTTP API)
 *   - Local Worker (CPU/GPU compute provider)
 *   - Live console status panel (replaces web dashboard)
 *   - mDNS auto-discovery
 *
 * Usage:
 *   distribox.exe                              # Full stack (VGPU + Worker + Console)
 *   distribox.exe --mode vgpu                  # VGPU Core only
 *   distribox.exe --mode worker                # Worker only
 *   distribox.exe --mode both                  # Full stack (default)
 *   distribox.exe --no-worker                  # VGPU Core only
 *   distribox.exe install                      # Install ICD + GPU interception
 *   distribox.exe status                       # Show cluster status
 *   distribox.exe version                      # Show version
 *   distribox.exe device create [--auto]       # Create virtual GPU
 *   distribox.exe device status                # Virtual GPU details
 *   distribox.exe device remove                # Remove virtual GPU
 *   distribox.exe worker list                  # List workers
 *   distribox.exe worker set <id> [opts]       # Configure worker
 */

package main

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"runtime"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"google.golang.org/grpc"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/pkg/discovery"
	"github.com/distribox/pkg/installer"
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

const (
	version = "v0.3.0"
)

var (
	mode             = flag.String("mode", "both", "Run mode: vgpu, worker, both")
	noWorker         = flag.Bool("no-worker", false, "Disable local worker")
	ipcAddr          = flag.String("ipc-addr", "127.0.0.1:9876", "IPC listen address")
	grpcPort         = flag.Int("grpc-port", 13800, "gRPC port")
	httpPort         = flag.Int("http-port", 13801, "HTTP API port")
	workerName       = flag.String("name", "", "Worker display name")
	intensity        = flag.Float64("intensity", 0.8, "Compute intensity (0.0-1.0)")
	orchestratorAddr = flag.String("orchestrator", "", "Remote orchestrator address")
	vgpuURL          = flag.String("vgpu-url", "", "VGPU Core HTTP URL (for CLI commands, default: http://localhost:13801)")
)

func main() {
	flag.Parse()

	if *noWorker {
		*mode = "vgpu"
	}

	// Handle subcommands (CLI mode)
	args := flag.Args()
	if len(args) > 0 {
		runSubcommand(args)
		return
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

// ── Subcommand dispatcher ────────────────────────────────

func runSubcommand(args []string) {
	switch args[0] {
	case "install":
		runInstall()
	case "status":
		runStatus()
	case "version":
		fmt.Printf("DistriBox %s — Distributed Virtual GPU\n", version)
		fmt.Println("Platform: " + runtime.GOOS + "/" + runtime.GOARCH)
	case "help", "--help", "-h":
		printUsage()
	case "device":
		handleDevice(args[1:])
	case "worker":
		handleWorker(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", args[0])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`DistriBox ` + version + ` — Distributed Virtual GPU

USAGE:
  distribox.exe                           Start full stack (VGPU + Worker + Console)
  distribox.exe --mode vgpu               VGPU Core only
  distribox.exe --mode worker             Worker only
  distribox.exe install                   Install ICD + GPU interception layers
  distribox.exe status                    Show cluster status
  distribox.exe version                   Show version
  distribox.exe device create [--auto]    Create virtual GPU device
  distribox.exe device status             Show virtual GPU details
  distribox.exe device remove             Remove virtual GPU device
  distribox.exe worker list               List connected workers
  distribox.exe worker set <id> [...]     Configure worker settings

FLAGS:
  --mode <vgpu|worker|both>    Run mode (default: both)
  --no-worker                  Disable local worker (same as --mode vgpu)
  --ipc-addr <addr>            IPC listen address (default: 127.0.0.1:9876)
  --grpc-port <port>           gRPC port (default: 13800)
  --http-port <port>           HTTP API port (default: 13801)
  --name <name>                Worker display name
  --intensity <0.0-1.0>        Compute intensity (default: 0.8)
  --orchestrator <addr>        Remote orchestrator address
  --vgpu-url <url>             VGPU Core URL for CLI commands

DEVICE CREATE OPTIONS:
  --name <name>     Virtual GPU name (default: "DistriBox Virtual GPU")
  --vram <gb>       VRAM size in GB (default: auto)
  --cu <n>          Compute units (default: auto)
  --clock <mhz>     Clock frequency in MHz (default: auto)
  --auto            Auto-compute specs from worker pool

WORKER SET OPTIONS:
  --intensity <0.0-1.0>   Compute intensity
  --only-charging          Only compute when charging
  --max-cores <n>          Max CPU cores
  --max-ram <mb>           Max RAM in MB

EXAMPLES:
  distribox.exe
  distribox.exe --mode worker --orchestrator 192.168.1.100:13800
  distribox.exe install
  distribox.exe device create --auto
  distribox.exe worker set w-abc123 --intensity 0.5`)
}

// ── Banner ───────────────────────────────────────────────

func printBanner() {
	enableVirtualTerminal()

	const cyan = "\033[36m"
	const purple = "\033[35m"
	const green = "\033[32m"
	const yellow = "\033[33m"
	const dim = "\033[2m"
	const reset = "\033[0m"
	const bold = "\033[1m"

	fmt.Print("\n")
	fmt.Print(purple + bold + "     ⚡ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ⚡\n" + reset)
	fmt.Print(purple + bold + "     ▐" + reset + cyan + "   DISTRIBOX — Distributed Virtual GPU   " + purple + "▌\n" + reset)
	fmt.Print(purple + bold + "     ▐" + reset + "  " + green + version + dim + "  |  Unified Launcher                 " + purple + "▌\n" + reset)
	fmt.Print(purple + bold + "     ▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌\n" + reset)
	fmt.Print(purple + bold + "     ▐" + reset + yellow + "  One GPU. Any Device. Zero Config.     " + purple + "▌\n" + reset)
	fmt.Print(purple + bold + "     ▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▌\n" + reset)
	fmt.Print("\n")
	fmt.Print(dim + "  Console  → live status panel\n" + reset)
	fmt.Print(dim + "  gRPC     → :13800  |  API → :13801  |  IPC → :9876\n" + reset)
	fmt.Print("\n")
}

func enableVirtualTerminal() {
	if runtime.GOOS != "windows" {
		return
	}
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	getStdHandle := kernel32.MustFindProc("GetStdHandle")
	getConsoleMode := kernel32.MustFindProc("GetConsoleMode")
	setConsoleMode := kernel32.MustFindProc("SetConsoleMode")

	const STD_OUTPUT_HANDLE = ^uintptr(0) - 11
	const ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004

	handle, _, _ := getStdHandle.Call(STD_OUTPUT_HANDLE)
	if handle == 0 || handle == ^uintptr(0) {
		return
	}
	var mode uint32
	ret, _, _ := getConsoleMode.Call(handle, uintptr(unsafe.Pointer(&mode)))
	if ret == 0 {
		return
	}
	setConsoleMode.Call(handle, uintptr(mode|ENABLE_VIRTUAL_TERMINAL_PROCESSING))
}

// ── Full Stack ───────────────────────────────────────────

func runFullStack() {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	log.Println("[MODE] Full stack — VGPU Core + Local Worker + Console")

	clusterToken := generateToken()
	log.Printf("Cluster token: %s", clusterToken)

	// Start VGPU Core
	vgpuReady := make(chan struct{})
	go func() {
		if err := startVGPU(ctx, clusterToken, vgpuReady); err != nil {
			log.Fatalf("VGPU Core failed: %v", err)
		}
	}()

	// Wait for VGPU to be ready
	select {
	case <-vgpuReady:
		log.Println("VGPU Core ready — starting local worker...")
	case <-time.After(5 * time.Second):
		log.Fatal("VGPU Core failed to start within 5s")
	}

	// Start Local Worker
	go func() {
		if err := startWorker(ctx, fmt.Sprintf("localhost:%d", *grpcPort), clusterToken); err != nil {
			log.Printf("Local worker error: %v", err)
		}
	}()

	// Start Console Panel (replaces web dashboard)
	stopConsole := make(chan struct{})
	console := NewConsolePanel(*httpPort)
	go console.Run(stopConsole)

	// Wait for shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")
	close(stopConsole)
	cancel()
	time.Sleep(500 * time.Millisecond)
	clearScreen()
	fmt.Println(cGreen + "  DistriBox stopped." + cReset)
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

	// Console in VGPU-only mode
	stopConsole := make(chan struct{})
	console := NewConsolePanel(*httpPort)
	go console.Run(stopConsole)

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	close(stopConsole)
	clearScreen()
	fmt.Println(cGreen + "  VGPU Core stopped." + cReset)
}

// ── Worker Only ─────────────────────────────────────────

func runWorkerOnly() {
	log.Println("[MODE] Worker only")

	ctx := context.Background()

	orchAddr := *orchestratorAddr
	if orchAddr == "" {
		orchAddr = fmt.Sprintf("localhost:%d", *grpcPort)
		mdnsDisc := discovery.New("worker", discovery.DeviceInfo{
			Name: getDefaultName(),
			Arch: runtime.GOARCH,
			OS:   runtime.GOOS,
		})
		ch, _ := mdnsDisc.Browse()
		select {
		case dev := <-ch:
			if dev.Role == "orchestrator" && dev.ClusterToken != "" {
				if dev.Port > 0 {
					orchAddr = fmt.Sprintf("%s:%d", dev.Host, dev.Port)
				} else {
					orchAddr = fmt.Sprintf("%s:%d", dev.Host, *grpcPort)
				}
				log.Printf("Discovered orchestrator: %s", orchAddr)
			}
		case <-time.After(10 * time.Second):
			log.Printf("No orchestrator found via mDNS, using %s", orchAddr)
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

	// HTTP API (no dashboard HTML — just REST + ICD management)
	apiHandler := server.NewAPIHandler(vram, sched, workerMon)
	http.HandleFunc("/api/v1/status", apiHandler.HandleStatus)
	http.HandleFunc("/api/v1/workers", apiHandler.HandleWorkers)
	http.HandleFunc("/api/v1/device", apiHandler.HandleDevice)
	http.HandleFunc("/", server.StatusPageHandler) // Simple JSON status instead of dashboard
	http.HandleFunc("/api/v1/icd/status", server.HandleICDStatus)
	http.HandleFunc("/api/v1/icd/install", server.HandleICDInstall)
	http.HandleFunc("/api/v1/icd/uninstall", server.HandleICDUninstall)
	http.HandleFunc("/api/v1/display/install", server.HandleDisplayAdapterInstall)
	http.HandleFunc("/api/v1/display/uninstall", server.HandleDisplayAdapterUninstall)
	http.HandleFunc("/api/v1/gl/install", server.HandleGLProxyInstall)
	http.HandleFunc("/sse", server.SSEHandler) // SSE for external monitoring
	go func() {
		log.Printf("API: http://localhost:%d", *httpPort)
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

func getVgpuURL() string {
	if *vgpuURL != "" {
		return *vgpuURL
	}
	return fmt.Sprintf("http://localhost:%d", *httpPort)
}

func runInstall() {
	fmt.Println(installer.InstallICD().Output)
}

func runStatus() {
	resp, err := httpGet(getVgpuURL() + "/api/v1/status")
	if err != nil {
		fmt.Println("VGPU Core not running. Start with: distribox.exe")
		return
	}

	var status struct {
		Device      map[string]interface{} `json:"device"`
		WorkerCount int                    `json:"worker_count"`
		Workers     []struct {
			ID     string  `json:"id"`
			Name   string  `json:"name"`
			Status string  `json:"status"`
			CPU    float64 `json:"cpu_pct"`
			GPU    float64 `json:"gpu_pct"`
			RAM    float64 `json:"ram_pct"`
			Score  float64 `json:"score"`
		} `json:"workers"`
		Uptime string `json:"uptime"`
	}
	if err := json.Unmarshal(resp, &status); err != nil {
		fmt.Println(string(resp))
		return
	}

	fmt.Println("╔════════════════════════════════════════════╗")
	fmt.Println("║      DistriBox — Cluster Status            ║")
	fmt.Println("╠════════════════════════════════════════════╣")
	if devName, ok := status.Device["name"].(string); ok {
		fmt.Printf("║  Virtual GPU: %-28s ║\n", devName)
	}
	if vram, ok := status.Device["vram_mb"].(float64); ok {
		fmt.Printf("║  VRAM:        %-8.0f MB               ║\n", vram)
	}
	if cu, ok := status.Device["compute_units"].(float64); ok {
		fmt.Printf("║  Compute Units: %-4.0f                    ║\n", cu)
	}
	fmt.Printf("║  Workers:     %-2d                        ║\n", status.WorkerCount)
	fmt.Println("╠════════════════════════════════════════════╣")
	if len(status.Workers) > 0 {
		fmt.Println("║  Worker List:                              ║")
		for _, w := range status.Workers {
			icon := "●"
			if w.Status != "idle" {
				icon = "○"
			}
			fmt.Printf("║    %s %-18s %-10s ║\n", icon, truncateStr(w.Name, 18), w.Status)
		}
	}
	fmt.Println("╚════════════════════════════════════════════╝")
}

// ── Device subcommands ──────────────────────────────────

func handleDevice(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: distribox device <create|status|remove>")
		os.Exit(1)
	}
	switch args[0] {
	case "create":
		deviceCreate(args[1:])
	case "status":
		deviceStatusCmd(args[1:])
	case "remove":
		deviceRemoveCmd(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown device command: %s\n", args[0])
		os.Exit(1)
	}
}

func deviceCreate(args []string) {
	fs := flag.NewFlagSet("device create", flag.ExitOnError)
	name := fs.String("name", "DistriBox Virtual GPU", "Device name")
	vramGB := fs.Float64("vram", 0, "VRAM size in GB")
	cu := fs.Int("cu", 0, "Compute units")
	clock := fs.Int("clock", 0, "Clock MHz")
	auto := fs.Bool("auto", false, "Auto-compute from workers")
	url := fs.String("vgpu-url", getVgpuURL(), "VGPU Core URL")
	fs.Parse(args)

	if *auto {
		fmt.Println("Auto-configuring virtual GPU from connected workers...")
		resp, err := httpGet(*url + "/api/v1/workers")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: cannot reach VGPU Core at %s: %v\n", *url, err)
			os.Exit(1)
		}

		var data struct {
			Workers []struct {
				CapabilityScore float64 `json:"capability_score"`
				AvailableRAMMB  int     `json:"available_ram_mb"`
			} `json:"workers"`
		}
		json.Unmarshal(resp, &data)

		totalRAM := 0
		totalScore := 0.0
		for _, w := range data.Workers {
			totalRAM += w.AvailableRAMMB
			totalScore += w.CapabilityScore
		}

		if len(data.Workers) == 0 {
			fmt.Println("No workers connected — using default specs")
			*vramGB = 8
			*cu = 2048
		} else {
			*vramGB = float64(totalRAM) * 0.8 / 1024.0
			*cu = int(totalScore * 1000)
			if *vramGB < 1 {
				*vramGB = 1
			}
			if *cu < 64 {
				*cu = 64
			}
			fmt.Printf("Detected %d workers, %.1f GB RAM, score=%.1f\n",
				len(data.Workers), float64(totalRAM)/1024.0, totalScore)
		}
	}

	spec := map[string]interface{}{
		"name":               *name,
		"vram_total":         uint64(*vramGB * 1024 * 1024 * 1024),
		"compute_units":      *cu,
		"max_clock_mhz":      *clock,
		"max_work_group_size": 256,
		"max_work_item_sizes": [3]uint64{1024, 1024, 64},
	}

	body, _ := json.Marshal(spec)
	_, err := httpPost(*url+"/api/v1/device", "application/json", body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error configuring device: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Virtual GPU created: %s\n", *name)
	fmt.Printf("  VRAM: %.1f GB | CU: %d | Clock: %d MHz\n", *vramGB, *cu, *clock)
}

func deviceStatusCmd(args []string) {
	fs := flag.NewFlagSet("device status", flag.ExitOnError)
	url := fs.String("vgpu-url", getVgpuURL(), "VGPU Core URL")
	fs.Parse(args)

	resp, err := httpGet(*url + "/api/v1/status")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: cannot reach VGPU Core: %v\n", err)
		os.Exit(1)
	}

	var data struct {
		Device struct {
			Name         string `json:"name"`
			VRAMTotalMB  int    `json:"vram_total_mb"`
			VRAMUsedMB   int    `json:"vram_used_mb"`
			ComputeUnits int    `json:"compute_units"`
			ClockMHz     int    `json:"clock_mhz"`
			BufferCount  int    `json:"buffer_count"`
		} `json:"device"`
		Workers []struct {
			ID     string  `json:"id"`
			Name   string  `json:"name"`
			Score  float64 `json:"score"`
			Status string  `json:"status"`
		} `json:"workers"`
		ActiveWorkers int `json:"active_workers"`
	}
	json.Unmarshal(resp, &data)

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("Virtual Device: %s\n", data.Device.Name)
	fmt.Printf("  VRAM: %d MB used / %d MB total\n", data.Device.VRAMUsedMB, data.Device.VRAMTotalMB)
	fmt.Printf("  CU: %d | Clock: %d MHz | Buffers: %d\n",
		data.Device.ComputeUnits, data.Device.ClockMHz, data.Device.BufferCount)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("Workers: %d active\n", data.ActiveWorkers)
	for _, w := range data.Workers {
		icon := "●"
		if w.Status != "idle" {
			icon = "○"
		}
		fmt.Printf("  %s %s (%s) score=%.2f [%s]\n", icon, w.Name, w.ID, w.Score, w.Status)
	}
}

func deviceRemoveCmd(args []string) {
	fs := flag.NewFlagSet("device remove", flag.ExitOnError)
	url := fs.String("vgpu-url", getVgpuURL(), "VGPU Core URL")
	fs.Parse(args)

	fmt.Print("Remove virtual GPU device? [y/N] ")
	var answer string
	fmt.Scanln(&answer)
	if answer != "y" && answer != "Y" {
		fmt.Println("Cancelled.")
		return
	}

	req, _ := http.NewRequest("DELETE", *url+"/api/v1/device", nil)
	http.DefaultClient.Do(req)
	fmt.Println("Virtual GPU device removed.")
}

// ── Worker subcommands ──────────────────────────────────

func handleWorker(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: distribox worker <list|set>")
		os.Exit(1)
	}
	switch args[0] {
	case "list":
		workerListCmd(args[1:])
	case "set":
		workerSetCmd(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown worker command: %s\n", args[0])
		os.Exit(1)
	}
}

func workerListCmd(args []string) {
	fs := flag.NewFlagSet("worker list", flag.ExitOnError)
	url := fs.String("vgpu-url", getVgpuURL(), "VGPU Core URL")
	fs.Parse(args)
	deviceStatusCmd([]string{"--vgpu-url", *url})
}

func workerSetCmd(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: distribox worker set <worker-id> [--intensity <0-1>] [--only-charging] [--max-cores <n>] [--max-ram <mb>]")
		os.Exit(1)
	}

	workerID := args[0]
	fs := flag.NewFlagSet("worker set", flag.ExitOnError)
	intensity := fs.Float64("intensity", 0, "Compute intensity")
	onlyCharging := fs.Bool("only-charging", false, "Only when charging")
	maxCores := fs.Int("max-cores", 0, "Max CPU cores")
	maxRAM := fs.Int("max-ram", 0, "Max RAM MB")
	url := fs.String("vgpu-url", getVgpuURL(), "VGPU Core URL")
	fs.Parse(args[1:])

	policy := map[string]interface{}{}
	if *intensity > 0 {
		policy["intensity"] = *intensity
	}
	if *onlyCharging {
		policy["only_when_charging"] = true
	}
	if *maxCores > 0 {
		policy["max_cpu_cores"] = *maxCores
	}
	if *maxRAM > 0 {
		policy["max_ram_mb"] = *maxRAM
	}

	body, _ := json.Marshal(policy)
	reqURL := fmt.Sprintf("%s/api/v1/workers/%s/policy", *url, workerID)
	_, err := httpPost(reqURL, "application/json", body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Worker %s updated.\n", workerID)
}

// ── HTTP helpers ────────────────────────────────────────

func httpGet(url string) ([]byte, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func httpPost(url string, contentType string, body []byte) ([]byte, error) {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(url, contentType, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
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
