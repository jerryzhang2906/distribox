/*
 * vgpu/main.go — Virtual GPU Core service entry point
 *
 * The VGPU Core is the central daemon running on the host PC.
 * It listens on:
 *   - TCP localhost for local ICD communication
 *   - gRPC for remote Worker communication
 *   - HTTP for user CLI/API control
 *
 * Usage:
 *   distribox-vgpu [--ipc-addr <addr>] [--grpc-port <port>] [--http-port <port>]
 */

package main

import (
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
	"syscall"
	"time"

	"google.golang.org/grpc"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/pkg/discovery"
	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/queue"
	"github.com/distribox/vgpu/scheduler"
	"github.com/distribox/vgpu/calibrate"
	"github.com/distribox/vgpu/server"
	"github.com/distribox/vgpu/monitor"
)

var (
	ipcAddr  = flag.String("ipc-addr", "127.0.0.1:9876", "IPC TCP listen address")
	grpcPort = flag.Int("grpc-port", 13800, "gRPC port for workers")
	httpPort = flag.Int("http-port", 13801, "HTTP API port for CLI/dashboard")
)

func main() {
	flag.Parse()
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	log.Println("DistriBox Virtual GPU Core starting...")

	// ── Initialize core subsystems ──────────────────────
	vram := mem.NewVRAMManager()
	sched := scheduler.NewScheduler()
	workerMon := monitor.NewWorkerMonitor()
	cmdQueue := queue.NewCommandQueueManager()
	calEngine := calibrate.NewEngine(sched)

	// ── Create orchestrator service (shared by gRPC and IPC) ──
	orchestratorSvc := server.NewOrchestratorService(sched)
	orchestratorSvc.SetWorkerMonitor(workerMon) // Wire worker health tracking into gRPC

	// ── Wire monitor callbacks to scheduler
	workerMon.OnWorkerLost = func(workerID string) {
		sched.RemoveWorker(workerID)
		log.Printf("Scheduler: removed lost worker %s", workerID)
	}
	workerMon.OnWorkerReturn = func(workerID string) {
		sched.RegisterWorker(&scheduler.WorkerInfo{
			ID:     workerID,
			Status: "idle",
		})
		log.Printf("Scheduler: re-registered recovered worker %s", workerID)
	}

	// ── Start IPC server (for local ICD communication) ──
	ipcSrv, err := server.NewIPCServer(*ipcAddr, vram, cmdQueue, sched)
	if err != nil {
		log.Fatalf("Failed to create IPC server: %v", err)
	}
	ipcSrv.SetOrchestrator(orchestratorSvc) // Wire gRPC dispatch into IPC
	go func() {
		log.Printf("IPC server listening on %s", *ipcAddr)
		if err := ipcSrv.Serve(); err != nil {
			log.Fatalf("IPC server error: %v", err)
		}
	}()

	// ── Start gRPC server (for remote Workers) ─────────
	grpcListener, err := net.Listen("tcp", fmt.Sprintf(":%d", *grpcPort))
	if err != nil {
		log.Fatalf("Failed to listen on gRPC port: %v", err)
	}
	grpcSrv := grpc.NewServer()
	distriv1.RegisterOrchestratorServer(grpcSrv, orchestratorSvc)
	go func() {
		log.Printf("gRPC server listening on :%d", *grpcPort)
		if err := grpcSrv.Serve(grpcListener); err != nil {
			log.Fatalf("gRPC server error: %v", err)
		}
	}()

	// ── mDNS advertisement (zero-config LAN discovery) ──
	clusterToken := generateToken()
	log.Printf("Cluster token: %s", clusterToken)

	hostname, _ := os.Hostname()
	mdnsDisc := discovery.New("orchestrator", discovery.DeviceInfo{
		Name:           hostname,
		Arch:           runtime.GOARCH,
		OS:             runtime.GOOS,
		ClusterToken:   clusterToken,
	})
	if err := mdnsDisc.Advertise(*grpcPort); err != nil {
		log.Printf("mDNS advertise failed (non-fatal): %v", err)
	}
	defer mdnsDisc.StopAdvertising()

	// ── Start HTTP API server (for CLI/dashboard) ──────
	apiHandler := server.NewAPIHandler(vram, sched, workerMon)
	http.HandleFunc("/api/v1/status", apiHandler.HandleStatus)
	http.HandleFunc("/api/v1/workers", apiHandler.HandleWorkers)
	http.HandleFunc("/api/v1/device", apiHandler.HandleDevice)
	// Dashboard + SSE + ICD management + Display Adapter
	http.HandleFunc("/", server.DashboardHandler)
	http.HandleFunc("/sse", server.SSEHandler)
	http.HandleFunc("/api/v1/icd/status", server.HandleICDStatus)
	http.HandleFunc("/api/v1/icd/install", server.HandleICDInstall)
	http.HandleFunc("/api/v1/icd/uninstall", server.HandleICDUninstall)
	http.HandleFunc("/api/v1/display/install", server.HandleDisplayAdapterInstall)
	http.HandleFunc("/api/v1/display/uninstall", server.HandleDisplayAdapterUninstall)
	http.HandleFunc("/api/v1/gl/install", server.HandleGLProxyInstall)
	go func() {
		log.Printf("HTTP Dashboard listening on http://localhost:%d", *httpPort)
		if err := http.ListenAndServe(fmt.Sprintf(":%d", *httpPort), nil); err != nil {
			log.Fatalf("HTTP API error: %v", err)
		}
	}()

	// ── Start dashboard data collector ─────────────────
	server.StartDashboardCollector(sched, vram, workerMon)

	// ── Auto-calibration loop: match GPU model to cluster ──
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		// Initial calibration
		time.Sleep(2 * time.Second)
		for range ticker.C {
			profile := calEngine.Recalibrate()
			if profile.WorkerCount > 0 {
				vram.UpdateSpec(mem.VirtualDeviceSpec{
					Name:           profile.MatchedGPU.Name,
					VRAMTotal:      profile.TotalVRAMMB * 1024 * 1024,
					ComputeUnits:   profile.MatchedGPU.ComputeUnits,
					MaxClockMHz:    profile.MatchedGPU.ClockMHz,
					MaxWorkGroupSize: 1024,
					MaxWorkItemSizes: [3]uint64{1024, 1024, 64},
				})
				log.Printf("🎯 Calibrated: %s (%.1f TFLOPS, %d MB VRAM, %d CUs)",
					profile.MatchedGPU.Name, profile.TotalTFLOPS,
					profile.TotalVRAMMB, profile.MatchedGPU.ComputeUnits)
			}
		}
	}()

	// ── Subscribe to lifecycle events ───────────────────
	go workerMon.Run()

	// Print status
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Println("Virtual GPU Core is ready.")
	log.Printf("  Dashboard: http://localhost:%d", *httpPort)
	log.Printf("  IPC:  %s", *ipcAddr)
	log.Printf("  gRPC: :%d", *grpcPort)
	log.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	log.Println("Waiting for workers to connect...")

	// ── Wait for shutdown signal ────────────────────────
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	<-sigCh

	log.Println("Shutting down...")
	grpcSrv.GracefulStop()
	ipcSrv.Close()
	log.Println("Virtual GPU Core stopped.")
}

func generateToken() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}
