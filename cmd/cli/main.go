/*
 * cmd/cli/main.go — Distribox CLI tool
 *
 * Usage:
 *   distribox device create [--name <n>] [--vram <size>] [--cu <n>] [--auto]
 *   distribox device status
 *   distribox device remove
 *   distribox worker list
 *   distribox worker set <id> --intensity <0.0-1.0>
 *   distribox version
 */

package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
)

const (
	defaultHTTPPort = 13801
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "device":
		handleDevice(os.Args[2:])
	case "worker":
		handleWorker(os.Args[2:])
	case "version":
		fmt.Println("distribox version 0.1.0")
	case "help", "--help", "-h":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`DistriBox — Distributed Virtual GPU

Usage:
  distribox device create   Create a virtual GPU device
  distribox device status   Show virtual GPU and worker status
  distribox device remove   Remove virtual GPU device
  distribox worker list     List connected workers
  distribox worker set      Configure worker settings
  distribox version         Show version

Device create options:
  --name <name>     Virtual GPU name (default: "DistriBox Virtual GPU")
  --vram <size>     VRAM size in GB (default: auto)
  --cu <n>          Compute units (default: auto)
  --clock <mhz>     Clock frequency in MHz (default: auto)
  --auto            Auto-compute specs from worker pool
  --vgpu-url <url>  VGPU Core HTTP address (default: http://localhost:13801)

Examples:
  distribox device create --auto
  distribox device status
  distribox worker set w-abc123 --intensity 0.5`)
}

// ── Device commands ────────────────────────────────────

func handleDevice(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: distribox device <create|status|remove>")
		os.Exit(1)
	}

	switch args[0] {
	case "create":
		deviceCreate(args[1:])
	case "status":
		deviceStatus(args[1:])
	case "remove":
		deviceRemove(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown device command: %s\n", args[0])
		os.Exit(1)
	}
}

func deviceCreate(args []string) {
	fs := flag.NewFlagSet("device create", flag.ExitOnError)
	name := fs.String("name", "DistriBox Virtual GPU", "Device name")
	vramGB := fs.Float64("vram", 0, "VRAM size in GB (0=auto)")
	cu := fs.Int("cu", 0, "Compute units (0=auto)")
	clock := fs.Int("clock", 0, "Clock MHz (0=auto)")
	auto := fs.Bool("auto", false, "Auto-compute from workers")
	vgpuURL := fs.String("vgpu-url", fmt.Sprintf("http://localhost:%d", defaultHTTPPort), "VGPU Core URL")
	fs.Parse(args)

	if *auto {
		fmt.Println("Auto-configuring virtual GPU from connected workers...")
		// Query VGPU Core for worker pool, compute optimal specs
		resp, err := httpGet(*vgpuURL + "/api/v1/workers")
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error: cannot reach VGPU Core at %s: %v\n", *vgpuURL, err)
			os.Exit(1)
		}

		var data struct {
			Workers []struct {
				ID              string  `json:"id"`
				Name            string  `json:"name"`
				CapabilityScore float64 `json:"capability_score"`
				AvailableRAMMB  int     `json:"available_ram_mb"`
				HasGPU          bool    `json:"has_gpu"`
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
			// Auto-compute: VRAM = 80% of total worker RAM, CU = sum of cores * 0.6
			*vramGB = float64(totalRAM) * 0.8 / 1024.0
			*cu = int(totalScore * 1000)
			if *vramGB < 1 {
				*vramGB = 1
			}
			if *cu < 64 {
				*cu = 64
			}
			fmt.Printf("Detected %d workers with %.1f GB total RAM, score=%.1f\n",
				len(data.Workers), float64(totalRAM)/1024.0, totalScore)
		}
	}

	spec := map[string]interface{}{
		"name":            *name,
		"vram_total":      uint64(*vramGB * 1024 * 1024 * 1024),
		"compute_units":   *cu,
		"max_clock_mhz":   *clock,
		"max_work_group_size": 256,
		"max_work_item_sizes": [3]uint64{1024, 1024, 64},
	}

	body, _ := json.Marshal(spec)
	_, err := httpPost(*vgpuURL+"/api/v1/device", "application/json", body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error configuring device: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Virtual GPU created: %s\n", *name)
	fmt.Printf("  VRAM: %.1f GB\n", *vramGB)
	fmt.Printf("  Compute Units: %d\n", *cu)
	fmt.Printf("  Clock: %d MHz\n", *clock)
	fmt.Println()
	fmt.Println("The device is now available for OpenCL applications.")
	fmt.Println("Run 'distribox device status' to check.")
}

func deviceStatus(args []string) {
	fs := flag.NewFlagSet("device status", flag.ExitOnError)
	vgpuURL := fs.String("vgpu-url", fmt.Sprintf("http://localhost:%d", defaultHTTPPort), "VGPU Core URL")
	fs.Parse(args)

	resp, err := httpGet(*vgpuURL + "/api/v1/status")
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: cannot reach VGPU Core: %v\n", err)
		fmt.Println("Is the Virtual GPU Core running? Start it with: distribox-vgpu")
		os.Exit(1)
	}

	var data struct {
		Device struct {
			Name          string `json:"name"`
			VRAMTotalMB   int    `json:"vram_total_mb"`
			VRAMUsedMB    int    `json:"vram_used_mb"`
			ComputeUnits  int    `json:"compute_units"`
			ClockMHz      int    `json:"clock_mhz"`
			BufferCount   int    `json:"buffer_count"`
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
	fmt.Printf("  VRAM: %d MB used / %d MB total\n",
		data.Device.VRAMUsedMB, data.Device.VRAMTotalMB)
	fmt.Printf("  Compute Units: %d\n", data.Device.ComputeUnits)
	fmt.Printf("  Clock: %d MHz\n", data.Device.ClockMHz)
	fmt.Printf("  Buffers: %d\n", data.Device.BufferCount)
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("Workers: %d active\n", data.ActiveWorkers)
	for _, w := range data.Workers {
		statusIcon := "●"
		if w.Status != "idle" {
			statusIcon = "○"
		}
		fmt.Printf("  %s %s (%s) — score=%.2f [%s]\n",
			statusIcon, w.Name, w.ID, w.Score, w.Status)
	}
}

func deviceRemove(args []string) {
	fs := flag.NewFlagSet("device remove", flag.ExitOnError)
	vgpuURL := fs.String("vgpu-url", fmt.Sprintf("http://localhost:%d", defaultHTTPPort), "VGPU Core URL")
	fs.Parse(args)

	fmt.Println("This will remove the virtual GPU device.")
	fmt.Print("Continue? [y/N] ")

	var answer string
	fmt.Scanln(&answer)
	if answer != "y" && answer != "Y" {
		fmt.Println("Cancelled.")
		return
	}

	req, _ := http.NewRequest("DELETE", *vgpuURL+"/api/v1/device", nil)
	http.DefaultClient.Do(req)
	fmt.Println("Virtual GPU device removed.")
}

// ── Worker commands ────────────────────────────────────

func handleWorker(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: distribox worker <list|set>")
		os.Exit(1)
	}

	switch args[0] {
	case "list":
		workerList(args[1:])
	case "set":
		workerSet(args[1:])
	default:
		fmt.Fprintf(os.Stderr, "Unknown worker command: %s\n", args[0])
		os.Exit(1)
	}
}

func workerList(args []string) {
	fs := flag.NewFlagSet("worker list", flag.ExitOnError)
	vgpuURL := fs.String("vgpu-url", fmt.Sprintf("http://localhost:%d", defaultHTTPPort), "VGPU Core URL")
	fs.Parse(args)

	// Reuse device status which includes worker info
	deviceStatus([]string{"--vgpu-url", *vgpuURL})
}

func workerSet(args []string) {
	if len(args) < 1 {
		fmt.Println("Usage: distribox worker set <worker-id> [--intensity <0.0-1.0>] [--only-charging] [--max-cores <n>] [--max-ram <mb>]")
		os.Exit(1)
	}

	workerID := args[0]
	fs := flag.NewFlagSet("worker set", flag.ExitOnError)
	intensity := fs.Float64("intensity", 0, "Compute intensity (0.0-1.0)")
	onlyCharging := fs.Bool("only-charging", false, "Only compute when charging")
	maxCores := fs.Int("max-cores", 0, "Max CPU cores")
	maxRAM := fs.Int("max-ram", 0, "Max RAM in MB")
	vgpuURL := fs.String("vgpu-url", fmt.Sprintf("http://localhost:%d", defaultHTTPPort), "VGPU Core URL")
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
	url := fmt.Sprintf("%s/api/v1/workers/%s/policy", *vgpuURL, workerID)
	_, err := httpPost(url, "application/json", body)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error setting worker policy: %v\n", err)
		os.Exit(1)
	}

	fmt.Printf("Worker %s updated.\n", workerID)
}

// ── HTTP helpers ───────────────────────────────────────

func httpGet(url string) ([]byte, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

func httpPost(url string, contentType string, body []byte) ([]byte, error) {
	resp, err := http.Post(url, contentType, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// ── stub for compilation until Go 1.22 ────────────────
func init() {
	// http.NewRequest exists but is unused in some code paths
	_ = http.NewRequest
}
