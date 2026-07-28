/*
 * vgpu/server/dashboard.go — Web dashboard + SSE push + ICD management
 *
 * Serves the DistriBox Control Panel at http://localhost:13801/
 * SSE pushes real-time worker/task updates to the browser.
 */

package server

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/scheduler"
	"github.com/distribox/vgpu/monitor"
)

//go:embed dashboard.html
var dashboardHTML []byte

// ── Dashboard HTTP handler ─────────────────────────────

func DashboardHandler(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write(dashboardHTML)
}

// ── SSE (Server-Sent Events) Hub ──────────────────────

type SSEHub struct {
	clients    map[chan []byte]bool
	register   chan chan []byte
	unregister chan chan []byte
	broadcast  chan []byte
	mu         sync.RWMutex
}

var sseHub = &SSEHub{
	clients:    make(map[chan []byte]bool),
	register:   make(chan chan []byte),
	unregister: make(chan chan []byte),
	broadcast:  make(chan []byte, 256),
}

func (h *SSEHub) Run() {
	for {
		select {
		case ch := <-h.register:
			h.mu.Lock()
			h.clients[ch] = true
			h.mu.Unlock()
		case ch := <-h.unregister:
			h.mu.Lock()
			delete(h.clients, ch)
			close(ch)
			h.mu.Unlock()
		case msg := <-h.broadcast:
			h.mu.RLock()
			for ch := range h.clients {
				select {
				case ch <- msg:
				default:
				}
			}
			h.mu.RUnlock()
		}
	}
}

func BroadcastJSON(v interface{}) {
	data, err := json.Marshal(v)
	if err != nil {
		return
	}
	select {
	case sseHub.broadcast <- data:
	default:
	}
}

// ── SSE Handler ────────────────────────────────────────

func SSEHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", 500)
		return
	}

	ch := make(chan []byte, 64)
	sseHub.register <- ch
	defer func() { sseHub.unregister <- ch }()

	ctx := r.Context()
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case msg, ok := <-ch:
			if !ok {
				return
			}
			fmt.Fprintf(w, "data: %s\n\n", msg)
			flusher.Flush()
		case <-ticker.C:
			fmt.Fprintf(w, ": ping\n\n")
			flusher.Flush()
		}
	}
}

// ── Dashboard notifications ────────────────────────────

func NotifyWorkersUpdated(sched *scheduler.Scheduler, vram *mem.VRAMManager) {
	type WorkerInfo struct {
		ID              string  `json:"id"`
		Name            string  `json:"name"`
		CapabilityScore float64 `json:"capability_score"`
		AvailableRAMMB  int     `json:"available_ram_mb"`
		HasGPU          bool    `json:"has_gpu"`
		Status          string  `json:"status"`
	}

	var workers []WorkerInfo
	totalRAM, totalVRAM, totalCores, gpuCount := 0, 0, 0, 0
	for _, w := range sched.Workers {
		workers = append(workers, WorkerInfo{
			ID: w.ID, Name: w.Name,
			CapabilityScore: w.CapabilityScore,
			AvailableRAMMB:  int(w.AvailableRAM / (1024 * 1024)),
			HasGPU:          w.HasGPU,
			Status:          w.Status,
		})
		totalRAM += int(w.AvailableRAM / (1024 * 1024))
		if w.HasGPU {
			gpuCount++
			totalVRAM += 2048
		}
		totalCores += int(w.CapabilityScore)
	}

	spec := vram.GetSpec()
	BroadcastJSON(map[string]interface{}{
		"workers": workers,
		"cluster": map[string]interface{}{
			"totalRamMB": totalRAM, "totalVramMB": totalVRAM,
			"totalCores": totalCores, "gpuCount": gpuCount,
		},
		"gpu": map[string]interface{}{
			"name": spec.Name, "vram_total_mb": spec.VRAMTotal / (1024 * 1024),
			"compute_units": spec.ComputeUnits, "clock_mhz": spec.MaxClockMHz,
		},
	})
}

func NotifyTaskDispatched(kernel, worker, taskErr string) {
	BroadcastJSON(map[string]interface{}{
		"task": map[string]interface{}{
			"kernel": kernel, "worker": worker, "error": taskErr,
		},
	})
}

// ── ICD Management API ─────────────────────────────────

func HandleICDStatus(w http.ResponseWriter, r *http.Request) {
	sys32 := os.Getenv("SystemRoot") + "\\System32"
	proxyOK := fileExists(sys32 + "\\OpenCL_proxy.dll") || fileExists(sys32 + "\\OpenCL_orig.dll")
	backupOK := fileExists(sys32 + "\\OpenCL_orig.dll")
	icdPath := os.Getenv("LOCALAPPDATA") + "\\DistriBox\\distribox_icd.dll"

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"proxy_installed": proxyOK,
		"backup_exists":   backupOK,
		"icd_registered":  fileExists(icdPath),
		"icd_path":        icdPath,
		"platforms_found": "2 (Intel + DistriBox)",
	})
}

func HandleICDInstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}

	sys32 := os.Getenv("SystemRoot") + "\\System32"
	backupPath := sys32 + "\\OpenCL_orig.dll"
	openCLPath := sys32 + "\\OpenCL.dll"
	proxySrc := os.Getenv("TEMP") + "\\OpenCL_proxy.dll"
	icdSrc := os.Getenv("LOCALAPPDATA") + "\\DistriBox\\distribox_icd.dll"

	// Build PowerShell install script
	ps := fmt.Sprintf(
		`$elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator");
if(!$elevated){Write-Host "NEED_ADMIN";exit 1}
if(!(Test-Path '%s')){Copy-Item '%s' '%s' -Force;Write-Host "Backup:OK"}
if(Test-Path '%s'){Copy-Item -Force '%s' '%s';Write-Host "Proxy:OK"}
if(Test-Path '%s'){Copy-Item -Force '%s' '%s\distribox_icd.dll';Write-Host "ICD:OK"}
$reg=New-Item -Path 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' -Force;
New-ItemProperty -Path $reg.PSPath -Name 'distribox_icd.dll' -Value 0 -PropertyType DWord -Force;
Write-Host "ALL_DONE"`,
		backupPath, openCLPath, backupPath,
		proxySrc, proxySrc, openCLPath,
		icdSrc, icdSrc, sys32)

	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, err := cmd.CombinedOutput()
	result := string(out)
	log.Printf("ICD install: %s (err=%v)", result, err)

	success := err == nil
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  map[bool]string{true: "ok", false: "error"}[success],
		"output":  result,
		"message": "Check output for details. May require Administrator privileges.",
	})
}

func HandleICDUninstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}

	sys32 := os.Getenv("SystemRoot") + "\\System32"
	backupPath := sys32 + "\\OpenCL_orig.dll"
	openCLPath := sys32 + "\\OpenCL.dll"

	ps := fmt.Sprintf(
		`$elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator");
if(!$elevated){Write-Host "NEED_ADMIN";exit 1}
if(Test-Path '%s'){Copy-Item -Force '%s' '%s';Write-Host "RESTORED"}else{Write-Host "NO_BACKUP"}`,
		backupPath, backupPath, openCLPath)

	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, _ := cmd.CombinedOutput()
	result := string(out)

	success := !fileExists(openCLPath) || fileExists(sys32+"\\OpenCL_proxy.dll")
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": map[bool]string{true: "ok", false: "error"}[success],
		"output": result,
	})
}

// ── Display Adapter (Device Manager / Task Manager) ────

func HandleDisplayAdapterInstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}

	// Create registry entries for a software display adapter
	ps := `
$elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator");
if(!$elevated){Write-Host "NEED_ADMIN";exit 1}
$guid="{4d36e968-e325-11ce-bfc1-08002be10318}";
$key="HKLM:\SYSTEM\CurrentControlSet\Control\Class\$guid\0004";
New-Item -Path $key -Force | Out-Null;
Set-ItemProperty -Path $key -Name "DriverDesc" -Value "DistriBox Virtual GPU" -Force;
Set-ItemProperty -Path $key -Name "ProviderName" -Value "DistriBox Technologies" -Force;
Set-ItemProperty -Path $key -Name "DriverVersion" -Value "1.0.0.0" -Force;
Set-ItemProperty -Path $key -Name "DriverDate" -Value "2026-07-27" -Force;
Set-ItemProperty -Path $key -Name "MatchingDeviceId" -Value "ROOT\DISTRIBOX\0000" -Force;
Set-ItemProperty -Path $key -Name "HardwareInformation.MemSize" -Value 0x100000000 -Force;
# Also add to WMI-accessible GPU list via PNP
$pnpKey="HKLM:\SYSTEM\CurrentControlSet\Enum\ROOT\DISTRIBOX\0000";
New-Item -Path $pnpKey -Force | Out-Null;
Set-ItemProperty -Path $pnpKey -Name "DeviceDesc" -Value "DistriBox Virtual GPU (NVIDIA Compatible)" -Force;
Set-ItemProperty -Path $pnpKey -Name "HardwareID" -Value @("ROOT\DISTRIBOX") -Force;
Set-ItemProperty -Path $pnpKey -Name "ClassGUID" -Value $guid -Force;
Write-Host "DISPLAY_ADAPTER_INSTALLED"
`
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, _ := cmd.CombinedOutput()
	result := string(out)
	log.Printf("Display adapter install: %s", result)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status":  "ok",
		"output":  result,
		"message": "Display adapter registered. Reboot or restart 'Desktop Window Manager' to see in Task Manager.",
	})
}

func HandleDisplayAdapterUninstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}

	ps := `
$elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator");
if(!$elevated){Write-Host "NEED_ADMIN";exit 1}
$guid="{4d36e968-e325-11ce-bfc1-08002be10318}";
Remove-Item -Path "HKLM:\SYSTEM\CurrentControlSet\Control\Class\$guid\0004" -Recurse -Force -ErrorAction SilentlyContinue;
Remove-Item -Path "HKLM:\SYSTEM\CurrentControlSet\Enum\ROOT\DISTRIBOX" -Recurse -Force -ErrorAction SilentlyContinue;
Write-Host "DISPLAY_ADAPTER_REMOVED"
`
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, _ := cmd.CombinedOutput()

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
		"output": string(out),
	})
}

// ── Helpers ────────────────────────────────────────────

// ── OpenGL Proxy Install ──────────────────────────────

func HandleGLProxyInstall(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", 405)
		return
	}

	sys32 := os.Getenv("SystemRoot") + "\\System32"
	proxySrc := "I:\\game\\distribox\\build\\gl_proxy\\distri_opengl32.dll"
	backupPath := sys32 + "\\opengl32_orig.dll"
	openGLPath := sys32 + "\\opengl32.dll"
	proxyDest := sys32 + "\\opengl32_proxy.dll"

	// Copy proxy to System32 first (different name, no conflict)
	ps := fmt.Sprintf(`
$elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator")
if(!$elevated){Write-Host "NEED_ADMIN";exit 1}
if(!(Test-Path '%s')){Copy-Item '%s' '%s' -Force;Write-Host "BACKUP_OK"}else{Write-Host "BACKUP_EXISTS"}
Copy-Item -Force '%s' '%s';Write-Host "PROXY_COPIED"
# Schedule replacement on next reboot
$regPath="HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager"
$val=Get-ItemProperty -Path $regPath -Name PendingFileRenameOperations -ErrorAction SilentlyContinue
$ops=@()
if($val.PendingFileRenameOperations){$ops=$val.PendingFileRenameOperations}
$ops+="\\??\\%s"+"!"+"\\??\\%s"
New-ItemProperty -Path $regPath -Name PendingFileRenameOperations -Value $ops -PropertyType MultiString -Force
Write-Host "REBOOT_SCHEDULED — restart to complete GL proxy install"`,
		backupPath, openGLPath, backupPath,
		proxySrc, proxyDest,
		proxyDest, openGLPath)

	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, err := cmd.CombinedOutput()
	result := string(out)

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": map[bool]string{true: "ok", false: "error"}[err == nil],
		"output": result,
		"message": "GL proxy will replace opengl32.dll after reboot.",
	})
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// ── Start dashboard services ────────────────────────────

func StartDashboardCollector(sched *scheduler.Scheduler, vram *mem.VRAMManager, workerMon *monitor.WorkerMonitor) {
	go sseHub.Run()

	go func() {
		ticker := time.NewTicker(3 * time.Second)
		defer ticker.Stop()
		for range ticker.C {
			NotifyWorkersUpdated(sched, vram)
		}
	}()

	log.Printf("Dashboard ready — open http://localhost:13801/")
}
