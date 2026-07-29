/*
 * cmd/distribox/console.go — Live Console Status Panel
 *
 * Replaces the web dashboard with a live-updating terminal display.
 * Uses ANSI escape codes for color, cursor control, and real-time refresh.
 * Refreshes every 2 seconds with VGPU specs, worker list, cluster stats.
 */

package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"time"
)

const (
	cReset  = "\033[0m"
	cBold   = "\033[1m"
	cDim    = "\033[2m"
	cCyan   = "\033[36m"
	cGreen  = "\033[32m"
	cYellow = "\033[33m"
	cRed    = "\033[31m"
	cPurple = "\033[35m"
	cBlue   = "\033[34m"
	cWhite  = "\033[37m"
)

// clearScreen clears the terminal and moves cursor to top-left
func clearScreen() {
	fmt.Print("\033[2J\033[H")
}

// hideCursor hides the terminal cursor
func hideCursor() {
	fmt.Print("\033[?25l")
}

// showCursor restores the terminal cursor
func showCursor() {
	fmt.Print("\033[?25h")
}

// ConsolePanel manages the live console display
type ConsolePanel struct {
	httpPort int
	apiURL   string
	running  bool
}

// NewConsolePanel creates a new console panel
func NewConsolePanel(httpPort int) *ConsolePanel {
	return &ConsolePanel{
		httpPort: httpPort,
		apiURL:   fmt.Sprintf("http://localhost:%d", httpPort),
	}
}

// Run starts the live console display, refreshing every 2 seconds
func (cp *ConsolePanel) Run(stopCh <-chan struct{}) {
	hideCursor()
	defer showCursor()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	// Draw immediately, then refresh on tick
	cp.draw()
	for {
		select {
		case <-stopCh:
			clearScreen()
			return
		case <-ticker.C:
			cp.draw()
		}
	}
}

func (cp *ConsolePanel) draw() {
	clearScreen()
	cp.drawHeader()
	cp.drawVGpuSpec()
	cp.drawWorkers()
	cp.drawFooter()
}

func (cp *ConsolePanel) drawHeader() {
	fmt.Print(cCyan + cBold)
	fmt.Println("  ⚡ DISTRIBOX — Distributed Virtual GPU")
	fmt.Print(cReset)
	fmt.Println(cDim + "  One GPU. Any Device. Zero Config." + cReset)
	fmt.Println(cDim + "  " + strings.Repeat("─", 56) + cReset)
	fmt.Println()
}

func (cp *ConsolePanel) drawVGpuSpec() {
	resp, err := cp.apiGet("/api/v1/status")
	if err != nil {
		fmt.Println(cRed + "  ⚠ VGPU Core not responding..." + cReset)
		fmt.Println()
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
	}

	if err := json.Unmarshal(resp, &status); err != nil {
		fmt.Println(cYellow + "  Status parse error" + cReset)
		return
	}

	// Virtual GPU card
	fmt.Print(cBold + cWhite)
	fmt.Println("  ┌── Virtual GPU " + strings.Repeat("─", 41) + "┐")
	fmt.Print(cReset)

	name := "DistriBox Virtual GPU"
	vramMB := 0.0
	cu := 0.0
	if n, ok := status.Device["name"].(string); ok {
		name = n
	}
	if v, ok := status.Device["vram_mb"].(float64); ok {
		vramMB = v
	}
	if c, ok := status.Device["compute_units"].(float64); ok {
		cu = c
	}

	fmt.Printf("  │  "+cCyan+"%-30s"+cReset+"  %10s │\n", "Model", name)
	fmt.Printf("  │  VRAM:       %8.0f MB"+strings.Repeat(" ", 18)+"│\n", vramMB)
	fmt.Printf("  │  Compute Units: %6.0f"+strings.Repeat(" ", 32)+"│\n", cu)
	fmt.Printf("  │  Workers:        %4d"+strings.Repeat(" ", 33)+"│\n", status.WorkerCount)
	fmt.Println(cWhite + "  └" + strings.Repeat("─", 54) + "┘" + cReset)
	fmt.Println()
}

func (cp *ConsolePanel) drawWorkers() {
	resp, err := cp.apiGet("/api/v1/workers")
	if err != nil {
		return
	}

	var data struct {
		Workers []struct {
			ID              string  `json:"id"`
			Name            string  `json:"name"`
			Status          string  `json:"status"`
			CapabilityScore float64 `json:"capability_score"`
			AvailableRAMMB  int     `json:"available_ram_mb"`
			HasGPU          bool    `json:"has_gpu"`
			CPU             float64 `json:"cpu_pct"`
			GPU             float64 `json:"gpu_pct"`
			RAM             float64 `json:"ram_pct"`
		} `json:"workers"`
	}
	json.Unmarshal(resp, &data)

	fmt.Print(cBold + cWhite)
	fmt.Println("  ┌── Workers " + strings.Repeat("─", 46) + "┐")
	fmt.Print(cReset)

	if len(data.Workers) == 0 {
		fmt.Println("  │  " + cDim + "No workers connected. Waiting..." + cReset + strings.Repeat(" ", 25) + "│")
	} else {
		// Header
		fmt.Printf("  │  %-20s %-8s %6s %6s %6s │\n", "Name", "Status", "CPU%", "GPU%", "RAM%")
		fmt.Println("  │  " + strings.Repeat("─", 49) + " │")
		for _, w := range data.Workers {
			statusColor := cGreen
			statusIcon := "●"
			switch w.Status {
			case "busy":
				statusColor = cYellow
				statusIcon = "◉"
			case "offline":
				statusColor = cRed
				statusIcon = "○"
			case "idle":
				statusColor = cGreen
				statusIcon = "●"
			}

			gpuIndicator := "  —"
			if w.HasGPU {
				gpuIndicator = "GPU"
			}

			fmt.Printf("  │  %-16s %s%-2s "+cReset+"%4s %5.0f%% %5.0f%% %5.0f%% │\n",
				truncateStr(w.Name, 16),
				statusColor, statusIcon,
				gpuIndicator,
				w.CPU, w.GPU, w.RAM)
		}
	}
	fmt.Println(cWhite + "  └" + strings.Repeat("─", 54) + "┘" + cReset)
	fmt.Println()
}

func (cp *ConsolePanel) drawFooter() {
	fmt.Println(cDim + "  " + strings.Repeat("─", 56) + cReset)
	fmt.Printf(cDim+"  gRPC :%d  |  API :%d  |  Refresh: 2s"+cReset+"\n", cp.httpPort-1, cp.httpPort)
	fmt.Println()
	fmt.Println(cDim + "  Commands: [q] quit  [i] install ICD  [s] status JSON" + cReset)
}

func (cp *ConsolePanel) apiGet(path string) ([]byte, error) {
	client := &http.Client{Timeout: 3 * time.Second}
	resp, err := client.Get(cp.apiURL + path)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	return io.ReadAll(resp.Body)
}

// ── Console helpers ──────────────────────────────────────

func truncateStr(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max-1] + "…"
}

// getTerminalWidth attempts to detect terminal width
func getTerminalWidth() int {
	if runtime.GOOS == "windows" {
		cmd := exec.Command("powershell", "-NoProfile", "-Command", "$host.UI.RawUI.WindowSize.Width")
		out, _ := cmd.Output()
		var w int
		fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &w)
		if w > 0 {
			return w
		}
	}
	// Fallback: try stty on unix-like
	cmd := exec.Command("stty", "size")
	out, _ := cmd.Output()
	var rows, cols int
	fmt.Sscanf(strings.TrimSpace(string(out)), "%d %d", &rows, &cols)
	if cols > 0 {
		return cols
	}
	return 80
}

// clearLine clears the current terminal line
func clearLine() {
	fmt.Print("\033[2K\r")
}

// moveUp moves cursor up n lines
func moveUp(n int) {
	fmt.Printf("\033[%dA", n)
}

// printKeyValue prints a formatted key-value pair
func printKeyValue(key, value string, indent int) {
	pad := strings.Repeat(" ", indent)
	fmt.Printf("%s%s%-20s%s %s\n", pad, cDim, key+":", cReset, value)
}

// printStatusLine prints a status line with colored status indicator
func printStatusLine(status string) {
	color := cGreen
	icon := "●"
	switch status {
	case "busy":
		color = cYellow
		icon = "◉"
	case "offline", "error":
		color = cRed
		icon = "○"
	case "connecting":
		color = cCyan
		icon = "◌"
	}
	fmt.Printf("  %s%s %s%s\n", color, icon, status, cReset)
}

// Spinner provides a simple ASCII spinner for progress indication
type Spinner struct {
	frames []string
	pos    int
	msg    string
}

func NewSpinner(msg string) *Spinner {
	return &Spinner{
		frames: []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"},
		msg:    msg,
	}
}

func (s *Spinner) Spin() string {
	frame := s.frames[s.pos%len(s.frames)]
	s.pos++
	return fmt.Sprintf("\r%s %s", frame, s.msg)
}

// notify sends a desktop notification (Windows only)
func notify(title, msg string) {
	if runtime.GOOS != "windows" {
		return
	}
	ps := fmt.Sprintf(`
	[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] > $null
	$template = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
	$template.GetElementsByTagName('text')[0].AppendChild($template.CreateTextNode('%s')) > $null
	$template.GetElementsByTagName('text')[1].AppendChild($template.CreateTextNode('%s')) > $null
	$toast = [Windows.UI.Notifications.ToastNotification]::new($template)
	[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier('DistriBox').Show($toast)
	`, title, msg)
	exec.Command("powershell", "-NoProfile", "-WindowStyle", "Hidden", "-Command", ps).Start()
}

// ── Simple console menu for subcommands ──────────────────

// ShowMenu displays an interactive menu when no subcommand is given
func ShowMenu(httpPort int) {
	apiURL := fmt.Sprintf("http://localhost:%d", httpPort)
	client := &http.Client{Timeout: 3 * time.Second}

	clearScreen()
	fmt.Print(cCyan + cBold)
	fmt.Println("  ⚡ DISTRIBOX v0.3.0 — Interactive Console")
	fmt.Print(cReset)
	fmt.Println()
	fmt.Println(cDim + "  Checking cluster status..." + cReset)

	resp, err := client.Get(apiURL + "/api/v1/status")
	if err != nil {
		fmt.Println(cRed + "  ✗ No VGPU Core running." + cReset)
		fmt.Println(cDim + "    Start with: distribox.exe" + cReset)
		os.Exit(1)
	}
	defer resp.Body.Close()

	var status struct {
		WorkerCount int `json:"worker_count"`
	}
	json.NewDecoder(resp.Body).Decode(&status)

	fmt.Printf(cGreen+"  ✓ VGPU Core running"+cReset+" — %d workers connected\n", status.WorkerCount)
	fmt.Println()
	fmt.Println("  Available commands:")
	fmt.Println(cCyan + "    status" + cReset + "         Show full cluster status")
	fmt.Println(cCyan + "    device create" + cReset + "  Create virtual GPU device (--auto)")
	fmt.Println(cCyan + "    device status" + cReset + "  Show virtual GPU details")
	fmt.Println(cCyan + "    device remove" + cReset + "  Remove virtual GPU device")
	fmt.Println(cCyan + "    worker list" + cReset + "    List connected workers")
	fmt.Println(cCyan + "    worker set" + cReset + "     Configure worker settings")
	fmt.Println(cCyan + "    install" + cReset + "        Install ICD + GPU interception layers")
	fmt.Println()
}
