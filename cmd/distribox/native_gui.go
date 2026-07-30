/*
 * cmd/distribox/native_gui.go — Native GUI via Edge WebView2 app mode
 *
 * Launches Microsoft Edge in "--app" mode pointing at the embedded dashboard.
 * This gives a clean, borderless, native-feeling window (identical to how
 * Tauri/Wails render on Windows — Edge WebView2 under the hood).
 *
 * Why this approach:
 *   - Edge is guaranteed on Windows 10+ (system component)
 *   - --app mode gives a borderless window with just the web content
 *   - Reuses existing dashboard.html (dark cyberpunk theme) — zero UI rewrite
 *   - Pure Go, no CGO, no external dependencies
 *
 * Build: go build -ldflags="-H windowsgui -s -w"
 */

package main

import (
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"syscall"
	"time"
	"unsafe"
)

const (
	nativeWindowWidth  = 960
	nativeWindowHeight = 680
)

// ── Entry ──────────────────────────────────────────────────

func RunNativeGUI(apiPort int) {
	url := fmt.Sprintf("http://localhost:%d/", apiPort)

	// Wait for server
	waitForServer(url, 15)

	// Try Edge --app mode first (cleanest, no browser chrome)
	if launchEdgeAppMode(url) {
		monitorServerHealth(url)
		return
	}

	// Fallback: open in default browser
	launchFallback(url)
}

// ── Edge --app mode ────────────────────────────────────────

func launchEdgeAppMode(url string) bool {
	edgePath := findEdge()
	if edgePath == "" {
		return false
	}

	cmd := exec.Command(edgePath,
		"--app="+url,
		"--new-window",
		fmt.Sprintf("--window-size=%d,%d", nativeWindowWidth, nativeWindowHeight),
		"--disable-extensions",
		"--disable-background-mode",
		"--disable-sync",
		"--no-first-run",
		"--no-default-browser-check",
	)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}

	if err := cmd.Start(); err != nil {
		return false
	}

	// Release the process — Edge manages its own lifecycle
	cmd.Process.Release()
	return true
}

func findEdge() string {
	// Check Edge Stable (most common)
	paths := []string{
		"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
		"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}

	// Try via registry
	return findEdgeViaRegistry()
}

func findEdgeViaRegistry() string {
	modadvapi32 := syscall.NewLazyDLL("advapi32.dll")
	procRegOpenKeyExW := modadvapi32.NewProc("RegOpenKeyExW")
	procRegQueryValueExW := modadvapi32.NewProc("RegQueryValueExW")
	procRegCloseKey := modadvapi32.NewProc("RegCloseKey")

	const HKEY_LOCAL_MACHINE = 0x80000002
	const KEY_READ = 0x20019

	subKey, _ := syscall.UTF16PtrFromString("SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\App Paths\\msedge.exe")

	var hkey uintptr
	ret, _, _ := procRegOpenKeyExW.Call(
		uintptr(HKEY_LOCAL_MACHINE),
		uintptr(unsafe.Pointer(subKey)),
		0, KEY_READ,
		uintptr(unsafe.Pointer(&hkey)),
	)
	if ret != 0 {
		return ""
	}
	defer procRegCloseKey.Call(hkey)

	buf := make([]uint16, 512)
	bufLen := uint32(len(buf) * 2)
	ret, _, _ = procRegQueryValueExW.Call(hkey,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(""))),
		0, 0,
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(&bufLen)),
	)
	if ret != 0 {
		return ""
	}

	return syscall.UTF16ToString(buf)
}

// ── Fallback ───────────────────────────────────────────────

func launchFallback(url string) {
	shell32 := syscall.NewLazyDLL("shell32.dll")
	shellExecute := shell32.NewProc("ShellExecuteW")

	shellExecute.Call(0,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr("open"))),
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(url))),
		0, 0, 5) // SW_SHOW
}

// ── Server health monitor ──────────────────────────────────

func monitorServerHealth(url string) {
	client := &http.Client{Timeout: 2 * time.Second}

	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	for range ticker.C {
		resp, err := client.Get(url + "api/v1/status")
		if err != nil {
			return // server stopped
		}
		resp.Body.Close()
	}
}

// ── Helpers ────────────────────────────────────────────────

func waitForServer(url string, maxRetries int) {
	client := &http.Client{Timeout: 1 * time.Second}
	for i := 0; i < maxRetries; i++ {
		time.Sleep(500 * time.Millisecond)
		resp, err := client.Get(url + "api/v1/status")
		if err == nil {
			resp.Body.Close()
			return
		}
	}
}

func init() {
	htaPath := filepath.Join(os.TempDir(), "distribox_dashboard.hta")
	os.Remove(htaPath)
}
