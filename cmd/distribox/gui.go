/*
 * cmd/distribox/gui.go — Windows GUI launcher
 *
 * Starts the DistriBox server in background and opens the dashboard
 * in the system default browser. No console window needed.
 *
 * Built with: go build -ldflags="-H windowsgui -s -w"
 */

package main

import (
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"syscall"
	"time"
	"unsafe"
)

func runGUI(apiPort int) {
	runtime.LockOSThread()

	url := fmt.Sprintf("http://localhost:%d", apiPort)

	// Wait for server to be fully ready
	client := &http.Client{Timeout: 1 * time.Second}
	for i := 0; i < 15; i++ {
		time.Sleep(500 * time.Millisecond)
		resp, err := client.Get(url + "/api/v1/status")
		if err == nil {
			resp.Body.Close()
			break
		}
	}

	// Open dashboard in default browser — no ugly Win32, no terminal
	openDefaultBrowser(url + "/")

	// Keep running — poll server health
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()
	for range ticker.C {
		resp, err := client.Get(url + "/api/v1/status")
		if err != nil {
			return // Server stopped
		}
		resp.Body.Close()
	}
}

func openDefaultBrowser(url string) {
	// Use ShellExecute on Windows — opens the user's default browser
	shell32 := syscall.NewLazyDLL("shell32.dll")
	shellExecute := shell32.NewProc("ShellExecuteW")
	shellExecute.Call(0,
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr("open"))),
		uintptr(unsafe.Pointer(syscall.StringToUTF16Ptr(url))),
		0, 0, 5) // SW_SHOW = 5
}

// Clean up any leftover HTA files from previous version
func init() {
	htaPath := filepath.Join(os.TempDir(), "distribox_dashboard.hta")
	os.Remove(htaPath)
}

func checkServerStatus() {}
