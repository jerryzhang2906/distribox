/*
 * cmd/distribox/gui.go — Native GUI launcher
 *
 * Runs the native GUI window (Edge WebView2 --app mode) instead of
 * opening the system default browser. The dashboard HTML is served
 * from the embedded HTTP server at localhost:<port>/.
 *
 * Built with: go build -ldflags="-H windowsgui -s -w"
 */

package main

import (
	"log"
	"os"
	"path/filepath"
)

func runGUI(apiPort int) {
	log.Println("[MODE] GUI — Native window with embedded dashboard")

	// RunNativeGUI launches Edge in --app mode for a clean borderless window
	// Falls back to default browser if Edge is not available
	RunNativeGUI(apiPort)

	// GUI window closed — shutdown
	log.Println("GUI closed — shutting down...")
}

// Clean up any leftover HTA files from previous version
func init() {
	htaPath := filepath.Join(os.TempDir(), "distribox_dashboard.hta")
	os.Remove(htaPath)
}
