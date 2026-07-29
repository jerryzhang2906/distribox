/*
 * cmd/distribox/gui.go — Modern GUI via HTA (HTML Application)
 *
 * Creates a native window using Windows built-in mshta.exe to render
 * the dashboard HTML. No console visible, no browser chrome —
 * looks like a native desktop app. Zero external dependencies.
 *
 * Built with: go build -ldflags="-H windowsgui -s -w"
 */

package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"syscall"
	"time"
)

const dashboardTemplate = `<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<title>DistriBox</title>
<hta:application
  id="distribox"
  applicationName="DistriBox"
  border="thick"
  borderStyle="normal"
  caption="yes"
  contextMenu="no"
  innerBorder="no"
  maximizeButton="yes"
  minimizeButton="yes"
  navigable="yes"
  scroll="no"
  showInTaskBar="yes"
  singleInstance="yes"
  sysMenu="yes"
  version="0.3.0"
  windowState="normal"
/>
<style>
  html, body { margin: 0; padding: 0; width: 100%%; height: 100%%; overflow: hidden; background: #0A0E1A; }
  iframe { width: 100%%; height: 100%%; border: none; }
  #loading { display: flex; align-items: center; justify-content: center; height: 100%%;
             font-family: 'Segoe UI', sans-serif; color: #8892B0; font-size: 16px; }
  .spinner { width: 40px; height: 40px; border: 3px solid #1E2A45; border-top-color: #00D4FF;
             border-radius: 50%%; animation: spin 0.8s linear infinite; margin-right: 16px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
<script>
  var url = "%s";
  function init() {
    var iframe = document.createElement('iframe');
    iframe.src = url;
    iframe.onload = function() {
      document.getElementById('loading').style.display = 'none';
    };
    document.body.appendChild(iframe);
  }
  // Retry connection
  var retries = 0;
  function tryConnect() {
    var xhr = new ActiveXObject('MSXML2.XMLHTTP');
    xhr.open('GET', url + 'api/v1/status', true);
    xhr.onreadystatechange = function() {
      if (xhr.readyState == 4 && xhr.status == 200) {
        document.getElementById('loading').style.display = 'none';
        init();
      }
    };
    xhr.send();
    if (retries++ < 10) setTimeout(tryConnect, 1500);
  }
  window.onload = function() {
    window.resizeTo(1150, 750);
    tryConnect();
  };
</script>
</head>
<body>
<div id="loading">
  <div class="spinner"></div>
  <div>Starting DistriBox Server...</div>
</div>
</body>
</html>`

var guiAPIURL string

func runGUI(apiPort int) {
	runtime.LockOSThread()
	guiAPIURL = fmt.Sprintf("http://localhost:%d", apiPort)

	// Wait briefly for server to be ready
	time.Sleep(500 * time.Millisecond)

	// Write HTA file to temp directory
	htaPath := filepath.Join(os.TempDir(), "distribox_dashboard.hta")
	htaContent := fmt.Sprintf(dashboardTemplate, guiAPIURL)
	os.WriteFile(htaPath, []byte(htaContent), 0644)
	defer os.Remove(htaPath)

	// Launch mshta.exe to render the dashboard in a native window
	cmd := exec.Command("mshta.exe", htaPath)
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true}
	cmd.Start()

	// Keep the Go process alive — poll server health
	ticker := time.NewTicker(3 * time.Second)
	defer ticker.Stop()

	client := &http.Client{Timeout: 2 * time.Second}
	for range ticker.C {
		resp, err := client.Get(guiAPIURL + "/api/v1/status")
		if err != nil {
			// Server stopped — exit
			return
		}
		io.Copy(io.Discard, resp.Body)
		resp.Body.Close()
	}
}

// checkServerStatus is called by timer in GUI mode
func checkServerStatus() {}
