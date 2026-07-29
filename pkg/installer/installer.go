/*
 * pkg/installer/installer.go — DistriBox Windows installer
 *
 * Provides Install() and Uninstall() functions that can be called
 * from both the CLI and the dashboard HTTP handler.
 */

package installer

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

// Result holds the outcome of an install/uninstall operation.
type Result struct {
	Success bool
	Output  string
	Steps   []StepResult
}

// StepResult holds a single step's outcome.
type StepResult struct {
	Step    string
	OK      bool
	Message string
}

// InstallICD installs the OpenCL ICD interception layer.
// Requires Administrator privileges.
func InstallICD() *Result {
	if runtime.GOOS != "windows" {
		return &Result{Success: false, Output: "ICD installation only supported on Windows"}
	}

	r := &Result{}
	admin := isAdmin()
	if !admin {
		r.Steps = append(r.Steps, StepResult{Step: "Admin Check", OK: false, Message: "Not running as Administrator"})
		r.Output = "ICD installation requires Administrator privileges.\nRun: Right-click → Run as Administrator"
		return r
	}
	r.Steps = append(r.Steps, StepResult{Step: "Admin Check", OK: true, Message: "Running as Administrator"})

	sys32 := os.Getenv("SystemRoot") + "\\System32"

	// Find distribox_icd.dll
	icdSrc := findICDDLL()
	if icdSrc == "" {
		r.Steps = append(r.Steps, StepResult{Step: "Find ICD DLL", OK: false, Message: "distribox_icd.dll not found in build/icd/ or current dir"})
		r.Output = "Cannot find distribox_icd.dll. Run from the DistriBox build directory."
		return r
	}
	r.Steps = append(r.Steps, StepResult{Step: "Find ICD DLL", OK: true, Message: icdSrc})

	// Step 1: Copy ICD DLL
	icdDst := sys32 + "\\distribox_icd.dll"
	if err := copyFile(icdSrc, icdDst); err != nil {
		r.Steps = append(r.Steps, StepResult{Step: "Copy ICD DLL", OK: false, Message: err.Error()})
	} else {
		r.Steps = append(r.Steps, StepResult{Step: "Copy ICD DLL", OK: true, Message: icdDst})
	}

	// Step 2: Register ICD in registry
	ps := `New-Item -Path 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' -Force | Out-Null;
New-ItemProperty -Path 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' -Name 'distribox_icd.dll' -Value 0 -PropertyType DWord -Force;
Write-Host "OK"`
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, err := cmd.CombinedOutput()
	output := strings.TrimSpace(string(out))
	if err != nil || !strings.Contains(output, "OK") {
		r.Steps = append(r.Steps, StepResult{Step: "Register ICD", OK: false, Message: output})
	} else {
		r.Steps = append(r.Steps, StepResult{Step: "Register ICD", OK: true, Message: "Registry key set"})
	}

	// Step 3: Copy engine DLL if exists
	engineSrc := filepath.Join(filepath.Dir(icdSrc), "distribox_engine.dll")
	if fileExists(engineSrc) {
		if err := copyFile(engineSrc, sys32+"\\distribox_engine.dll"); err != nil {
			r.Steps = append(r.Steps, StepResult{Step: "Copy Engine DLL", OK: false, Message: err.Error()})
		} else {
			r.Steps = append(r.Steps, StepResult{Step: "Copy Engine DLL", OK: true, Message: "OK"})
		}
	}

	// Step 4: Install CUDA proxy if exists
	cudaSrc := filepath.Join(filepath.Dir(icdSrc), "nvcuda.dll")
	if fileExists(cudaSrc) {
		cudaBackup := sys32 + "\\nvcuda_orig.dll"
		cudaDst := sys32 + "\\nvcuda.dll"
		backupFile(cudaDst, cudaBackup)
		if err := copyFile(cudaSrc, cudaDst); err != nil {
			r.Steps = append(r.Steps, StepResult{Step: "Install CUDA Proxy", OK: false, Message: err.Error()})
		} else {
			r.Steps = append(r.Steps, StepResult{Step: "Install CUDA Proxy", OK: true, Message: "nvcuda.dll → System32 (original backed up)"})
		}
	}

	// Step 5: Register Vulkan layer
	vkJSONSrc := filepath.Join(filepath.Dir(icdSrc), "distribox_vk_layer.json")
	vkDLLSrc := filepath.Join(filepath.Dir(icdSrc), "distribox_vk_layer.dll")
	if fileExists(vkJSONSrc) && fileExists(vkDLLSrc) {
		copyFile(vkJSONSrc, sys32+"\\distribox_vk_layer.json")
		copyFile(vkDLLSrc, sys32+"\\distribox_vk_layer.dll")
		psVk := fmt.Sprintf(
			`New-Item -Path 'HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers' -Force | Out-Null;
New-ItemProperty -Path 'HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers' -Name 'distribox' -Value '%s\distribox_vk_layer.json' -PropertyType String -Force;
Write-Host "OK"`, sys32)
		cmdVk := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", psVk)
		outVk, _ := cmdVk.CombinedOutput()
		if strings.Contains(string(outVk), "OK") {
			r.Steps = append(r.Steps, StepResult{Step: "Register Vulkan Layer", OK: true, Message: "OK"})
		} else {
			r.Steps = append(r.Steps, StepResult{Step: "Register Vulkan Layer", OK: false, Message: string(outVk)})
		}
	}

	r.Success = true
	r.Output = formatResult(r)
	return r
}

// UninstallICD removes the OpenCL ICD interception layer.
func UninstallICD() *Result {
	if runtime.GOOS != "windows" {
		return &Result{Success: false, Output: "Windows only"}
	}

	r := &Result{}
	admin := isAdmin()
	if !admin {
		r.Output = "Uninstall requires Administrator privileges."
		return r
	}

	sys32 := os.Getenv("SystemRoot") + "\\System32"

	// Restore CUDA proxy backup
	cudaBackup := sys32 + "\\nvcuda_orig.dll"
	cudaDst := sys32 + "\\nvcuda.dll"
	if fileExists(cudaBackup) {
		copyFile(cudaBackup, cudaDst)
		os.Remove(cudaBackup)
		r.Steps = append(r.Steps, StepResult{Step: "Restore nvcuda.dll", OK: true, Message: "Restored from backup"})
	}

	// Remove registry entries
	ps := `Remove-ItemProperty -Path 'HKLM:\SOFTWARE\Khronos\OpenCL\Vendors' -Name 'distribox_icd.dll' -Force -ErrorAction SilentlyContinue;
Remove-ItemProperty -Path 'HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers' -Name 'distribox' -Force -ErrorAction SilentlyContinue;
Write-Host "OK"`
	cmd := exec.Command("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps)
	out, _ := cmd.CombinedOutput()
	if strings.Contains(string(out), "OK") {
		r.Steps = append(r.Steps, StepResult{Step: "Remove Registry", OK: true, Message: "Cleaned up"})
	}

	r.Success = true
	r.Output = "DistriBox uninstalled. System DLLs restored."
	return r
}

// Status returns the current ICD installation status.
func Status() string {
	sys32 := os.Getenv("SystemRoot") + "\\System32"
	icdPath := sys32 + "\\distribox_icd.dll"
	cudaProxy := sys32 + "\\nvcuda.dll"
	cudaBackup := sys32 + "\\nvcuda_orig.dll"

	var lines []string
	lines = append(lines, "DistriBox Installation Status")
	lines = append(lines, "─────────────────────────────")
	lines = append(lines, fmt.Sprintf("ICD DLL:    %s", checkMark(fileExists(icdPath))))
	lines = append(lines, fmt.Sprintf("ICD Path:   %s", icdPath))
	lines = append(lines, fmt.Sprintf("CUDA Proxy: %s", checkMark(fileExists(cudaProxy))))
	lines = append(lines, fmt.Sprintf("CUDA Bkup:  %s", checkMark(fileExists(cudaBackup))))
	return strings.Join(lines, "\n")
}

// ── Helpers ─────────────────────────────────────────

func isAdmin() bool {
	// Quick check: try writing a test registry key
	ps := `$elevated=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]"Administrator");if($elevated){Write-Host "YES"}else{Write-Host "NO"}`
	cmd := exec.Command("powershell", "-NoProfile", "-Command", ps)
	out, _ := cmd.CombinedOutput()
	return strings.TrimSpace(string(out)) == "YES"
}

func findICDDLL() string {
	// Check common locations
	candidates := []string{
		"build\\icd\\distribox_icd.dll",
		"build\\dist-windows\\distribox_icd.dll",
		"distribox_icd.dll",
	}
	for _, p := range candidates {
		if fileExists(p) {
			abs, _ := filepath.Abs(p)
			return abs
		}
	}
	return ""
}

func copyFile(src, dst string) error {
	data, err := os.ReadFile(src)
	if err != nil {
		return fmt.Errorf("read %s: %w", src, err)
	}
	if err := os.WriteFile(dst, data, 0644); err != nil {
		return fmt.Errorf("write %s: %w", dst, err)
	}
	return nil
}

func backupFile(src, backup string) {
	if fileExists(src) && !fileExists(backup) {
		copyFile(src, backup)
	}
}

func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

func checkMark(ok bool) string {
	if ok {
		return "✓ Installed"
	}
	return "✗ Not installed"
}

func formatResult(r *Result) string {
	var lines []string
	lines = append(lines, "DistriBox ICD Installation")
	lines = append(lines, "──────────────────────────")
	for _, s := range r.Steps {
		mark := "✗"
		if s.OK {
			mark = "✓"
		}
		lines = append(lines, fmt.Sprintf("  %s %s: %s", mark, s.Step, s.Message))
	}
	if r.Success {
		lines = append(lines, "\nInstallation complete! Start VGPU Core with: distribox.exe")
	}
	return strings.Join(lines, "\n")
}
