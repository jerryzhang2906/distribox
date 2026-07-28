# DistriBox — One-click installer for Windows
# Run as Administrator to install all GPU interception layers.
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\install.ps1

param(
    [switch]$Uninstall = $false,
    [string]$DistDir = "$PSScriptRoot\..\build\dist-windows"
)

$ErrorActionPreference = "Stop"
Write-Host "=== DistriBox Installer ===" -ForegroundColor Cyan

if (-not (Test-Path $DistDir)) {
    Write-Host "Error: Distribution not found at $DistDir" -ForegroundColor Red
    Write-Host "  Run 'make dist-windows' first, or specify -DistDir"
    exit 1
}

function Install-File($src, $dst) {
    if (Test-Path $src) {
        Copy-Item -Force $src $dst
        Write-Host "  OK: $dst" -ForegroundColor Green
    } else {
        Write-Host "  SKIP: $src not found" -ForegroundColor Yellow
    }
}

if ($Uninstall) {
    Write-Host "`nUninstalling..." -ForegroundColor Yellow
    # Restore original DLLs
    if (Test-Path "$env:WINDIR\System32\OpenCL_orig.dll") {
        Copy-Item -Force "$env:WINDIR\System32\OpenCL_orig.dll" "$env:WINDIR\System32\OpenCL.dll"
        Write-Host "  Restored original OpenCL.dll"
    }
    if (Test-Path "$env:WINDIR\System32\nvcuda_orig.dll") {
        Copy-Item -Force "$env:WINDIR\System32\nvcuda_orig.dll" "$env:WINDIR\System32\nvcuda.dll"
        Write-Host "  Restored original nvcuda.dll"
    }
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors" -Name "distribox_icd.dll" -Force -ErrorAction SilentlyContinue
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers" -Name "distribox" -Force -ErrorAction SilentlyContinue
    Remove-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows" -Name "AppInit_DLLs" -Force -ErrorAction SilentlyContinue
    Write-Host "DistriBox uninstalled." -ForegroundColor Green
    exit 0
}

Write-Host "`nInstalling DistriBox GPU layers..." -ForegroundColor Cyan

# Create DistriBox program directory
$progDir = "$env:ProgramData\DistriBox"
New-Item -ItemType Directory -Force -Path $progDir | Out-Null

# Copy all DLLs to System32 and program directory
$dlls = @("distribox_icd.dll", "distribox_engine.dll", "nvcuda.dll",
          "distribox_vk_layer.dll", "distribox_hook.dll")
foreach ($dll in $dlls) {
    Install-File "$DistDir\$dll" "$env:WINDIR\System32\"
    Install-File "$DistDir\$dll" "$progDir\"
}

# Vulkan layer JSON
Install-File "$DistDir\distribox_vk_layer.json" "$env:WINDIR\System32\"
Install-File "$DistDir\distribox_vk_layer.json" "$progDir\"

# Copy main executable
Install-File "$DistDir\distribox.exe" "$progDir\"

# Register OpenCL ICD
Write-Host "`nRegistering OpenCL ICD..." -ForegroundColor Cyan
New-Item -Path "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors" -Force | Out-Null
Set-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\OpenCL\Vendors" -Name "distribox_icd.dll" -Value 0 -Type DWord -Force
Write-Host "  OK: OpenCL ICD registered"

# Register Vulkan layer
Write-Host "`nRegistering Vulkan Layer..." -ForegroundColor Cyan
New-Item -Path "HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers" -Force | Out-Null
$vkJson = "$env:WINDIR\System32\distribox_vk_layer.json"
Set-ItemProperty -Path "HKLM:\SOFTWARE\Khronos\Vulkan\ImplicitLayers" -Name "distribox" -Value $vkJson -Type String -Force
Write-Host "  OK: Vulkan layer registered"

# Register AppInit for OpenGL hook
Write-Host "`nRegistering OpenGL Hook (AppInit)..." -ForegroundColor Cyan
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows" -Name "AppInit_DLLs" -Value "C:\ProgramData\DistriBox\distribox_hook.dll" -Type String -Force
Set-ItemProperty -Path "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows" -Name "LoadAppInit_DLLs" -Value 1 -Type DWord -Force
Write-Host "  OK: OpenGL hook registered"

# Create desktop shortcut
Write-Host "`nCreating shortcuts..." -ForegroundColor Cyan
$WshShell = New-Object -ComObject WScript.Shell
$shortcut = $WshShell.CreateShortcut("$env:USERPROFILE\Desktop\DistriBox.lnk")
$shortcut.TargetPath = "$progDir\distribox.exe"
$shortcut.WorkingDirectory = $progDir
$shortcut.Description = "DistriBox — Distributed Virtual GPU"
$shortcut.Save()
Write-Host "  OK: Desktop shortcut created"

Write-Host @"

╔══════════════════════════════════════╗
║  DistriBox installed successfully!  ║
║                                      ║
║  Dashboard: http://localhost:13801   ║
║  Start:     C:\ProgramData\DistriBox\distribox.exe
║                                      ║
║  Double-click the desktop shortcut   ║
║  to start sharing GPU power!         ║
╚══════════════════════════════════════╝

"@ -ForegroundColor Green
