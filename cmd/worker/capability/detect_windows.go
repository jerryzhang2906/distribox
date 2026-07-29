/*
 * cmd/worker/capability/detect_windows.go — Windows hardware detection
 *
 * Uses Win32 APIs (via syscall.LoadDLL) to detect real CPU, GPU,
 * and memory information. Follows the same pattern as gpu_windows.go.
 */

//go:build windows

package capability

import (
	"fmt"
	"syscall"
	"unsafe"
)

// ── Win32 types ───────────────────────────────────────

type memoryStatusEx struct {
	Length               uint32
	MemoryLoad           uint32
	TotalPhys            uint64
	AvailPhys            uint64
	TotalPageFile        uint64
	AvailPageFile        uint64
	TotalVirtual         uint64
	AvailVirtual         uint64
	AvailExtendedVirtual uint64
}

type systemInfo struct {
	ProcessorArchitecture uint16
	Reserved              uint16
	PageSize              uint32
	MinAppAddress         uintptr
	MaxAppAddress         uintptr
	ActiveProcessorMask   uintptr
	NumberOfProcessors    uint32
	ProcessorType         uint32
	AllocationGranularity uint32
	ProcessorLevel        uint16
	ProcessorRevision     uint16
}

// ── CPU detection ─────────────────────────────────────

func detectCPUPlatform(info *HardwareInfo) {
	// CPU model name from registry
	info.CPUModel = readRegString(
		"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
		"ProcessorNameString",
	)
	if info.CPUModel == "" {
		info.CPUModel = "Unknown x64 CPU"
	}

	// CPU frequency from registry
	info.CPUFreqMHz = readRegDWORD(
		"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
		"~MHz",
	)
	if info.CPUFreqMHz == 0 {
		info.CPUFreqMHz = 2400
	}

	// Physical and logical cores
	info.CPUCores, info.CPULogical = getProcessorCounts()

	// ISA features
	if info.Arch == "amd64" {
		info.ISAFatures = []string{"SSE", "SSE2", "SSE3", "SSE4", "AVX", "AVX2", "FMA"}
	} else if info.Arch == "arm64" {
		info.ISAFatures = []string{"NEON", "FP16"}
	}

	// Estimate GFLOPS
	info.BenchGFLOPS = float64(info.CPUCores) * float64(info.CPUFreqMHz) * 4.0 / 1000.0
}

func getProcessorCounts() (physical, logical int) {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	getLogicalProcessorInformationEx := kernel32.MustFindProc("GetLogicalProcessorInformationEx")
	getActiveProcessorCount := kernel32.MustFindProc("GetActiveProcessorCount")

	const RelationProcessorCore = 0
	const ALL_PROCESSOR_GROUPS = 0xFFFF

	// Get logical processors
	ret, _, _ := getActiveProcessorCount.Call(uintptr(ALL_PROCESSOR_GROUPS))
	if ret > 0 {
		logical = int(ret)
	}

	// Get physical cores via GetLogicalProcessorInformationEx
	var bufLen uint32
	getLogicalProcessorInformationEx.Call(
		uintptr(RelationProcessorCore),
		0,
		uintptr(unsafe.Pointer(&bufLen)),
	)
	if bufLen == 0 {
		if logical == 0 {
			logical = 1
		}
		return logical, logical
	}

	buf := make([]byte, bufLen)
	ret, _, _ = getLogicalProcessorInformationEx.Call(
		uintptr(RelationProcessorCore),
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(&bufLen)),
	)
	if ret != 0 {
		physical = 0
		offset := uint32(0)
		for offset < bufLen {
			// Each entry starts with Relationship(uint32) + Size(uint32)
			if offset+8 > bufLen {
				break
			}
			entrySize := *(*uint32)(unsafe.Pointer(&buf[offset+4]))
			if entrySize == 0 {
				break
			}
			physical++
			offset += entrySize
		}
	}

	if physical == 0 {
		physical = logical
	}
	if logical == 0 {
		logical = 1
	}
	return physical, logical
}

// ── Memory detection ──────────────────────────────────

func detectMemoryPlatform(info *HardwareInfo) {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	globalMemoryStatusEx := kernel32.MustFindProc("GlobalMemoryStatusEx")

	var mem memoryStatusEx
	mem.Length = uint32(unsafe.Sizeof(mem))
	ret, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&mem)))
	if ret != 0 {
		info.TotalRAMMB = int(mem.TotalPhys / (1024 * 1024))
		info.AvailableRAMMB = int(mem.AvailPhys / (1024 * 1024))
	} else {
		info.TotalRAMMB = 8192
		info.AvailableRAMMB = 4096
	}

	info.MemoryBwGBps = 25.0 // DDR4 dual-channel estimate
}

// ── GPU detection ─────────────────────────────────────

func detectGPUPlatform(info *HardwareInfo) {
	// Try OpenCL first (most accurate)
	if detectGPUOpenCL(info) {
		return
	}

	// Fallback: try registry for display adapters
	if detectGPURegistry(info) {
		return
	}

	info.HasGPU = false
}

func detectGPUOpenCL(info *HardwareInfo) bool {
	opencl, err := syscall.LoadDLL("OpenCL.dll")
	if err != nil {
		return false
	}
	defer opencl.Release()

	getPlatformIDs, err := opencl.FindProc("clGetPlatformIDs")
	if err != nil {
		return false
	}
	getDeviceIDs, err := opencl.FindProc("clGetDeviceIDs")
	if err != nil {
		return false
	}
	getDeviceInfo, err := opencl.FindProc("clGetDeviceInfo")
	if err != nil {
		return false
	}

	// Get number of platforms
	var numPlatforms uint32
	getPlatformIDs.Call(0, 0, uintptr(unsafe.Pointer(&numPlatforms)))
	if numPlatforms == 0 {
		return false
	}

	platforms := make([]uintptr, numPlatforms)
	getPlatformIDs.Call(uintptr(numPlatforms), uintptr(unsafe.Pointer(&platforms[0])), 0)
	if numPlatforms == 0 || platforms[0] == 0 {
		return false
	}

	// Get number of GPU devices (CL_DEVICE_TYPE_GPU = 0xFFFFFFFF for ALL)
	var numDevices uint32
	getDeviceIDs.Call(platforms[0], uintptr(0xFFFFFFFF), 0, uintptr(unsafe.Pointer(&numDevices)))
	if numDevices == 0 {
		return false
	}

	devices := make([]uintptr, numDevices)
	getDeviceIDs.Call(platforms[0], uintptr(0xFFFFFFFF), uintptr(numDevices), uintptr(unsafe.Pointer(&devices[0])), 0)
	if devices[0] == 0 {
		return false
	}

	info.HasGPU = true
	info.GPUModel = oclString(getDeviceInfo, devices[0], 0x102B)       // CL_DEVICE_NAME
	info.GPUVendor = oclString(getDeviceInfo, devices[0], 0x102C)      // CL_DEVICE_VENDOR
	info.OpenCLVersion = oclString(getDeviceInfo, devices[0], 0x103C)  // CL_DEVICE_OPENCL_C_VERSION

	// CL_DEVICE_GLOBAL_MEM_SIZE = 0x101F
	var memSize uint64
	getDeviceInfo.Call(devices[0], uintptr(0x101F), uintptr(8), uintptr(unsafe.Pointer(&memSize)), 0)
	info.GPUVramMB = int(memSize / (1024 * 1024))

	// CL_DEVICE_MAX_COMPUTE_UNITS = 0x1002
	var cu uint32
	getDeviceInfo.Call(devices[0], uintptr(0x1002), uintptr(4), uintptr(unsafe.Pointer(&cu)), 0)
	info.GPUComputeUnits = int(cu)

	// CL_DEVICE_MAX_CLOCK_FREQUENCY = 0x100C
	var freq uint32
	getDeviceInfo.Call(devices[0], uintptr(0x100C), uintptr(4), uintptr(unsafe.Pointer(&freq)), 0)
	info.GPUFreqMHz = int(freq)

	return true
}

func oclString(getDeviceInfo *syscall.Proc, device uintptr, param uintptr) string {
	var size uint64
	getDeviceInfo.Call(device, param, 0, 0, uintptr(unsafe.Pointer(&size)))
	if size == 0 {
		return ""
	}
	buf := make([]byte, size)
	getDeviceInfo.Call(device, param, uintptr(size), uintptr(unsafe.Pointer(&buf[0])), 0)
	for i, b := range buf {
		if b == 0 {
			return string(buf[:i])
		}
	}
	return string(buf)
}

func detectGPURegistry(info *HardwareInfo) bool {
	// Enumerate display adapter subkeys under the GPU class GUID
	classGUID := "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}"

	for i := 0; i < 10; i++ {
		subKey := fmt.Sprintf("%s\\%04d", classGUID, i)
		desc := readRegString(subKey, "DriverDesc")
		if desc == "" {
			continue
		}

		info.HasGPU = true
		info.GPUModel = desc

		// Parse vendor from description
		if strContains(desc, "NVIDIA") {
			info.GPUVendor = "NVIDIA"
		} else if strContains(desc, "AMD") || strContains(desc, "Radeon") {
			info.GPUVendor = "AMD"
		} else if strContains(desc, "Intel") {
			info.GPUVendor = "Intel"
		} else {
			info.GPUVendor = "Unknown"
		}

		info.GPUVramMB = 4096 // Conservative default
		info.GPUComputeUnits = 16
		info.GPUFreqMHz = 1000
		return true
	}

	return false
}

// ── Registry helpers (via advapi32.dll) ───────────────

func readRegString(keyPath, valueName string) string {
	advapi32 := syscall.MustLoadDLL("advapi32.dll")
	regOpenKeyEx := advapi32.MustFindProc("RegOpenKeyExW")
	regQueryValueEx := advapi32.MustFindProc("RegQueryValueExW")
	regCloseKey := advapi32.MustFindProc("RegCloseKey")

	const HKEY_LOCAL_MACHINE = 0x80000002
	const KEY_READ = 0x20019

	var hKey uintptr
	keyPathPtr, _ := syscall.UTF16PtrFromString(keyPath)
	ret, _, _ := regOpenKeyEx.Call(
		uintptr(HKEY_LOCAL_MACHINE),
		uintptr(unsafe.Pointer(keyPathPtr)),
		0,
		uintptr(KEY_READ),
		uintptr(unsafe.Pointer(&hKey)),
	)
	if ret != 0 {
		return ""
	}
	defer regCloseKey.Call(hKey)

	// Get required buffer size
	var bufLen uint32
	valueNamePtr, _ := syscall.UTF16PtrFromString(valueName)
	regQueryValueEx.Call(
		hKey,
		uintptr(unsafe.Pointer(valueNamePtr)),
		0,
		0,
		0,
		uintptr(unsafe.Pointer(&bufLen)),
	)

	buf := make([]uint16, bufLen/2+1)
	ret, _, _ = regQueryValueEx.Call(
		hKey,
		uintptr(unsafe.Pointer(valueNamePtr)),
		0,
		0,
		uintptr(unsafe.Pointer(&buf[0])),
		uintptr(unsafe.Pointer(&bufLen)),
	)
	if ret != 0 {
		return ""
	}

	return syscall.UTF16ToString(buf)
}

func readRegDWORD(keyPath, valueName string) int {
	advapi32 := syscall.MustLoadDLL("advapi32.dll")
	regOpenKeyEx := advapi32.MustFindProc("RegOpenKeyExW")
	regQueryValueEx := advapi32.MustFindProc("RegQueryValueExW")
	regCloseKey := advapi32.MustFindProc("RegCloseKey")

	const HKEY_LOCAL_MACHINE = 0x80000002
	const KEY_READ = 0x20019

	var hKey uintptr
	keyPathPtr, _ := syscall.UTF16PtrFromString(keyPath)
	ret, _, _ := regOpenKeyEx.Call(
		uintptr(HKEY_LOCAL_MACHINE),
		uintptr(unsafe.Pointer(keyPathPtr)),
		0,
		uintptr(KEY_READ),
		uintptr(unsafe.Pointer(&hKey)),
	)
	if ret != 0 {
		return 0
	}
	defer regCloseKey.Call(hKey)

	var val uint32
	var valLen uint32 = 4
	valueNamePtr, _ := syscall.UTF16PtrFromString(valueName)
	ret, _, _ = regQueryValueEx.Call(
		hKey,
		uintptr(unsafe.Pointer(valueNamePtr)),
		0,
		0,
		uintptr(unsafe.Pointer(&val)),
		uintptr(unsafe.Pointer(&valLen)),
	)
	if ret != 0 {
		return 0
	}
	return int(val)
}

func strContains(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		match := true
		for j := 0; j < len(substr); j++ {
			if s[i+j] != substr[j] {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}
