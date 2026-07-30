/*
 * cmd/worker/monitor/resources_windows.go — Windows real-time resource monitoring
 *
 * Uses Win32 APIs for real CPU, GPU, memory, and battery metrics.
 * CPU: GetSystemTimes diff-based calculation
 * GPU: PDH (Performance Data Helper) counters
 * Battery: GetSystemPowerStatus
 */

//go:build windows

package monitor

import (
	"math"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

// ── Filetime helpers ──────────────────────────────────

type filetime struct {
	LowDateTime  uint32
	HighDateTime uint32
}

func ftToUint64(ft filetime) uint64 {
	return uint64(ft.HighDateTime)<<32 | uint64(ft.LowDateTime)
}

// ── CPU sampling ──────────────────────────────────────

var (
	prevIdle  uint64
	prevTotal uint64
	cpuMu     sync.Mutex
)

func sampleCPU() float64 {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	getSystemTimes := kernel32.MustFindProc("GetSystemTimes")

	var idle, kernel, user filetime
	ret, _, _ := getSystemTimes.Call(
		uintptr(unsafe.Pointer(&idle)),
		uintptr(unsafe.Pointer(&kernel)),
		uintptr(unsafe.Pointer(&user)),
	)
	if ret == 0 {
		return 0
	}

	idleNow := ftToUint64(idle)
	kernelNow := ftToUint64(kernel)
	userNow := ftToUint64(user)
	totalNow := kernelNow + userNow

	cpuMu.Lock()
	defer cpuMu.Unlock()

	if prevTotal == 0 {
		prevIdle = idleNow
		prevTotal = totalNow
		return 0
	}

	idleDelta := idleNow - prevIdle
	totalDelta := totalNow - prevTotal

	prevIdle = idleNow
	prevTotal = totalNow

	if totalDelta == 0 {
		return 0
	}

	return math.Min(100.0, float64(totalDelta-idleDelta)/float64(totalDelta)*100.0)
}

// ── Memory sampling ───────────────────────────────────

type memStatusEx struct {
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

func sampleMemory() (usedMB int64, availMB int64) {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	globalMemoryStatusEx := kernel32.MustFindProc("GlobalMemoryStatusEx")

	var mem memStatusEx
	mem.Length = uint32(unsafe.Sizeof(mem))
	ret, _, _ := globalMemoryStatusEx.Call(uintptr(unsafe.Pointer(&mem)))
	if ret == 0 {
		return 512, 4096
	}

	totalMB := int64(mem.TotalPhys / (1024 * 1024))
	availMB = int64(mem.AvailPhys / (1024 * 1024))
	usedMB = totalMB - availMB
	if usedMB < 0 {
		usedMB = 0
	}
	return usedMB, availMB
}

// ── GPU sampling (via PDH) ────────────────────────────

var (
	pdhQuery   uintptr
	pdhCounter uintptr
	pdhInited  bool
	pdhMu      sync.Mutex
)

func sampleGPU() float64 {
	pdhMu.Lock()
	defer pdhMu.Unlock()

	if !pdhInited {
		initPDH()
		pdhInited = true
	}

	if pdhQuery == 0 || pdhCounter == 0 {
		return 0
	}

	pdh := syscall.MustLoadDLL("pdh.dll")
	collectQueryData := pdh.MustFindProc("PdhCollectQueryData")
	getFormattedCounterValue := pdh.MustFindProc("PdhGetFormattedCounterValue")

	// Collect current data
	collectQueryData.Call(pdhQuery)

	// PDH_FMT_DOUBLE = 0x00000200
	var counterType uint32
	var value float64
	ret, _, _ := getFormattedCounterValue.Call(
		pdhCounter,
		uintptr(0x00000200),
		0,
		uintptr(unsafe.Pointer(&value)),
	)
	if ret != 0 {
		return 0
	}
	_ = counterType

	return value
}

func initPDH() {
	pdh := syscall.MustLoadDLL("pdh.dll")
	pdhOpenQuery := pdh.MustFindProc("PdhOpenQueryW")
	pdhAddCounter := pdh.MustFindProc("PdhAddCounterW")
	pdhCollectQueryData := pdh.MustFindProc("PdhCollectQueryData")

	// Open a query
	var query uintptr
	ret, _, _ := pdhOpenQuery.Call(0, 0, uintptr(unsafe.Pointer(&query)))
	if ret != 0 {
		return
	}
	pdhQuery = query

	// Try to add GPU engine utilization counter
	// The counter path format is: \GPU Engine(*engtype_3D)\Utilization Percentage
	// We use a wildcard to find any GPU engine
	counterPath := "\\GPU Engine(*engtype_3D)\\Utilization Percentage"
	counterPathPtr, _ := syscall.UTF16PtrFromString(counterPath)

	var counter uintptr
	ret, _, _ = pdhAddCounter.Call(
		query,
		uintptr(unsafe.Pointer(counterPathPtr)),
		0,
		uintptr(unsafe.Pointer(&counter)),
	)
	if ret != 0 {
		// Fallback: try simpler GPU counter
		counterPath2 := "\\GPU Adapter Memory(*)\\Shared Usage"
		counterPathPtr2, _ := syscall.UTF16PtrFromString(counterPath2)
		ret2, _, _ := pdhAddCounter.Call(
			query,
			uintptr(unsafe.Pointer(counterPathPtr2)),
			0,
			uintptr(unsafe.Pointer(&counter)),
		)
		if ret2 != 0 {
			pdhCounter = 0
			return
		}
	}
	pdhCounter = counter

	// First collection to initialize
	pdhCollectQueryData.Call(query)
	time.Sleep(100 * time.Millisecond)
	pdhCollectQueryData.Call(query)
}

// ── Battery sampling ──────────────────────────────────

type systemPowerStatus struct {
	ACLineStatus        byte
	BatteryFlag         byte
	BatteryLifePercent  byte
	SystemStatusFlag    byte
	BatteryLifeTime     uint32
	BatteryFullLifeTime uint32
}

func sampleThermal() (tempC float64, throttled bool) {
	// No standard user-mode API for CPU/GPU temperature on Windows.
	// Requires kernel driver or WMI provider (e.g., OpenHardwareMonitor).
	return 0, false
}

func sampleBattery() (pct float64, charging bool) {
	kernel32 := syscall.MustLoadDLL("kernel32.dll")
	getSystemPowerStatus := kernel32.MustFindProc("GetSystemPowerStatus")

	var status systemPowerStatus
	ret, _, _ := getSystemPowerStatus.Call(uintptr(unsafe.Pointer(&status)))
	if ret == 0 {
		return 100, true // No battery (desktop) = always "full + charging"
	}

	pct = float64(status.BatteryLifePercent)
	if status.BatteryLifePercent == 255 {
		pct = 100 // Unknown status
	}
	charging = status.ACLineStatus == 1 // 1=online, 0=offline, 255=unknown
	if status.ACLineStatus == 255 {
		charging = true
	}
	return pct, charging
}
