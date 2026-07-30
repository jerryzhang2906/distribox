/*
 * cmd/worker/monitor/resources_linux.go — Linux/Android real-time resource monitoring
 *
 * Reads real system metrics from /proc and /sys filesystems.
 * No hardcoded fake data — if a metric can't be read, returns 0.
 *
 * CPU: /proc/stat diff-based idle/total calculation
 * Memory: /proc/meminfo
 * GPU: /sys/class/kgsl/kgsl-3d0/gpu_busy_percentage (Qualcomm) or similar
 * Battery: /sys/class/power_supply
 * Thermal: /sys/class/thermal/thermal_zoneX/temp
 */

//go:build linux && cgo

package monitor

import (
	"math"
	"os"
	"strconv"
	"strings"
	"sync"
)

// ── CPU sampling via /proc/stat ────────────────────────

var (
	prevIdleLinux  uint64
	prevTotalLinux uint64
	cpuLinuxMu     sync.Mutex
)

func sampleCPU() float64 {
	data, err := os.ReadFile("/proc/stat")
	if err != nil {
		return 0
	}

	// First line is aggregate: "cpu  user nice system idle iowait irq softirq steal guest guest_nice"
	lines := strings.SplitN(string(data), "\n", 2)
	if len(lines) == 0 {
		return 0
	}

	fields := strings.Fields(lines[0])
	if len(fields) < 5 || fields[0] != "cpu" {
		return 0
	}

	// Parse fields: user, nice, system, idle, iowait, irq, softirq, steal...
	var vals [8]uint64
	for i := 1; i < len(fields) && i <= 8; i++ {
		v, err := strconv.ParseUint(fields[i], 10, 64)
		if err != nil {
			continue
		}
		vals[i-1] = v
	}

	idle := vals[3] + vals[4] // idle + iowait
	total := vals[0] + vals[1] + vals[2] + vals[3] + vals[4] + vals[5] + vals[6] + vals[7]

	cpuLinuxMu.Lock()
	defer cpuLinuxMu.Unlock()

	if prevTotalLinux == 0 {
		prevIdleLinux = idle
		prevTotalLinux = total
		return 0
	}

	idleDelta := idle - prevIdleLinux
	totalDelta := total - prevTotalLinux

	prevIdleLinux = idle
	prevTotalLinux = total

	if totalDelta == 0 {
		return 0
	}

	usage := float64(totalDelta-idleDelta) / float64(totalDelta) * 100.0
	return math.Min(100.0, math.Max(0.0, usage))
}

// ── Memory sampling via /proc/meminfo ───────────────────

func sampleMemory() (usedMB int64, availMB int64) {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0, 0
	}

	var totalKB, availKB, freeKB, buffersKB, cachedKB int64
	lines := strings.Split(string(data), "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(trimmed, "MemTotal"):
			totalKB = parseMemKB(trimmed)
		case strings.HasPrefix(trimmed, "MemAvailable"):
			availKB = parseMemKB(trimmed)
		case strings.HasPrefix(trimmed, "MemFree"):
			freeKB = parseMemKB(trimmed)
		case strings.HasPrefix(trimmed, "Buffers"):
			buffersKB = parseMemKB(trimmed)
		case strings.HasPrefix(trimmed, "Cached"):
			cachedKB = parseMemKB(trimmed)
		}
	}

	if availKB > 0 {
		availMB = availKB / 1024
	} else {
		// Older kernels without MemAvailable: estimate from free + buffers + cache
		availKB = freeKB + buffersKB + cachedKB
		availMB = availKB / 1024
	}

	if totalKB > 0 {
		totalMB := totalKB / 1024
		usedMB = totalMB - availMB
		if usedMB < 0 {
			usedMB = 0
		}
	}

	return usedMB, availMB
}

func parseMemKB(line string) int64 {
	// Format: "MemTotal:       3928448 kB"
	parts := strings.Fields(line)
	if len(parts) >= 2 {
		val, err := strconv.ParseInt(parts[1], 10, 64)
		if err == nil {
			return val
		}
	}
	return 0
}

// ── GPU sampling via sysfs ──────────────────────────────

func sampleGPU() float64 {
	// Try Qualcomm Adreno GPU utilization
	paths := []string{
		"/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage",
		"/sys/class/kgsl/kgsl-3d0/gpubusy",
		"/sys/kernel/gpu/gpu_busy",
	}

	for _, p := range paths {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		val, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
		if err != nil {
			// Some paths have "X%" format
			s := strings.TrimSpace(string(data))
			s = strings.TrimSuffix(s, "%")
			val, err = strconv.ParseFloat(s, 64)
			if err != nil {
				continue
			}
		}
		if val >= 0 && val <= 100 {
			return val
		}
	}

	// Try Mali GPU utilization
	// Mali devices expose devfreq stats
	maliPaths := []string{
		"/sys/class/misc/mali0/device/devfreq/device0/load",
		"/sys/devices/platform/mali*/devfreq/*/load",
	}

	for _, p := range maliPaths {
		data, err := os.ReadFile(p)
		if err != nil {
			continue
		}
		// Format: "frequency load%" or just a percentage
		s := strings.TrimSpace(string(data))
		// Try direct percentage
		s = strings.TrimSuffix(s, "%")
		if idx := strings.IndexByte(s, ' '); idx > 0 {
			// "freq load%" format, take the second part
			s = s[idx+1:]
			s = strings.TrimSuffix(s, "%")
		}
		val, err := strconv.ParseFloat(s, 64)
		if err == nil && val >= 0 && val <= 100 {
			return val
		}
	}

	return 0 // GPU utilization not available
}

// ── Battery sampling via /sys/class/power_supply ────────

func sampleBattery() (pct float64, charging bool) {
	// Find battery device
	batteryDirs := findBatteryDirs()
	if len(batteryDirs) == 0 {
		return 100, true // No battery = always "full" (desktop/server)
	}

	batDir := batteryDirs[0] // Use first battery found

	// Read capacity
	data, err := os.ReadFile(batDir + "/capacity")
	if err == nil {
		cap, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
		if err == nil {
			pct = cap
		}
	}

	// Read status (Charging/Discharging/Full)
	data, err = os.ReadFile(batDir + "/status")
	if err == nil {
		status := strings.TrimSpace(string(data))
		switch status {
		case "Charging", "Full":
			charging = true
		case "Discharging":
			charging = false
		default:
			charging = true
		}
	} else {
		charging = true
	}

	// If no capacity could be read, assume full
	if pct == 0 {
		pct = 100
	}

	return pct, charging
}

func findBatteryDirs() []string {
	var dirs []string
	entries, err := os.ReadDir("/sys/class/power_supply")
	if err != nil {
		return nil
	}
	for _, entry := range entries {
		if entry.IsDir() {
			dirs = append(dirs, "/sys/class/power_supply/"+entry.Name())
		}
	}
	return dirs
}

// ── Thermal sampling via /sys/class/thermal ─────────────

func sampleThermal() (tempC float64, throttled bool) {
	// Read thermal zone temperatures
	entries, err := os.ReadDir("/sys/class/thermal")
	if err != nil {
		return 0, false
	}

	var maxTemp float64
	for _, entry := range entries {
		if !entry.IsDir() || !strings.HasPrefix(entry.Name(), "thermal_zone") {
			continue
		}

		// Read temperature (in millidegrees Celsius)
		data, err := os.ReadFile("/sys/class/thermal/" + entry.Name() + "/temp")
		if err != nil {
			continue
		}

		val, err := strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
		if err != nil {
			continue
		}

		// Convert from millidegrees to degrees
		temp := val / 1000.0
		if temp > maxTemp {
			maxTemp = temp
		}

		// Check type to determine if this is CPU/GPU thermal zone
		typeData, err := os.ReadFile("/sys/class/thermal/" + entry.Name() + "/type")
		if err != nil {
			continue
		}
		zoneType := strings.TrimSpace(string(typeData))

		// Check for throttling trip points
		tripData, err := os.ReadFile("/sys/class/thermal/" + entry.Name() + "/trip_point_0_temp")
		if err == nil {
			tripTemp, err := strconv.ParseFloat(strings.TrimSpace(string(tripData)), 64)
			if err == nil && temp >= tripTemp/1000.0 {
				if zoneType == "cpu-thermal" || zoneType == "gpu-thermal" || zoneType == "tsens_tz_sensor0" {
					throttled = true
				}
			}
		}
	}

	if maxTemp > 125 {
		maxTemp = 0 // Unreasonable reading, ignore
	}

	return maxTemp, throttled
}
