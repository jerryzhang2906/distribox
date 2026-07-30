/*
 * cmd/worker/monitor/resources_defaults.go — Default resource monitoring stubs
 *
 * Used on non-Windows platforms. Provides conservative defaults.
 */

//go:build (!windows && !linux) || (linux && !cgo)

package monitor

func sampleCPU() float64 {
	return 0
}

func sampleGPU() float64 {
	return 0
}

func sampleBattery() (pct float64, charging bool) {
	return 100, true
}

func sampleMemory() (usedMB int64, availMB int64) {
	return 0, 0
}

func sampleThermal() (tempC float64, throttled bool) {
	return 0, false
}
