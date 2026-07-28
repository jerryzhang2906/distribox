/*
 * Go worker entry point for Android shared library.
 * Compiled with -buildmode=c-shared to produce libdistribox_worker.so
 * which is loaded by the Android Java activity via System.loadLibrary().
 */
package main

import "C"
import (
	"github.com/distribox/cmd/worker/agent"
	"github.com/distribox/cmd/worker/capability"
	"github.com/distribox/cmd/worker/monitor"
)

var worker *agent.GRPCWorkerClient

//export StartWorker
func StartWorker(orchestratorAddr *C.char, token *C.char, name *C.char) *C.char {
	addr := C.GoString(orchestratorAddr)
	tok := C.GoString(token)
	n := C.GoString(name)

	if tok == "" {
		tok = "insecure-lan-mode"
	}

	detector := capability.NewDetector()
	caps := detector.Detect()

	resMon := monitor.NewResourceMonitor()
	go resMon.Run()

	worker = agent.NewGRPCWorkerClient(agent.WorkerConfig{
		Name:             n,
		OrchestratorAddr: addr,
		ClusterToken:     tok,
		Capabilities:     caps,
		Policy:           agent.UserPolicy{Intensity: 0.8, OnlyWhenCharging: true},
		ResourceMon:      resMon,
	})

	err := worker.Connect(nil)
	if err != nil {
		return C.CString(err.Error())
	}

	go worker.Run()
	return C.CString("")
}

//export StopWorker
func StopWorker() {
	if worker != nil {
		worker.Disconnect()
		worker = nil
	}
}

//export WorkerStatus
func WorkerStatus() *C.char {
	if worker == nil {
		return C.CString("stopped")
	}
	return C.CString("running")
}

func main() {}
