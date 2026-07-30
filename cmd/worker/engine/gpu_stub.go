//go:build (!windows && !linux) || (linux && !cgo)

/*
 * cmd/worker/engine/gpu_stub.go — Stub GPU engine for non-Windows platforms
 *
 * On Linux/macOS/Android, we can't use syscall.LoadDLL to access OpenCL.
 * The worker falls back to the pure-Go CPU engine (fallback.go).
 * CGO-enabled builds use opencl_engine.go instead.
 */

package engine

import "fmt"

// GPUEngine stub — always unavailable on non-Windows without CGO
type GPUEngine struct {
	available bool
}

// GPUBuffer stub
type GPUBuffer struct {
	mem  uintptr
	size uint64
}

func (b *GPUBuffer) Size() uint64 { return b.size }

// GPUKernel stub (defined in gpu_windows.go for Windows)
type GPUKernel struct {
	kernel  uintptr
	program uintptr
	name    string
}

func (k *GPUKernel) Name() string { return k.name }

// NewGPUEngine always returns an error on non-Windows (use CGO for real GPU)
func NewGPUEngine() (*GPUEngine, error) {
	return nil, fmt.Errorf("GPU engine requires Windows (syscall) or CGO (opencl_engine.go)")
}

func (e *GPUEngine) BackendName() string       { return "GPU (unavailable)" }
func (e *GPUEngine) DeviceInfo() string         { return "" }
func (e *GPUEngine) RunMicroBenchmark() float64 { return 0 }
func (e *GPUEngine) Close()                     {}

func (e *GPUEngine) CreateBuffer(size uint64, flags uint32, data []byte) (*GPUBuffer, error) {
	return nil, fmt.Errorf("stub")
}
func (e *GPUEngine) ReleaseBuffer(b *GPUBuffer) {}
func (e *GPUEngine) CreateKernelFromSource(source, kernelName string) (*GPUKernel, error) {
	return nil, fmt.Errorf("stub")
}
func (e *GPUEngine) SetKernelArg(k *GPUKernel, index uint32, value interface{}) error {
	return fmt.Errorf("stub")
}
func (e *GPUEngine) ExecuteNDRange(k *GPUKernel, workDim uint32,
	globalSize, globalOffset, localSize []uint64, outputBuffers []*GPUBuffer) error {
	return fmt.Errorf("stub")
}
func (e *GPUEngine) ReadBuffer(b *GPUBuffer, offset uint64, size uint64) ([]byte, error) {
	return nil, fmt.Errorf("stub")
}
func (e *GPUEngine) WriteBuffer(b *GPUBuffer, offset uint64, data []byte) error {
	return fmt.Errorf("stub")
}
func (e *GPUEngine) Finish() error   { return nil }
func (e *GPUEngine) ReleaseKernel(k *GPUKernel) {}
