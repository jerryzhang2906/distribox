/*
 * cmd/worker/engine/gpu_linux.go — Real GPU execution on Linux/Android via CGO OpenCL
 *
 * Wraps OCLEngine (opencl_engine.go + opencl_shim.c) as a GPUEngine-compatible
 * implementation. Uses CGO to call libOpenCL.so for real Mali/Adreno GPU execution.
 *
 * Build: CGO_ENABLED=1 GOOS=android GOARCH=arm64 go build
 *        CGO_ENABLED=1 GOOS=linux go build
 */

//go:build linux && cgo

package engine

import (
	"fmt"
	"log"
)

// ── GPUEngine on Linux wraps OCLEngine ─────────────────

type GPUEngine struct {
	ocl       *OCLEngine
	available bool
}

func NewGPUEngine() (*GPUEngine, error) {
	ocl, err := NewOCLEngine()
	if err != nil {
		return nil, fmt.Errorf("GPU engine unavailable: %w", err)
	}

	log.Printf("REAL GPU (CGO/OpenCL): %s", ocl.BackendName())
	return &GPUEngine{
		ocl:       ocl,
		available: true,
	}, nil
}

func (e *GPUEngine) BackendName() string {
	if e.available && e.ocl != nil {
		return "OpenCL (CGO): " + e.ocl.BackendName()
	}
	return "GPU (unavailable)"
}

func (e *GPUEngine) DeviceInfo() string {
	if e.ocl != nil {
		return e.ocl.DeviceInfo()
	}
	return ""
}

func (e *GPUEngine) RunMicroBenchmark() float64 {
	if e.ocl != nil {
		return e.ocl.RunMicroBenchmark()
	}
	return 0
}

func (e *GPUEngine) Close() {
	if e.ocl != nil {
		e.ocl.Close()
		e.ocl = nil
	}
	e.available = false
}

// ── Buffer wrappers ────────────────────────────────────

// GPUBuffer is defined per-platform. On Linux, wraps OCLBuffer.
type GPUBuffer struct {
	oclBuf *OCLBuffer
	size   uint64
}

func (b *GPUBuffer) Size() uint64 { return b.size }

func (e *GPUEngine) CreateBuffer(size uint64, flags uint32, data []byte) (*GPUBuffer, error) {
	oclBuf, err := e.ocl.CreateBuffer(size, flags, data)
	if err != nil {
		return nil, err
	}
	return &GPUBuffer{oclBuf: oclBuf, size: size}, nil
}

func (e *GPUEngine) ReleaseBuffer(b *GPUBuffer) {
	if b != nil && b.oclBuf != nil {
		e.ocl.ReleaseBuffer(b.oclBuf)
		b.oclBuf = nil
	}
}

func (e *GPUEngine) ReadBuffer(b *GPUBuffer, offset uint64, size uint64) ([]byte, error) {
	return e.ocl.ReadBuffer(b.oclBuf, offset, size)
}

func (e *GPUEngine) WriteBuffer(b *GPUBuffer, offset uint64, data []byte) error {
	return e.ocl.WriteBuffer(b.oclBuf, offset, data)
}

// ── Kernel wrappers ────────────────────────────────────

type GPUKernel struct {
	oclKernel *OCLKernel
	name      string
}

func (k *GPUKernel) Name() string { return k.name }

func (e *GPUEngine) CreateKernelFromSource(source, kernelName string) (*GPUKernel, error) {
	oclKern, err := e.ocl.CreateKernelFromSource(source, kernelName)
	if err != nil {
		return nil, err
	}
	return &GPUKernel{oclKernel: oclKern, name: kernelName}, nil
}

func (e *GPUEngine) SetKernelArg(k *GPUKernel, index uint32, value interface{}) error {
	// Adapt the value: if caller passes a *GPUBuffer, unwrap to *OCLBuffer
	switch v := value.(type) {
	case *GPUBuffer:
		return e.ocl.SetKernelArg(k.oclKernel, index, v.oclBuf)
	default:
		return e.ocl.SetKernelArg(k.oclKernel, index, value)
	}
}

func (e *GPUEngine) ExecuteNDRange(k *GPUKernel, workDim uint32,
	globalSize, globalOffset, localSize []uint64,
	outputBuffers []*GPUBuffer) error {

	// Unwrap output buffers
	oclOutputs := make([]*OCLBuffer, len(outputBuffers))
	for i, buf := range outputBuffers {
		if buf != nil {
			oclOutputs[i] = buf.oclBuf
		}
	}

	return e.ocl.ExecuteNDRange(k.oclKernel, workDim,
		globalSize, globalOffset, localSize, oclOutputs)
}

func (e *GPUEngine) Finish() error {
	return e.ocl.Finish()
}

func (e *GPUEngine) ReleaseKernel(k *GPUKernel) {
	if k != nil && k.oclKernel != nil {
		e.ocl.ReleaseKernel(k.oclKernel)
		k.oclKernel = nil
	}
}
