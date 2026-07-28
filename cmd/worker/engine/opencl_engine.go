//go:build cgo

/*
 * cmd/worker/engine/opencl_engine.go — Real GPU execution via CGO OpenCL shim
 *
 * Uses the local OpenCL GPU (Intel/Mali/NVIDIA) for actual kernel execution.
 * Falls back to Go-CPU engine if OpenCL is unavailable.
 */

package engine

/*
#cgo LDFLAGS: -lOpenCL
#include "opencl_shim.c"
*/
import "C"

import (
	"fmt"
	"log"
	"unsafe"
)

// ── OpenCL Engine ─────────────────────────────────────

type OCLEngine struct {
	initialized bool
	backendName string
}

func NewOCLEngine() (*OCLEngine, error) {
	e := &OCLEngine{}
	ret := C.ocl_init()
	if ret != 0 {
		return nil, fmt.Errorf("OpenCL init failed")
	}
	e.initialized = true
	e.backendName = C.GoString(C.ocl_device_name())
	log.Printf("OpenCL Engine: %s", e.backendName)
	return e, nil
}

func (e *OCLEngine) BackendName() string       { return e.backendName }
func (e *OCLEngine) DeviceInfo() string         { return e.backendName }
func (e *OCLEngine) RunMicroBenchmark() float64 { return 0 }

func (e *OCLEngine) Close() {
	if e.initialized {
		C.ocl_close()
		e.initialized = false
	}
}

// ── Buffer ────────────────────────────────────────────

type OCLBuffer struct {
	handle unsafe.Pointer // cl_mem
	size   uint64
}

func (b *OCLBuffer) Size() uint64 { return b.size }

func (e *OCLEngine) CreateBuffer(size uint64, flags uint32, data []byte) (*OCLBuffer, error) {
	var cData unsafe.Pointer
	if len(data) > 0 {
		cData = unsafe.Pointer(&data[0])
	}
	handle := C.ocl_create_buffer(C.size_t(size), cData)
	if handle == nil {
		return nil, fmt.Errorf("OCL buffer create failed")
	}
	return &OCLBuffer{handle: handle, size: size}, nil
}

func (e *OCLEngine) WriteBuffer(b *OCLBuffer, offset uint64, data []byte) error {
	if len(data) == 0 { return nil }
	ret := C.ocl_write_buffer(b.handle, C.size_t(offset), C.size_t(len(data)), unsafe.Pointer(&data[0]))
	if ret != 0 { return fmt.Errorf("OCL write failed") }
	return nil
}

func (e *OCLEngine) ReadBuffer(b *OCLBuffer, offset uint64, size uint64) ([]byte, error) {
	out := make([]byte, size)
	ret := C.ocl_read_buffer(b.handle, C.size_t(offset), C.size_t(size), unsafe.Pointer(&out[0]))
	if ret != 0 { return nil, fmt.Errorf("OCL read failed") }
	return out, nil
}

func (e *OCLEngine) ReleaseBuffer(b *OCLBuffer) {
	if b.handle != nil {
		C.ocl_release_buffer(b.handle)
		b.handle = nil
	}
}

// ── Kernel ────────────────────────────────────────────

type OCLKernel struct {
	handle unsafe.Pointer
	name   string
}

func (k *OCLKernel) Name() string { return k.name }

func (e *OCLEngine) CreateKernelFromSource(source, kernelName string) (*OCLKernel, error) {
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))
	cName := C.CString(kernelName)
	defer C.free(unsafe.Pointer(cName))

	handle := C.ocl_create_kernel(cSource, cName)
	if handle == nil {
		return nil, fmt.Errorf("OCL kernel '%s' create failed", kernelName)
	}
	return &OCLKernel{handle: handle, name: kernelName}, nil
}

func (e *OCLEngine) SetKernelArg(k *OCLKernel, index uint32, value interface{}) error {
	switch v := value.(type) {
	case *OCLBuffer:
		bufHandle := v.handle
		ret := C.ocl_set_kernel_arg(k.handle, C.int(index), 0, unsafe.Pointer(&bufHandle), 1)
		if ret != 0 { return fmt.Errorf("set arg %d failed", index) }
	case []byte:
		ret := C.ocl_set_kernel_arg(k.handle, C.int(index), C.size_t(len(v)), unsafe.Pointer(&v[0]), 0)
		if ret != 0 { return fmt.Errorf("set arg %d failed", index) }
	case int32:
		ret := C.ocl_set_kernel_arg(k.handle, C.int(index), 4, unsafe.Pointer(&v), 0)
		if ret != 0 { return fmt.Errorf("set arg %d failed", index) }
	case float32:
		ret := C.ocl_set_kernel_arg(k.handle, C.int(index), 4, unsafe.Pointer(&v), 0)
		if ret != 0 { return fmt.Errorf("set arg %d failed", index) }
	default:
		return fmt.Errorf("unsupported arg type for arg %d", index)
	}
	return nil
}

func (e *OCLEngine) ExecuteNDRange(k *OCLKernel, workDim uint32,
	globalSize, globalOffset, localSize []uint64,
	outputBuffers []*OCLBuffer) error {

	var gs, goff, ls C.size_t
	var pgs, pgoff, pls *C.size_t

	if len(globalSize) > 0 {
		gs = C.size_t(globalSize[0])
		pgs = &gs
	}
	if len(globalOffset) > 0 {
		goff = C.size_t(globalOffset[0])
		pgoff = &goff
	}
	if len(localSize) > 0 && localSize[0] > 0 {
		ls = C.size_t(localSize[0])
		pls = &ls
	}

	ret := C.ocl_execute_ndrange(k.handle, C.int(workDim), pgs, pgoff, pls)
	if ret != 0 { return fmt.Errorf("OCL execute failed") }

	// Read back output buffers
	for i, buf := range outputBuffers {
		if buf != nil && buf.handle != nil {
			_ = i // output read-back handled by caller
			_ = buf
		}
	}

	return nil
}

func (e *OCLEngine) Finish() error {
	ret := C.ocl_finish()
	if ret != 0 { return fmt.Errorf("OCL finish failed") }
	return nil
}

func (e *OCLEngine) ReleaseKernel(k *OCLKernel) {
	if k.handle != nil {
		C.ocl_release_kernel(k.handle)
		k.handle = nil
	}
}
