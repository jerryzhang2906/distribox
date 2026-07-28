//go:build cgo
// +build cgo

// cmd/worker/engine/bridge.go 鈥- CGO bridge to C compute engine
// This file wraps the C worker_engine.h API via CGO.
// Only compiled when CGO is enabled AND a C compiler is available.
// When CGO is disabled, Go uses the pure-Go CPU engine in fallback.go.

package engine

// #cgo LDFLAGS: -ldistribox_engine -lOpenCL
// #cgo linux LDFLAGS: -ldistribox_engine -lOpenCL -lm
// #cgo darwin LDFLAGS: -ldistribox_engine -framework OpenCL
// #cgo windows LDFLAGS: -ldistribox_engine -lOpenCL
/*
#include <stdlib.h>
#include "distribox/worker_engine.h"

// Helper to convert Go string to C string and back
static char* engine_device_info_helper(engine_ctx_t* ctx) {
    return (char*)engine_get_device_info(ctx);
}
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// 鈹€鈹€ Engine context (CGO-backed) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

type CGOEngine struct {
	ctx *C.engine_ctx_t
}

// NewCGOEngine initializes the C compute engine
func NewCGOEngine(preferredBackend string) (*CGOEngine, error) {
	var backend *C.char
	if preferredBackend != "" {
		backend = C.CString(preferredBackend)
		defer C.free(unsafe.Pointer(backend))
	}

	ctx := C.engine_init(backend)
	if ctx == nil {
		return nil, fmt.Errorf("C engine init failed")
	}

	return &CGOEngine{ctx: ctx}, nil
}

func (e *CGOEngine) BackendName() string {
	return C.GoString(C.engine_backend_name(e.ctx))
}

func (e *CGOEngine) DeviceInfo() string {
	return C.GoString(C.engine_device_info_helper(e.ctx))
}

func (e *CGOEngine) Close() {
	if e.ctx != nil {
		C.engine_destroy(e.ctx)
		e.ctx = nil
	}
}

// 鈹€鈹€ Interface implementations 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
func (b *CGOBuffer) Size() uint64 {
	if b.buf == nil { return 0 }
	return uint64(C.engine_buffer_get_size(b.buf))
}
func (p *CGOProgram) ID() string    { return "cgo-prog" }
func (p *CGOProgram) IsBuilt() bool { return true }
func (k *CGOKernel) Name() string   { return k.name }

// 鈹€鈹€ Buffer 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

type CGOBuffer struct {
	buf *C.engine_buffer_t
}



func (e *CGOEngine) CreateBuffer(size uint64, flags uint32, data []byte) (*CGOBuffer, error) {
	var cData unsafe.Pointer
	if len(data) > 0 {
		cData = unsafe.Pointer(&data[0])
	}
	buf := C.engine_buffer_create(e.ctx, C.uint64_t(size), C.uint32_t(flags), cData)
	if buf == nil {
		return nil, fmt.Errorf("buffer create failed")
	}
	return &CGOBuffer{buf: buf}, nil
}

func (e *CGOEngine) WriteBuffer(b *CGOBuffer, offset uint64, data []byte) error {
	if len(data) == 0 {
		return nil
	}
	ret := C.engine_buffer_write(e.ctx, b.buf, C.uint64_t(offset),
		C.uint64_t(len(data)), unsafe.Pointer(&data[0]))
	if ret != 0 {
		return fmt.Errorf("buffer write failed")
	}
	return nil
}

func (e *CGOEngine) ReadBuffer(b *CGOBuffer, offset uint64, size uint64) ([]byte, error) {
	buf := C.malloc(C.size_t(size))
	defer C.free(buf)
	ret := C.engine_buffer_read(e.ctx, b.buf, C.uint64_t(offset),
		C.uint64_t(size), buf)
	if ret != 0 {
		return nil, fmt.Errorf("buffer read failed")
	}
	return C.GoBytes(buf, C.int(size)), nil
}

func (e *CGOEngine) ReleaseBuffer(b *CGOBuffer) {
	if b.buf != nil {
		C.engine_buffer_release(e.ctx, b.buf)
		b.buf = nil
	}
}

// 鈹€鈹€ Program 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

type CGOProgram struct {
	prog *C.engine_program_t
}

func (e *CGOEngine) CreateProgramFromSource(source string, options string) (*CGOProgram, error) {
	cSource := C.CString(source)
	defer C.free(unsafe.Pointer(cSource))

	var cOptions *C.char
	if options != "" {
		cOptions = C.CString(options)
		defer C.free(unsafe.Pointer(cOptions))
	}

	prog := C.engine_program_create_from_source(e.ctx, cSource,
		C.uint64_t(len(source)), cOptions)
	if prog == nil {
		return nil, fmt.Errorf("program create failed")
	}
	return &CGOProgram{prog: prog}, nil
}

func (e *CGOEngine) BuildProgram(p *CGOProgram, options string) (string, error) {
	var cOptions *C.char
	if options != "" {
		cOptions = C.CString(options)
		defer C.free(unsafe.Pointer(cOptions))
	}

	var buildLog *C.char
	ret := C.engine_program_build(e.ctx, p.prog, cOptions, &buildLog)
	log := C.GoString(buildLog)
	if buildLog != nil {
		C.free(unsafe.Pointer(buildLog))
	}
	if ret != 0 {
		return log, fmt.Errorf("program build failed: %s", log)
	}
	return log, nil
}

func (e *CGOEngine) ReleaseProgram(p *CGOProgram) {
	if p.prog != nil {
		C.engine_program_release(e.ctx, p.prog)
		p.prog = nil
	}
}

// 鈹€鈹€ Kernel 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

type CGOKernel struct {
	kernel *C.engine_kernel_t
	name   string
}

func (e *CGOEngine) CreateKernel(p *CGOProgram, name string) (*CGOKernel, error) {
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	k := C.engine_kernel_create(e.ctx, p.prog, cName)
	if k == nil {
		return nil, fmt.Errorf("kernel create failed: %s", name)
	}
	return &CGOKernel{kernel: k, name: name}, nil
}

func (e *CGOEngine) SetKernelArg(k *CGOKernel, index uint32, value interface{}) error {
	// Type switch to handle buffer vs scalar arguments
	switch v := value.(type) {
	case *CGOBuffer:
		ret := C.engine_kernel_set_arg(e.ctx, k.kernel, C.uint32_t(index),
			C.uint64_t(0), unsafe.Pointer(v.buf), true)
		if ret != 0 {
			return fmt.Errorf("set kernel arg %d (buffer) failed", index)
		}
	case []byte:
		ret := C.engine_kernel_set_arg(e.ctx, k.kernel, C.uint32_t(index),
			C.uint64_t(len(v)), unsafe.Pointer(&v[0]), false)
		if ret != 0 {
			return fmt.Errorf("set kernel arg %d (scalar) failed", index)
		}
	case int32:
		ret := C.engine_kernel_set_arg(e.ctx, k.kernel, C.uint32_t(index),
			C.uint64_t(4), unsafe.Pointer(&v), false)
		if ret != 0 {
			return fmt.Errorf("set kernel arg %d (int32) failed", index)
		}
	default:
		return fmt.Errorf("unsupported arg type for arg %d", index)
	}
	return nil
}

func (e *CGOEngine) ReleaseKernel(k *CGOKernel) {
	if k.kernel != nil {
		C.engine_kernel_release(e.ctx, k.kernel)
		k.kernel = nil
	}
}

// 鈹€鈹€ Execute NDRange 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

func (e *CGOEngine) ExecuteNDRange(kernel *CGOKernel, workDim uint32,
	globalSize, globalOffset, localSize []uint64,
	outputBuffers []*CGOBuffer) error {

	var gSize, gOffset, lSize *C.uint64_t
	if len(globalSize) > 0 {
		gSize = (*C.uint64_t)(unsafe.Pointer(&globalSize[0]))
	}
	if len(globalOffset) > 0 {
		gOffset = (*C.uint64_t)(unsafe.Pointer(&globalOffset[0]))
	}
	if len(localSize) > 0 {
		lSize = (*C.uint64_t)(unsafe.Pointer(&localSize[0]))
	}

	// Build C array of output buffer pointers
	var outBufs **C.engine_buffer_t
	if len(outputBuffers) > 0 {
		outBufs = (**C.engine_buffer_t)(C.malloc(C.size_t(len(outputBuffers)) * C.size_t(unsafe.Sizeof((*C.engine_buffer_t)(nil)))))
		defer C.free(unsafe.Pointer(outBufs))
		for i, b := range outputBuffers {
			*(**C.engine_buffer_t)(unsafe.Pointer(uintptr(unsafe.Pointer(outBufs)) + uintptr(i)*unsafe.Sizeof((*C.engine_buffer_t)(nil)))) = b.buf
		}
	}

	ret := C.engine_execute_ndrange(e.ctx, kernel.kernel,
		C.uint32_t(workDim), gSize, gOffset, lSize,
		outBufs, C.uint32_t(len(outputBuffers)))
	if ret != 0 {
		return fmt.Errorf("NDRange execution failed")
	}
	return nil
}

func (e *CGOEngine) Finish() error {
	ret := C.engine_finish(e.ctx)
	if ret != 0 {
		return fmt.Errorf("engine finish failed")
	}
	return nil
}

// 鈹€鈹€ Benchmark 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

func (e *CGOEngine) RunMicroBenchmark() float64 {
	return float64(C.engine_run_micro_benchmark(e.ctx))
}

