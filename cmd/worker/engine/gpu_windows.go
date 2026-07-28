//go:build windows

/*
 * cmd/worker/engine/gpu_windows.go — Pure Go OpenCL GPU engine (no CGO)
 *
 * Windows-only: Uses syscall.LoadLibrary + syscall.Syscall to call OpenCL.dll directly.
 * No C compiler required. Falls back to Go-CPU if OpenCL is unavailable.
 *
 * This gives REAL GPU execution on Intel/NVIDIA/AMD GPUs without CGO.
 */

package engine

import (
	"fmt"
	"log"
	"syscall"
	"unsafe"
)

var (
	openclDLL *syscall.DLL
)

func init() {
	// Try loading any available OpenCL runtime
	tryDLL := func(path string) bool {
		dll, err := syscall.LoadDLL(path)
		if err == nil { openclDLL = dll; return true }
		return false
	}
	// Try OpenCL.dll (our proxy or system default)
	if tryDLL("OpenCL.dll") { return }
	// Try Intel's ICD directly
	if tryDLL("IntelOpenCL64.dll") { return }
	// Try the backup original
	if tryDLL("C:\\Windows\\System32\\OpenCL_orig.dll") { return }
	if tryDLL("OpenCL_orig.dll") { return }
}

// ── OpenCL GPU Engine ──────────────────────────────────

type GPUEngine struct {
	platform   uintptr
	device     uintptr
	context    uintptr
	queue      uintptr
	deviceName string
	available  bool
}

func NewGPUEngine() (*GPUEngine, error) {
	if openclDLL == nil {
		return nil, fmt.Errorf("OpenCL.dll not found")
	}

	e := &GPUEngine{}

	// clGetPlatformIDs
	proc, _ := openclDLL.FindProc("clGetPlatformIDs")
	var numPlatforms uint32
	var platform uintptr
	ret, _, _ := proc.Call(1, uintptr(unsafe.Pointer(&platform)), uintptr(unsafe.Pointer(&numPlatforms)))
	if ret != 0 || numPlatforms == 0 {
		return nil, fmt.Errorf("no OpenCL platform (err=%d)", ret)
	}
	e.platform = platform

	// clGetDeviceIDs (GPU first, CPU fallback)
	proc, _ = openclDLL.FindProc("clGetDeviceIDs")
	var device uintptr
	var numDevices uint32
	const CL_DEVICE_TYPE_GPU = 1 << 2
	ret, _, _ = proc.Call(platform, CL_DEVICE_TYPE_GPU, 1,
		uintptr(unsafe.Pointer(&device)), uintptr(unsafe.Pointer(&numDevices)))
	if ret != 0 {
		const CL_DEVICE_TYPE_CPU = 1 << 1
		ret, _, _ = proc.Call(platform, CL_DEVICE_TYPE_CPU, 1,
			uintptr(unsafe.Pointer(&device)), uintptr(unsafe.Pointer(&numDevices)))
	}
	if ret != 0 {
		return nil, fmt.Errorf("no OpenCL device (err=%d)", ret)
	}
	e.device = device

	// clCreateContext
	proc, _ = openclDLL.FindProc("clCreateContext")
	var errCode int32
	ctx, _, _ := proc.Call(0, 1, uintptr(unsafe.Pointer(&device)), 0, 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		return nil, fmt.Errorf("context create failed (err=%d)", errCode)
	}
	e.context = ctx

	// clCreateCommandQueueWithProperties
	proc, _ = openclDLL.FindProc("clCreateCommandQueueWithProperties")
	q, _, _ := proc.Call(ctx, device, 0, uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		openclDLL.MustFindProc("clReleaseContext").Call(ctx)
		return nil, fmt.Errorf("queue create failed (err=%d)", errCode)
	}
	e.queue = q

	// Get device name
	var name [256]byte
	proc, _ = openclDLL.FindProc("clGetDeviceInfo")
	const CL_DEVICE_NAME = 0x102B
	var retSize uint64
	proc.Call(device, CL_DEVICE_NAME, 256, uintptr(unsafe.Pointer(&name[0])), uintptr(unsafe.Pointer(&retSize)))
	e.deviceName = string(name[:retSize-1]) // strip null

	e.available = true
	log.Printf("GPU Engine: OpenCL -> %s", e.deviceName)
	return e, nil
}

func (e *GPUEngine) BackendName() string       {
	if e.available { return "OpenCL: " + e.deviceName }
	return "CPU Fallback"
}
func (e *GPUEngine) DeviceInfo() string         { return e.deviceName }
func (e *GPUEngine) RunMicroBenchmark() float64 { return 0 }

func (e *GPUEngine) Close() {
	if e.queue != 0 {
		openclDLL.MustFindProc("clReleaseCommandQueue").Call(e.queue)
	}
	if e.context != 0 {
		openclDLL.MustFindProc("clReleaseContext").Call(e.context)
	}
	e.available = false
}

// ── GPU Buffer ────────────────────────────────────────

type GPUBuffer struct {
	mem  uintptr // cl_mem
	size uint64
}

func (b *GPUBuffer) Size() uint64 { return b.size }

func (e *GPUEngine) CreateBuffer(size uint64, flags uint32, data []byte) (*GPUBuffer, error) {
	const CL_MEM_READ_WRITE = 1 << 0
	const CL_MEM_COPY_HOST_PTR = 1 << 5
	memFlags := uintptr(CL_MEM_READ_WRITE)
	var hostPtr unsafe.Pointer
	if len(data) > 0 {
		memFlags |= CL_MEM_COPY_HOST_PTR
		hostPtr = unsafe.Pointer(&data[0])
	}
	proc, _ := openclDLL.FindProc("clCreateBuffer")
	var errCode int32
	mem, _, _ := proc.Call(e.context, memFlags, uintptr(size), uintptr(hostPtr), uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		return nil, fmt.Errorf("OCL buffer create failed (err=%d)", errCode)
	}
	return &GPUBuffer{mem: mem, size: size}, nil
}

func (e *GPUEngine) ReleaseBuffer(b *GPUBuffer) {
	if b.mem != 0 {
		openclDLL.MustFindProc("clReleaseMemObject").Call(b.mem)
		b.mem = 0
	}
}

// ── GPU Kernel ────────────────────────────────────────

type GPUKernel struct {
	kernel     uintptr
	program    uintptr
	name       string
}

func (k *GPUKernel) Name() string { return k.name }

func (e *GPUEngine) CreateKernelFromSource(source, kernelName string) (*GPUKernel, error) {
	// Create program
	cSource := syscall.StringBytePtr(source)
	proc, _ := openclDLL.FindProc("clCreateProgramWithSource")
	sLen := uint64(len(source))
	var errCode int32
	prog, _, _ := proc.Call(e.context, 1, uintptr(unsafe.Pointer(&cSource)), uintptr(unsafe.Pointer(&sLen)), uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		return nil, fmt.Errorf("program create failed (err=%d)", errCode)
	}

	// Build program
	proc, _ = openclDLL.FindProc("clBuildProgram")
	ret, _, _ := proc.Call(prog, 1, uintptr(unsafe.Pointer(&e.device)), 0, 0, 0)
	if ret != 0 {
		var log [4096]byte
		proc2, _ := openclDLL.FindProc("clGetProgramBuildInfo")
		const CL_PROGRAM_BUILD_LOG = 0x1183
		var logSize uint64
		proc2.Call(prog, e.device, CL_PROGRAM_BUILD_LOG, 4096, uintptr(unsafe.Pointer(&log[0])), uintptr(unsafe.Pointer(&logSize)))
		openclDLL.MustFindProc("clReleaseProgram").Call(prog)
		return nil, fmt.Errorf("build failed: %s", string(log[:logSize]))
	}

	// Create kernel
	cName := syscall.StringBytePtr(kernelName)
	proc, _ = openclDLL.FindProc("clCreateKernel")
	kern, _, _ := proc.Call(prog, uintptr(unsafe.Pointer(cName)), uintptr(unsafe.Pointer(&errCode)))
	if errCode != 0 {
		openclDLL.MustFindProc("clReleaseProgram").Call(prog)
		return nil, fmt.Errorf("kernel '%s' create failed (err=%d)", kernelName, errCode)
	}

	return &GPUKernel{kernel: kern, program: prog, name: kernelName}, nil
}

func (e *GPUEngine) SetKernelArg(k *GPUKernel, index uint32, value interface{}) error {
	proc, _ := openclDLL.FindProc("clSetKernelArg")
	switch v := value.(type) {
	case *GPUBuffer:
		mem := v.mem
		ret, _, _ := proc.Call(k.kernel, uintptr(index), unsafe.Sizeof(mem), uintptr(unsafe.Pointer(&mem)))
		if ret != 0 { return fmt.Errorf("set arg %d (buffer) failed", index) }
	case []byte:
		ret, _, _ := proc.Call(k.kernel, uintptr(index), uintptr(len(v)), uintptr(unsafe.Pointer(&v[0])))
		if ret != 0 { return fmt.Errorf("set arg %d (scalar) failed", index) }
	case int32:
		ret, _, _ := proc.Call(k.kernel, uintptr(index), 4, uintptr(unsafe.Pointer(&v)))
		if ret != 0 { return fmt.Errorf("set arg %d (int32) failed", index) }
	case float32:
		ret, _, _ := proc.Call(k.kernel, uintptr(index), 4, uintptr(unsafe.Pointer(&v)))
		if ret != 0 { return fmt.Errorf("set arg %d (float32) failed", index) }
	}
	return nil
}

func (e *GPUEngine) ExecuteNDRange(k *GPUKernel, workDim uint32,
	globalSize, globalOffset, localSize []uint64,
	outputBuffers []*GPUBuffer) error {

	proc, _ := openclDLL.FindProc("clEnqueueNDRangeKernel")
	var gOffset unsafe.Pointer
	if len(globalOffset) > 0 && globalOffset[0] > 0 {
		gOffset = unsafe.Pointer(&globalOffset[0])
	}
	var lSize unsafe.Pointer
	if len(localSize) > 0 {
		lSize = unsafe.Pointer(&localSize[0])
	}
	ret, _, _ := proc.Call(e.queue, k.kernel, uintptr(workDim),
		uintptr(gOffset), uintptr(unsafe.Pointer(&globalSize[0])),
		uintptr(lSize), 0, 0, 0)
	if ret != 0 {
		return fmt.Errorf("NDRange failed for '%s' (err=%d)", k.name, ret)
	}

	// Read back output buffers
	for _, buf := range outputBuffers {
		if buf != nil && buf.mem != 0 {
			bf := make([]byte, buf.size)
			proc2, _ := openclDLL.FindProc("clEnqueueReadBuffer")
			const CL_TRUE = 1
			ret, _, _ = proc2.Call(e.queue, buf.mem, CL_TRUE, uintptr(0),
				uintptr(buf.size), uintptr(unsafe.Pointer(&bf[0])), 0, 0, 0)
			_ = ret // result in bf
		}
	}

	return nil
}

func (e *GPUEngine) ReadBuffer(b *GPUBuffer, offset uint64, size uint64) ([]byte, error) {
	out := make([]byte, size)
	proc, _ := openclDLL.FindProc("clEnqueueReadBuffer")
	const CL_TRUE = 1
	ret, _, _ := proc.Call(e.queue, b.mem, CL_TRUE, uintptr(offset), uintptr(size), uintptr(unsafe.Pointer(&out[0])), 0, 0, 0)
	if ret != 0 { return nil, fmt.Errorf("OCL read failed (err=%d)", ret) }
	return out, nil
}

func (e *GPUEngine) WriteBuffer(b *GPUBuffer, offset uint64, data []byte) error {
	proc, _ := openclDLL.FindProc("clEnqueueWriteBuffer")
	const CL_TRUE = 1
	ret, _, _ := proc.Call(e.queue, b.mem, CL_TRUE, uintptr(offset), uintptr(len(data)), uintptr(unsafe.Pointer(&data[0])), 0, 0, 0)
	if ret != 0 { return fmt.Errorf("OCL write failed (err=%d)", ret) }
	return nil
}

func (e *GPUEngine) Finish() error {
	proc, _ := openclDLL.FindProc("clFinish")
	proc.Call(e.queue)
	return nil
}

func (e *GPUEngine) ReleaseKernel(k *GPUKernel) {
	if k.kernel != 0 {
		openclDLL.MustFindProc("clReleaseKernel").Call(k.kernel)
	}
	if k.program != 0 {
		openclDLL.MustFindProc("clReleaseProgram").Call(k.program)
	}
}
