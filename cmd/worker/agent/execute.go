/*
 * cmd/worker/agent/execute.go — Task execution using compute engine
 *
 * Updated to use the common Engine interface (CGO or pure-Go).
 */

package agent

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/distribox/cmd/worker/engine"
)

// ── Task execution types ──────────────────────────────

type TaskRequest struct {
	TaskID    string
	QueueID   string
	KernelID  string
	KernelName string

	// NDRange specification (this worker's portion)
	WorkDim      uint32
	GlobalSize   []uint64
	GlobalOffset []uint64
	LocalSize    []uint64

	// Kernel arguments
	Args []KernelArg

	// Buffer data (may be included inline or already on worker)
	InputBuffers  map[string][]byte
	OutputBufferIDs []string

	// Event dependencies
	WaitEventIDs []string
}

type KernelArg struct {
	Index    uint32
	IsBuffer bool
	BufferID string
	Scalar   []byte
	Size     uint64
}

type TaskResult struct {
	TaskID   string
	Success  bool
	ErrorMsg string
	OutputBuffers map[string][]byte
	StartTimeNs   int64
	EndTimeNs     int64
}

// ── Task executor ─────────────────────────────────────

type TaskExecutor struct {
	eng      *engine.GoEngine
	gpuEng   *engine.GPUEngine
	hasGPU   bool
	worker   *Worker
	buffers  map[string]*engine.GoBuffer
	gpuBufs  map[string]*engine.GPUBuffer
	mu       sync.Mutex
}

func NewTaskExecutor(w *Worker) *TaskExecutor {
	e := &TaskExecutor{
		eng:     engine.NewGoEngine(),
		buffers: make(map[string]*engine.GoBuffer),
		gpuBufs: make(map[string]*engine.GPUBuffer),
		worker:  w,
	}

	gpuEng, err := engine.NewGPUEngine()
	if err == nil {
		e.gpuEng = gpuEng
		e.hasGPU = true
		log.Printf("🚀 REAL GPU: %s", gpuEng.BackendName())
	} else {
		log.Printf("Compute engine: %s (GPU: %v)", e.eng.BackendName(), err)
	}

	return e
}

func (e *TaskExecutor) Close() {
	e.mu.Lock()
	defer e.mu.Unlock()
	for _, buf := range e.buffers {
		e.eng.ReleaseBuffer(buf)
	}
	e.eng.Close()
}

// Execute runs a compute task using the engine
func (e *TaskExecutor) Execute(ctx context.Context, req *TaskRequest) (*TaskResult, error) {
	log.Printf("Task %s: kernel=%s global=%v offset=%v",
		req.TaskID, req.KernelName, req.GlobalSize, req.GlobalOffset)

	startTime := time.Now()

	// ── Step 1: Ensure input buffers are on-device ─────
	e.mu.Lock()
	for bufID, data := range req.InputBuffers {
		if _, exists := e.buffers[bufID]; !exists {
			buf, err := e.eng.CreateBuffer(uint64(len(data)), 0, data)
			if err != nil {
				e.mu.Unlock()
				return e.fail(req.TaskID, fmt.Sprintf("buffer %s: %v", bufID, err)), err
			}
			e.buffers[bufID] = buf
		}
	}

	// ── Step 2: Create output buffers ──────────────────
	var outputBufs []*engine.GoBuffer
	for _, bufID := range req.OutputBufferIDs {
		buf, exists := e.buffers[bufID]
		if !exists {
			buf, _ = e.eng.CreateBuffer(1024*1024, 0, nil)
			e.buffers[bufID] = buf
		}
		outputBufs = append(outputBufs, buf)
	}
	e.mu.Unlock()

	// ── Step 3: Build kernel with args ─────────────────
	kernel := &engine.GoKernel{NameVal: req.KernelName}
	for _, arg := range req.Args {
		var val interface{}
		if arg.IsBuffer {
			if buf, ok := e.buffers[arg.BufferID]; ok {
				val = buf
			} else {
				// Buffer not pre-loaded — create an empty buffer for the kernel arg
				buf, err := e.eng.CreateBuffer(arg.Size, 0, nil)
				if err != nil {
					return e.fail(req.TaskID, fmt.Sprintf("kernel arg %d: create buffer %s: %v", arg.Index, arg.BufferID, err)), err
				}
				e.buffers[arg.BufferID] = buf
				val = buf
			}
		} else {
			val = arg.Scalar
		}
		if err := e.eng.SetKernelArg(kernel, arg.Index, val); err != nil {
			return e.fail(req.TaskID, fmt.Sprintf("set kernel arg %d: %v", arg.Index, err)), err
		}
	}

	// ── Step 4: Execute — GPU if available, else CPU ────
	var err error
	startTime = time.Now()

	if e.hasGPU && e.gpuEng != nil {
		err = e.executeOnGPU(req)
	} else {
		err = e.eng.ExecuteNDRange(kernel, req.WorkDim,
			req.GlobalSize, req.GlobalOffset, req.LocalSize, outputBufs)
	}
	endTime := time.Now()

	if err != nil {
		return e.fail(req.TaskID, err.Error()), err
	}

	// ── Step 5: Read output buffers ────────────────────
	outputs := make(map[string][]byte)
	e.mu.Lock()
	for i, bufID := range req.OutputBufferIDs {
		if i < len(outputBufs) {
			data, rerr := e.eng.ReadBuffer(outputBufs[i], 0, outputBufs[i].Size())
			if rerr == nil {
				outputs[bufID] = data
			}
		}
	}
	e.mu.Unlock()

	return &TaskResult{
		TaskID:        req.TaskID,
		Success:       true,
		OutputBuffers: outputs,
		StartTimeNs:   startTime.UnixNano(),
		EndTimeNs:     endTime.UnixNano(),
	}, nil
}

func (e *TaskExecutor) fail(taskID, msg string) *TaskResult {
	log.Printf("Task %s FAILED: %s", taskID, msg)
	return &TaskResult{TaskID: taskID, Success: false, ErrorMsg: msg}
}

// executeOnGPU runs the task on real OpenCL GPU hardware
func (e *TaskExecutor) executeOnGPU(req *TaskRequest) error {
	// Build OpenCL kernel source for the requested kernel name
	kernelSrc := getKernelSource(req.KernelName)
	if kernelSrc == "" {
		return fmt.Errorf("no OpenCL source for kernel '%s'", req.KernelName)
	}

	gpuKern, err := e.gpuEng.CreateKernelFromSource(kernelSrc, req.KernelName)
	if err != nil {
		return fmt.Errorf("GPU kernel compile failed: %w", err)
	}
	defer e.gpuEng.ReleaseKernel(gpuKern)

	// Create GPU buffers from Go buffers
	gpuBufs := make(map[string]*engine.GPUBuffer)
	e.mu.Lock()
	for bufID, goBuf := range e.buffers {
		data, _ := e.eng.ReadBuffer(goBuf, 0, goBuf.Size())
		gpuBuf, gpuErr := e.gpuEng.CreateBuffer(goBuf.Size(), 0, data)
		if gpuErr == nil {
			gpuBufs[bufID] = gpuBuf
		}
	}

	// Set kernel args
	for _, arg := range req.Args {
		if arg.IsBuffer {
			if gb, ok := gpuBufs[arg.BufferID]; ok {
				e.gpuEng.SetKernelArg(gpuKern, arg.Index, gb)
			}
		} else {
			e.gpuEng.SetKernelArg(gpuKern, arg.Index, arg.Scalar)
		}
	}
	e.mu.Unlock()

	// Build output GPU buffer list
	var gpuOutputs []*engine.GPUBuffer
	for _, bufID := range req.OutputBufferIDs {
		if gb, ok := gpuBufs[bufID]; ok {
			gpuOutputs = append(gpuOutputs, gb)
		}
	}

	// Execute on real GPU!
	err = e.gpuEng.ExecuteNDRange(gpuKern, req.WorkDim,
		req.GlobalSize, req.GlobalOffset, req.LocalSize, gpuOutputs)
	e.gpuEng.Finish()

	// Copy GPU results back to Go buffers
	e.mu.Lock()
	for bufID, gb := range gpuBufs {
		if goBuf, ok := e.buffers[bufID]; ok {
			data, rerr := e.gpuEng.ReadBuffer(gb, 0, gb.Size())
			if rerr == nil {
				e.eng.WriteBuffer(goBuf, 0, data)
			}
		}
		e.gpuEng.ReleaseBuffer(gb)
	}
	e.mu.Unlock()

	return err
}

var kernelSources = map[string]string{
	"vector_add": `__kernel void vector_add(__global const float* a, __global const float* b, __global float* c) {
		int i = get_global_id(0); c[i] = a[i] + b[i]; }`,
	"gelu": `__kernel void gelu(__global const float* in, __global float* out) {
		int i = get_global_id(0); float x = in[i];
		out[i] = 0.5f * x * (1.0f + tanh(0.797884f * (x + 0.044715f * x * x * x))); }`,
	"relu": `__kernel void relu(__global const float* in, __global float* out) {
		int i = get_global_id(0); out[i] = fmax(0.0f, in[i]); }`,
	"matmul": `__kernel void matmul(__global const float* A, __global const float* B, __global float* C, const int N) {
		int row = get_global_id(0); int col = get_global_id(1);
		if (row < N && col < N) { float s = 0; for (int k = 0; k < N; k++) s += A[row*N+k] * B[k*N+col]; C[row*N+col] = s; } }`,
	"sigmoid": `__kernel void sigmoid(__global const float* in, __global float* out) {
		int i = get_global_id(0); out[i] = 1.0f / (1.0f + exp(-in[i])); }`,
	"scalar_mul": `__kernel void scalar_mul(__global const float* in, __global float* out, const float scalar) {
		int i = get_global_id(0); out[i] = in[i] * scalar; }`,
	"element_wise_mul": `__kernel void element_wise_mul(__global const float* a, __global const float* b, __global float* out) {
		int i = get_global_id(0); out[i] = a[i] * b[i]; }`,
	"transpose": `__kernel void transpose(__global const float* in, __global float* out, const int cols) {
		int idx = get_global_id(0); int total = get_global_size(0);
		int col = idx % cols; int row = idx / cols;
		out[col * (total / cols) + row] = in[idx]; }`,
	"reduce_sum": `__kernel void reduce_sum(__global const float* in, __global float* out, const int N) {
		int gid = get_global_id(0);
		if (gid == 0) {
			float sum = 0.0f;
			for (int i = 0; i < N; i++) sum += in[i];
			out[0] = sum;
		}
	}`,
	"add_bias": `__kernel void add_bias(__global const float* in, __global float* out, __global const float* bias, const int cols) {
		int i = get_global_id(0); int row = i / cols; int col = i % cols;
		out[i] = in[i] + bias[col]; }`,
	"rope": `__kernel void rope(__global const float* in, __global float* out, const int pos, const int head_dim) {
		int i = get_global_id(0); int half = head_dim / 2;
		int row = i / head_dim; int j = i % half;
		float theta = 1.0f / pow(10000.0f, (float)(2 * j) / (float)head_dim);
		float cos_theta = cos((float)pos * theta);
		float sin_theta = sin((float)pos * theta);
		float x0 = in[row * head_dim + j];
		float x1 = in[row * head_dim + j + half];
		out[row * head_dim + j] = x0 * cos_theta - x1 * sin_theta;
		out[row * head_dim + j + half] = x0 * sin_theta + x1 * cos_theta;
	}`,
	"layer_norm": `__kernel void layer_norm(__global const float* in, __global float* out,
		__global const float* gamma, __global const float* beta, const int cols) {
		int row = get_global_id(0);
		if (row >= get_global_size(0)) return;
		int base = row * cols;
		// Compute mean
		float mean = 0.0f;
		for (int j = 0; j < cols; j++) mean += in[base + j];
		mean /= (float)cols;
		// Compute variance
		float var = 0.0f;
		for (int j = 0; j < cols; j++) {
			float diff = in[base + j] - mean;
			var += diff * diff;
		}
		var /= (float)cols;
		float inv_std = 1.0f / sqrt(var + 1e-5f);
		// Normalize + scale + shift
		for (int j = 0; j < cols; j++) {
			float x = (in[base + j] - mean) * inv_std;
			float g = gamma ? gamma[j] : 1.0f;
			float b = beta ? beta[j] : 0.0f;
			out[base + j] = x * g + b;
		}
	}`,
	"rms_norm": `__kernel void rms_norm(__global const float* in, __global float* out,
		__global const float* gamma, const int dim) {
		int i = get_global_id(0);
		// Compute RMS over this row
		float sum_sq = 0.0f;
		for (int j = 0; j < dim; j++) {
			float v = in[i * dim + j];
			sum_sq += v * v;
		}
		float rms = sqrt(sum_sq / (float)dim + 1e-6f);
		float inv_rms = 1.0f / rms;
		for (int j = 0; j < dim; j++) {
			float g = gamma ? gamma[j] : 1.0f;
			out[i * dim + j] = in[i * dim + j] * inv_rms * g;
		}
	}`,
	"softmax": `__kernel void softmax(__global const float* in, __global float* out, const int N) {
		int i = get_global_id(0);
		// Numerically stable softmax
		float max_val = -INFINITY;
		for (int j = 0; j < N; j++) {
			float v = in[i * N + j];
			if (v > max_val) max_val = v;
		}
		float sum = 0.0f;
		for (int j = 0; j < N; j++) {
			float e = exp(in[i * N + j] - max_val);
			out[i * N + j] = e;
			sum += e;
		}
		for (int j = 0; j < N; j++) {
			out[i * N + j] /= sum;
		}
	}`,
}

func getKernelSource(name string) string {
	if src, ok := kernelSources[name]; ok { return src }
	return ""
}

