/*
 * cmd/worker/engine/fallback.go — Pure-Go CPU engine fallback
 *
 * Used when CGO is disabled (no C compiler available).
 * Implements basic compute operations in pure Go.
 * Much slower than the C engine but allows the worker to function.
 *
 * Build tag: !cgo_enabled (default when no C compiler)
 */

package engine

import (
	"fmt"
	"math"
	"sync"
	"time"
	"unsafe"
)

// ── GoEngine — pure-Go compute engine ─────────────────

type GoEngine struct {
	mu       sync.RWMutex
	buffers  map[string]*GoBuffer
	programs map[string]*GoProgram
}

type GoBuffer struct {
	id    string
	size  uint64
	flags uint32
	data  []byte
}

type GoProgram struct {
	id      string
	source  string
	built   bool
	binaries map[string][]byte // kernel_name → binary (Go function reference)
}

type GoKernel struct {
	NameVal string
	program *GoProgram
	args    []GoKernelArg
}

type GoKernelArg struct {
	isBuffer bool
	bufferID string
	scalar   []byte
	size     uint64
}

func NewGoEngine() *GoEngine {
	return &GoEngine{
		buffers:  make(map[string]*GoBuffer),
		programs: make(map[string]*GoProgram),
	}
}

func (e *GoEngine) BackendName() string { return "Go-CPU" }

func (e *GoEngine) DeviceInfo() string {
	return `{"vendor":"DistriBox","model":"Go CPU Engine","vram_mb":0,"type":"CPU"}`
}

func (e *GoEngine) Close() {}

// ── Buffer operations ─────────────────────────────────

func (e *GoEngine) CreateBuffer(size uint64, flags uint32, data []byte) (*GoBuffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	buf := &GoBuffer{
		id:    fmt.Sprintf("buf-%d", len(e.buffers)+1),
		size:  size,
		flags: flags,
		data:  make([]byte, size),
	}
	if len(data) > 0 {
		copy(buf.data, data)
	}
	e.buffers[buf.id] = buf
	return buf, nil
}

func (e *GoEngine) WriteBuffer(b *GoBuffer, offset uint64, data []byte) error {
	if uint64(len(data))+offset > b.size {
		return fmt.Errorf("write out of bounds")
	}
	copy(b.data[offset:], data)
	return nil
}

func (e *GoEngine) ReadBuffer(b *GoBuffer, offset uint64, size uint64) ([]byte, error) {
	if offset+size > b.size {
		return nil, fmt.Errorf("read out of bounds")
	}
	result := make([]byte, size)
	copy(result, b.data[offset:offset+size])
	return result, nil
}

func (e *GoEngine) ReleaseBuffer(b *GoBuffer) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.buffers, b.id)
}

func (e *GoEngine) BufferSize(b *GoBuffer) uint64 {
	return b.size
}

// ── Program operations ────────────────────────────────

func (e *GoEngine) CreateProgramFromSource(source string, options string) (*GoProgram, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	prog := &GoProgram{
		id:     fmt.Sprintf("prog-%d", len(e.programs)+1),
		source: source,
		built:  true, // Go "compilation" is instant — we just remember the source
	}
	e.programs[prog.id] = prog
	return prog, nil
}

func (e *GoEngine) BuildProgram(p *GoProgram, options string) (string, error) {
	p.built = true
	return "Go engine: kernel source stored (no JIT compilation needed)", nil
}

func (e *GoEngine) ReleaseProgram(p *GoProgram) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.programs, p.id)
}

// ── Kernel operations ─────────────────────────────────

func (e *GoEngine) CreateKernel(p *GoProgram, name string) (*GoKernel, error) {
	if !p.built {
		return nil, fmt.Errorf("program not built")
	}
	return &GoKernel{
		NameVal: name,
		program: p,
		args:    make([]GoKernelArg, 0, 8),
	}, nil
}

func (e *GoEngine) SetKernelArg(k *GoKernel, index uint32, value interface{}) error {
	// Expand args slice if needed
	for uint32(len(k.args)) <= index {
		k.args = append(k.args, GoKernelArg{})
	}

	switch v := value.(type) {
	case *GoBuffer:
		k.args[index] = GoKernelArg{
			isBuffer: true,
			bufferID: v.id,
		}
	case int32:
		k.args[index] = GoKernelArg{
			isBuffer: false,
			scalar:   int32ToBytes(v),
			size:     4,
		}
	case float32:
		k.args[index] = GoKernelArg{
			isBuffer: false,
			scalar:   float32ToBytes(v),
			size:     4,
		}
	case []byte:
		k.args[index] = GoKernelArg{
			isBuffer: false,
			scalar:   v,
			size:     uint64(len(v)),
		}
	default:
		return fmt.Errorf("unsupported arg type for arg %d: %T", index, value)
	}
	return nil
}

func (e *GoEngine) ReleaseKernel(k *GoKernel) {
	k.args = nil
}

// ── Execute NDRange (pure Go implementation) ──────────

func (e *GoEngine) ExecuteNDRange(kernel *GoKernel, workDim uint32,
	globalSize, globalOffset, localSize []uint64,
	outputBuffers []*GoBuffer) error {

	// Pure Go: execute the kernel function by name
	// For MVP, we recognize common kernel patterns and execute them in Go

	switch kernel.NameVal {
	case "vector_add":
		return e.executeVectorAdd(kernel, globalSize, globalOffset)
	case "matmul", "matrix_mul", "gemm":
		return e.executeMatMul(kernel, globalSize, globalOffset)
	case "softmax":
		return e.executeSoftmax(kernel, globalSize, globalOffset)
	case "relu":
		return e.executeReLU(kernel, globalSize, globalOffset)
	case "gelu":
		return e.executeGELU(kernel, globalSize, globalOffset)
	case "scalar_mul", "scalar_multiply":
		return e.executeScalarMul(kernel, globalSize, globalOffset)
	case "element_wise_mul", "hadamard":
		return e.executeElementWiseMul(kernel, globalSize, globalOffset)
	case "transpose":
		return e.executeTranspose(kernel, globalSize, globalOffset)
	case "reduce_sum":
		return e.executeReduceSum(kernel, globalSize, globalOffset)
	case "sigmoid":
		return e.executeSigmoid(kernel, globalSize, globalOffset)
	case "layer_norm":
		return e.executeLayerNorm(kernel, globalSize, globalOffset)
	case "rms_norm":
		return e.executeRMSNorm(kernel, globalSize, globalOffset)
	case "rope":
		return e.executeRoPE(kernel, globalSize, globalOffset)
	case "add_bias":
		return e.executeAddBias(kernel, globalSize, globalOffset)
	default:
		return fmt.Errorf("kernel '%s' not implemented in Go fallback", kernel.NameVal)
	}
}

// vector_add: C[i] = A[i] + B[i]
func (e *GoEngine) executeVectorAdd(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 3 {
		return fmt.Errorf("vector_add needs at least 3 args")
	}

	bufA := e.buffers[k.args[0].bufferID]
	bufB := e.buffers[k.args[1].bufferID]
	bufC := e.buffers[k.args[2].bufferID]

	n := int(globalSize[0])
	if len(k.args) >= 4 && !k.args[3].isBuffer {
		n = int(bytesToInt32(k.args[3].scalar))
	}

	if bufA == nil || bufB == nil || bufC == nil {
		return fmt.Errorf("buffer not found")
	}

	a := bytesAsFloat32Slice(bufA.data)
	b := bytesAsFloat32Slice(bufB.data)
	c := bytesAsFloat32Slice(bufC.data)

	offset := uint64(0)
	if len(globalOffset) > 0 {
		offset = globalOffset[0]
	}

	for i := 0; i < n && i < len(a) && i < len(b) && i < len(c); i++ {
		c[offset+uint64(i)] = a[offset+uint64(i)] + b[offset+uint64(i)]
	}

	return nil
}

// matrix_mul: C[M,N] = A[M,K] * B[K,N]
// NDRange global = [M, N] (2D) or [M*N] (1D flattened)
// K is either passed as arg[3] (int32 scalar) or inferred from buffer sizes
func (e *GoEngine) executeMatMul(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 3 {
		return fmt.Errorf("matmul needs at least 3 args (A, B, C)")
	}

	bufA := e.buffers[k.args[0].bufferID]
	bufB := e.buffers[k.args[1].bufferID]
	bufC := e.buffers[k.args[2].bufferID]
	if bufA == nil || bufB == nil || bufC == nil {
		return fmt.Errorf("buffer not found")
	}

	// Determine M, N from NDRange
	M := int(globalSize[0])
	N := int(globalSize[1])
	if N == 0 {
		N = 1
	}

	// Determine K: either from arg[3] or inferred from buffer sizes
	K := 0
	if len(k.args) >= 4 && !k.args[3].isBuffer && len(k.args[3].scalar) >= 4 {
		K = int(bytesToInt32(k.args[3].scalar))
	}
	if K == 0 {
		// Infer K from A's size: A has M*K elements
		elemSize := uint64(4) // float32
		totalElems := bufA.size / elemSize
		if M > 0 {
			K = int(totalElems / uint64(M))
		}
	}
	if K == 0 {
		// Infer from B: B has K*N elements
		totalElems := bufB.size / 4
		if N > 0 {
			K = int(totalElems / uint64(N))
		}
	}
	if K == 0 {
		return fmt.Errorf("cannot determine K dimension for matmul (M=%d, N=%d, sizeA=%d, sizeB=%d)",
			M, N, bufA.size, bufB.size)
	}

	a := bytesAsFloat32Slice(bufA.data)
	b := bytesAsFloat32Slice(bufB.data)
	c := bytesAsFloat32Slice(bufC.data)

	// Check bounds
	if len(c) < M*N {
		return fmt.Errorf("output buffer too small: need %d, have %d", M*N, len(c))
	}
	if len(a) < M*K {
		return fmt.Errorf("A buffer too small: need %d, have %d", M*K, len(a))
	}
	if len(b) < K*N {
		return fmt.Errorf("B buffer too small: need %d, have %d", K*N, len(b))
	}

	// Compute offset for this worker's portion
	rowOffset := 0
	colOffset := 0
	if len(globalOffset) > 0 {
		rowOffset = int(globalOffset[0])
	}
	if len(globalOffset) > 1 {
		colOffset = int(globalOffset[1])
	}

	// C[row][col] = sum_k A[row][k] * B[k][col]
	for row := 0; row < M; row++ {
		for col := 0; col < N; col++ {
			var sum float32
			for kk := 0; kk < K; kk++ {
				sum += a[(rowOffset+row)*K+kk] * b[kk*N+(colOffset+col)]
			}
			c[(rowOffset+row)*N+(colOffset+col)] = sum
		}
	}

	return nil
}

// softmax: output[i] = exp(input[i]) / sum(exp(input[j]))
func (e *GoEngine) executeSoftmax(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("softmax needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufIn.size / 4)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	if len(in) < n || len(out) < n {
		return fmt.Errorf("buffer too small: need %d elements", n)
	}

	// Find max for numerical stability
	maxVal := float32(math.Inf(-1))
	for i := 0; i < n; i++ {
		if in[i] > maxVal {
			maxVal = in[i]
		}
	}

	// Compute exp and sum
	var sum float64
	for i := 0; i < n; i++ {
		out[i] = float32(math.Exp(float64(in[i] - maxVal)))
		sum += float64(out[i])
	}

	// Normalize
	for i := 0; i < n; i++ {
		out[i] = float32(float64(out[i]) / sum)
	}

	return nil
}

// relu: output[i] = max(0, input[i])
func (e *GoEngine) executeReLU(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("relu needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufIn.size / 4)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	count := n
	if count > len(in) {
		count = len(in)
	}
	if count > len(out) {
		count = len(out)
	}

	for i := 0; i < count; i++ {
		if in[i] > 0 {
			out[i] = in[i]
		} else {
			out[i] = 0
		}
	}

	return nil
}

// scalar_mul: output[i] = input[i] * scalar
func (e *GoEngine) executeScalarMul(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 3 {
		return fmt.Errorf("scalar_mul needs at least 3 args (input, output, scalar)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	scalar := float32(1.0)
	if !k.args[2].isBuffer && len(k.args[2].scalar) >= 4 {
		scalar = math.Float32frombits(uint32(bytesToInt32(k.args[2].scalar)))
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufIn.size / 4)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	count := n
	if count > len(in) {
		count = len(in)
	}
	if count > len(out) {
		count = len(out)
	}

	for i := 0; i < count; i++ {
		out[i] = in[i] * scalar
	}

	return nil
}

// element_wise_mul (Hadamard product): output[i] = A[i] * B[i]
func (e *GoEngine) executeElementWiseMul(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 3 {
		return fmt.Errorf("element_wise_mul needs at least 3 args (A, B, output)")
	}

	bufA := e.buffers[k.args[0].bufferID]
	bufB := e.buffers[k.args[1].bufferID]
	bufOut := e.buffers[k.args[2].bufferID]
	if bufA == nil || bufB == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufA.size / 4)
	}

	a := bytesAsFloat32Slice(bufA.data)
	b := bytesAsFloat32Slice(bufB.data)
	out := bytesAsFloat32Slice(bufOut.data)

	count := n
	if count > len(a) {
		count = len(a)
	}
	if count > len(b) {
		count = len(b)
	}
	if count > len(out) {
		count = len(out)
	}

	for i := 0; i < count; i++ {
		out[i] = a[i] * b[i]
	}

	return nil
}

// transpose: output[j][i] = input[i][j] for 2D matrix
// NDRange global = [rows, cols] or [rows*cols] for flattened
func (e *GoEngine) executeTranspose(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("transpose needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	// Determine dimensions
	rows := int(globalSize[0])
	cols := int(globalSize[1])
	if cols == 0 {
		// 1D: try to infer from arg[2] (columns scalar)
		if len(k.args) >= 3 && !k.args[2].isBuffer && len(k.args[2].scalar) >= 4 {
			cols = int(bytesToInt32(k.args[2].scalar))
		}
	}
	if cols == 0 || rows == 0 {
		n := int(bufIn.size / 4)
		// Assume square or reasonable dimensions
		rows = n
		cols = 1
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			srcIdx := i*cols + j
			dstIdx := j*rows + i
			if srcIdx < len(in) && dstIdx < len(out) {
				out[dstIdx] = in[srcIdx]
			}
		}
	}

	return nil
}

// reduce_sum: reduces input along a dimension
// For 1D: output[0] = sum(input[0..n-1])
// NDRange global = [n] for 1D input
func (e *GoEngine) executeReduceSum(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("reduce_sum needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufIn.size / 4)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	var sum float32
	for i := 0; i < n && i < len(in); i++ {
		sum += in[i]
	}

	if len(out) > 0 {
		out[0] = sum
	}

	return nil
}

// sigmoid: output[i] = 1 / (1 + exp(-input[i]))
func (e *GoEngine) executeSigmoid(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("sigmoid needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufIn.size / 4)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	count := n
	if count > len(in) {
		count = len(in)
	}
	if count > len(out) {
		count = len(out)
	}

	for i := 0; i < count; i++ {
		// Clamp to avoid overflow in exp
		x := float64(in[i])
		if x > 20 {
			out[i] = 1.0
		} else if x < -20 {
			out[i] = 0.0
		} else {
			out[i] = float32(1.0 / (1.0 + math.Exp(-x)))
		}
	}

	return nil
}

// gelu: output[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Uses the tanh approximation (same as PyTorch/Fairseq).
func (e *GoEngine) executeGELU(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("gelu needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	n := int(globalSize[0])
	if n == 0 {
		n = int(bufIn.size / 4)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	count := n
	if count > len(in) {
		count = len(in)
	}
	if count > len(out) {
		count = len(out)
	}

	// GELU constants for tanh approximation
	// sqrt(2/pi) ≈ 0.7978845608
	// 0.044715
	const sqrt2OverPi = 0.7978845608028654
	const coeff = 0.044715

	for i := 0; i < count; i++ {
		x := float64(in[i])
		x3 := x * x * x
		inner := sqrt2OverPi * (x + coeff*x3)
		out[i] = float32(0.5 * x * (1.0 + math.Tanh(inner)))
	}

	return nil
}

// layer_norm: output[i*D+j] = (input[i*D+j] - mean_i) / sqrt(var_i + eps) * gamma[j] + beta[j]
// Operates on 2D input [rows, D]. D is passed as kernel arg[4] (int32).
// gamma (arg[2]) and beta (arg[3]) are optional weight/bias vectors of size D.
func (e *GoEngine) executeLayerNorm(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("layer_norm needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	// Determine dimensions from NDRange
	rows := int(globalSize[0])
	dim := int(globalSize[1])
	if dim == 0 {
		// 1D NDRange: dim passed as arg[4]
		if len(k.args) >= 5 && !k.args[4].isBuffer && len(k.args[4].scalar) >= 4 {
			dim = int(bytesToInt32(k.args[4].scalar))
		}
	}
	if rows == 0 {
		totalElems := int(bufIn.size / 4)
		if dim > 0 {
			rows = totalElems / dim
		} else {
			rows = totalElems
			dim = 1
		}
	}

	// Optional gamma/beta
	gamma := float32(1.0)
	beta := float32(0.0)
	var gammaBuf, betaBuf []float32
	if len(k.args) >= 3 && k.args[2].isBuffer {
		if buf := e.buffers[k.args[2].bufferID]; buf != nil {
			gammaBuf = bytesAsFloat32Slice(buf.data)
		}
	}
	if len(k.args) >= 4 && k.args[3].isBuffer {
		if buf := e.buffers[k.args[3].bufferID]; buf != nil {
			betaBuf = bytesAsFloat32Slice(buf.data)
		}
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	const eps = 1e-5

	for row := 0; row < rows; row++ {
		base := row * dim

		// Compute mean
		var sum float64
		for j := 0; j < dim; j++ {
			if base+j < len(in) {
				sum += float64(in[base+j])
			}
		}
		mean := float32(sum / float64(dim))

		// Compute variance
		var varSum float64
		for j := 0; j < dim; j++ {
			if base+j < len(in) {
				diff := float64(in[base+j] - mean)
				varSum += diff * diff
			}
		}
		invStd := float32(1.0 / math.Sqrt(varSum/float64(dim)+eps))

		// Normalize and apply affine
		for j := 0; j < dim; j++ {
			idx := base + j
			if idx >= len(out) {
				break
			}
			if j < len(gammaBuf) {
				gamma = gammaBuf[j]
			}
			if j < len(betaBuf) {
				beta = betaBuf[j]
			}
			_ = gamma
			_ = beta
			// Clamp beta to 0 when no betaBuf provided
			var actualBeta float32
			if betaBuf == nil {
				actualBeta = 0
			} else if j < len(betaBuf) {
				actualBeta = betaBuf[j]
			}
			var actualGamma float32 = 1.0
			if gammaBuf != nil && j < len(gammaBuf) {
				actualGamma = gammaBuf[j]
			}

			if idx < len(in) {
				out[idx] = (in[idx]-mean)*invStd*actualGamma + actualBeta
			}
		}
	}

	return nil
}

// rms_norm: output[i] = input[i] / sqrt(mean(x^2) + eps) * gamma[i]
// Used in LLaMA, Mistral, Qwen transformer architectures.
// gamma is an optional weight vector (arg[2]).
func (e *GoEngine) executeRMSNorm(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("rms_norm needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	// Determine dimensions
	rows := int(globalSize[0])
	dim := int(globalSize[1])
	if dim == 0 {
		if len(k.args) >= 4 && !k.args[3].isBuffer && len(k.args[3].scalar) >= 4 {
			dim = int(bytesToInt32(k.args[3].scalar))
		}
	}
	if rows == 0 {
		totalElems := int(bufIn.size / 4)
		if dim > 0 {
			rows = totalElems / dim
		} else {
			rows = totalElems
			dim = 1
		}
	}

	var gammaBuf []float32
	if len(k.args) >= 3 && k.args[2].isBuffer {
		if buf := e.buffers[k.args[2].bufferID]; buf != nil {
			gammaBuf = bytesAsFloat32Slice(buf.data)
		}
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	const eps = 1e-6

	for row := 0; row < rows; row++ {
		base := row * dim

		// Compute RMS (root mean square)
		var sumSq float64
		for j := 0; j < dim; j++ {
			if base+j < len(in) {
				v := float64(in[base+j])
				sumSq += v * v
			}
		}
		rms := float32(math.Sqrt(sumSq/float64(dim) + eps))

		// Normalize and scale
		for j := 0; j < dim; j++ {
			idx := base + j
			if idx >= len(out) {
				break
			}
			scale := float32(1.0)
			if j < len(gammaBuf) {
				scale = gammaBuf[j]
			}
			if idx < len(in) {
				out[idx] = (in[idx] / rms) * scale
			}
		}
	}

	return nil
}

// rope: Rotary Position Embedding for half of the head dimension.
// Input is [rows, dim] where dim is head_dim. The second half of each row
// is rotated by position-dependent angles.
// Args: [0]=input, [1]=output, [2]=position (int32), [3]=head_dim (int32)
// Theta base config is hardcoded to 10000.0 (standard for LLaMA/Mistral).
func (e *GoEngine) executeRoPE(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 2 {
		return fmt.Errorf("rope needs at least 2 args (input, output)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	// Position from arg[2], default 0
	position := int32(0)
	if len(k.args) >= 3 && !k.args[2].isBuffer && len(k.args[2].scalar) >= 4 {
		position = bytesToInt32(k.args[2].scalar)
	}

	// Head dim from arg[3] or NDRange
	headDim := int(globalSize[1])
	if headDim == 0 {
		if len(k.args) >= 4 && !k.args[3].isBuffer && len(k.args[3].scalar) >= 4 {
			headDim = int(bytesToInt32(k.args[3].scalar))
		}
	}
	if headDim == 0 {
		headDim = int(bufIn.size / 4) // single row
	}

	rows := int(globalSize[0])
	if rows == 0 {
		rows = int(bufIn.size / 4 / uint64(headDim))
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	const thetaBase = 10000.0
	half := headDim / 2
	pos := float64(position)

	for row := 0; row < rows; row++ {
		base := row * headDim
		for i := 0; i < half; i++ {
			// theta_i = pos / thetaBase^(2*i/headDim)
			theta := pos / math.Pow(thetaBase, 2.0*float64(i)/float64(headDim))
			cos, sin := math.Cos(theta), math.Sin(theta)

			idx0 := base + i
			idx1 := base + i + half

			if idx0 < len(in) && idx1 < len(in) {
				x0, x1 := in[idx0], in[idx1]
				out[idx0] = float32(float64(x0)*cos - float64(x1)*sin)
				out[idx1] = float32(float64(x0)*sin + float64(x1)*cos)
			}
		}
	}

	return nil
}

// add_bias: output[row * D + j] = input[row * D + j] + bias[j]
// Adds a bias vector to each row of a 2D matrix.
// Args: [0]=input, [1]=output, [2]=bias (vector of size D), [3]=D (int32, optional)
func (e *GoEngine) executeAddBias(k *GoKernel, globalSize, globalOffset []uint64) error {
	if len(k.args) < 3 {
		return fmt.Errorf("add_bias needs at least 3 args (input, output, bias)")
	}

	bufIn := e.buffers[k.args[0].bufferID]
	bufOut := e.buffers[k.args[1].bufferID]
	if bufIn == nil || bufOut == nil {
		return fmt.Errorf("buffer not found")
	}

	var biasBuf []float32
	if k.args[2].isBuffer {
		if buf := e.buffers[k.args[2].bufferID]; buf != nil {
			biasBuf = bytesAsFloat32Slice(buf.data)
		}
	}

	if biasBuf == nil {
		return fmt.Errorf("bias buffer not found")
	}

	// Determine dimensions
	rows := int(globalSize[0])
	dim := len(biasBuf)
	if rows == 0 {
		totalElems := int(bufIn.size / 4)
		if dim > 0 {
			rows = totalElems / dim
		} else {
			rows = totalElems
			dim = 1
		}
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)

	for row := 0; row < rows; row++ {
		base := row * dim
		for j := 0; j < dim; j++ {
			idx := base + j
			if idx >= len(in) || idx >= len(out) {
				break
			}
			out[idx] = in[idx] + biasBuf[j]
		}
	}

	return nil
}

func (e *GoEngine) Finish() error { return nil }

func (e *GoEngine) RunMicroBenchmark() float64 {
	// Real matmul micro-benchmark with wall-clock timing.
	// Result is stored to prevent compiler optimization.
	const size = 256
	a := make([]float32, size*size)
	b := make([]float32, size*size)
	c := make([]float32, size*size)

	// Initialize with random-ish data
	for i := 0; i < size*size; i++ {
		a[i] = float32(i%100) * 0.01
		b[i] = float32((i*7)%100) * 0.01
	}

	// Warm up (1 iteration)
	for i := 0; i < size; i++ {
		for k := 0; k < size; k++ {
			aik := a[i*size+k]
			for j := 0; j < size; j++ {
				c[i*size+j] += aik * b[k*size+j]
			}
		}
	}

	// Measure
	iters := 5
	start := time.Now()
	for iter := 0; iter < iters; iter++ {
		for i := 0; i < size; i++ {
			for k := 0; k < size; k++ {
				aik := a[i*size+k]
				for j := 0; j < size; j++ {
					c[i*size+j] += aik * b[k*size+j]
				}
			}
		}
	}
	elapsed := time.Since(start).Seconds()

	// 2*M*N*K operations per matmul
	totalOps := int64(2) * int64(size) * int64(size) * int64(size) * int64(iters)
	gflops := float64(totalOps) / elapsed / 1e9

	// Prevent dead-code elimination
	if c[0] < 0 {
		return 0
	}

	return math.Round(gflops*100) / 100
}

// ── Helpers ──────────────────────────────────────────

func int32ToBytes(v int32) []byte {
	b := make([]byte, 4)
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	return b
}

func float32ToBytes(v float32) []byte {
	bits := math.Float32bits(v)
	b := make([]byte, 4)
	b[0] = byte(bits)
	b[1] = byte(bits >> 8)
	b[2] = byte(bits >> 16)
	b[3] = byte(bits >> 24)
	return b
}

func bytesToInt32(b []byte) int32 {
	if len(b) < 4 {
		return 0
	}
	return int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16 | int32(b[3])<<24
}

// bytesAsFloat32Slice returns a mutable float32 view into the underlying byte slice.
// Modifications to the returned slice directly update the buffer data.
func bytesAsFloat32Slice(data []byte) []float32 {
	if len(data) == 0 {
		return nil
	}
	return unsafe.Slice((*float32)(unsafe.Pointer(&data[0])), len(data)/4)
}
