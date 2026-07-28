/*
 * vgpu/server/ipc_server_test.go — Integration tests for IPC + local execution
 *
 * Tests the full local execution path: IPC message → NDRange handler →
 * VRAM buffer read → GoEngine execute → VRAM buffer write-back.
 *
 * Run: go test ./vgpu/server/ -v
 */

package server

import (
	"encoding/json"
	"math"
	"testing"

	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/queue"
	"github.com/distribox/vgpu/scheduler"
)

// ── test helpers ────────────────────────────────────────

func newTestIPCServer(t *testing.T) *IPCServer {
	t.Helper()
	vram := mem.NewVRAMManager()
	cmdQueue := queue.NewCommandQueueManager()
	sched := scheduler.NewScheduler()

	srv, err := NewIPCServer("127.0.0.1:0", vram, cmdQueue, sched)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	return srv
}

func mustMarshalTest(v interface{}) []byte {
	data, _ := json.Marshal(v)
	return data
}

func makeNDRangeMsg(kernelName string, global []uint64, args []map[string]interface{}) []byte {
	rawArgs := make([]json.RawMessage, len(args))
	for i, arg := range args {
		rawArgs[i], _ = json.Marshal(arg)
	}

	msg := IPCMessage{
		Type:       "ndrange",
		MsgID:      "test-msg-1",
		QueueID:    "q1",
		KernelID:   "k1",
		KernelName: kernelName,
		ProgramID:  "p1",
		WorkDim:    uint32(len(global)),
		Global:     global,
		Args:       rawArgs,
	}
	return mustMarshalTest(msg)
}

// ── integration: vector_add via IPC ──────────────────────

func TestIPCLocalVectorAdd(t *testing.T) {
	srv := newTestIPCServer(t)

	// Allocate buffers in VRAM and write data
	n := 8
	const elemSize = 4 // float32

	_, err := srv.vram.Allocate("buf_a", uint64(n*elemSize), 0, mem.BufferReadOnly)
	if err != nil {
		t.Fatalf("Failed to allocate buf_a: %v", err)
	}
	_, err = srv.vram.Allocate("buf_b", uint64(n*elemSize), 0, mem.BufferReadOnly)
	if err != nil {
		t.Fatalf("Failed to allocate buf_b: %v", err)
	}
	_, err = srv.vram.Allocate("buf_c", uint64(n*elemSize), 0, mem.BufferReadWrite)
	if err != nil {
		t.Fatalf("Failed to allocate buf_c: %v", err)
	}

	// Write test data: A=[1,2,3,4,5,6,7,8], B=[10,20,30,40,50,60,70,80]
	dataA := make([]byte, n*elemSize)
	dataB := make([]byte, n*elemSize)
	for i := 0; i < n; i++ {
		writeFloat32(dataA, i, float32(i+1))
		writeFloat32(dataB, i, float32((i+1)*10))
	}
	srv.vram.Write("buf_a", 0, dataA)
	srv.vram.Write("buf_b", 0, dataB)

	// Build NDRange message
	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_a"},
		{"index": 1, "type": "buffer", "id": "buf_b"},
		{"index": 2, "type": "buffer", "id": "buf_c"},
		{"index": 3, "type": "int32", "value": float64(n)}, // JSON numbers are float64
	}
	ndrangeJSON := makeNDRangeMsg("vector_add", []uint64{uint64(n)}, args)

	// Process the message
	respJSON := srv.processMessage(string(ndrangeJSON))

	// Parse response
	var resp map[string]interface{}
	if err := json.Unmarshal([]byte(respJSON), &resp); err != nil {
		t.Fatalf("Failed to parse response: %v", err)
	}
	if resp["success"] != true {
		t.Fatalf("NDRange failed: %v", resp["error"])
	}

	// Verify output in VRAM
	resultData, err := srv.vram.Read("buf_c", 0, uint64(n*elemSize))
	if err != nil {
		t.Fatalf("Failed to read buf_c: %v", err)
	}

	for i := 0; i < n; i++ {
		expected := float32(i+1) + float32((i+1)*10) // A[i] + B[i]
		actual := readFloat32(resultData, i)
		if math.Abs(float64(actual-expected)) > 0.001 {
			t.Errorf("vector_add result[%d]: expected %.0f, got %.1f", i, expected, actual)
		}
	}
}

// ── integration: matmul via IPC ─────────────────────────

func TestIPCLocalMatMul(t *testing.T) {
	srv := newTestIPCServer(t)

	// A = 2x3, B = 3x2 → C = 2x2
	M, K, N := 2, 3, 2
	const elemSize = 4

	_, err := srv.vram.Allocate("buf_a", uint64(M*K*elemSize), 0, mem.BufferReadOnly)
	if err != nil {
		t.Fatalf("Failed to allocate buf_a: %v", err)
	}
	_, err = srv.vram.Allocate("buf_b", uint64(K*N*elemSize), 0, mem.BufferReadOnly)
	if err != nil {
		t.Fatalf("Failed to allocate buf_b: %v", err)
	}
	_, err = srv.vram.Allocate("buf_c", uint64(M*N*elemSize), 0, mem.BufferReadWrite)
	if err != nil {
		t.Fatalf("Failed to allocate buf_c: %v", err)
	}

	// A = [[1, 2, 3],
	//      [4, 5, 6]]
	dataA := make([]byte, M*K*elemSize)
	writeFloat32(dataA, 0, 1); writeFloat32(dataA, 1, 2); writeFloat32(dataA, 2, 3)
	writeFloat32(dataA, 3, 4); writeFloat32(dataA, 4, 5); writeFloat32(dataA, 5, 6)

	// B = [[7, 8],
	//      [9, 10],
	//      [11, 12]]
	dataB := make([]byte, K*N*elemSize)
	writeFloat32(dataB, 0, 7); writeFloat32(dataB, 1, 8)
	writeFloat32(dataB, 2, 9); writeFloat32(dataB, 3, 10)
	writeFloat32(dataB, 4, 11); writeFloat32(dataB, 5, 12)

	srv.vram.Write("buf_a", 0, dataA)
	srv.vram.Write("buf_b", 0, dataB)

	// NDRange with 2D global = [M, N], K passed as arg[3]
	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_a"},
		{"index": 1, "type": "buffer", "id": "buf_b"},
		{"index": 2, "type": "buffer", "id": "buf_c"},
		{"index": 3, "type": "int32", "value": float64(K)},
	}
	ndrangeJSON := makeNDRangeMsg("matmul", []uint64{uint64(M), uint64(N)}, args)

	respJSON := srv.processMessage(string(ndrangeJSON))

	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["success"] != true {
		t.Fatalf("NDRange failed: %v", resp["error"])
	}

	// Verify: C = A × B
	// C[0][0] = 1*7 + 2*9 + 3*11 = 7 + 18 + 33 = 58
	// C[0][1] = 1*8 + 2*10 + 3*12 = 8 + 20 + 36 = 64
	// C[1][0] = 4*7 + 5*9 + 6*11 = 28 + 45 + 66 = 139
	// C[1][1] = 4*8 + 5*10 + 6*12 = 32 + 50 + 72 = 154
	resultData, _ := srv.vram.Read("buf_c", 0, uint64(M*N*elemSize))
	expected := []float32{58, 64, 139, 154}
	for i := 0; i < M*N; i++ {
		actual := readFloat32(resultData, i)
		if math.Abs(float64(actual-expected[i])) > 0.5 {
			t.Errorf("matmul result[%d]: expected %.0f, got %.1f", i, expected[i], actual)
		}
	}
}

// ── integration: softmax via IPC ────────────────────────

func TestIPCLocalSoftmax(t *testing.T) {
	srv := newTestIPCServer(t)

	n := 4
	const elemSize = 4

	srv.vram.Allocate("buf_in", uint64(n*elemSize), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_out", uint64(n*elemSize), 0, mem.BufferReadWrite)

	dataIn := make([]byte, n*elemSize)
	writeFloat32(dataIn, 0, 1)
	writeFloat32(dataIn, 1, 2)
	writeFloat32(dataIn, 2, 3)
	writeFloat32(dataIn, 3, 4)
	srv.vram.Write("buf_in", 0, dataIn)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_in"},
		{"index": 1, "type": "buffer", "id": "buf_out"},
	}
	ndrangeJSON := makeNDRangeMsg("softmax", []uint64{uint64(n)}, args)

	respJSON := srv.processMessage(string(ndrangeJSON))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["success"] != true {
		t.Fatalf("softmax failed: %v", resp["error"])
	}

	resultData, _ := srv.vram.Read("buf_out", 0, uint64(n*elemSize))
	result := bytesAsFloatSlice(resultData)

	// Check sum ≈ 1.0
	var sum float32
	for i := 0; i < n; i++ {
		sum += result[i]
	}
	if math.Abs(float64(sum-1.0)) > 0.01 {
		t.Errorf("softmax sum: expected 1.0, got %.4f", sum)
	}
}

// ── integration: relu via IPC ───────────────────────────

func TestIPCLocalReLU(t *testing.T) {
	srv := newTestIPCServer(t)

	n := 6
	const elemSize = 4

	srv.vram.Allocate("buf_in", uint64(n*elemSize), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_out", uint64(n*elemSize), 0, mem.BufferReadWrite)

	dataIn := make([]byte, n*elemSize)
	writeFloat32(dataIn, 0, -2)
	writeFloat32(dataIn, 1, -1)
	writeFloat32(dataIn, 2, 0)
	writeFloat32(dataIn, 3, 1)
	writeFloat32(dataIn, 4, 2)
	writeFloat32(dataIn, 5, 3)
	srv.vram.Write("buf_in", 0, dataIn)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_in"},
		{"index": 1, "type": "buffer", "id": "buf_out"},
	}
	ndrangeJSON := makeNDRangeMsg("relu", []uint64{uint64(n)}, args)

	respJSON := srv.processMessage(string(ndrangeJSON))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["success"] != true {
		t.Fatalf("relu failed: %v", resp["error"])
	}

	resultData, _ := srv.vram.Read("buf_out", 0, uint64(n*elemSize))
	expected := []float32{0, 0, 0, 1, 2, 3}
	for i := 0; i < n; i++ {
		actual := readFloat32(resultData, i)
		if math.Abs(float64(actual-expected[i])) > 0.001 {
			t.Errorf("relu result[%d]: expected %.0f, got %.1f", i, expected[i], actual)
		}
	}
}

// ── integration: gelu via IPC ───────────────────────────

func TestIPCLocalGELU(t *testing.T) {
	srv := newTestIPCServer(t)

	n := 5
	const elemSize = 4

	srv.vram.Allocate("buf_in", uint64(n*elemSize), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_out", uint64(n*elemSize), 0, mem.BufferReadWrite)

	dataIn := make([]byte, n*elemSize)
	writeFloat32(dataIn, 0, 0)
	writeFloat32(dataIn, 1, 1)
	writeFloat32(dataIn, 2, -1)
	writeFloat32(dataIn, 3, 2)
	writeFloat32(dataIn, 4, -2)
	srv.vram.Write("buf_in", 0, dataIn)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_in"},
		{"index": 1, "type": "buffer", "id": "buf_out"},
	}
	ndrangeJSON := makeNDRangeMsg("gelu", []uint64{uint64(n)}, args)

	respJSON := srv.processMessage(string(ndrangeJSON))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["success"] != true {
		t.Fatalf("gelu failed: %v", resp["error"])
	}

	// GELU(0) ≈ 0, GELU(1) > 0, GELU(-1) < 0 (small magnitude)
	resultData, _ := srv.vram.Read("buf_out", 0, uint64(n*elemSize))
	if math.Abs(float64(readFloat32(resultData, 0))) > 0.01 {
		t.Errorf("GELU(0): expected ~0, got %.4f", readFloat32(resultData, 0))
	}
	if readFloat32(resultData, 1) <= 0 {
		t.Error("GELU(1) should be positive")
	}
	if readFloat32(resultData, 3) <= 0 {
		t.Error("GELU(2) should be positive")
	}
}

// ── integration: rms_norm via IPC ────────────────────────

func TestIPCLocalRMSNorm(t *testing.T) {
	srv := newTestIPCServer(t)

	dim := 4
	const elemSize = 4

	srv.vram.Allocate("buf_in", uint64(dim*elemSize), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_out", uint64(dim*elemSize), 0, mem.BufferReadWrite)

	// Input: all 2s → rms = 2, output = all 1s
	dataIn := make([]byte, dim*elemSize)
	for i := 0; i < dim; i++ {
		writeFloat32(dataIn, i, 2.0)
	}
	srv.vram.Write("buf_in", 0, dataIn)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_in"},
		{"index": 1, "type": "buffer", "id": "buf_out"},
	}
	ndrangeJSON := makeNDRangeMsg("rms_norm", []uint64{1, uint64(dim)}, args)

	respJSON := srv.processMessage(string(ndrangeJSON))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["success"] != true {
		t.Fatalf("rms_norm failed: %v", resp["error"])
	}

	resultData, _ := srv.vram.Read("buf_out", 0, uint64(dim*elemSize))
	for i := 0; i < dim; i++ {
		if math.Abs(float64(readFloat32(resultData, i)-1.0)) > 0.01 {
			t.Errorf("rms_norm[%d]: expected 1.0, got %.4f", i, readFloat32(resultData, i))
		}
	}
}

// ── integration: add_bias via IPC ────────────────────────

func TestIPCLocalAddBias(t *testing.T) {
	srv := newTestIPCServer(t)

	rows, dim := 2, 3
	const elemSize = 4

	srv.vram.Allocate("buf_in", uint64(rows*dim*elemSize), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_out", uint64(rows*dim*elemSize), 0, mem.BufferReadWrite)
	srv.vram.Allocate("buf_bias", uint64(dim*elemSize), 0, mem.BufferReadOnly)

	// Input: [0,1,2,3,4,5]
	dataIn := make([]byte, rows*dim*elemSize)
	for i := 0; i < rows*dim; i++ {
		writeFloat32(dataIn, i, float32(i))
	}
	srv.vram.Write("buf_in", 0, dataIn)

	// Bias: [10, 20, 30]
	dataBias := make([]byte, dim*elemSize)
	writeFloat32(dataBias, 0, 10)
	writeFloat32(dataBias, 1, 20)
	writeFloat32(dataBias, 2, 30)
	srv.vram.Write("buf_bias", 0, dataBias)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_in"},
		{"index": 1, "type": "buffer", "id": "buf_out"},
		{"index": 2, "type": "buffer", "id": "buf_bias"},
	}
	ndrangeJSON := makeNDRangeMsg("add_bias", []uint64{uint64(rows), uint64(dim)}, args)

	respJSON := srv.processMessage(string(ndrangeJSON))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["success"] != true {
		t.Fatalf("add_bias failed: %v", resp["error"])
	}

	resultData, _ := srv.vram.Read("buf_out", 0, uint64(rows*dim*elemSize))
	expected := []float32{10, 21, 32, 13, 24, 35}
	for i := 0; i < rows*dim; i++ {
		actual := readFloat32(resultData, i)
		if math.Abs(float64(actual-expected[i])) > 0.01 {
			t.Errorf("add_bias[%d]: expected %.0f, got %.1f", i, expected[i], actual)
		}
	}
}

// ── integration: error handling ─────────────────────────

func TestIPCLocalUnknownKernel(t *testing.T) {
	srv := newTestIPCServer(t)

	ndrangeJSON := makeNDRangeMsg("nonexistent", []uint64{4}, []map[string]interface{}{})

	respJSON := srv.processMessage(string(ndrangeJSON))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)

	if resp["success"] == true {
		t.Error("Expected failure for unknown kernel")
	}
}

// ── integration: hello message ──────────────────────────

func TestIPCHello(t *testing.T) {
	srv := newTestIPCServer(t)

	msg := IPCMessage{Type: "hello", MsgID: "hello-1"}
	respJSON := srv.processMessage(string(mustMarshalTest(msg)))

	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["type"] != "ok" {
		t.Errorf("Expected 'ok' type, got '%v'", resp["type"])
	}
}

// ── integration: device_config ──────────────────────────

func TestIPCDeviceConfig(t *testing.T) {
	srv := newTestIPCServer(t)

	msg := IPCMessage{Type: "device_config", MsgID: "dc-1"}
	respJSON := srv.processMessage(string(mustMarshalTest(msg)))

	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	if resp["type"] != "device_info" {
		t.Errorf("Expected 'device_info' type, got '%v'", resp["type"])
	}
	if resp["device_name"] != "DistriBox Virtual GPU" {
		t.Errorf("Expected 'DistriBox Virtual GPU', got '%v'", resp["device_name"])
	}
}

// ── helpers: float32 ↔ bytes conversion ────────────────

// writeFloat32 writes float32 values into a byte slice at given index
func writeFloat32(data []byte, index int, val float32) {
	bits := math.Float32bits(val)
	off := index * 4
	data[off] = byte(bits)
	data[off+1] = byte(bits >> 8)
	data[off+2] = byte(bits >> 16)
	data[off+3] = byte(bits >> 24)
}

// readFloat32 reads a float32 from a byte slice at given index
func readFloat32(data []byte, index int) float32 {
	off := index * 4
	bits := uint32(data[off]) | uint32(data[off+1])<<8 |
		uint32(data[off+2])<<16 | uint32(data[off+3])<<24
	return math.Float32frombits(bits)
}

// bytesAsFloatSlice reads all floats from a byte slice (read-only copy)
func bytesAsFloatSlice(data []byte) []float32 {
	count := len(data) / 4
	result := make([]float32, count)
	for i := 0; i < count; i++ {
		result[i] = readFloat32(data, i)
	}
	return result
}

// ── benchmark ───────────────────────────────────────────

func BenchmarkIPCLocalVectorAdd(b *testing.B) {
	srv, _ := NewIPCServer("127.0.0.1:0",
		mem.NewVRAMManager(),
		queue.NewCommandQueueManager(),
		scheduler.NewScheduler())

	n := 1024
	srv.vram.Allocate("buf_a", uint64(n*4), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_b", uint64(n*4), 0, mem.BufferReadOnly)
	srv.vram.Allocate("buf_c", uint64(n*4), 0, mem.BufferReadWrite)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_a"},
		{"index": 1, "type": "buffer", "id": "buf_b"},
		{"index": 2, "type": "buffer", "id": "buf_c"},
	}
	msg := string(makeNDRangeMsg("vector_add", []uint64{uint64(n)}, args))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv.processMessage(msg)
	}
}
