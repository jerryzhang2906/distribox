/*
 * vgpu/server/ipc_buffer_test.go — E2E tests for buffer data execution
 *
 * Tests the full IPC → VRAM → Kernel → Output cycle with real buffer data.
 */

package server

import (
	"encoding/hex"
	"encoding/json"
	"math"
	"testing"
)

// float32ToHex converts a float32 slice to hex-encoded bytes
func float32ToHex(data []float32) string {
	buf := make([]byte, len(data)*4)
	for i, v := range data {
		bits := math.Float32bits(v)
		buf[i*4] = byte(bits)
		buf[i*4+1] = byte(bits >> 8)
		buf[i*4+2] = byte(bits >> 16)
		buf[i*4+3] = byte(bits >> 24)
	}
	return hex.EncodeToString(buf)
}

// hexToFloat32 converts hex-encoded bytes back to float32 slice
func hexToFloat32(h string) []float32 {
	raw, _ := hex.DecodeString(h)
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		bits := uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 |
			uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24
		out[i] = math.Float32frombits(bits)
	}
	return out
}

func TestIPC_CUDABuffers_VectorAdd(t *testing.T) {
	s := newTestIPCServer(t)
	defer s.Close()

	// a = [1, 2, 3, 4], b = [5, 6, 7, 8]
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}
	// Expected: c = [6, 8, 10, 12]

	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "buf-vecadd-1",
		KernelName: "vector_add",
		Grid:       []uint32{4, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "buf-a", Size: 16, DataB64: float32ToHex(a), Direction: "in"},
			{ID: "buf-b", Size: 16, DataB64: float32ToHex(b), Direction: "in"},
			{ID: "buf-c", Size: 16, DataB64: "00000000000000000000000000000000", Direction: "out"},
		},
	}
	resp := s.processMessage(mustMarshal(msg))

	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)

	if !result["success"].(bool) {
		t.Fatalf("vector_add failed: %v", result["error"])
	}

	// Verify output
	outputs, ok := result["outputs"].([]interface{})
	if !ok || len(outputs) == 0 {
		t.Fatal("No output buffers returned")
	}

	output := outputs[0].(map[string]interface{})
	dataHex := output["data_b64"].(string)
	resultFloats := hexToFloat32(dataHex)

	expected := []float32{6, 8, 10, 12}
	for i, exp := range expected {
		if i >= len(resultFloats) {
			t.Fatalf("Output too short: got %d elements, expected %d", len(resultFloats), len(expected))
		}
		if math.Abs(float64(resultFloats[i]-exp)) > 0.001 {
			t.Errorf("result[%d]: expected %.0f, got %.4f", i, exp, resultFloats[i])
		}
	}
	t.Logf("vector_add: %v + %v = %v ✅", a, b, resultFloats)
}

func TestIPC_CUDABuffers_ReLU(t *testing.T) {
	s := newTestIPCServer(t)
	defer s.Close()

	// input = [-1, 0, 1, 2, -3, 4]
	input := []float32{-1, 0, 1, 2, -3, 4}
	// Expected: [0, 0, 1, 2, 0, 4]

	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "buf-relu-1",
		KernelName: "relu",
		Grid:       []uint32{6, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "buf-in", Size: uint64(len(input) * 4), DataB64: float32ToHex(input), Direction: "in"},
			{ID: "buf-out", Size: uint64(len(input) * 4), DataB64: float32ToHex(make([]float32, len(input))), Direction: "out"},
		},
	}
	resp := s.processMessage(mustMarshal(msg))

	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)

	if !result["success"].(bool) {
		t.Fatalf("relu failed: %v", result["error"])
	}

	outputs := result["outputs"].([]interface{})
	output := outputs[0].(map[string]interface{})
	resultFloats := hexToFloat32(output["data_b64"].(string))

	expected := []float32{0, 0, 1, 2, 0, 4}
	for i, exp := range expected {
		if math.Abs(float64(resultFloats[i]-exp)) > 0.001 {
			t.Errorf("relu[%d]: expected %.0f, got %.4f", i, exp, resultFloats[i])
		}
	}
	t.Logf("relu: %v → %v ✅", input, resultFloats)
}

func TestIPC_CUDABuffers_GELU(t *testing.T) {
	s := newTestIPCServer(t)
	defer s.Close()

	// GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
	input := []float32{0, 1, -1}
	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "buf-gelu-1",
		KernelName: "gelu",
		Grid:       []uint32{3, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "in", Size: 12, DataB64: float32ToHex(input), Direction: "in"},
			{ID: "out", Size: 12, DataB64: float32ToHex(make([]float32, 3)), Direction: "out"},
		},
	}
	resp := s.processMessage(mustMarshal(msg))

	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)
	if !result["success"].(bool) {
		t.Fatalf("gelu failed: %v", result["error"])
	}

	outputs := result["outputs"].([]interface{})
	output := outputs[0].(map[string]interface{})
	r := hexToFloat32(output["data_b64"].(string))

	// GELU approximations from the known implementation
	if math.Abs(float64(r[0])) > 0.01 {
		t.Errorf("GELU(0): expected ~0, got %.4f", r[0])
	}
	if math.Abs(float64(r[1]-0.841)) > 0.1 {
		t.Errorf("GELU(1): expected ~0.841, got %.4f", r[1])
	}
	if r[2] > 0 {
		t.Errorf("GELU(-1): expected negative, got %.4f", r[2])
	}
	t.Logf("gelu: %v → [%.4f, %.4f, %.4f] ✅", input, r[0], r[1], r[2])
}

func TestIPC_CUDABuffers_ScalarMul(t *testing.T) {
	s := newTestIPCServer(t)
	defer s.Close()

	// input * 2.0
	input := []float32{1, 2, 3, 4}
	scalar := []float32{2.0}

	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "buf-smul-1",
		KernelName: "scalar_mul",
		Grid:       []uint32{4, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "in", Size: 16, DataB64: float32ToHex(input), Direction: "in"},
			{ID: "out", Size: 16, DataB64: float32ToHex(make([]float32, 4)), Direction: "out"},
			{ID: "scalar", Size: 4, DataB64: float32ToHex(scalar), Direction: "in", Kind: "scalar_float32"},
		},
	}
	resp := s.processMessage(mustMarshal(msg))

	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)
	if !result["success"].(bool) {
		t.Fatalf("scalar_mul failed: %v", result["error"])
	}

	outputs := result["outputs"].([]interface{})
	output := outputs[0].(map[string]interface{})
	r := hexToFloat32(output["data_b64"].(string))

	expected := []float32{2, 4, 6, 8}
	for i, exp := range expected {
		if math.Abs(float64(r[i]-exp)) > 0.01 {
			t.Errorf("scalar_mul[%d]: expected %.0f, got %.4f", i, exp, r[i])
		}
	}
	t.Logf("scalar_mul: %v * 2 = %v ✅", input, r)
}

func TestIPC_CUDABuffers_Sigmoid(t *testing.T) {
	s := newTestIPCServer(t)
	defer s.Close()

	// sigmoid(0)=0.5, sigmoid(5)≈1, sigmoid(-5)≈0
	input := []float32{0, 5, -5}
	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "buf-sigmoid-1",
		KernelName: "sigmoid",
		Grid:       []uint32{3, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "in", Size: 12, DataB64: float32ToHex(input), Direction: "in"},
			{ID: "out", Size: 12, DataB64: float32ToHex(make([]float32, 3)), Direction: "out"},
		},
	}
	resp := s.processMessage(mustMarshal(msg))

	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)
	if !result["success"].(bool) {
		t.Fatalf("sigmoid failed: %v", result["error"])
	}

	outputs := result["outputs"].([]interface{})
	output := outputs[0].(map[string]interface{})
	r := hexToFloat32(output["data_b64"].(string))

	if math.Abs(float64(r[0]-0.5)) > 0.01 {
		t.Errorf("sigmoid(0): expected ~0.5, got %.4f", r[0])
	}
	if r[1] < 0.99 {
		t.Errorf("sigmoid(5): expected ~1, got %.4f", r[1])
	}
	if r[2] > 0.01 {
		t.Errorf("sigmoid(-5): expected ~0, got %.4f", r[2])
	}
	t.Logf("sigmoid: %v → [%.4f, %.4f, %.4f] ✅", input, r[0], r[1], r[2])
}

func TestIPC_CUDABuffers_ElementWiseMul(t *testing.T) {
	s := newTestIPCServer(t)
	defer s.Close()

	a := []float32{1, 2, 3, 4}
	b := []float32{2, 3, 4, 5}

	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "buf-ewmul-1",
		KernelName: "element_wise_mul",
		Grid:       []uint32{4, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "a", Size: 16, DataB64: float32ToHex(a), Direction: "in"},
			{ID: "b", Size: 16, DataB64: float32ToHex(b), Direction: "in"},
			{ID: "out", Size: 16, DataB64: float32ToHex(make([]float32, 4)), Direction: "out"},
		},
	}
	resp := s.processMessage(mustMarshal(msg))

	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)
	if !result["success"].(bool) {
		t.Fatalf("element_wise_mul failed: %v", result["error"])
	}

	outputs := result["outputs"].([]interface{})
	output := outputs[0].(map[string]interface{})
	r := hexToFloat32(output["data_b64"].(string))

	expected := []float32{2, 6, 12, 20}
	for i, exp := range expected {
		if math.Abs(float64(r[i]-exp)) > 0.01 {
			t.Errorf("hadamard[%d]: expected %.0f, got %.4f", i, exp, r[i])
		}
	}
	t.Logf("element_wise_mul: %v ⊙ %v = %v ✅", a, b, r)
}

func TestIPC_CUDABuffers_RoundTrip(t *testing.T) {
	// Test: multiple sequential launches work correctly on the same server
	s := newTestIPCServer(t)
	defer s.Close()

	kernels := []struct {
		name  string
		input []float32
	}{
		{"relu", []float32{-1, 2, -3, 4, -5}},
		{"gelu", []float32{0, 1, 2}},
		{"sigmoid", []float32{-2, 0, 2}},
	}

	for _, k := range kernels {
		size := uint64(len(k.input) * 4)
		msg := IPCMessage{
			Type:       "cuda_launch",
			MsgID:      "roundtrip-" + k.name,
			KernelName: k.name,
			Grid:       []uint32{uint32(len(k.input)), 1, 1},
			Block:      []uint32{1, 1, 1},
			Buffers: []IPCBufferData{
				{ID: "in", Size: size, DataB64: float32ToHex(k.input), Direction: "in"},
				{ID: "out", Size: size, DataB64: float32ToHex(make([]float32, len(k.input))), Direction: "out"},
			},
		}
		resp := s.processMessage(mustMarshal(msg))
		var result map[string]interface{}
		json.Unmarshal([]byte(resp), &result)
		if !result["success"].(bool) {
			t.Errorf("%s round-trip failed: %v", k.name, result["error"])
		}
	}
}
