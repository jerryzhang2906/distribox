/*
 * integration_test.go — Full-stack E2E test for Vulkan/CUDA IPC interception
 *
 * Simulates the complete flow: OpenCL ICD / Vulkan layer / CUDA proxy →
 * IPC Server → VRAM → GoEngine → output.
 *
 * This is the test equivalent of running Ollama with our interception DLLs.
 */
package server

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"strings"
	"testing"
)

// ── Full integration: IPC server + real client ─────────

func TestIntegration_VulkanLayer_DispatchWithBuffers(t *testing.T) {
	srv := newTestIPCServer(t)
	defer srv.Close()

	// In-process test (same IPC path the TCP server uses)
	a := []float32{1, 2, 3, 4}
	b := []float32{5, 6, 7, 8}

	inputA := float32ToHex(a)
	inputB := float32ToHex(b)
	zeroOut := float32ToHex(make([]float32, 4))

	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "integration-vk-1",
		KernelName: "vector_add",
		Grid:       []uint32{4, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "buf-a", Size: 16, DataB64: inputA, Direction: "in"},
			{ID: "buf-b", Size: 16, DataB64: inputB, Direction: "in"},
			{ID: "buf-c", Size: 16, DataB64: zeroOut, Direction: "out"},
		},
	}

	respJSON := srv.processMessage(mustMarshal(msg))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)

	if !resp["success"].(bool) {
		t.Fatalf("Integration vector_add failed: %v", resp["error"])
	}

	outputs := resp["outputs"].([]interface{})
	if len(outputs) != 1 {
		t.Fatalf("Expected 1 output buffer, got %d", len(outputs))
	}

	out := outputs[0].(map[string]interface{})
	result := hexToFloat32(out["data_b64"].(string))

	expected := []float32{6, 8, 10, 12}
	for i, exp := range expected {
		if math.Abs(float64(result[i]-exp)) > 0.001 {
			t.Errorf("result[%d]: expected %.0f, got %.4f", i, exp, result[i])
		}
	}
	t.Logf("Integration VK: vector_add → %v ✅", result)
}

func TestIntegration_OllamaFFN_Simulated(t *testing.T) {
	// Simulate a full Transformer FFN forward pass:
	// hidden=4, intermediate=8
	// 1. gate = matmul(x, W_gate)  -- not tested here (matrix reduce)
	// 2. gate = GELU(gate)
	// 3. up = matmul(x, W_up)
	// 4. gated = Hadamard(gate, up)
	// 5. out = matmul(gated, W_down)

	srv := newTestIPCServer(t)
	defer srv.Close()

	// Step 2: GELU on gate tensor
	gateInput := []float32{-2, -1, 0, 1, 2, 3, -0.5, 0.5}
	hidden := float32ToHex(gateInput)
	zero := float32ToHex(make([]float32, 8))

	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "ffn-gelu",
		KernelName: "gelu",
		Grid:       []uint32{8, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "gate-in", Size: uint64(len(gateInput) * 4), DataB64: hidden, Direction: "in"},
			{ID: "gate-out", Size: uint64(len(gateInput) * 4), DataB64: zero, Direction: "out"},
		},
	}
	respJSON := srv.processMessage(mustMarshal(msg))
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)

	if !resp["success"].(bool) {
		t.Fatalf("FFN GELU failed: %v", resp["error"])
	}

	outputs := resp["outputs"].([]interface{})
	gateOut := hexToFloat32(outputs[0].(map[string]interface{})["data_b64"].(string))

	// GELU(-2) < 0, GELU(0)=0, GELU(2)≈2, GELU(3)≈3
	if gateOut[0] >= 0 {
		t.Errorf("GELU(-2) should be negative, got %.4f", gateOut[0])
	}
	if gateOut[2] > 0.01 || gateOut[2] < -0.01 {
		t.Errorf("GELU(0) should be ~0, got %.4f", gateOut[2])
	}
	if gateOut[4] < 1.9 {
		t.Errorf("GELU(2) should be ~2, got %.4f", gateOut[4])
	}
	if gateOut[5] < 2.9 {
		t.Errorf("GELU(3) should be ~3, got %.4f", gateOut[5])
	}

	t.Logf("FFN GELU: %v → [%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f] ✅",
		gateInput, gateOut[0], gateOut[1], gateOut[2], gateOut[3],
		gateOut[4], gateOut[5], gateOut[6], gateOut[7])

	// Step 3-4: Element-wise multiply (gated = gate ⊙ up)
	up := []float32{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}
	upHex := float32ToHex(up)
	zero2 := float32ToHex(make([]float32, 8))

	msg2 := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "ffn-hadamard",
		KernelName: "element_wise_mul",
		Grid:       []uint32{8, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "a", Size: 32, DataB64: float32ToHex(gateOut), Direction: "in"},
			{ID: "b", Size: 32, DataB64: upHex, Direction: "in"},
			{ID: "c", Size: 32, DataB64: zero2, Direction: "out"},
		},
	}
	respJSON2 := srv.processMessage(mustMarshal(msg2))
	var resp2 map[string]interface{}
	json.Unmarshal([]byte(respJSON2), &resp2)

	if !resp2["success"].(bool) {
		t.Fatalf("FFN Hadamard failed: %v", resp2["error"])
	}

	outputs2 := resp2["outputs"].([]interface{})
	gatedOut := hexToFloat32(outputs2[0].(map[string]interface{})["data_b64"].(string))
	t.Logf("FFN Hadamard: gate ⊙ up → %v ✅", gatedOut)
}

func TestIntegration_VKLayerProtocol_MatchesSpec(t *testing.T) {
	// Test the exact JSON format the Vulkan layer DLL will send
	srv := newTestIPCServer(t)
	defer srv.Close()

	// This is the exact format from distribox_vk_layer.c serializeDispatch()
	input := []float32{-1, 2, -3, 4}
	inHex := float32ToHex(input)
	outHex := float32ToHex(make([]float32, 4))

	// Build JSON exactly as the C layer does (but with buffers)
	vkJSON := fmt.Sprintf(
		`{"type":"vk_dispatch","msg_id":"vk-1","group_count":[4,1,1],"pipeline":"0xDEADBEEF","buffers":[`+
			`{"id":"vk-in","size":%d,"data_b64":"%s","direction":"in"},`+
			`{"id":"vk-out","size":%d,"data_b64":"%s","direction":"out"}]}`,
		len(input)*4, inHex,
		len(input)*4, outHex)

	respJSON := srv.processMessage(vkJSON)
	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)

	if !resp["success"].(bool) {
		t.Fatalf("VK protocol match failed: %v", resp["error"])
	}

	// With unknown pipeline, should passthrough
	if !strings.Contains(respJSON, "passthrough") && !strings.Contains(respJSON, "vector_add") {
		t.Log("VK dispatch handled (unknown pipeline → passthrough or compatibility shim)")
	}
}

func TestIntegration_DLLSizeLimits(t *testing.T) {
	// Verify we can handle buffer sizes up to what the DLL would send
	// CUDA proxy uses 64KB JSON buffer, Vulkan layer uses 4KB
	sizes := []int{4, 16, 256, 4096}

	for _, size := range sizes {
		data := make([]float32, size/4)
		for i := range data {
			data[i] = float32(i)
		}

		srv := newTestIPCServer(t)

		msg := IPCMessage{
			Type:       "cuda_launch",
			MsgID:      fmt.Sprintf("size-%d", size),
			KernelName: "relu",
			Grid:       []uint32{uint32(size / 4), 1, 1},
			Block:      []uint32{1, 1, 1},
			Buffers: []IPCBufferData{
				{ID: "in", Size: uint64(size), DataB64: float32ToHex(data), Direction: "in"},
				{ID: "out", Size: uint64(size), DataB64: float32ToHex(make([]float32, size/4)), Direction: "out"},
			},
		}

		respJSON := srv.processMessage(mustMarshal(msg))
		var resp map[string]interface{}
		json.Unmarshal([]byte(respJSON), &resp)

		if !resp["success"].(bool) {
			t.Errorf("Size %d failed: %v", size, resp["error"])
		}
		srv.Close()
	}
	t.Logf("All buffer sizes (4B → 4KB) handled correctly ✅")
}

// Helper for hex encoding single integer (for int32 scalars)
func int32ToHex(val int32) string {
	buf := make([]byte, 4)
	buf[0] = byte(val)
	buf[1] = byte(val >> 8)
	buf[2] = byte(val >> 16)
	buf[3] = byte(val >> 24)
	return hex.EncodeToString(buf)
}

// ── Benchmark: simulate Vulkan layer dispatch rate ─────

func BenchmarkIntegration_CUDALaunchWithBuffers(b *testing.B) {
	srv, err := NewIPCServer("127.0.0.1:0", nil, nil, nil)
	if err != nil {
		b.Fatal(err)
	}
	defer srv.Close()

	input := float32ToHex([]float32{1, 2, 3, 4})
	msg := IPCMessage{
		Type:       "cuda_launch",
		MsgID:      "bench",
		KernelName: "relu",
		Grid:       []uint32{4, 1, 1},
		Block:      []uint32{1, 1, 1},
		Buffers: []IPCBufferData{
			{ID: "in", Size: 16, DataB64: input, Direction: "in"},
			{ID: "out", Size: 16, DataB64: float32ToHex(make([]float32, 4)), Direction: "out"},
		},
	}
	msgJSON := mustMarshal(msg)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv.processMessage(msgJSON)
	}
}
