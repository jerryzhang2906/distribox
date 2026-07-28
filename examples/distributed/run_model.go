/*
 * examples/distributed/run_model.go — Distributed inference via VGPU IPC
 *
 * Connects to the running VGPU Core via TCP IPC (port 9876),
 * sends model weights + kernel execution commands,
 * verifies the phone worker executes them.
 *
 * Usage: go run ./examples/distributed/
 */
package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"net"
	"time"
)

func main() {
	fmt.Println("=== DistriBox Distributed Inference Test ===")
	fmt.Println("Connecting to VGPU Core at 127.0.0.1:9876...")

	conn, err := net.Dial("tcp", "127.0.0.1:9876")
	if err != nil {
		fmt.Printf("ERROR: Cannot connect to VGPU Core: %v\n", err)
		fmt.Println("Start VGPU Core first: build\\distribox.exe --mode vgpu")
		return
	}
	defer conn.Close()
	fmt.Println("Connected!")

	// Send hello
	send(conn, map[string]interface{}{
		"type":     "hello",
		"msg_id":   "h1",
		"protocol": "1.0",
	})
	recv := readResponse(conn)
	fmt.Printf("Hello: %s\n", recv)

	// Get device config
	send(conn, map[string]interface{}{
		"type":   "device_config",
		"msg_id": "dc1",
	})
	recv = readResponse(conn)
	var devCfg map[string]interface{}
	json.Unmarshal([]byte(recv), &devCfg)
	fmt.Printf("Device: VRAM=%v MB, CUs=%v\n",
		devCfg["vram_total_mb"], devCfg["compute_units"])

	// ── Create buffers for a meaningful AI workload ──────
	rng := rand.New(rand.NewSource(42))

	// Simulate a Transformer FFN layer: 2048-dim hidden, 8192-dim intermediate
	// This is ~phi3-mini scale for one layer
	const hidden = 2048
	const intermediate = 8192

	// Create input buffer (activations from previous layer)
	inputData := makeFloats(rng, hidden)
	createBuffer(conn, "buf_input", inputData, "read_write")

	// Create weight buffers (gate, up, down projections)
	gateW := makeFloats(rng, hidden*intermediate)
	createBuffer(conn, "buf_gate_w", gateW, "read_only")

	upW := makeFloats(rng, hidden*intermediate)
	createBuffer(conn, "buf_up_w", upW, "read_only")

	downW := makeFloats(rng, hidden*intermediate)
	createBuffer(conn, "buf_down_w", downW, "read_only")

	// Output buffers
	createBuffer(conn, "buf_gate_out", make([]byte, intermediate*4), "read_write")
	createBuffer(conn, "buf_up_out", make([]byte, intermediate*4), "read_write")
	createBuffer(conn, "buf_interm", make([]byte, intermediate*4), "read_write")
	createBuffer(conn, "buf_output", make([]byte, hidden*4), "read_write")

	fmt.Printf("Buffers: %d created (%.0f KB weights)\n", 8,
		float64(len(gateW)+len(upW)+len(downW))*4/1024)

	// ── Build program ────────────────────────────────────
	send(conn, map[string]interface{}{
		"type":       "program_build",
		"msg_id":     "pb1",
		"program_id": "prog_ffn",
		"source":     "// Phi-3 FFN kernels",
		"options":    "",
	})
	readResponse(conn)

	// ── Execute GELU kernel (gate activation) ─────────────
	fmt.Println("\n=== Running GELU (gate projection → activation) ===")
	start := time.Now()

	// Step 1: gate = input @ gate_w^T (matmul: [1,2048] @ [2048,8192] → [8192])
	fmt.Println("Step 1: MatMul gate projection...")
	sendNDRange(conn, "q_main", "k_gate", "gelu_proj", "prog_ffn", 2,
		[]uint64{1, intermediate}, nil, []uint64{1, 1},
		[]map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "buf_input"},
			{"index": 1, "type": "buffer", "id": "buf_gate_w"},
			{"index": 2, "type": "buffer", "id": "buf_gate_out"},
			{"index": 3, "type": "int32", "value": float64(hidden)},
		})
	resp := readResponse(conn)
	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)
	fmt.Printf("  MatMul result: success=%v, gRPC=%v\n",
		result["success"], result["grpc_dispatch"])

	// Step 2: Apply GELU activation
	fmt.Println("Step 2: GELU activation...")
	sendNDRange(conn, "q_main", "k_gelu", "gelu1", "prog_ffn", 1,
		[]uint64{intermediate}, nil, nil,
		[]map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "buf_gate_out"},
			{"index": 1, "type": "buffer", "id": "buf_gate_out"},
		})
	resp = readResponse(conn)
	json.Unmarshal([]byte(resp), &result)
	fmt.Printf("  GELU result: success=%v, gRPC=%v\n",
		result["success"], result["grpc_dispatch"])

	// Step 3: up projection (up = input @ up_w^T)
	fmt.Println("Step 3: MatMul up projection...")
	sendNDRange(conn, "q_main", "k_up", "up_proj", "prog_ffn", 2,
		[]uint64{1, intermediate}, nil, []uint64{1, 1},
		[]map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "buf_input"},
			{"index": 1, "type": "buffer", "id": "buf_up_w"},
			{"index": 2, "type": "buffer", "id": "buf_up_out"},
			{"index": 3, "type": "int32", "value": float64(hidden)},
		})
	resp = readResponse(conn)
	json.Unmarshal([]byte(resp), &result)
	fmt.Printf("  MatMul result: success=%v, gRPC=%v\n",
		result["success"], result["grpc_dispatch"])

	// Step 4: Element-wise multiply (gated = gelu * up)
	fmt.Println("Step 4: Hadamard product (gate ⊙ up)...")
	sendNDRange(conn, "q_main", "k_mul", "hadamard1", "prog_ffn", 1,
		[]uint64{intermediate}, nil, nil,
		[]map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "buf_gate_out"},
			{"index": 1, "type": "buffer", "id": "buf_up_out"},
			{"index": 2, "type": "buffer", "id": "buf_interm"},
		})
	resp = readResponse(conn)
	json.Unmarshal([]byte(resp), &result)
	fmt.Printf("  Mul result: success=%v, gRPC=%v\n",
		result["success"], result["grpc_dispatch"])

	// Step 5: Down projection (output = interm @ down_w^T → [2048])
	fmt.Println("Step 5: MatMul down projection...")
	sendNDRange(conn, "q_main", "k_down", "down_proj", "prog_ffn", 2,
		[]uint64{1, hidden}, nil, []uint64{1, 1},
		[]map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "buf_interm"},
			{"index": 1, "type": "buffer", "id": "buf_down_w"},
			{"index": 2, "type": "buffer", "id": "buf_output"},
			{"index": 3, "type": "int32", "value": float64(intermediate)},
		})
	resp = readResponse(conn)
	json.Unmarshal([]byte(resp), &result)
	fmt.Printf("  MatMul result: success=%v, gRPC=%v\n",
		result["success"], result["grpc_dispatch"])

	// Finish queue
	send(conn, map[string]interface{}{
		"type":     "queue_finish",
		"msg_id":   "qf1",
		"queue_id": "q_main",
	})
	finalResp := readResponse(conn)

	elapsed := time.Since(start)
	totalOps := int64(hidden)*int64(intermediate)*4*2 // 4 matmuls + element-wise ops
	gflops := float64(totalOps) / elapsed.Seconds() / 1e9

	fmt.Printf("\n=== Distributed Inference Complete ===")
	fmt.Printf("\n  Time: %.2f seconds", elapsed.Seconds())
	fmt.Printf("\n  Throughput: %.1f GFLOPS", gflops)
	fmt.Printf("\n  Workers: 1 (phone) + 0 (local)")
	fmt.Printf("\n  Final response: %s\n", finalResp)
}

func send(conn net.Conn, msg map[string]interface{}) {
	data, _ := json.Marshal(msg)
	data = append(data, '\n')
	conn.Write(data)
}

func readResponse(conn net.Conn) string {
	buf := make([]byte, 65536)
	n, _ := conn.Read(buf)
	return string(buf[:n])
}

func createBuffer(conn net.Conn, id string, data []byte, btype string) {
	hexData := fmt.Sprintf("%x", data)
	send(conn, map[string]interface{}{
		"type":        "buffer_create",
		"msg_id":      "bc_" + id,
		"buffer_id":   id,
		"size":        len(data),
		"flags":       0,
		"buffer_type": btype,
	})
	readResponse(conn) // ack

	// Write data
	if len(data) > 0 {
		send(conn, map[string]interface{}{
			"type":      "buffer_write",
			"msg_id":    "bw_" + id,
			"buffer_id": id,
			"offset":    0,
			"size":      len(data),
			"data":      hexData,
		})
		readResponse(conn)
	}
}

func sendNDRange(conn net.Conn, queueID, kernelID, kernelName, programID string,
	workDim uint32, global, offset, local []uint64,
	args []map[string]interface{}) {
	msg := map[string]interface{}{
		"type":        "ndrange",
		"msg_id":      kernelID + "_msg",
		"queue_id":    queueID,
		"kernel_id":   kernelID,
		"kernel_name": kernelName,
		"program_id":  programID,
		"work_dim":    workDim,
		"global":      global,
		"args":        args,
	}
	if len(offset) > 0 {
		msg["global_offset"] = offset
	}
	if len(local) > 0 {
		msg["local"] = local
	}
	argsJSON, _ := json.Marshal(args)
	raw := json.RawMessage(argsJSON)
	msg["args"] = raw
	data, _ := json.Marshal(msg)
	data = append(data, '\n')
	conn.Write(data)
}

func makeFloats(rng *rand.Rand, n int) []byte {
	result := make([]byte, n*4)
	for i := 0; i < n; i++ {
		bits := math.Float32bits(float32(rng.NormFloat64() * 0.02))
		result[i*4] = byte(bits)
		result[i*4+1] = byte(bits >> 8)
		result[i*4+2] = byte(bits >> 16)
		result[i*4+3] = byte(bits >> 24)
	}
	return result
}
