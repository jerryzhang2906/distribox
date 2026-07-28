package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net"
	"time"
)

func main() {
	fmt.Println("=== DistriBox Distributed E2E Test ===")
	fmt.Println("Testing: ICD → IPC → VGPU → gRPC → Phone Worker → Result")

	conn, err := net.Dial("tcp", "127.0.0.1:9876")
	if err != nil {
		fmt.Printf("ERROR: %v\n", err)
		return
	}
	defer conn.Close()

	send(conn, map[string]interface{}{"type": "hello", "msg_id": "h1", "protocol": "1.0"})
	recv := readLine(conn)
	fmt.Printf("  Connected: %s\n", recv[:40])

	// Create input buffer A: [1, 2, 3, 4, 5, 6, 7, 8]
	inputA := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	hexA := hex.EncodeToString(floatsToBytes(inputA))
	sendBuf(conn, "buf_a", 32, "read_write")
	send(conn, map[string]interface{}{"type": "buffer_write", "msg_id": "wa", "buffer_id": "buf_a", "offset": 0, "size": 32, "data_b64": hexA})
	readLine(conn)
	fmt.Printf("  Buffer A: %v\n", inputA)

	// Create input buffer B: [10, 20, 30, 40, 50, 60, 70, 80]
	inputB := []float32{10, 20, 30, 40, 50, 60, 70, 80}
	hexB := hex.EncodeToString(floatsToBytes(inputB))
	sendBuf(conn, "buf_b", 32, "read_write")
	send(conn, map[string]interface{}{"type": "buffer_write", "msg_id": "wb", "buffer_id": "buf_b", "offset": 0, "size": 32, "data_b64": hexB})
	readLine(conn)
	fmt.Printf("  Buffer B: %v\n", inputB)

	// Create output buffer C (will be filled by worker)
	sendBuf(conn, "buf_c", 32, "read_write")
	readLine(conn)
	fmt.Println("  Buffer C: empty (will hold result)")

	// Execute vector_add: C = A + B
	fmt.Println("\n  Executing vector_add on phone worker...")
	start := time.Now()
	send(conn, map[string]interface{}{
		"type":       "ndrange",
		"msg_id":     "ndr",
		"queue_id":   "q",
		"kernel_id":  "k1",
		"kernel_name": "vector_add",
		"program_id": "p1",
		"work_dim":   1,
		"global":     []int{8},
		"args": []map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "buf_a"},
			{"index": 1, "type": "buffer", "id": "buf_b"},
			{"index": 2, "type": "buffer", "id": "buf_c"},
		},
	})
	readLine(conn)

	// Finish queue
	send(conn, map[string]interface{}{"type": "queue_finish", "msg_id": "qf", "queue_id": "q"})
	readLine(conn)

	// Read result
	send(conn, map[string]interface{}{"type": "buffer_read", "msg_id": "rd", "buffer_id": "buf_c", "offset": 0, "size": 32})
	resp := readLine(conn)
	elapsed := time.Since(start)

	// Parse response
	var result map[string]interface{}
	json.Unmarshal([]byte(resp), &result)

	fmt.Println("\n=== Results ===")
	fmt.Printf("  Time: %.0f ms\n", float64(elapsed.Microseconds())/1000)

	if dataB64, ok := result["data_b64"].(string); ok {
		data, _ := hex.DecodeString(dataB64)
		output := bytesToFloats(data)
		fmt.Println("  Output values:")
		allCorrect := true
		for i := 0; i < 8 && i < len(output); i++ {
			expected := inputA[i] + inputB[i]
			correct := "OK"
			if math.Abs(float64(output[i]-expected)) > 0.01 {
				correct = "FAIL"
				allCorrect = false
			}
			fmt.Printf("    [%d] %.1f + %.1f = %.1f (expected %.1f) %s\n",
				i, inputA[i], inputB[i], output[i], expected, correct)
		}
		if allCorrect {
			fmt.Println("\n  ✅ ALL CORRECT — Distributed GPU computing works!")
		} else {
			fmt.Println("\n  ❌ MISMATCH")
		}
	} else {
		fmt.Printf("  No data_b64 in response: %s\n", resp[:min(200, len(resp))])
	}
}

func send(conn net.Conn, msg map[string]interface{}) {
	data, _ := json.Marshal(msg)
	data = append(data, '\n')
	conn.Write(data)
}

func sendBuf(conn net.Conn, id string, size int, btype string) {
	send(conn, map[string]interface{}{
		"type": "buffer_create", "msg_id": "bc_" + id,
		"buffer_id": id, "size": size, "flags": 0, "buffer_type": btype,
	})
	readLine(conn)
}

func readLine(conn net.Conn) string {
	buf := make([]byte, 65536)
	conn.SetReadDeadline(time.Now().Add(5 * time.Second))
	n, _ := conn.Read(buf)
	return string(buf[:n])
}

func floatsToBytes(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return b
}

func bytesToFloats(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(
			uint32(b[i*4]) | uint32(b[i*4+1])<<8 |
				uint32(b[i*4+2])<<16 | uint32(b[i*4+3])<<24)
	}
	return f
}

func min(a, b int) int {
	if a < b { return a }
	return b
}
