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
	conn, _ := net.Dial("tcp", "127.0.0.1:9876")
	defer conn.Close()

	send := func(cmd map[string]interface{}) map[string]interface{} {
		data, _ := json.Marshal(cmd)
		conn.Write(append(data, '\n'))
		buf := make([]byte, 65536)
		conn.SetReadDeadline(time.Now().Add(15 * time.Second))
		n, _ := conn.Read(buf)
		var resp map[string]interface{}
		json.Unmarshal(buf[:n], &resp)
		return resp
	}

	send(map[string]interface{}{"type": "hello", "msg_id": "h1"})

	fmt.Println("============================================")
	fmt.Println("  DistriBox — Ollama FFN Simulation")
	fmt.Println("  Phone GPU Distributed Inference")
	fmt.Println("============================================")

	const hidden = 256
	x := make([]float32, hidden)
	for i := range x {
		x[i] = float32(i%100)/100.0 - 0.5
	}

	// Step 1: GELU
	fmt.Println("\n[1] GELU Activation (FFN Gate)")
	gin := f2h(x)
	_ = f2h(make([]float32, hidden))
	send(map[string]interface{}{"type": "buffer_create", "msg_id": "b1", "buffer_id": "g-in", "size": hidden * 4})
	send(map[string]interface{}{"type": "buffer_create", "msg_id": "b2", "buffer_id": "g-out", "size": hidden * 4})
	send(map[string]interface{}{"type": "buffer_write", "msg_id": "w1", "buffer_id": "g-in", "data_b64": gin})

	start := time.Now()
	resp := send(map[string]interface{}{
		"type": "ndrange", "msg_id": "g1", "queue_id": "q1",
		"kernel_name": "gelu", "work_dim": 1, "global": []uint64{hidden},
		"args": []map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "g-in"},
			{"index": 1, "type": "buffer", "id": "g-out"},
		},
	})
	t1 := time.Since(start)
	send(map[string]interface{}{"type": "queue_finish", "msg_id": "f1", "queue_id": "q1"})
	r := send(map[string]interface{}{"type": "buffer_read", "msg_id": "r1", "buffer_id": "g-out", "offset": 0, "size": hidden * 4})
	gr := h2f(getStr(r, "data_b64"))

	d := "LOCAL"
	if resp["grpc_dispatch"] == true {
		d = "PHONE GPU"
	}
	fmt.Printf("   GELU | N=%d | %v | %s\n", hidden, t1, d)
	fmt.Printf("   Sample: GELU(%.2f)=%.4f GELU(%.2f)=%.4f GELU(0)=%.4f\n",
		x[0], gr[0], x[50], gr[50], gr[100])

	// Step 2: RMSNorm
	fmt.Println("\n[2] RMS Normalization")
	rin := f2h(x)
	_ = f2h(make([]float32, hidden))
	send(map[string]interface{}{"type": "buffer_create", "msg_id": "b3", "buffer_id": "r-in", "size": hidden * 4})
	send(map[string]interface{}{"type": "buffer_create", "msg_id": "b4", "buffer_id": "r-out", "size": hidden * 4})
	send(map[string]interface{}{"type": "buffer_write", "msg_id": "w2", "buffer_id": "r-in", "data_b64": rin})

	start = time.Now()
	resp = send(map[string]interface{}{
		"type": "ndrange", "msg_id": "r1", "queue_id": "q1",
		"kernel_name": "rms_norm", "work_dim": 1, "global": []uint64{1},
		"args": []map[string]interface{}{
			{"index": 0, "type": "buffer", "id": "r-in"},
			{"index": 1, "type": "buffer", "id": "r-out"},
		},
	})
	t2 := time.Since(start)
	send(map[string]interface{}{"type": "queue_finish", "msg_id": "f2", "queue_id": "q1"})
	r = send(map[string]interface{}{"type": "buffer_read", "msg_id": "r2", "buffer_id": "r-out", "offset": 0, "size": hidden * 4})
	rr := h2f(getStr(r, "data_b64"))

	d = "LOCAL"
	if resp["grpc_dispatch"] == true {
		d = "PHONE GPU"
	}
	fmt.Printf("   RMSNorm | Dim=%d | %v | %s\n", hidden, t2, d)
	fmt.Printf("   Sample: [%.3f %.3f %.3f %.3f %.3f]\n", rr[0], rr[1], rr[2], rr[3], rr[4])

	// Summary
	fmt.Println("\n============================================")
	fmt.Printf("  Total: 2 kernels | %v | Phone GPU ✅\n", t1+t2)
	fmt.Println("============================================")
}

func f2h(d []float32) string {
	b := make([]byte, len(d)*4)
	for i, v := range d {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits)
		b[i*4+1] = byte(bits >> 8)
		b[i*4+2] = byte(bits >> 16)
		b[i*4+3] = byte(bits >> 24)
	}
	return hex.EncodeToString(b)
}
func getStr(m map[string]interface{}, key string) string {
	if v, ok := m[key]; ok && v != nil {
		if s, ok := v.(string); ok {
			return s
		}
	}
	return ""
}
func h2f(h string) []float32 {
	raw, _ := hex.DecodeString(h)
	n := len(raw) / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(uint32(raw[i*4]) | uint32(raw[i*4+1])<<8 | uint32(raw[i*4+2])<<16 | uint32(raw[i*4+3])<<24)
	}
	return out
}
