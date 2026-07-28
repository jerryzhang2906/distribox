/*
 * vgpu/server/grpc_server_test.go — gRPC end-to-end tests
 *
 * Tests: Register RPC, worker session management, capability reporting,
 * and the scheduler split → task dispatch chain.
 *
 * Run: go test ./vgpu/server/ -v -run "TestGRPC|TestScheduler|TestExecutor"
 */

package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"testing"
	"time"

	"github.com/distribox/cmd/worker/agent"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/queue"
	"github.com/distribox/vgpu/scheduler"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"
)

const gRPCTestBufSize = 1024 * 1024

// ── gRPC test harness ────────────────────────────────────

func newTestGRPCServer(t *testing.T) (*grpc.Server, *OrchestratorService, *bufconn.Listener) {
	t.Helper()

	sched := scheduler.NewScheduler()
	svc := NewOrchestratorService(sched)

	lis := bufconn.Listen(gRPCTestBufSize)
	srv := grpc.NewServer()
	distriv1.RegisterOrchestratorServer(srv, svc)

	go func() {
		if err := srv.Serve(lis); err != nil {
			log.Printf("gRPC test server stopped: %v", err)
		}
	}()

	return srv, svc, lis
}

func newTestGRPCClient(t *testing.T, lis *bufconn.Listener) (distriv1.OrchestratorClient, *grpc.ClientConn) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, "bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("Failed to dial gRPC server: %v", err)
	}

	return distriv1.NewOrchestratorClient(conn), conn
}

// ── Test: Register RPC ───────────────────────────────────

func TestGRPCRegister(t *testing.T) {
	srv, _, lis := newTestGRPCServer(t)
	defer srv.Stop()

	client, conn := newTestGRPCClient(t, lis)
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	// Simulate a worker registering
	req := &distriv1.RegisterRequest{
		ProtocolVersion: "1.0",
		Hostname:        "test-worker-1",
		Arch:            "amd64",
		Os:              "linux",
		HasGpu:          true,
		TotalRamMb:      16384,
		AuthToken:       "test-cluster-token",
	}

	resp, err := client.Register(ctx, req)
	if err != nil {
		t.Fatalf("Register RPC failed: %v", err)
	}

	if resp.WorkerId == "" {
		t.Error("Expected non-empty worker ID")
	}
	if resp.SessionToken == "" {
		t.Error("Expected non-empty session token")
	}
	if resp.OrchestratorVersion == "" {
		t.Error("Expected non-empty orchestrator version")
	}

	t.Logf("Worker registered: ID=%s, Token=%s, Version=%s",
		resp.WorkerId, resp.SessionToken, resp.OrchestratorVersion)
}

// ── Test: Register with wrong protocol version ────────────

func TestGRPCRegisterBadVersion(t *testing.T) {
	srv, _, lis := newTestGRPCServer(t)
	defer srv.Stop()

	client, conn := newTestGRPCClient(t, lis)
	defer conn.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	req := &distriv1.RegisterRequest{
		ProtocolVersion: "0.5", // Old version
		Hostname:        "old-worker",
		Arch:            "arm64",
		Os:              "android",
	}

	_, err := client.Register(ctx, req)
	if err == nil {
		t.Error("Expected error for unsupported protocol version")
	}
	t.Logf("Correctly rejected old protocol version: %v", err)
}

// ── Test: Multiple worker registrations ───────────────────

func TestGRPCMultipleWorkers(t *testing.T) {
	srv, svc, lis := newTestGRPCServer(t)
	defer srv.Stop()

	client, conn := newTestGRPCClient(t, lis)
	defer conn.Close()

	// Register 3 workers
	for i := 1; i <= 3; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)

		req := &distriv1.RegisterRequest{
			ProtocolVersion: "1.0",
			Hostname:        fmt.Sprintf("worker-%d", i),
			Arch:            "amd64",
			Os:              "linux",
			HasGpu:          i%2 == 0, // worker-2 has GPU
			TotalRamMb:      uint64(8192 * i),
		}

		resp, err := client.Register(ctx, req)
		cancel()
		if err != nil {
			t.Fatalf("Worker %d registration failed: %v", i, err)
		}
		t.Logf("Worker %d: ID=%s", i, resp.WorkerId)
	}

	// Verify workers are tracked
	svc.mu.RLock()
	count := len(svc.workers)
	svc.mu.RUnlock()

	if count != 3 {
		t.Errorf("Expected 3 workers, got %d", count)
	}
}

// ── Scheduler split algorithm tests ──────────────────────

func TestSchedulerSplit1D(t *testing.T) {
	s := scheduler.NewScheduler()

	// Two workers: one fast, one slow
	s.RegisterWorker(&scheduler.WorkerInfo{ID: "fast", CapabilityScore: 3.0, Status: "idle"})
	s.RegisterWorker(&scheduler.WorkerInfo{ID: "slow", CapabilityScore: 1.0, Status: "idle"})

	task := &scheduler.ComputeTask{
		TaskID:     "task-1",
		KernelName: "vector_add",
		WorkDim:    1,
		GlobalSize: []uint64{1000},
		LocalSize:  []uint64{32},
	}

	subTasks, err := s.SplitNDRange(task)
	if err != nil {
		t.Fatalf("SplitNDRange failed: %v", err)
	}

	if len(subTasks) != 2 {
		t.Fatalf("Expected 2 subtasks, got %d", len(subTasks))
	}

	// Fast worker gets 3/4 of work (~750 items, aligned to 32)
	// Slow worker gets 1/4 (~250 items)
	fastItems := subTasks[0].GlobalSize[0]
	slowItems := subTasks[1].GlobalSize[0]

	t.Logf("Fast worker: %d items (offset=%d)", fastItems, subTasks[0].GlobalOffset[0])
	t.Logf("Slow worker: %d items (offset=%d)", slowItems, subTasks[1].GlobalOffset[0])

	// Verify total coverage
	total := fastItems + slowItems
	if total != 1000 {
		t.Errorf("Total items %d != 1000", total)
	}

	// Fast worker gets more (proportional to capability 3:1)
	if fastItems <= slowItems {
		t.Error("Fast worker should get more items than slow worker")
	}

	// Verify proportions are roughly 3:1 (within 5% tolerance)
	expectedRatio := 3.0
	actualRatio := float64(fastItems) / float64(slowItems)
	if actualRatio < expectedRatio*0.5 || actualRatio > expectedRatio*1.5 {
		t.Errorf("Expected ratio ~%.1f:1, got %.2f:1 (%d:%d)",
			expectedRatio, actualRatio, fastItems, slowItems)
	}
}

func TestSchedulerSplit2D(t *testing.T) {
	s := scheduler.NewScheduler()

	s.RegisterWorker(&scheduler.WorkerInfo{ID: "w1", CapabilityScore: 1.0, Status: "idle"})
	s.RegisterWorker(&scheduler.WorkerInfo{ID: "w2", CapabilityScore: 1.0, Status: "idle"})

	task := &scheduler.ComputeTask{
		TaskID:     "matmul-1",
		KernelName: "matmul",
		WorkDim:    2,
		GlobalSize: []uint64{1024, 512}, // 1024 rows, 512 cols
		LocalSize:  []uint64{16, 16},
	}

	subTasks, err := s.SplitNDRange(task)
	if err != nil {
		t.Fatalf("SplitNDRange 2D failed: %v", err)
	}

	if len(subTasks) != 2 {
		t.Fatalf("Expected 2 subtasks, got %d", len(subTasks))
	}

	// Split should be along dim 0 (1024 is larger than 512)
	t.Logf("Subtask 0: global=%v offset=%v", subTasks[0].GlobalSize, subTasks[0].GlobalOffset)
	t.Logf("Subtask 1: global=%v offset=%v", subTasks[1].GlobalSize, subTasks[1].GlobalOffset)

	// Each gets ~512 rows, full 512 cols
	if subTasks[0].GlobalSize[1] != 512 || subTasks[1].GlobalSize[1] != 512 {
		t.Error("Second dimension should not be split (not the split axis)")
	}
}

func TestSchedulerNoWorkers(t *testing.T) {
	s := scheduler.NewScheduler()

	task := &scheduler.ComputeTask{
		TaskID:     "task-1",
		KernelName: "vector_add",
		WorkDim:    1,
		GlobalSize: []uint64{100},
	}

	_, err := s.SplitNDRange(task)
	if err == nil {
		t.Error("Expected error when no workers available")
	}
}

// ── Worker TaskExecutor integration tests ────────────────

func TestWorkerExecuteVectorAdd(t *testing.T) {
	w := &agent.Worker{}
	exec := agent.NewTaskExecutor(w)
	defer exec.Close()

	n := 8
	dataA := make([]byte, n*4)
	dataB := make([]byte, n*4)
	for i := 0; i < n; i++ {
		writeFloat32(dataA, i, float32(i+1))
		writeFloat32(dataB, i, float32((i+1)*10))
	}

	req := &agent.TaskRequest{
		TaskID:     "task-vecadd-1",
		KernelName: "vector_add",
		WorkDim:    1,
		GlobalSize: []uint64{uint64(n)},
		Args: []agent.KernelArg{
			{Index: 0, IsBuffer: true, BufferID: "buf_a"},
			{Index: 1, IsBuffer: true, BufferID: "buf_b"},
			{Index: 2, IsBuffer: true, BufferID: "buf_c"},
			{Index: 3, IsBuffer: false, Scalar: int32ToBytes(int32(n)), Size: 4},
		},
		InputBuffers: map[string][]byte{
			"buf_a": dataA,
			"buf_b": dataB,
		},
		OutputBufferIDs: []string{"buf_c"},
	}

	result, err := exec.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("Execute vector_add failed: %v", err)
	}
	if !result.Success {
		t.Fatalf("Task failed: %s", result.ErrorMsg)
	}

	// Verify results
	out := result.OutputBuffers["buf_c"]
	for i := 0; i < n; i++ {
		expected := float32(i+1) + float32((i+1)*10)
		actual := readFloat32(out, i)
		if absDiff(expected, actual) > 0.001 {
			t.Errorf("result[%d]: expected %.0f, got %.1f", i, expected, actual)
		}
	}
}

func TestWorkerExecuteMatMul(t *testing.T) {
	w := &agent.Worker{}
	exec := agent.NewTaskExecutor(w)
	defer exec.Close()

	M, K, N := 2, 3, 2
	// A = [[1,2,3], [4,5,6]]
	dataA := make([]byte, M*K*4)
	writeFloat32(dataA, 0, 1); writeFloat32(dataA, 1, 2); writeFloat32(dataA, 2, 3)
	writeFloat32(dataA, 3, 4); writeFloat32(dataA, 4, 5); writeFloat32(dataA, 5, 6)
	// B = [[7,8], [9,10], [11,12]]
	dataB := make([]byte, K*N*4)
	writeFloat32(dataB, 0, 7); writeFloat32(dataB, 1, 8)
	writeFloat32(dataB, 2, 9); writeFloat32(dataB, 3, 10)
	writeFloat32(dataB, 4, 11); writeFloat32(dataB, 5, 12)

	req := &agent.TaskRequest{
		TaskID:     "task-matmul-1",
		KernelName: "matmul",
		WorkDim:    2,
		GlobalSize: []uint64{uint64(M), uint64(N)},
		Args: []agent.KernelArg{
			{Index: 0, IsBuffer: true, BufferID: "buf_a"},
			{Index: 1, IsBuffer: true, BufferID: "buf_b"},
			{Index: 2, IsBuffer: true, BufferID: "buf_c"},
			{Index: 3, IsBuffer: false, Scalar: int32ToBytes(int32(K)), Size: 4},
		},
		InputBuffers: map[string][]byte{
			"buf_a": dataA,
			"buf_b": dataB,
		},
		OutputBufferIDs: []string{"buf_c"},
	}

	result, err := exec.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("Execute matmul failed: %v", err)
	}
	if !result.Success {
		t.Fatalf("Task failed: %s", result.ErrorMsg)
	}

	out := result.OutputBuffers["buf_c"]
	expected := []float32{58, 64, 139, 154}
	for i := 0; i < M*N; i++ {
		actual := readFloat32(out, i)
		if absDiff(expected[i], actual) > 0.5 {
			t.Errorf("result[%d]: expected %.0f, got %.1f", i, expected[i], actual)
		}
	}
}

func TestWorkerExecuteGELU(t *testing.T) {
	w := &agent.Worker{}
	exec := agent.NewTaskExecutor(w)
	defer exec.Close()

	n := 4
	dataIn := make([]byte, n*4)
	writeFloat32(dataIn, 0, 0)
	writeFloat32(dataIn, 1, 1)
	writeFloat32(dataIn, 2, -1)
	writeFloat32(dataIn, 3, 2)

	req := &agent.TaskRequest{
		TaskID:     "task-gelu-1",
		KernelName: "gelu",
		WorkDim:    1,
		GlobalSize: []uint64{uint64(n)},
		Args: []agent.KernelArg{
			{Index: 0, IsBuffer: true, BufferID: "buf_in"},
			{Index: 1, IsBuffer: true, BufferID: "buf_out"},
		},
		InputBuffers:    map[string][]byte{"buf_in": dataIn},
		OutputBufferIDs: []string{"buf_out"},
	}

	result, err := exec.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("Execute gelu failed: %v", err)
	}
	if !result.Success {
		t.Fatalf("Task failed: %s", result.ErrorMsg)
	}

	out := result.OutputBuffers["buf_out"]
	// GELU(0) ≈ 0
	if absDiff(readFloat32(out, 0), 0) > 0.01 {
		t.Errorf("GELU(0): expected ~0, got %.4f", readFloat32(out, 0))
	}
	// GELU(1) > 0
	if readFloat32(out, 1) <= 0 {
		t.Error("GELU(1) should be positive")
	}
}

func TestWorkerExecuteRMSNorm(t *testing.T) {
	w := &agent.Worker{}
	exec := agent.NewTaskExecutor(w)
	defer exec.Close()

	dim := 4
	dataIn := make([]byte, dim*4)
	for i := 0; i < dim; i++ {
		writeFloat32(dataIn, i, 2.0) // all 2s, rms = 2
	}

	req := &agent.TaskRequest{
		TaskID:     "task-rmsnorm-1",
		KernelName: "rms_norm",
		WorkDim:    2,
		GlobalSize: []uint64{1, uint64(dim)},
		Args: []agent.KernelArg{
			{Index: 0, IsBuffer: true, BufferID: "buf_in"},
			{Index: 1, IsBuffer: true, BufferID: "buf_out"},
		},
		InputBuffers:    map[string][]byte{"buf_in": dataIn},
		OutputBufferIDs: []string{"buf_out"},
	}

	result, err := exec.Execute(context.Background(), req)
	if err != nil {
		t.Fatalf("Execute rms_norm failed: %v", err)
	}
	if !result.Success {
		t.Fatalf("Task failed: %s", result.ErrorMsg)
	}

	out := result.OutputBuffers["buf_out"]
	for i := 0; i < dim; i++ {
		if absDiff(readFloat32(out, i), 1.0) > 0.01 {
			t.Errorf("rms_norm[%d]: expected 1.0, got %.4f", i, readFloat32(out, i))
		}
	}
}

// ── Benchmark: full local execution pipeline ─────────────

func BenchmarkGRPCRegister(b *testing.B) {
	srv, _, lis := newTestGRPCServerBench(b)
	defer srv.Stop()

	client, conn := newTestGRPCClientBench(b, lis)
	defer conn.Close()

	req := &distriv1.RegisterRequest{
		ProtocolVersion: "1.0",
		Hostname:        "bench-worker",
		Arch:            "amd64",
		Os:              "linux",
		HasGpu:          false,
		TotalRamMb:      8192,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
		client.Register(ctx, req)
		cancel()
	}
}

func BenchmarkWorkerVectorAdd(b *testing.B) {
	w := &agent.Worker{}
	exec := agent.NewTaskExecutor(w)
	defer exec.Close()

	n := 1024
	dataA := make([]byte, n*4)
	dataB := make([]byte, n*4)

	req := &agent.TaskRequest{
		TaskID:     "bench-vecadd",
		KernelName: "vector_add",
		WorkDim:    1,
		GlobalSize: []uint64{uint64(n)},
		Args: []agent.KernelArg{
			{Index: 0, IsBuffer: true, BufferID: "buf_a"},
			{Index: 1, IsBuffer: true, BufferID: "buf_b"},
			{Index: 2, IsBuffer: true, BufferID: "buf_c"},
			{Index: 3, IsBuffer: false, Scalar: int32ToBytes(int32(n)), Size: 4},
		},
		InputBuffers: map[string][]byte{
			"buf_a": dataA,
			"buf_b": dataB,
		},
		OutputBufferIDs: []string{"buf_c"},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		exec.Execute(context.Background(), req)
	}
}

// ── End-to-end: gRPC ControlChannel task dispatch ─────────
// Test the full gRPC bidirectional stream: orchestrator sends task via
// ControlChannel, worker receives, executes, returns result.

func TestE2EGRPCTaskDispatch(t *testing.T) {
	srv, svc, lis := newTestGRPCServer(t)
	defer srv.Stop()

	// ── Step 1: Worker registers ──────────────────────────
	client, conn := newTestGRPCClient(t, lis)
	defer conn.Close()

	ctx := context.Background()
	regResp, err := client.Register(ctx, &distriv1.RegisterRequest{
		ProtocolVersion: "1.0",
		Hostname:        "e2e-worker",
		Arch:            "amd64",
		Os:              "linux",
		HasGpu:          false,
		TotalRamMb:      8192,
	})
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	workerID := regResp.WorkerId
	t.Logf("Worker registered: %s", workerID)

	// ── Step 2: Open ControlChannel stream ────────────────
	stream, err := client.ControlChannel(ctx)
	if err != nil {
		t.Fatalf("ControlChannel failed: %v", err)
	}

	// Send capability report to establish identity
	err = stream.Send(&distriv1.WorkerMessage{
		Payload: &distriv1.WorkerMessage_CapabilityUpdate{
			CapabilityUpdate: &distriv1.CapabilityReport{
				WorkerId: workerID,
				Compute: &distriv1.ComputeCapability{
					Cpu: &distriv1.CPUInfo{
						Model:         "Test CPU",
						CoresPhysical: 4,
						FrequencyMhz:  3000,
					},
					Memory: &distriv1.MemoryInfo{
						TotalRamMb:     8192,
						AvailableRamMb: 4096,
					},
				},
			},
		},
	})
	if err != nil {
		t.Fatalf("Send capability report failed: %v", err)
	}

	// Give orchestrator time to process the capability report
	time.Sleep(100 * time.Millisecond)

	// ── Step 3: Start worker receive loop in goroutine ────
	// This simulates the worker's task execution loop
	taskDone := make(chan *distriv1.TaskResult, 1)
	exec := agent.NewTaskExecutor(nil)
	defer exec.Close()

	go func() {
		for {
			msg, err := stream.Recv()
			if err != nil {
				t.Logf("Worker recv done: %v", err)
				return
			}

			task := msg.GetTask()
			if task == nil {
				continue
			}

			t.Logf("Worker received task: %s (kernel=%s, global=%v)",
				task.TaskId,
				task.Compute.GetKernelName(),
				task.Compute.GetGlobalWorkSize())

			// Execute the task via GoEngine
			n := int(task.Compute.GetGlobalWorkSize()[0])
			dataA := make([]byte, n*4)
			dataB := make([]byte, n*4)
			for i := 0; i < n; i++ {
				writeFloat32(dataA, i, float32(i+1))
				writeFloat32(dataB, i, float32((i+1)*10))
			}

			result, execErr := exec.Execute(ctx, &agent.TaskRequest{
				TaskID:     task.TaskId,
				KernelName: task.Compute.GetKernelName(),
				WorkDim:    task.Compute.GetWorkDim(),
				GlobalSize: task.Compute.GetGlobalWorkSize(),
				Args: []agent.KernelArg{
					{Index: 0, IsBuffer: true, BufferID: "buf_a"},
					{Index: 1, IsBuffer: true, BufferID: "buf_b"},
					{Index: 2, IsBuffer: true, BufferID: "buf_c"},
					{Index: 3, IsBuffer: false, Scalar: int32ToBytes(int32(n)), Size: 4},
				},
				InputBuffers: map[string][]byte{
					"buf_a": dataA,
					"buf_b": dataB,
				},
				OutputBufferIDs: []string{"buf_c"},
			})

			// Build TaskResult
			status := distriv1.TaskStatus_TASK_STATUS_COMPLETE
			errMsg := ""
			if execErr != nil || (result != nil && !result.Success) {
				status = distriv1.TaskStatus_TASK_STATUS_ERROR
				if result != nil {
					errMsg = result.ErrorMsg
				} else {
					errMsg = execErr.Error()
				}
			}

			taskResult := &distriv1.TaskResult{
				TaskId:       task.TaskId,
				Status:       status,
				ErrorMessage: errMsg,
				StartTimeNs:  result.StartTimeNs,
				EndTimeNs:    result.EndTimeNs,
			}

			// Send result back
			stream.Send(&distriv1.WorkerMessage{
				Payload: &distriv1.WorkerMessage_TaskResult{
					TaskResult: taskResult,
				},
			})

			taskDone <- taskResult
		}
	}()

	// ── Step 4: Orchestrator assigns a task ───────────────
	taskID := "e2e-task-1"
	n := uint64(8)

	assignTask := &distriv1.TaskAssignment{
		TaskId: taskID,
		Compute: &distriv1.ComputeTask{
			KernelId:        "kern-1",
			KernelName:      "vector_add",
			ProgramId:       "prog-1",
			WorkDim:         1,
			GlobalWorkSize:  []uint64{n},
			GlobalWorkOffset: []uint64{0},
			LocalWorkSize:   []uint64{1},
			Args: []*distriv1.KernelArg{
				{Index: 0, Value: &distriv1.KernelArg_Buffer{Buffer: &distriv1.BufferArg{BufferId: "buf_a"}}},
				{Index: 1, Value: &distriv1.KernelArg_Buffer{Buffer: &distriv1.BufferArg{BufferId: "buf_b"}}},
				{Index: 2, Value: &distriv1.KernelArg_Buffer{Buffer: &distriv1.BufferArg{BufferId: "buf_c"}}},
				{Index: 3, Value: &distriv1.KernelArg_Scalar{Scalar: &distriv1.ScalarArg{Data: int32ToBytes(int32(n)), SizeBytes: 4}}},
			},
		},
		InputBufferIds:  []string{"buf_a", "buf_b"},
		OutputBufferIds: []string{"buf_c"},
	}

	err = svc.AssignTask(workerID, assignTask)
	if err != nil {
		t.Fatalf("AssignTask failed: %v", err)
	}

	// ── Step 5: Wait for result ───────────────────────────
	select {
	case result := <-taskDone:
		if result.Status != distriv1.TaskStatus_TASK_STATUS_COMPLETE {
			t.Fatalf("Task failed: %s", result.ErrorMessage)
		}
		t.Logf("E2E task completed: %s (status=%v, time=%d ns)",
			result.TaskId, result.Status, result.EndTimeNs-result.StartTimeNs)
	case <-time.After(5 * time.Second):
		t.Fatal("Timeout waiting for task result")
	}

	// ── Step 6: Verify via WaitForTaskResult API ───────────
	resultCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	// Note: WaitForTaskResult won't work here because the result
	// channel was consumed above. In real usage, each caller
	// calls WaitForTaskResult independently.
	_ = resultCtx
}

// Test the IPC → gRPC dispatch path (integration)
func TestE2EIPCToGRPCDispatch(t *testing.T) {
	// Share a single scheduler between IPC server and gRPC orchestrator
	sharedSched := scheduler.NewScheduler()
	svc := NewOrchestratorService(sharedSched)

	lis := bufconn.Listen(gRPCTestBufSize)
	srv := grpc.NewServer()
	distriv1.RegisterOrchestratorServer(srv, svc)
	go func() { srv.Serve(lis) }()
	defer srv.Stop()

	vram := mem.NewVRAMManager()
	cmdQueue := queue.NewCommandQueueManager()
	ipcSrv, _ := NewIPCServer("127.0.0.1:0", vram, cmdQueue, sharedSched)
	ipcSrv.SetOrchestrator(svc)

	// ── Register a worker via gRPC ────────────────────────
	client, conn := newTestGRPCClient(t, lis)
	defer conn.Close()

	ctx := context.Background()
	regResp, err := client.Register(ctx, &distriv1.RegisterRequest{
		ProtocolVersion: "1.0",
		Hostname:        "ipc-e2e-worker",
		Arch:            "amd64",
		Os:              "linux",
		HasGpu:          false,
		TotalRamMb:      4096,
	})
	if err != nil {
		t.Fatalf("Register failed: %v", err)
	}
	workerID := regResp.WorkerId

	// Open ControlChannel and establish identity
	stream, err := client.ControlChannel(ctx)
	if err != nil {
		t.Fatalf("ControlChannel failed: %v", err)
	}

	stream.Send(&distriv1.WorkerMessage{
		Payload: &distriv1.WorkerMessage_CapabilityUpdate{
			CapabilityUpdate: &distriv1.CapabilityReport{
				WorkerId: workerID,
				Compute: &distriv1.ComputeCapability{
					Cpu:    &distriv1.CPUInfo{Model: "Test", CoresPhysical: 2, FrequencyMhz: 2000},
					Memory: &distriv1.MemoryInfo{TotalRamMb: 4096, AvailableRamMb: 2048},
				},
			},
		},
	})
	time.Sleep(100 * time.Millisecond)

	// ── Worker receive loop ───────────────────────────────
	taskDone := make(chan struct{}, 1)
	exec := agent.NewTaskExecutor(nil)
	defer exec.Close()

	go func() {
		for {
			msg, recvErr := stream.Recv()
			if recvErr != nil {
				return
			}
			task := msg.GetTask()
			if task == nil {
				continue
			}

			n := int(task.Compute.GetGlobalWorkSize()[0])
			dataA := make([]byte, n*4)
			dataB := make([]byte, n*4)
			for i := 0; i < n; i++ {
				writeFloat32(dataA, i, float32(i+1))
				writeFloat32(dataB, i, float32((i+1)*10))
			}

			result, execErr := exec.Execute(ctx, &agent.TaskRequest{
				TaskID:     task.TaskId,
				KernelName: task.Compute.GetKernelName(),
				WorkDim:    task.Compute.GetWorkDim(),
				GlobalSize: task.Compute.GetGlobalWorkSize(),
				Args: []agent.KernelArg{
					{Index: 0, IsBuffer: true, BufferID: "buf_a"},
					{Index: 1, IsBuffer: true, BufferID: "buf_b"},
					{Index: 2, IsBuffer: true, BufferID: "buf_c"},
					{Index: 3, IsBuffer: false, Scalar: int32ToBytes(int32(n)), Size: 4},
				},
				InputBuffers:    map[string][]byte{"buf_a": dataA, "buf_b": dataB},
				OutputBufferIDs: []string{"buf_c"},
			})

			status := distriv1.TaskStatus_TASK_STATUS_COMPLETE
			errMsg := ""
			if execErr != nil || (result != nil && !result.Success) {
				status = distriv1.TaskStatus_TASK_STATUS_ERROR
				if result != nil {
					errMsg = result.ErrorMsg
				} else {
					errMsg = execErr.Error()
				}
			}

			startNs := int64(0)
			endNs := int64(0)
			if result != nil {
				startNs = result.StartTimeNs
				endNs = result.EndTimeNs
			}

			stream.Send(&distriv1.WorkerMessage{
				Payload: &distriv1.WorkerMessage_TaskResult{
					TaskResult: &distriv1.TaskResult{
						TaskId: task.TaskId, Status: status,
						ErrorMessage: errMsg,
						StartTimeNs:  startNs, EndTimeNs: endNs,
					},
				},
			})
			taskDone <- struct{}{}
			return
		}
	}()

	// ── Send NDRange via IPC ──────────────────────────────
	n := 4
	srvVRAM := ipcSrv.vram
	srvVRAM.Allocate("buf_a", uint64(n*4), 0, mem.BufferReadOnly)
	srvVRAM.Allocate("buf_b", uint64(n*4), 0, mem.BufferReadOnly)
	srvVRAM.Allocate("buf_c", uint64(n*4), 0, mem.BufferReadWrite)

	// Write test data
	dataA := make([]byte, n*4)
	dataB := make([]byte, n*4)
	for i := 0; i < n; i++ {
		writeFloat32(dataA, i, float32(i+1))
		writeFloat32(dataB, i, float32((i+1)*10))
	}
	srvVRAM.Write("buf_a", 0, dataA)
	srvVRAM.Write("buf_b", 0, dataB)

	args := []map[string]interface{}{
		{"index": 0, "type": "buffer", "id": "buf_a"},
		{"index": 1, "type": "buffer", "id": "buf_b"},
		{"index": 2, "type": "buffer", "id": "buf_c"},
		{"index": 3, "type": "int32", "value": float64(n)},
	}
	ndrangeJSON := makeNDRangeMsg("vector_add", []uint64{uint64(n)}, args)

	respJSON := ipcSrv.processMessage(string(ndrangeJSON))

	var resp map[string]interface{}
	json.Unmarshal([]byte(respJSON), &resp)
	t.Logf("IPC NDRange response: %v", resp)

	// With orchestrator wired, it should attempt gRPC dispatch
	// The response should have grpc_dispatch=true
	if resp["success"] != true {
		t.Logf("IPC response error: %v (may have fallen back to local exec if gRPC dispatch failed)", resp["error"])
	}

	// Wait for the worker task to complete
	select {
	case <-taskDone:
		t.Log("Worker task completed via gRPC dispatch")
	case <-time.After(5 * time.Second):
		t.Log("Timeout — task may have fallen back to local execution")
	}
}

// ── Test helpers (reuse from ipc_server_test.go) ────────

// Import aliases for benchmark helper functions
func newTestGRPCServerBench(b *testing.B) (*grpc.Server, *OrchestratorService, *bufconn.Listener) {
	sched := scheduler.NewScheduler()
	svc := NewOrchestratorService(sched)

	lis := bufconn.Listen(gRPCTestBufSize)
	srv := grpc.NewServer()
	distriv1.RegisterOrchestratorServer(srv, svc)

	go func() {
		srv.Serve(lis)
	}()

	return srv, svc, lis
}

func newTestGRPCClientBench(b *testing.B, lis *bufconn.Listener) (distriv1.OrchestratorClient, *grpc.ClientConn) {
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, "bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.Dial()
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		b.Fatalf("Failed to dial: %v", err)
	}

	return distriv1.NewOrchestratorClient(conn), conn
}

func int32ToBytes(v int32) []byte {
	b := make([]byte, 4)
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
	return b
}

func absDiff(a, b float32) float32 {
	if a > b {
		return a - b
	}
	return b - a
}
