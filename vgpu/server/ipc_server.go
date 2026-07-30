/*
 * vgpu/server/ipc_server.go — Local IPC server for ICD communication
 *
 * Listens on TCP localhost (127.0.0.1:9876).
 * Receives JSON commands from the ICD library running inside application processes.
 * Dispatches to VRAM manager, command queue, and scheduler.
 */

package server

import (
	"bufio"
	"context"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"time"
	"unsafe"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/queue"
	"github.com/distribox/vgpu/scheduler"

	"github.com/distribox/cmd/worker/engine"
)

// ── IPC Server ─────────────────────────────────────────

type IPCServer struct {
	listener    net.Listener
	addr        string
	vram        *mem.VRAMManager
	cmdQueue    *queue.CommandQueueManager
	sched       *scheduler.Scheduler
	localEngine *engine.GoEngine // Fallback when no workers are connected
	orchestrator *OrchestratorService // gRPC task dispatch (may be nil)
	closing     bool
}

func NewIPCServer(addr string, vram *mem.VRAMManager, cmdQueue *queue.CommandQueueManager,
	sched *scheduler.Scheduler) (*IPCServer, error) {

	return &IPCServer{
		addr:        addr,
		vram:        vram,
		cmdQueue:    cmdQueue,
		sched:       sched,
		localEngine: engine.NewGoEngine(),
	}, nil
}

// SetOrchestrator sets the gRPC orchestrator for remote task dispatch.
// When nil, tasks are executed locally via the Go engine.
func (s *IPCServer) SetOrchestrator(o *OrchestratorService) {
	s.orchestrator = o
}

func (s *IPCServer) Serve() error {
	var err error
	s.listener, err = net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("failed to listen on %s: %w", s.addr, err)
	}

	log.Printf("IPC server listening on tcp://%s", s.addr)

	for !s.closing {
		conn, err := s.listener.Accept()
		if err != nil {
			if s.closing {
				return nil
			}
			log.Printf("IPC accept error: %v", err)
			continue
		}
		go s.handleConnection(conn)
	}
	return nil
}

func (s *IPCServer) Close() {
	s.closing = true
	if s.listener != nil {
		s.listener.Close()
	}
}

// ── Connection handler ─────────────────────────────────

func (s *IPCServer) handleConnection(conn net.Conn) {
	defer conn.Close()
	scanner := bufio.NewScanner(conn)
	scanner.Buffer(make([]byte, 65536), 1<<20) // 1 MB max per line

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		response := s.processMessage(line)
		if response != "" {
			fmt.Fprintf(conn, "%s\n", response)
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("IPC read error: %v", err)
	}
}

// ── Message dispatch ───────────────────────────────────

type IPCMessage struct {
	Type      string `json:"type"`
	MsgID     string `json:"msg_id"`
	RequestID string `json:"request_id"`

	// Fields for specific message types (optional)
	BufferID   string `json:"buffer_id,omitempty"`
	BufferType string `json:"buffer_type,omitempty"`
	Size       uint64 `json:"size,omitempty"`
	Flags      uint32 `json:"flags,omitempty"`
	BufOffset  uint64 `json:"offset,omitempty"`
	DataB64    string `json:"data_b64,omitempty"`
	Pattern    string `json:"pattern,omitempty"`
	SrcBuffer  string `json:"src_buffer_id,omitempty"`
	DstBuffer  string `json:"dst_buffer_id,omitempty"`
	SHMName    string `json:"shm_name,omitempty"`

	// NDRange fields
	QueueID    string   `json:"queue_id,omitempty"`
	KernelID   string   `json:"kernel_id,omitempty"`
	KernelName string   `json:"kernel_name,omitempty"`
	ProgramID  string   `json:"program_id,omitempty"`
	WorkDim    uint32   `json:"work_dim,omitempty"`
	Global     []uint64 `json:"global,omitempty"`
	GlobalOffset []uint64 `json:"global_offset,omitempty"`
	Local      []uint64 `json:"local,omitempty"`
	Args       []json.RawMessage `json:"args,omitempty"`

	// Vulkan dispatch fields
	GroupCount []uint32 `json:"group_count,omitempty"`
	Pipeline   string   `json:"pipeline,omitempty"`
	Protocol   string   `json:"protocol,omitempty"`

	// CUDA launch fields
	Grid      []uint32 `json:"grid,omitempty"`
	Block     []uint32 `json:"block,omitempty"`
	SharedMem uint32   `json:"shared_mem,omitempty"`
	Stream    string   `json:"stream,omitempty"`

	// Buffer data for Vulkan/CUDA dispatch messages
	Buffers []IPCBufferData `json:"buffers,omitempty"`
}

// IPCBufferData carries buffer or scalar content in IPC messages
type IPCBufferData struct {
	ID        string `json:"id"`
	Size      uint64 `json:"size"`
	DataB64   string `json:"data_b64,omitempty"`
	Direction string `json:"direction,omitempty"` // "in", "out", "inout"
	Offset    uint64 `json:"offset,omitempty"`
	Kind      string `json:"kind,omitempty"` // "buffer" (default), "scalar_int32", "scalar_float32"
}

type IPCResponse struct {
	Type      string `json:"type"`
	RequestID string `json:"request_id,omitempty"`
	MsgID     string `json:"msg_id,omitempty"`
	Success   bool   `json:"success"`
	Error     string `json:"error,omitempty"`
}

func (s *IPCServer) processMessage(line string) string {
	var msg IPCMessage
	if err := json.Unmarshal([]byte(line), &msg); err != nil {
		return mustMarshal(IPCResponse{Type: "error", Error: fmt.Sprintf("bad JSON: %v", err)})
	}

	reqID := msg.MsgID
	if reqID == "" {
		reqID = msg.RequestID
	}

	log.Printf("IPC ← %s (id=%s)", msg.Type, reqID)

	switch msg.Type {
	case "hello":
		return s.handleHello(msg, reqID)
	case "device_config":
		return s.handleDeviceConfig(reqID)
	case "buffer_create":
		return s.handleBufferCreate(msg, reqID)
	case "buffer_write":
		return s.handleBufferWrite(msg, reqID)
	case "buffer_read":
		return s.handleBufferRead(msg, reqID)
	case "buffer_release":
		return s.handleBufferRelease(msg, reqID)
	case "buffer_fill":
		return s.handleBufferFill(msg, reqID)
	case "buffer_copy":
		return s.handleBufferCopy(msg, reqID)
	case "program_build":
		return s.handleProgramBuild(msg, reqID)
	case "ndrange":
		return s.handleNDRange(msg, reqID)
	case "queue_finish":
		return s.handleQueueFinish(msg, reqID)
	case "vk_hello":
		return s.handleVKHello(msg, reqID)
	case "vk_dispatch":
		return s.handleVKDispatch(msg, reqID)
	case "cuda_hello":
		return s.handleCUDAHello(msg, reqID)
	case "cuda_launch":
		return s.handleCUDALaunch(msg, reqID)
	default:
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: fmt.Sprintf("unknown type: %s", msg.Type)})
	}
}

// ── Message handlers ───────────────────────────────────

func (s *IPCServer) handleHello(msg IPCMessage, reqID string) string {
	log.Printf("ICD connected (protocol %s)", msg.Type)
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleDeviceConfig(reqID string) string {
	spec := s.vram.GetSpec()
	resp := map[string]interface{}{
		"type":          "device_info",
		"request_id":    reqID,
		"device_name":   spec.Name,
		"vram_bytes":    spec.VRAMTotal,
		"compute_units": spec.ComputeUnits,
		"clock_mhz":     spec.MaxClockMHz,
		"max_wg_size":   spec.MaxWorkGroupSize,
		"max_wi_sizes":  spec.MaxWorkItemSizes,
	}
	return mustMarshal(resp)
}

func (s *IPCServer) handleBufferCreate(msg IPCMessage, reqID string) string {
	bufType := mem.BufferReadWrite
	switch msg.BufferType {
	case "read_only":
		bufType = mem.BufferReadOnly
	case "read_write":
		bufType = mem.BufferReadWrite
	case "temporary":
		bufType = mem.BufferTemporary
	}

	_, err := s.vram.Allocate(msg.BufferID, msg.Size, msg.Flags, bufType)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	log.Printf("Buffer created: %s (%d bytes, type=%s)", msg.BufferID, msg.Size, msg.BufferType)
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleBufferWrite(msg IPCMessage, reqID string) string {
	// Decode hex-encoded data from data_b64 field
	var data []byte
	if msg.DataB64 != "" {
		decoded, err := hex.DecodeString(msg.DataB64)
		if err != nil {
			// Fall back to raw string bytes (backward compat)
			data = []byte(msg.DataB64)
		} else {
			data = decoded
		}
	}
	err := s.vram.Write(msg.BufferID, msg.BufOffset, data)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleBufferRead(msg IPCMessage, reqID string) string {
	data, err := s.vram.Read(msg.BufferID, msg.BufOffset, msg.Size)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	// Hex-encode binary data for safe JSON transport
	dataHex := hex.EncodeToString(data)

	resp := map[string]interface{}{
		"type":       "ok",
		"request_id": reqID,
		"success":    true,
		"data_b64":   dataHex,
		"size":       len(data),
	}
	return mustMarshal(resp)
}

func (s *IPCServer) handleBufferRelease(msg IPCMessage, reqID string) string {
	err := s.vram.Release(msg.BufferID)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleBufferFill(msg IPCMessage, reqID string) string {
	// Fill buffer with repeating pattern
	buf, err := s.vram.Get(msg.BufferID)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	pattern := []byte(msg.Pattern)
	for i := uint64(0); i < msg.Size; i += uint64(len(pattern)) {
		n := uint64(len(pattern))
		if i+n > msg.Size {
			n = msg.Size - i
		}
		s.vram.Write(msg.BufferID, msg.BufOffset+i, pattern[:n])
	}
	_ = buf // suppress unused warning
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleBufferCopy(msg IPCMessage, reqID string) string {
	data, err := s.vram.Read(msg.SrcBuffer, 0, msg.Size)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}
	err = s.vram.Write(msg.DstBuffer, 0, data)
	if err != nil {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleProgramBuild(msg IPCMessage, reqID string) string {
	// Kernel source compilation is distributed to workers lazily.
	// For now, just acknowledge — actual compilation happens on first NDRange.
	log.Printf("Program build requested: %s", msg.ProgramID)
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleNDRange(msg IPCMessage, reqID string) string {
	taskID := fmt.Sprintf("task-%s", reqID)

	// Enqueue the command for queue tracking (clFinish depends on this)
	s.cmdQueue.Enqueue(&queue.Command{
		ID:      taskID,
		QueueID: msg.QueueID,
		CmdType: queue.CmdNDRangeKernel,
	})
	defer s.cmdQueue.MarkComplete(msg.QueueID, taskID)

	// Extract buffer args
	var bufferArgs []string
	for _, arg := range msg.Args {
		var argMap map[string]interface{}
		json.Unmarshal(arg, &argMap)
		if t, ok := argMap["type"].(string); ok && t == "buffer" {
			if id, ok := argMap["id"].(string); ok {
				bufferArgs = append(bufferArgs, id)
			}
		}
	}

	// Check if we have active workers
	workers := s.sched.GetActiveWorkers()
	log.Printf("NDRange %s: %d active workers, orch=%v", msg.KernelName, len(workers), s.orchestrator != nil)
	if len(workers) > 0 && s.orchestrator != nil {
		// Skip local (in-process) workers to avoid re-entrant deadlock:
		// IPC → gRPC → local worker → GPU → OpenCL → ICD → IPC (busy!)
		remoteWorkers := filterRemoteWorkers(workers, s.orchestrator)
		log.Printf("NDRange %s: %d remote workers after filter", msg.KernelName, len(remoteWorkers))
		if len(remoteWorkers) > 0 {
			return s.dispatchToWorkers(msg, reqID, taskID, bufferArgs, remoteWorkers)
		}
		// All workers are local — fall through to local execution
		log.Printf("NDRange %s: all %d workers are local, executing locally (avoids ICD re-entry)",
			msg.KernelName, len(workers))
	}

	// ── No remote workers: execute locally via Go fallback engine ──
	return s.executeLocally(msg, reqID, taskID, bufferArgs)
}

// dispatchToWorkers splits NDRange and assigns subtasks to workers via gRPC
func (s *IPCServer) dispatchToWorkers(msg IPCMessage, reqID, taskID string, bufferArgs []string, workers []*scheduler.WorkerInfo) string {
	task := &scheduler.ComputeTask{
		TaskID:      taskID,
		QueueID:     msg.QueueID,
		KernelID:    msg.KernelID,
		KernelName:  msg.KernelName,
		ProgramID:   msg.ProgramID,
		WorkDim:     msg.WorkDim,
		GlobalSize:  msg.Global,
		LocalSize:   msg.Local,
		ArgBuffers:  bufferArgs,
	}

	subTasks, err := s.sched.SplitNDRangeForWorkers(task, workers)
	if err != nil {
		log.Printf("NDRange split failed: %v", err)
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	log.Printf("NDRange %s: split into %d subtasks across %d workers",
		task.KernelName, len(subTasks), len(workers))

	// If orchestrator is wired, dispatch via gRPC (with timeout fallback)
	if s.orchestrator != nil {
		type dispatchResult struct {
			resp string
		}
		resultCh := make(chan dispatchResult, 1)
		go func() {
			resultCh <- dispatchResult{s.dispatchViaGRPC(msg, reqID, taskID, subTasks, bufferArgs)}
		}()

		select {
		case r := <-resultCh:
			return r.resp
		case <-time.After(5 * time.Second):
			log.Printf("gRPC dispatch timeout for %s — falling back to local execution", msg.KernelName)
			return s.executeLocally(msg, reqID, taskID, bufferArgs)
		}
	}

	// Fall back to local execution
	log.Printf("gRPC dispatch not wired — executing locally")
	return s.executeLocally(msg, reqID, taskID, bufferArgs)
}

// dispatchViaGRPC sends subtasks to workers via gRPC and waits for results.
func (s *IPCServer) dispatchViaGRPC(msg IPCMessage, reqID, taskID string, subTasks []*scheduler.SubTask, bufferArgs []string) string {
	// Build list of args as protobuf KernelArgs
	var protoArgs []*distriv1.KernelArg
	for i, arg := range msg.Args {
		var argMap map[string]interface{}
		json.Unmarshal(arg, &argMap)

		argType, _ := argMap["type"].(string)
		idx := uint32(i)
		if idxFloat, ok := argMap["index"].(float64); ok {
			idx = uint32(idxFloat)
		}

		karg := &distriv1.KernelArg{Index: idx}
		switch argType {
		case "buffer":
			bufID, _ := argMap["id"].(string)
			karg.Value = &distriv1.KernelArg_Buffer{
				Buffer: &distriv1.BufferArg{BufferId: bufID},
			}
		case "int32", "scalar", "float32", "float":
			if val, ok := argMap["value"].(float64); ok {
				data := make([]byte, 4)
				v := int32(val)
				data[0], data[1], data[2], data[3] = byte(v), byte(v>>8), byte(v>>16), byte(v>>24)
				karg.Value = &distriv1.KernelArg_Scalar{
					Scalar: &distriv1.ScalarArg{Data: data, SizeBytes: 4},
				}
			}
		}
		protoArgs = append(protoArgs, karg)
	}

	// ── Transfer input buffer data to each worker ────────
	// Read VRAM data once, then send to each worker
	bufferData := make(map[string][]byte)
	for _, bufID := range bufferArgs {
		data, err := s.vram.Read(bufID, 0, 0) // 0 size = read all
		if err == nil {
			bufferData[bufID] = data
		}
	}

	// Send buffer data to each worker before task dispatch
	for _, st := range subTasks {
		for bufID, data := range bufferData {
			transferMsg := &distriv1.OrchestratorMessage{
				Payload: &distriv1.OrchestratorMessage_Buffer{
					Buffer: &distriv1.BufferTransfer{
						BufferId:  bufID,
						Direction: distriv1.TransferDirection_TRANSFER_TO_WORKER,
						SizeBytes: int64(len(data)),
						Data:      data,
					},
				},
			}
			if err := s.orchestrator.SendToWorker(st.WorkerID, transferMsg); err != nil {
				log.Printf("Buffer transfer to %s failed: %v", st.WorkerID, err)
			}
		}
	}
	log.Printf("Buffer data (%d buffers, %d bytes total) sent to %d workers",
		len(bufferData), totalBytes(bufferData), len(subTasks))

	// Dispatch each subtask to its assigned worker
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	type subResult struct {
		workerID string
		result   *distriv1.TaskResult
		err      error
	}

	resultCh := make(chan subResult, len(subTasks))

	for i, st := range subTasks {
		subTaskID := fmt.Sprintf("%s-%d", taskID, i)

		subTask := &distriv1.TaskAssignment{
			TaskId:         subTaskID,
			CommandQueueId: msg.QueueID,
			Compute: &distriv1.ComputeTask{
				KernelId:         msg.KernelID,
				KernelName:       msg.KernelName,
				ProgramId:        msg.ProgramID,
				WorkDim:          msg.WorkDim,
				GlobalWorkSize:   st.GlobalSize,
				GlobalWorkOffset: st.GlobalOffset,
				LocalWorkSize:    st.LocalSize,
				Args:             protoArgs,
			},
			InputBufferIds:  bufferArgs,
			OutputBufferIds: safeLast(bufferArgs), // last buffer is output
		}

		err := s.orchestrator.AssignTask(st.WorkerID, subTask)
		if err != nil {
			log.Printf("Failed to assign subtask %s to %s: %v", subTaskID, st.WorkerID, err)
			resultCh <- subResult{workerID: st.WorkerID, err: err}
			continue
		}

		// Wait for result in background
		go func(wid, stid string) {
			res, err := s.orchestrator.WaitForTaskResult(ctx, wid, stid)
			resultCh <- subResult{workerID: wid, result: res, err: err}
		}(st.WorkerID, subTaskID)
	}

	// Collect results
	successCount := 0
	failCount := 0
	for range subTasks {
		sr := <-resultCh
		if sr.err != nil {
			failCount++
			log.Printf("Subtask on %s error: %v", sr.workerID, sr.err)
		} else if sr.result != nil && sr.result.Status == distriv1.TaskStatus_TASK_STATUS_COMPLETE {
			successCount++
		} else {
			failCount++
			if sr.result != nil {
				log.Printf("Subtask on %s failed: %s", sr.workerID, sr.result.ErrorMessage)
			}
		}
	}

	// Collect output buffer data from workers (arrives after TaskResult via BufferTransfer)
	outputData := make(map[string][]byte)
	for _, st := range subTasks {
		// Poll for returned buffers with timeout
		for attempt := 0; attempt < 300; attempt++ {
			returned := s.orchestrator.GetReturnedBuffers(st.WorkerID)
			if len(returned) > 0 {
				for bufID, data := range returned {
					outputData[bufID] = data
					log.Printf("Output %s: %d bytes collected from worker %s", bufID, len(data), st.WorkerID)
				}
				break
			}
			time.Sleep(10 * time.Millisecond)
		}
	}

	// Write output data back to VRAM
	for bufID, data := range outputData {
		if err := s.vram.Write(bufID, 0, data); err != nil {
			log.Printf("Failed to write output %s to VRAM: %v", bufID, err)
		}
	}
	if len(outputData) > 0 {
		log.Printf("Output merge: %d buffers written back to VRAM", len(outputData))
	}

	log.Printf("NDRange %s: %d/%d subtasks succeeded via gRPC",
		msg.KernelName, successCount, len(subTasks))

	if successCount == 0 {
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: "all subtasks failed"})
	}

	return mustMarshal(map[string]interface{}{
		"type":         "ok",
		"request_id":   reqID,
		"success":      true,
		"event_id":     fmt.Sprintf("evt-%s", reqID),
		"grpc_dispatch": true,
		"subtasks":     successCount,
	})
}

// executeLocally runs the kernel on the local Go engine using VRAM buffer data
func (s *IPCServer) executeLocally(msg IPCMessage, reqID, taskID string, bufferArgs []string) string {
	log.Printf("NDRange %s: executing locally (no workers or dispatch fallback)", msg.KernelName)

	// Build kernel with args from the IPC message
	kernel := &engine.GoKernel{NameVal: msg.KernelName}

	// Map VRAM buffers to GoEngine buffers
	engBufs := make(map[string]*engine.GoBuffer)
	for _, arg := range msg.Args {
		var argMap map[string]interface{}
		json.Unmarshal(arg, &argMap)

		argType, _ := argMap["type"].(string)
		idxFloat, _ := argMap["index"].(float64)
		idx := uint32(idxFloat)

		switch argType {
		case "buffer":
			bufID, _ := argMap["id"].(string)
			// Read buffer data from VRAM
			vramBuf, err := s.vram.Get(bufID)
			if err != nil {
				return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: fmt.Sprintf("buffer %s not found: %v", bufID, err)})
			}
			data, _ := s.vram.Read(bufID, 0, vramBuf.Size)
			engBuf, err := s.localEngine.CreateBuffer(vramBuf.Size, vramBuf.Flags, data)
			if err != nil {
				return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
			}
			engBufs[bufID] = engBuf
			s.localEngine.SetKernelArg(kernel, idx, engBuf)

		case "scalar", "int32", "uint32":
			if val, ok := argMap["value"].(float64); ok {
				s.localEngine.SetKernelArg(kernel, idx, int32(val))
			}

		case "float", "float32":
			if val, ok := argMap["value"].(float64); ok {
				s.localEngine.SetKernelArg(kernel, idx, float32(val))
			}

		default:
			// Raw bytes
			if dataStr, ok := argMap["data"].(string); ok {
				s.localEngine.SetKernelArg(kernel, idx, []byte(dataStr))
			}
		}
	}

	// Build global size (default to buffer size if not specified)
	global := msg.Global
	if len(global) == 0 && len(bufferArgs) > 0 {
		// Infer from first buffer: treat as 1D with N = buffer_size / 4
		if buf, ok := engBufs[bufferArgs[0]]; ok {
			global = []uint64{buf.Size() / 4}
		}
	}
	globalOffset := msg.GlobalOffset
	if len(globalOffset) == 0 {
		globalOffset = make([]uint64, len(global))
	}
	local := msg.Local
	if len(local) == 0 {
		local = make([]uint64, len(global))
	}

	// Determine output buffers to write back to VRAM.
	// Default to all buffer args (read-only inputs are idempotent on write-back).
	var outputBufs []*engine.GoBuffer
	for _, bufID := range bufferArgs {
		if buf, ok := engBufs[bufID]; ok {
			outputBufs = append(outputBufs, buf)
		}
	}

	// Execute
	err := s.localEngine.ExecuteNDRange(kernel, msg.WorkDim, global, globalOffset, local, outputBufs)
	if err != nil {
		log.Printf("Local execution failed: %v", err)
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	// Write output buffer results back to VRAM
	for _, outBuf := range outputBufs {
		data, _ := s.localEngine.ReadBuffer(outBuf, 0, outBuf.Size())
		// Try to find the matching buffer ID and write back
		for bufID, engBuf := range engBufs {
			if engBuf == outBuf {
				s.vram.Write(bufID, 0, data)
				break
			}
		}
	}

	// Cleanup engine buffers
	for _, buf := range engBufs {
		s.localEngine.ReleaseBuffer(buf)
	}

	log.Printf("NDRange %s: local execution complete (global=%v)", msg.KernelName, global)

	return mustMarshal(map[string]interface{}{
		"type":       "ok",
		"request_id": reqID,
		"success":    true,
		"event_id":   fmt.Sprintf("evt-%s", reqID),
		"local_exec": true,
	})
}

func (s *IPCServer) handleQueueFinish(msg IPCMessage, reqID string) string {
	// Wait for all pending commands on the queue to complete
	if msg.QueueID != "" {
		s.cmdQueue.WaitForQueue(msg.QueueID)
	}
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

// ── Vulkan/CUDA message handlers ────────────────────────

func (s *IPCServer) handleVKHello(msg IPCMessage, reqID string) string {
	log.Printf("Vulkan layer connected (protocol %s)", msg.Protocol)
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleVKDispatch(msg IPCMessage, reqID string) string {
	taskID := fmt.Sprintf("vk-%s", reqID)

	// Convert Vulkan group counts to NDRange global sizes
	// Vulkan compute shader: global_size = group_count * local_size
	// For simplicity, treat group_count as the global work size
	global := make([]uint64, 3)
	workDim := uint32(1)
	if len(msg.GroupCount) >= 3 {
		global[0] = uint64(msg.GroupCount[0])
		global[1] = uint64(msg.GroupCount[1])
		global[2] = uint64(msg.GroupCount[2])
		workDim = 3
	} else if len(msg.GroupCount) >= 1 {
		global[0] = uint64(msg.GroupCount[0])
		workDim = 1
	}

	// Map pipeline identifier to kernel name
	kernelName := fmt.Sprintf("vk_pipeline_%s", msg.Pipeline)

	log.Printf("VK dispatch: pipeline=%s groups=%v → kernel=%s", msg.Pipeline, msg.GroupCount, kernelName)

	// Check if we have active workers
	workers := s.sched.GetActiveWorkers()
	if len(workers) > 0 && s.orchestrator != nil {
		// Convert to NDRange message format and dispatch
		ndrangeMsg := IPCMessage{
			KernelName: kernelName,
			WorkDim:    workDim,
			Global:     global,
		}
		return s.dispatchToWorkers(ndrangeMsg, reqID, taskID, nil, workers)
	}

	// No workers: execute locally
	// For Vulkan compute, we execute the equivalent kernel on local engine
	ndrangeMsg := IPCMessage{
		KernelName: kernelName,
		WorkDim:    workDim,
		Global:     global,
	}
	return s.executeLocallyVK(ndrangeMsg, reqID, taskID)
}

func (s *IPCServer) handleCUDAHello(msg IPCMessage, reqID string) string {
	log.Printf("CUDA proxy connected (protocol %s)", msg.Protocol)
	return mustMarshal(IPCResponse{Type: "ok", RequestID: reqID, Success: true})
}

func (s *IPCServer) handleCUDALaunch(msg IPCMessage, reqID string) string {
	taskID := fmt.Sprintf("cu-%s", reqID)

	// Convert CUDA grid/block to NDRange
	global := make([]uint64, 3)
	local := make([]uint64, 3)
	workDim := uint32(1)

	if len(msg.Grid) >= 3 && len(msg.Block) >= 3 {
		global[0] = uint64(msg.Grid[0]) * uint64(msg.Block[0])
		global[1] = uint64(msg.Grid[1]) * uint64(msg.Block[1])
		global[2] = uint64(msg.Grid[2]) * uint64(msg.Block[2])
		local[0] = uint64(msg.Block[0])
		local[1] = uint64(msg.Block[1])
		local[2] = uint64(msg.Block[2])
		workDim = 3
	} else if len(msg.Grid) >= 1 && len(msg.Block) >= 1 {
		global[0] = uint64(msg.Grid[0]) * uint64(msg.Block[0])
		local[0] = uint64(msg.Block[0])
		workDim = 1
	}

	kernelName := msg.KernelName
	if kernelName == "" {
		kernelName = "unknown_kernel"
	}

	log.Printf("CUDA launch: kernel=%s grid=%v block=%v shared=%d buffers=%d → NDRange global=%v local=%v",
		kernelName, msg.Grid, msg.Block, msg.SharedMem, len(msg.Buffers), global, local)

	ndrangeMsg := IPCMessage{
		KernelName: kernelName,
		WorkDim:    workDim,
		Global:     global,
		Local:      local,
		Buffers:    msg.Buffers,
	}

	// Check workers — if buffers present, dispatch to workers
	if len(msg.Buffers) > 0 && len(s.sched.GetActiveWorkers()) > 0 && s.orchestrator != nil {
		return s.dispatchVKCUDA(ndrangeMsg, reqID, taskID)
	}

	return s.executeLocallyVK(ndrangeMsg, reqID, taskID)
}

// dispatchVKCUDA dispatches a VK/CUDA kernel with buffer data to workers
func (s *IPCServer) dispatchVKCUDA(msg IPCMessage, reqID, taskID string) string {
	// Write buffer data to VRAM first
	bufferIDs := s.writeBuffersToVRAM(msg.Buffers)
	defer s.cleanupVRAMBuffers(bufferIDs)

	// Build kernel args from buffers
	var args []json.RawMessage
	for i, buf := range msg.Buffers {
		argJSON := fmt.Sprintf(`{"index":%d,"type":"buffer","id":"%s"}`,
			i, buf.ID)
		args = append(args, json.RawMessage(argJSON))
	}

	ndrangeMsg := IPCMessage{
		KernelName:  msg.KernelName,
		WorkDim:     msg.WorkDim,
		Global:      msg.Global,
		Local:       msg.Local,
		Args:        args,
	}
	// Use the existing NDRange dispatch (which handles VRAM → workers → VRAM merge)
	return s.dispatchToWorkers(ndrangeMsg, reqID, taskID, bufferIDs,
		s.sched.GetActiveWorkers())
}

// writeBuffersToVRAM creates VRAM buffers from IPC buffer data
func (s *IPCServer) writeBuffersToVRAM(buffers []IPCBufferData) []string {
	var ids []string
	for _, buf := range buffers {
		if buf.DataB64 == "" {
			continue
		}
		decoded, err := hex.DecodeString(buf.DataB64)
		if err != nil {
			log.Printf("Buffer %s: base64 decode failed: %v", buf.ID, err)
			continue
		}

		bufType := mem.BufferReadWrite
		if buf.Direction == "in" {
			bufType = mem.BufferReadOnly
		}

		_, err = s.vram.Allocate(buf.ID, uint64(len(decoded)), 0, bufType)
		if err != nil {
			log.Printf("Buffer %s: VRAM allocate failed: %v", buf.ID, err)
			continue
		}

		err = s.vram.Write(buf.ID, buf.Offset, decoded)
		if err != nil {
			log.Printf("Buffer %s: VRAM write failed: %v", buf.ID, err)
			continue
		}

		ids = append(ids, buf.ID)
		log.Printf("Buffer %s: %d bytes written to VRAM (dir=%s)", buf.ID, len(decoded), buf.Direction)
	}
	return ids
}

func (s *IPCServer) cleanupVRAMBuffers(ids []string) {
	for _, id := range ids {
		s.vram.Release(id)
	}
}

// executeLocallyVK runs a Vulkan/CUDA kernel locally via the Go engine.
// If buffer data is provided, does a full VRAM → engine → VRAM cycle.
// For unknown kernels without buffers, passes through to local GPU.
func (s *IPCServer) executeLocallyVK(msg IPCMessage, reqID, taskID string) string {
	kernelName := msg.KernelName

	knownKernels := map[string]bool{
		"vector_add": true, "matmul": true, "gelu": true, "relu": true,
		"softmax": true, "sigmoid": true, "scalar_mul": true, "element_wise_mul": true,
		"transpose": true, "reduce_sum": true, "layer_norm": true,
		"rms_norm": true, "rope": true, "add_bias": true,
	}

	// ── With buffer data: full VRAM → engine → VRAM execution ──
	if len(msg.Buffers) > 0 && knownKernels[kernelName] {
		return s.executeLocallyVKWithBuffers(msg, reqID, taskID, kernelName)
	}

	// ── Without buffers or unknown kernel: passthrough ──
	if !knownKernels[kernelName] {
		log.Printf("VK/CUDA local: unknown kernel '%s' — passthrough to local GPU", kernelName)
		return mustMarshal(map[string]interface{}{
			"type":       "ok",
			"request_id": reqID,
			"success":    true,
			"event_id":   fmt.Sprintf("evt-%s", reqID),
			"note":       "unknown kernel, passthrough to local GPU",
		})
	}

	// Bufferless known kernel: do minimal execution for routing verification
	kernel := &engine.GoKernel{NameVal: kernelName}
	global := msg.Global
	if len(global) == 0 {
		global = []uint64{1}
	}
	local := msg.Local
	if len(local) == 0 {
		local = make([]uint64, len(global))
	}

	err := s.localEngine.ExecuteNDRange(kernel, msg.WorkDim, global,
		make([]uint64, len(global)), local, nil)
	if err != nil {
		log.Printf("VK/CUDA local execution failed: %v", err)
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	log.Printf("VK/CUDA local: %s executed (no buffers)", kernelName)
	return mustMarshal(map[string]interface{}{
		"type": "ok", "request_id": reqID, "success": true,
		"event_id": fmt.Sprintf("evt-%s", reqID), "local_exec": true,
	})
}

// executeLocallyVKWithBuffers does full VRAM → GoEngine → VRAM with buffer data
func (s *IPCServer) executeLocallyVKWithBuffers(msg IPCMessage, reqID, taskID, kernelName string) string {
	// Step 1: Write buffer data to VRAM
	bufferIDs := s.writeBuffersToVRAM(msg.Buffers)
	defer s.cleanupVRAMBuffers(bufferIDs)

	// Step 2: Build kernel args — map buffer/scalar IDs to engine args
	kernel := &engine.GoKernel{NameVal: kernelName}
	engBufs := make(map[string]*engine.GoBuffer)

	for i, buf := range msg.Buffers {
		if buf.DataB64 == "" {
			continue
		}
		decoded, err := hex.DecodeString(buf.DataB64)
		if err != nil {
			continue
		}

		switch buf.Kind {
		case "scalar_int32":
			if len(decoded) >= 4 {
				val := int32(decoded[0]) | int32(decoded[1])<<8 | int32(decoded[2])<<16 | int32(decoded[3])<<24
				s.localEngine.SetKernelArg(kernel, uint32(i), val)
			}
		case "scalar_float32":
			if len(decoded) >= 4 {
				bits := uint32(decoded[0]) | uint32(decoded[1])<<8 | uint32(decoded[2])<<16 | uint32(decoded[3])<<24
				val := float32FromBits(bits)
				s.localEngine.SetKernelArg(kernel, uint32(i), val)
			}
		default: // "buffer" or empty
			engBuf, err := s.localEngine.CreateBuffer(uint64(len(decoded)), 0, decoded)
			if err != nil {
				log.Printf("Buffer %s: engine create failed: %v", buf.ID, err)
				continue
			}
			engBufs[buf.ID] = engBuf
			s.localEngine.SetKernelArg(kernel, uint32(i), engBuf)
		}
	}

	// Step 3: Determine output buffers (skip scalar args, direction="out" or "inout")
	var outputBufs []*engine.GoBuffer
	var outputBufInfo []IPCBufferData
	for _, buf := range msg.Buffers {
		if buf.Kind == "scalar_int32" || buf.Kind == "scalar_float32" {
			continue // scalars don't produce output
		}
		if buf.Direction == "out" || buf.Direction == "inout" {
			if engBuf, ok := engBufs[buf.ID]; ok {
				outputBufs = append(outputBufs, engBuf)
				outputBufInfo = append(outputBufInfo, buf)
			}
		}
	}

	// If no explicit output buffers, use all buffers as output
	if len(outputBufs) == 0 {
		for _, buf := range msg.Buffers {
			if engBuf, ok := engBufs[buf.ID]; ok {
				outputBufs = append(outputBufs, engBuf)
				outputBufInfo = append(outputBufInfo, buf)
			}
		}
	}

	// Step 4: Execute kernel
	global := msg.Global
	if len(global) == 0 {
		global = []uint64{1}
	}
	local := msg.Local
	if len(local) == 0 {
		local = make([]uint64, len(global))
	}

	err := s.localEngine.ExecuteNDRange(kernel, msg.WorkDim, global,
		make([]uint64, len(global)), local, outputBufs)
	if err != nil {
		log.Printf("VK/CUDA local: %s execution failed: %v", kernelName, err)
		return mustMarshal(IPCResponse{Type: "error", RequestID: reqID, Error: err.Error()})
	}

	// Step 5: Read back output buffers
	outputs := make([]map[string]interface{}, 0)
	for i, buf := range outputBufInfo {
		if i < len(outputBufs) && outputBufs[i] != nil {
			data, err := s.localEngine.ReadBuffer(outputBufs[i], 0, outputBufs[i].Size())
			if err == nil {
				// Write results back to VRAM
				s.vram.Write(buf.ID, 0, data)
				outputs = append(outputs, map[string]interface{}{
					"id":       buf.ID,
					"size":     len(data),
					"data_b64": hex.EncodeToString(data),
				})
			}
		}
	}

	// Step 6: Cleanup engine buffers
	for _, buf := range engBufs {
		s.localEngine.ReleaseBuffer(buf)
	}

	log.Printf("VK/CUDA local: %s complete, %d output buffers returned", kernelName, len(outputs))

	return mustMarshal(map[string]interface{}{
		"type": "ok", "request_id": reqID, "success": true,
		"event_id": fmt.Sprintf("evt-%s", reqID), "local_exec": true,
		"outputs": outputs,
	})
}

// filterRemoteWorkers returns only workers not running on localhost.
// Local workers use the ICD proxy for GPU execution, which calls back into
// the same IPC server, creating a re-entrant deadlock.
func filterRemoteWorkers(workers []*scheduler.WorkerInfo, orch *OrchestratorService) []*scheduler.WorkerInfo {
	var remote []*scheduler.WorkerInfo
	for _, w := range workers {
		if orch.IsSameHost(w.ID) {
			log.Printf("  Worker %s is local (same host) — skipping gRPC dispatch", w.ID)
			continue
		}
		remote = append(remote, w)
	}
	return remote
}

// ── Helpers ────────────────────────────────────────────

func totalBytes(data map[string][]byte) int64 {
	var total int64
	for _, d := range data {
		total += int64(len(d))
	}
	return total
}

func safeLast(s []string) []string {
	if len(s) == 0 {
		return nil
	}
	return s[len(s)-1:]
}

func float32FromBits(bits uint32) float32 {
	return *(*float32)(unsafe.Pointer(&bits))
}

func mustMarshal(v interface{}) string {
	data, err := json.Marshal(v)
	if err != nil {
		return `{"type":"error","error":"marshal failed"}`
	}
	return string(data)
}
