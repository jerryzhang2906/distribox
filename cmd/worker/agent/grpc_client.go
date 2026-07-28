/*
 * cmd/worker/agent/grpc_client.go — Worker gRPC client implementation
 *
 * Full gRPC client lifecycle:
 *   1. Dial orchestrator → Register RPC
 *   2. Open ControlChannel bidirectional stream
 *   3. Send capability report
 *   4. Receive loop: TaskAssignment → execute → TaskResult
 *
 * Sprint 4: Full bidirectional stream implementation.
 */

package agent

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/cmd/worker/capability"
	"github.com/distribox/cmd/worker/monitor"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
)

// ── gRPC Worker Client ────────────────────────────────

type GRPCWorkerClient struct {
	config        WorkerConfig
	workerID      string
	sessionToken  string

	conn          *grpc.ClientConn
	client        distriv1.OrchestratorClient
	stream        distriv1.Orchestrator_ControlChannelClient

	executor      *TaskExecutor
	resourceMon   *monitor.ResourceMonitor

	bufferCache   map[string][]byte // buffer_id → data, populated by BufferTransfer messages

	stopCh        chan struct{}
	connected     bool
}

// NewGRPCWorkerClient creates a gRPC-connected worker client.
func NewGRPCWorkerClient(config WorkerConfig) *GRPCWorkerClient {
	return &GRPCWorkerClient{
		config:      config,
		resourceMon: config.ResourceMon,
		bufferCache: make(map[string][]byte),
		stopCh:      make(chan struct{}),
	}
}

// Connect dials the orchestrator, registers, and opens the ControlChannel.
func (c *GRPCWorkerClient) Connect(ctx context.Context) error {
	// ── Step 1: Dial orchestrator ─────────────────────────
	var err error
	c.conn, err = grpc.DialContext(ctx, c.config.OrchestratorAddr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
	)
	if err != nil {
		return fmt.Errorf("dial %s: %w", c.config.OrchestratorAddr, err)
	}
	c.client = distriv1.NewOrchestratorClient(c.conn)

	// ── Step 2: Register ──────────────────────────────────
	caps := c.config.Capabilities
	hasGPU := caps != nil && caps.HasGPU
	gpuVendor := ""
	if caps != nil {
		gpuVendor = caps.GPUVendor
	}
	totalRAM := uint64(0)
	if caps != nil {
		totalRAM = uint64(caps.TotalRAMMB)
	}

	regResp, err := c.client.Register(ctx, &distriv1.RegisterRequest{
		ProtocolVersion: "1.0",
		Hostname:        c.config.Name,
		Arch:            capsArch(caps),
		Os:              capsOS(caps),
		HasGpu:          hasGPU,
		GpuVendor:       gpuVendor,
		TotalRamMb:      totalRAM,
		AuthToken:       c.config.ClusterToken,
	})
	if err != nil {
		c.conn.Close()
		return fmt.Errorf("register: %w", err)
	}

	c.workerID = regResp.WorkerId
	c.sessionToken = regResp.SessionToken

	log.Printf("Registered as %s (token=%s, orchestrator=%s)",
		c.workerID, c.sessionToken, regResp.OrchestratorVersion)

	// ── Step 3: Open ControlChannel stream ────────────────
	stream, err := c.client.ControlChannel(ctx)
	if err != nil {
		c.conn.Close()
		return fmt.Errorf("open ControlChannel: %w", err)
	}
	c.stream = stream

	// ── Step 4: Send initial capability report ────────────
	if err := c.sendCapabilityReport(); err != nil {
		log.Printf("Warning: failed to send capability report: %v", err)
	}

	// ── Step 5: Initialize task executor ──────────────────
	c.executor = NewTaskExecutor(nil) // nil Worker ref — we use our own config
	c.connected = true

	log.Printf("Worker %s fully connected and ready for tasks", c.workerID)
	return nil
}

// Run starts the task receive loop. Blocks until Disconnect is called.
func (c *GRPCWorkerClient) Run() error {
	if c.stream == nil {
		return fmt.Errorf("not connected — call Connect first")
	}

	// Start heartbeat goroutine
	go c.heartbeatLoop()

	// ── Main receive loop ─────────────────────────────────
	for {
		msg, err := c.stream.Recv()
		if err == io.EOF {
			log.Printf("ControlChannel closed by orchestrator")
			return nil
		}
		if err != nil {
			select {
			case <-c.stopCh:
				return nil
			default:
				return fmt.Errorf("ControlChannel recv error: %w", err)
			}
		}

		// Dispatch received message
		switch {
		case msg.GetTask() != nil:
			go c.handleTaskAssignment(msg.GetTask())
		case msg.GetCompile() != nil:
			go c.handleCompileRequest(msg.GetCompile())
		case msg.GetBuffer() != nil:
			c.handleBufferTransfer(msg.GetBuffer())
		case msg.GetHeartbeatAck() != nil:
			// Heartbeat acknowledged
		case msg.GetShutdown() != nil:
			log.Printf("Received shutdown: %s", msg.GetShutdown().Reason)
			if msg.GetShutdown().Graceful {
				// TODO: finish current tasks before shutting down
			}
			return nil
		}
	}
}

// Disconnect gracefully shuts down the worker connection.
func (c *GRPCWorkerClient) Disconnect() {
	close(c.stopCh)
	c.connected = false

	if c.stream != nil {
		c.stream.CloseSend()
	}
	if c.executor != nil {
		c.executor.Close()
	}
	if c.conn != nil {
		c.conn.Close()
	}
	log.Printf("Worker %s disconnected", c.workerID)
}

// ── Task handling ──────────────────────────────────────

func (c *GRPCWorkerClient) handleTaskAssignment(task *distriv1.TaskAssignment) {
	log.Printf("Task received: %s (kernel=%s, global=%v, offset=%v)",
		task.TaskId,
		task.Compute.GetKernelName(),
		task.Compute.GetGlobalWorkSize(),
		task.Compute.GetGlobalWorkOffset())

	// Convert protobuf args to engine args
	var engineArgs []KernelArg
	for _, arg := range task.Compute.GetArgs() {
		ka := KernelArg{Index: arg.Index}
		switch v := arg.Value.(type) {
		case *distriv1.KernelArg_Buffer:
			ka.IsBuffer = true
			ka.BufferID = v.Buffer.BufferId
			ka.Size = v.Buffer.SizeBytes
		case *distriv1.KernelArg_Scalar:
			ka.IsBuffer = false
			ka.Scalar = v.Scalar.Data
			ka.Size = uint64(v.Scalar.SizeBytes)
		}
		engineArgs = append(engineArgs, ka)
	}

	// Populate input buffers from the cache (populated by BufferTransfer messages)
	inputBufs := make(map[string][]byte)
	for _, bufID := range task.InputBufferIds {
		if data, ok := c.bufferCache[bufID]; ok {
			inputBufs[bufID] = data
		}
	}

	req := &TaskRequest{
		TaskID:          task.TaskId,
		QueueID:         task.CommandQueueId,
		KernelID:        task.Compute.GetKernelId(),
		KernelName:      task.Compute.GetKernelName(),
		WorkDim:         task.Compute.GetWorkDim(),
		GlobalSize:      task.Compute.GetGlobalWorkSize(),
		GlobalOffset:    task.Compute.GetGlobalWorkOffset(),
		LocalSize:       task.Compute.GetLocalWorkSize(),
		Args:            engineArgs,
		OutputBufferIDs: task.OutputBufferIds,
		InputBuffers:    inputBufs,
	}

	// Execute
	result, err := c.executor.Execute(context.Background(), req)

	// Send result back
	status := distriv1.TaskStatus_TASK_STATUS_COMPLETE
	errMsg := ""
	if err != nil || (result != nil && !result.Success) {
		status = distriv1.TaskStatus_TASK_STATUS_ERROR
		if result != nil {
			errMsg = result.ErrorMsg
		} else if err != nil {
			errMsg = err.Error()
		}
	}

	var regions []*distriv1.BufferRegion
	if result != nil {
		for bufID, data := range result.OutputBuffers {
			regions = append(regions, &distriv1.BufferRegion{
				BufferId:   bufID,
				SizeBytes:  uint64(len(data)),
				OffsetBytes: 0,
			})
		}
	}

	startNs := int64(0)
	endNs := int64(0)
	if result != nil {
		startNs = result.StartTimeNs
		endNs = result.EndTimeNs
	}

	taskResult := &distriv1.TaskResult{
		TaskId:         task.TaskId,
		Status:         status,
		ErrorMessage:   errMsg,
		OutputRegions:  regions,
		StartTimeNs:    startNs,
		EndTimeNs:      endNs,
	}

	workerMsg := &distriv1.WorkerMessage{
		Payload: &distriv1.WorkerMessage_TaskResult{
			TaskResult: taskResult,
		},
	}

	if err := c.stream.Send(workerMsg); err != nil {
		log.Printf("Failed to send task result for %s: %v", task.TaskId, err)
	} else {
		log.Printf("Task %s: result sent (status=%v, outputs=%d)",
			task.TaskId, status, len(regions))
	}

	// Send output buffer data back via BufferTransfer
	if result != nil {
		for bufID, data := range result.OutputBuffers {
			transferMsg := &distriv1.WorkerMessage{
				Payload: &distriv1.WorkerMessage_Buffer{
					Buffer: &distriv1.BufferTransfer{
						BufferId:  bufID,
						Direction: distriv1.TransferDirection_TRANSFER_FROM_WORKER,
						SizeBytes: int64(len(data)),
						Data:      data,
					},
				},
			}
			if err := c.stream.Send(transferMsg); err != nil {
				log.Printf("Failed to send output %s: %v", bufID, err)
			} else {
				log.Printf("Output %s: %d bytes sent back to orchestrator", bufID, len(data))
			}
		}
	}
}

// handleBufferTransfer stores buffer data received from the orchestrator.
func (c *GRPCWorkerClient) handleBufferTransfer(msg *distriv1.BufferTransfer) {
	if msg.Direction == distriv1.TransferDirection_TRANSFER_TO_WORKER {
		c.bufferCache[msg.BufferId] = msg.Data
		log.Printf("Buffer %s: received %d bytes from orchestrator", msg.BufferId, len(msg.Data))
	} else {
		// FROM_WORKER: orchestrator requesting data back — not implemented yet
		log.Printf("Buffer %s: FROM_WORKER request (size=%d)", msg.BufferId, msg.SizeBytes)
	}
}

func (c *GRPCWorkerClient) handleCompileRequest(req *distriv1.CompileRequest) {
	log.Printf("Compile request: %s/%s", req.ProgramId, req.KernelName)

	// For Go fallback engine, compilation is a no-op (interpreted)
	result := &distriv1.CompileResult{
		ProgramId:  req.ProgramId,
		KernelName: req.KernelName,
		Success:    true,
		BuildLog:   "Go fallback engine: no compilation needed",
	}

	msg := &distriv1.WorkerMessage{
		Payload: &distriv1.WorkerMessage_CompileResult{
			CompileResult: result,
		},
	}
	c.stream.Send(msg)
}

// ── Heartbeat ──────────────────────────────────────────

func (c *GRPCWorkerClient) heartbeatLoop() {
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-c.stopCh:
			return
		case <-ticker.C:
			if !c.connected {
				return
			}

			var cpuUtil, gpuUtil, memPct, batteryPct float64
			var charging, throttled bool

			if c.resourceMon != nil {
				snap := c.resourceMon.Snapshot()
				cpuUtil = snap.CPUPct
				gpuUtil = snap.GPUPct
				memPct = snap.MemoryPct
				batteryPct = snap.BatteryPct
				charging = snap.Charging
				throttled = snap.ThermalThrottled
			}

			hb := &distriv1.Heartbeat{
				TimestampUnixMs:  time.Now().UnixMilli(),
				CpuUtilization:   cpuUtil,
				GpuUtilization:   gpuUtil,
				MemoryUsedPct:    memPct,
				ThermalThrottled: throttled,
				BatteryPct:       batteryPct,
				Charging:         charging,
			}

			msg := &distriv1.WorkerMessage{
				Payload: &distriv1.WorkerMessage_Heartbeat{
					Heartbeat: hb,
				},
			}

			if err := c.stream.Send(msg); err != nil {
				log.Printf("Heartbeat send failed: %v", err)
				c.connected = false
				return
			}
		}
	}
}

// ── Capability reporting ───────────────────────────────

func (c *GRPCWorkerClient) sendCapabilityReport() error {
	caps := c.config.Capabilities

	report := &distriv1.CapabilityReport{
		WorkerId: c.workerID,
	}

	if caps != nil {
		report.Compute = &distriv1.ComputeCapability{
			Cpu: &distriv1.CPUInfo{
				Model:         caps.CPUModel,
				CoresPhysical: uint32(caps.CPUCores),
				FrequencyMhz:  uint32(caps.CPUFreqMHz),
			},
			Gpu: &distriv1.GPUInfo{
				Available:    caps.HasGPU,
				Vendor:       caps.GPUVendor,
				Model:        caps.GPUModel,
				VramMb:       uint64(caps.GPUVramMB),
				ComputeUnits: uint32(caps.GPUComputeUnits),
				FrequencyMhz: uint32(caps.GPUFreqMHz),
			},
			Memory: &distriv1.MemoryInfo{
				TotalRamMb:     uint64(caps.TotalRAMMB),
				AvailableRamMb: uint64(caps.AvailableRAMMB),
			},
		}
	}

	msg := &distriv1.WorkerMessage{
		Payload: &distriv1.WorkerMessage_CapabilityUpdate{
			CapabilityUpdate: report,
		},
	}

	return c.stream.Send(msg)
}

// ── Helpers ────────────────────────────────────────────

func capsArch(c *capability.HardwareInfo) string {
	if c == nil {
		return "unknown"
	}
	return c.Arch
}

func capsOS(c *capability.HardwareInfo) string {
	if c == nil {
		return "unknown"
	}
	return c.OS
}
