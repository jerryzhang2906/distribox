/*
 * vgpu/server/grpc_server.go — gRPC server implementing generated Orchestrator service
 *
 * Implements the OrchestratorServer interface from the generated protobuf code.
 * Workers connect to register, stream control messages, and receive tasks.
 *
 * Sprint 4: Full bidirectional stream dispatch — tasks are now sent via
 * ControlChannel and results are received asynchronously.
 */

package server

import (
	"context"
	"fmt"
	"io"
	"log"
	"sync"
	"sync/atomic"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	distriv1 "github.com/distribox/pkg/protocol/distri/v1"
	"github.com/distribox/vgpu/monitor"
	"github.com/distribox/vgpu/scheduler"
)

// ── Orchestrator gRPC Service ─────────────────────────
// Implements the generated distri.v1.OrchestratorServer interface

type OrchestratorService struct {
	distriv1.UnimplementedOrchestratorServer
	sched     *scheduler.Scheduler
	workers   map[string]*WorkerSession
	workerMon *monitor.WorkerMonitor
	mu        sync.RWMutex
}

// SetWorkerMonitor wires the worker monitor for health tracking
func (s *OrchestratorService) SetWorkerMonitor(m *monitor.WorkerMonitor) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.workerMon = m
}

// WorkerSession tracks a connected worker's state and control stream
type WorkerSession struct {
	ID            string
	Hostname      string
	Arch          string
	OS            string
	HasGPU        bool
	SessionToken  string
	ConnectedAt   time.Time
	LastHeartbeat time.Time
	Tasks         map[string]struct{} // pending task IDs

	// Stream communication
	sendCh      chan *distriv1.OrchestratorMessage // outgoing messages to worker
	taskResults map[string]chan *distriv1.TaskResult // taskID → result channel
	stream      distriv1.Orchestrator_ControlChannelServer // the gRPC stream

	// Buffer data returned from workers (keyed by buffer_id)
	returnedBuffers map[string][]byte

	mu sync.RWMutex
	connected bool
}

func NewOrchestratorService(sched *scheduler.Scheduler) *OrchestratorService {
	return &OrchestratorService{
		sched:   sched,
		workers: make(map[string]*WorkerSession),
	}
}

// ── Register RPC ──────────────────────────────────────

func (s *OrchestratorService) Register(ctx context.Context, req *distriv1.RegisterRequest) (*distriv1.RegisterResponse, error) {
	if req.ProtocolVersion != "1.0" {
		return nil, status.Error(codes.InvalidArgument, "unsupported protocol version")
	}

	workerID := generateWorkerID()
	sessionToken := fmt.Sprintf("sess-%s", generateShortID())

	ws := &WorkerSession{
		ID:              workerID,
		Hostname:        req.Hostname,
		Arch:            req.Arch,
		OS:              req.Os,
		HasGPU:          req.HasGpu,
		SessionToken:    sessionToken,
		ConnectedAt:     time.Now(),
		LastHeartbeat:   time.Now(),
		Tasks:           make(map[string]struct{}),
		sendCh:          make(chan *distriv1.OrchestratorMessage, 64),
		taskResults:     make(map[string]chan *distriv1.TaskResult),
		returnedBuffers: make(map[string][]byte),
	}

	s.mu.Lock()
	s.workers[workerID] = ws
	s.mu.Unlock()

	// Register with scheduler
	s.sched.RegisterWorker(&scheduler.WorkerInfo{
		ID:     workerID,
		Name:   req.Hostname,
		Status: "idle",
	})

	// Register with health monitor
	s.mu.RLock()
	if s.workerMon != nil {
		s.workerMon.Register(workerID, req.Hostname)
	}
	s.mu.RUnlock()

	log.Printf("Worker registered: %s (%s/%s, GPU=%v, RAM=%d MB)",
		req.Hostname, req.Os, req.Arch, req.HasGpu, req.TotalRamMb)

	return &distriv1.RegisterResponse{
		WorkerId:            workerID,
		SessionToken:        sessionToken,
		OrchestratorVersion: "0.1.0",
	}, nil
}

// ── ControlChannel — bidirectional stream ─────────────

func (s *OrchestratorService) ControlChannel(stream distriv1.Orchestrator_ControlChannelServer) error {
	var workerID string
	var ws *WorkerSession

	// Send goroutine: reads from sendCh and writes to stream
	sendDone := make(chan struct{})
	go func() {
		// Wait until workerID is set before sending
		for workerID == "" {
			time.Sleep(10 * time.Millisecond)
			select {
			case <-sendDone:
				return
			default:
			}
		}

		ws = s.getWorker(workerID)
		if ws == nil {
			return
		}

		for {
			select {
			case <-sendDone:
				return
			case msg, ok := <-ws.sendCh:
				if !ok {
					return
				}
				if err := stream.Send(msg); err != nil {
					log.Printf("ControlChannel send to %s failed: %v", workerID, err)
					return
				}
			}
		}
	}()

	// Receive loop: process messages from worker
	for {
		msg, err := stream.Recv()
		if err == io.EOF {
			log.Printf("ControlChannel: worker %s closed stream", workerID)
			break
		}
		if err != nil {
			log.Printf("ControlChannel: worker %s error: %v", workerID, err)
			break
		}

		// Handle first message to establish worker identity
		if workerID == "" {
			if capReport := msg.GetCapabilityUpdate(); capReport != nil {
				workerID = capReport.WorkerId
				ws = s.getWorker(workerID)
				if ws != nil {
					ws.mu.Lock()
					ws.stream = stream
					ws.connected = true
					ws.mu.Unlock()
				}
				log.Printf("ControlChannel: worker %s stream established", workerID)
			}
		}

		if workerID != "" {
			s.updateHeartbeat(workerID)
		}

		// Dispatch message
		switch {
		case msg.GetCapabilityUpdate() != nil:
			s.handleCapabilityUpdate(workerID, msg.GetCapabilityUpdate())
		case msg.GetTaskResult() != nil:
			s.handleTaskResult(workerID, msg.GetTaskResult())
		case msg.GetHeartbeat() != nil:
			// Already updated heartbeat above
		case msg.GetCompileResult() != nil:
			s.handleCompileResult(workerID, msg.GetCompileResult())
		case msg.GetBuffer() != nil:
			s.handleWorkerBufferTransfer(workerID, msg.GetBuffer())
		}
	}

	// Cleanup
	close(sendDone)
	if ws != nil {
		ws.mu.Lock()
		ws.connected = false
		ws.stream = nil
		ws.mu.Unlock()
	}

	// Remove worker from scheduler and monitor
	if workerID != "" {
		s.sched.RemoveWorker(workerID)
		s.mu.RLock()
		if s.workerMon != nil {
			s.workerMon.Remove(workerID)
		}
		s.mu.RUnlock()
	}

	return nil
}

// ── Message handlers ──────────────────────────────────

func (s *OrchestratorService) handleCapabilityUpdate(workerID string, report *distriv1.CapabilityReport) {
	log.Printf("Capability update from %s: CPU=%s, GPU=%v, RAM=%d MB",
		workerID,
		report.GetCompute().GetCpu().GetModel(),
		report.GetCompute().GetGpu().GetAvailable(),
		report.GetCompute().GetMemory().GetAvailableRamMb())

	ws := s.getWorker(workerID)
	if ws != nil {
		caps := report.GetCompute()
		gpuInfo := caps.GetGpu()
		memInfo := caps.GetMemory()

		score := float64(caps.GetCpu().GetCoresPhysical()) * float64(caps.GetCpu().GetFrequencyMhz()) / 1e6
		if gpuInfo.GetAvailable() {
			score += float64(gpuInfo.GetVramMb()) / 1024.0 * 0.5
		}

		s.sched.RegisterWorker(&scheduler.WorkerInfo{
			ID:              workerID,
			Name:            ws.Hostname,
			CapabilityScore: score,
			AvailableRAM:    memInfo.GetAvailableRamMb() * 1024 * 1024,
			HasGPU:          gpuInfo.GetAvailable(),
			Status:          "idle",
		})
	}
}

func (s *OrchestratorService) handleTaskResult(workerID string, result *distriv1.TaskResult) {
	log.Printf("Task result from %s: task=%s status=%v regions=%d",
		workerID, result.TaskId, result.Status, len(result.OutputRegions))

	ws := s.getWorker(workerID)
	if ws != nil {
		ws.mu.Lock()
		delete(ws.Tasks, result.TaskId)
		ws.mu.Unlock()

		// Notify via result channel if someone is waiting
		ws.mu.RLock()
		ch, ok := ws.taskResults[result.TaskId]
		ws.mu.RUnlock()
		if ok && ch != nil {
			select {
			case ch <- result:
			default:
			}
		}
	}
}

func (s *OrchestratorService) handleHeartbeat(workerID string, hb *distriv1.Heartbeat) {
	_ = hb
}

func (s *OrchestratorService) handleCompileResult(workerID string, cr *distriv1.CompileResult) {
	if cr.Success {
		log.Printf("Kernel %s compiled on %s (%d bytes binary)",
			cr.KernelName, workerID, len(cr.Binary))
	} else {
		log.Printf("Kernel %s compile FAILED on %s: %s",
			cr.KernelName, workerID, cr.BuildLog)
	}
}

// handleWorkerBufferTransfer stores output buffer data returned from a worker.
func (s *OrchestratorService) handleWorkerBufferTransfer(workerID string, bt *distriv1.BufferTransfer) {
	if bt.Direction == distriv1.TransferDirection_TRANSFER_FROM_WORKER {
		ws := s.getWorker(workerID)
		if ws != nil {
			ws.mu.Lock()
			ws.returnedBuffers[bt.BufferId] = bt.Data
			ws.mu.Unlock()
			log.Printf("Worker %s returned buffer %s: %d bytes", workerID, bt.BufferId, len(bt.Data))
		}
	}
}

// GetReturnedBuffers returns and clears the returned buffer data for a worker.
func (s *OrchestratorService) GetReturnedBuffers(workerID string) map[string][]byte {
	ws := s.getWorker(workerID)
	if ws == nil {
		return nil
	}
	ws.mu.Lock()
	defer ws.mu.Unlock()
	result := ws.returnedBuffers
	ws.returnedBuffers = make(map[string][]byte)
	return result
}

// ── Worker management helpers ─────────────────────────

func (s *OrchestratorService) getWorker(workerID string) *WorkerSession {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.workers[workerID]
}

func (s *OrchestratorService) updateHeartbeat(workerID string) {
	ws := s.getWorker(workerID)
	if ws != nil {
		ws.mu.Lock()
		ws.LastHeartbeat = time.Now()
		ws.mu.Unlock()
	}
	// Also update the worker monitor
	s.mu.RLock()
	if s.workerMon != nil {
		s.workerMon.Heartbeat(workerID)
	}
	s.mu.RUnlock()
}

// ── Task assignment (FULL IMPLEMENTATION) ─────────────

// AssignTask sends a compute task to a worker via the ControlChannel stream.
// Returns immediately; the result can be waited on via WaitForTaskResult.
func (s *OrchestratorService) AssignTask(workerID string, task *distriv1.TaskAssignment) error {
	ws := s.getWorker(workerID)
	if ws == nil {
		return fmt.Errorf("worker %s not found", workerID)
	}

	ws.mu.Lock()
	defer ws.mu.Unlock()

	if !ws.connected {
		return fmt.Errorf("worker %s is not connected (no ControlChannel)", workerID)
	}

	// Track pending task
	ws.Tasks[task.TaskId] = struct{}{}

	// Send via stream
	msg := &distriv1.OrchestratorMessage{
		Payload: &distriv1.OrchestratorMessage_Task{
			Task: task,
		},
	}

	select {
	case ws.sendCh <- msg:
		log.Printf("Task %s (%s) dispatched to worker %s via ControlChannel",
			task.TaskId, task.Compute.GetKernelName(), workerID)
		return nil
	default:
		delete(ws.Tasks, task.TaskId)
		return fmt.Errorf("worker %s send buffer full", workerID)
	}
}

// WaitForTaskResult blocks until a task result is received or the context is done.
// Returns the TaskResult, or an error if the context is cancelled.
func (s *OrchestratorService) WaitForTaskResult(ctx context.Context, workerID, taskID string) (*distriv1.TaskResult, error) {
	ws := s.getWorker(workerID)
	if ws == nil {
		return nil, fmt.Errorf("worker %s not found", workerID)
	}

	ch := make(chan *distriv1.TaskResult, 1)

	ws.mu.Lock()
	ws.taskResults[taskID] = ch
	ws.mu.Unlock()

	defer func() {
		ws.mu.Lock()
		delete(ws.taskResults, taskID)
		ws.mu.Unlock()
	}()

	select {
	case result := <-ch:
		return result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// ── Compile request ───────────────────────────────────

func (s *OrchestratorService) SendCompileRequest(workerID string, req *distriv1.CompileRequest) error {
	ws := s.getWorker(workerID)
	if ws == nil {
		return fmt.Errorf("worker %s not found", workerID)
	}

	ws.mu.RLock()
	defer ws.mu.RUnlock()

	if !ws.connected {
		return fmt.Errorf("worker %s is not connected", workerID)
	}

	msg := &distriv1.OrchestratorMessage{
		Payload: &distriv1.OrchestratorMessage_Compile{
			Compile: req,
		},
	}

	select {
	case ws.sendCh <- msg:
		log.Printf("Compile %s dispatched to worker %s", req.KernelName, workerID)
		return nil
	default:
		return fmt.Errorf("worker %s send buffer full", workerID)
	}
}

// SendToWorker sends an arbitrary message to a worker via the ControlChannel.
func (s *OrchestratorService) SendToWorker(workerID string, msg *distriv1.OrchestratorMessage) error {
	ws := s.getWorker(workerID)
	if ws == nil {
		return fmt.Errorf("worker %s not found", workerID)
	}

	ws.mu.RLock()
	defer ws.mu.RUnlock()

	if !ws.connected {
		return fmt.Errorf("worker %s is not connected", workerID)
	}

	select {
	case ws.sendCh <- msg:
		return nil
	default:
		return fmt.Errorf("worker %s send buffer full", workerID)
	}
}

// ── Helpers ───────────────────────────────────────────

var idCounter atomic.Uint64

func generateWorkerID() string {
	return fmt.Sprintf("w-%s", generateShortID())
}

func generateShortID() string {
	seq := idCounter.Add(1)
	return fmt.Sprintf("%x-%04x", time.Now().UnixNano(), seq%0xFFFF)
}
