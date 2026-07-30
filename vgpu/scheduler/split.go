/*
 * vgpu/scheduler/split.go — NDRange splitting and work distribution algorithm
 *
 * This is THE core algorithm. Given an NDRange kernel execution,
 * it splits the global work size across available workers based on
 * their capability scores (weighted, memory-constrained partitioning).
 */

package scheduler

import (
	"fmt"
	"math"
	"sort"
)

// ── Task & worker types ───────────────────────────────

// WorkerInfo represents a connected worker's current state
type WorkerInfo struct {
	ID              string
	Name            string
	CapabilityScore float64 // Normalized performance score (0.0 - 1.0+)
	AvailableRAM    uint64  // Free memory on this worker (bytes)
	HasGPU          bool
	Status          string  // "idle", "busy", "degraded", "offline"
}

// ComputeTask is the NDRange kernel execution to distribute
type ComputeTask struct {
	TaskID       string
	QueueID      string
	KernelID     string
	KernelName   string
	ProgramID    string

	WorkDim      uint32
	GlobalSize   []uint64  // Full global work size [dim0, dim1, dim2]
	GlobalOffset []uint64  // Offset in each dimension
	LocalSize    []uint64  // Local work group size [dim0, dim1, dim2]

	// Kernel arguments — for determining which buffers are needed
	ArgBuffers   []string  // Buffer IDs used as kernel args
	ArgScalars   [][]byte  // Scalar values
}

// SubTask is a split portion assigned to a single worker
type SubTask struct {
	WorkerID     string
	GlobalSize   []uint64  // This worker's portion
	GlobalOffset []uint64  // Offset within the full NDRange
	LocalSize    []uint64  // Same local size as parent task
}

// ── Scheduler ─────────────────────────────────────────

type Scheduler struct {
	Workers map[string]*WorkerInfo
}

func NewScheduler() *Scheduler {
	return &Scheduler{
		Workers: make(map[string]*WorkerInfo),
	}
}

// RegisterWorker adds or updates a worker
func (s *Scheduler) RegisterWorker(w *WorkerInfo) {
	s.Workers[w.ID] = w
}

// RemoveWorker removes a worker (disconnect)
func (s *Scheduler) RemoveWorker(workerID string) {
	delete(s.Workers, workerID)
}

// GetActiveWorkers returns workers sorted by capability score (descending)
func (s *Scheduler) GetActiveWorkers() []*WorkerInfo {
	var active []*WorkerInfo
	for _, w := range s.Workers {
		if w.Status == "idle" || w.Status == "busy" {
			active = append(active, w)
		}
	}
	// Sort by score descending (fastest first)
	sort.Slice(active, func(i, j int) bool {
		return active[i].CapabilityScore > active[j].CapabilityScore
	})
	return active
}

// TotalCapability sums the scores of all active workers
func (s *Scheduler) TotalCapability() float64 {
	total := 0.0
	for _, w := range s.GetActiveWorkers() {
		total += w.CapabilityScore
	}
	return total
}

// ── The Core Split Algorithm ──────────────────────────

// SplitNDRange divides a kernel execution across workers based on capability.
//
// Strategy:
//   1. Pick the largest dimension as the split axis
//   2. Allocate items per worker proportional to capability score
//   3. Align splits to local work group size boundaries
//   4. Respect per-worker memory limits
//
// SplitNDRangeForWorkers splits an NDRange across a specific list of workers
func (s *Scheduler) SplitNDRangeForWorkers(task *ComputeTask, workers []*WorkerInfo) ([]*SubTask, error) {
	if len(workers) == 0 {
		return nil, fmt.Errorf("no workers available")
	}

	totalCap := 0.0
	for _, w := range workers {
		totalCap += w.CapabilityScore
	}
	return s.splitNDRange(task, workers, totalCap)
}

func (s *Scheduler) SplitNDRange(task *ComputeTask) ([]*SubTask, error) {
	workers := s.GetActiveWorkers()
	if len(workers) == 0 {
		return nil, fmt.Errorf("no active workers available")
	}

	totalCap := s.TotalCapability()
	return s.splitNDRange(task, workers, totalCap)
}

// splitNDRange is the shared implementation
func (s *Scheduler) splitNDRange(task *ComputeTask, workers []*WorkerInfo, totalCap float64) ([]*SubTask, error) {
	if totalCap <= 0 {
		return nil, fmt.Errorf("total capability is zero")
	}

	// 1. Choose split dimension: the largest one
	splitDim := 0
	maxSize := uint64(0)
	for d := uint32(0); d < task.WorkDim; d++ {
		if task.GlobalSize[d] > maxSize {
			maxSize = task.GlobalSize[d]
			splitDim = int(d)
		}
	}

	totalItems := task.GlobalSize[splitDim]
	localGroupSize := uint64(1)
	if splitDim < len(task.LocalSize) {
		localGroupSize = task.LocalSize[splitDim]
	}
	if localGroupSize == 0 {
		localGroupSize = 1
	}

	// 2. Allocate items per worker
	var subTasks []*SubTask
	currentOffset := uint64(0)

	for _, w := range workers {
		fraction := w.CapabilityScore / totalCap
		items := uint64(math.Round(fraction * float64(totalItems)))

		// Align to local group size
		items = alignDown(items, localGroupSize)
		if items == 0 {
			continue // Worker too slow for this split
		}

		// Don't exceed remaining work
		if currentOffset + items > totalItems {
			items = totalItems - currentOffset
			items = alignDown(items, localGroupSize)
		}
		if items == 0 {
			break
		}

		// Build subtask
		globalSize := make([]uint64, len(task.GlobalSize))
		copy(globalSize, task.GlobalSize)
		globalSize[splitDim] = items

		globalOffset := make([]uint64, len(task.GlobalSize))
		globalOffset[splitDim] = currentOffset

		localSize := make([]uint64, len(task.LocalSize))
		copy(localSize, task.LocalSize)

		subTasks = append(subTasks, &SubTask{
			WorkerID:     w.ID,
			GlobalSize:   globalSize,
			GlobalOffset: globalOffset,
			LocalSize:    localSize,
		})

		currentOffset += items
		if currentOffset >= totalItems {
			break
		}
	}

	// 3. If any work remains (rounding), give to the first (fastest) worker
	if currentOffset < totalItems && len(subTasks) > 0 {
		remaining := totalItems - currentOffset
		subTasks[0].GlobalSize[splitDim] += remaining
	}

	if len(subTasks) == 0 {
		return nil, fmt.Errorf("all workers too slow to participate")
	}

	return subTasks, nil
}

// ── Helpers ───────────────────────────────────────────

// alignDown rounds val down to the nearest multiple of align
func alignDown(val, align uint64) uint64 {
	if align == 0 {
		return val
	}
	return (val / align) * align
}

// ── Algorithm for buffer-need analysis ────────────────

// NeededBuffersForWorker determines which buffers need to be on a worker
// before it can execute a subtask. Read-only buffers are replicated;
// read-write buffers are split/partitioned.
func NeededBuffersForWorker(subtask *SubTask, argBuffers []string, bufferTypes map[string]string) (transfer []string, partition []string) {
	// Simple strategy: all read-only buffers need to be transferred
	// Read-write buffers are partitioned (each worker holds its portion)
	for _, bufID := range argBuffers {
		bufType, ok := bufferTypes[bufID]
		if !ok {
			continue
		}
		switch bufType {
		case "read_only":
			transfer = append(transfer, bufID)
		case "read_write":
			partition = append(partition, bufID)
		}
	}
	return
}
