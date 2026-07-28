/*
 * vgpu/mem/vram.go — Virtual VRAM manager
 *
 * Manages "virtual GPU memory" — actually host RAM that is tracked
 * as if it were GPU VRAM. Buffers are allocated here and distributed
 * to workers as needed.
 */

package mem

import (
	"fmt"
	"sync"
)

// ── Buffer types ──────────────────────────────────────

// BufferType classifies how a buffer is used (affects distribution strategy)
type BufferType int

const (
	BufferReadOnly  BufferType = iota // Model weights: write once, replicate to workers
	BufferReadWrite                   // Activations: partitioned, merge on read back
	BufferTemporary                   // Small/temporary: sent inline with kernel args
)

// Buffer represents a chunk of virtual VRAM
type Buffer struct {
	ID       string
	Size     uint64
	Flags    uint32        // cl_mem_flags
	BufType  BufferType
	HostPtr  []byte        // Local staging copy (in host RAM)
	Dirty    bool          // True if local data differs from worker copies

	// Worker locations: which workers have a copy of this buffer
	Workers  map[string]bool // worker_id → has_copy

	mu       sync.RWMutex
}

// ── VRAM Manager ──────────────────────────────────────

// VirtualDeviceSpec describes the virtual GPU presented to applications
type VirtualDeviceSpec struct {
	Name          string
	VRAMTotal     uint64
	VRAMUsed      uint64
	ComputeUnits  uint32
	MaxClockMHz   uint32
	MaxWorkGroupSize uint64
	MaxWorkItemSizes [3]uint64
}

// VRAMManager tracks all buffer allocations in virtual VRAM
type VRAMManager struct {
	Spec     VirtualDeviceSpec
	Buffers  map[string]*Buffer // buffer_id → Buffer
	TotalAllocated uint64

	mu       sync.RWMutex
}

// NewVRAMManager creates a VRAM manager with default specs
func NewVRAMManager() *VRAMManager {
	return &VRAMManager{
		Spec: VirtualDeviceSpec{
			Name:          "DistriBox Virtual GPU",
			VRAMTotal:     8 * 1024 * 1024 * 1024, // 8 GB default
			VRAMUsed:      0,
			ComputeUnits:  2048,
			MaxClockMHz:   1500,
			MaxWorkGroupSize: 256,
			MaxWorkItemSizes: [3]uint64{1024, 1024, 64},
		},
		Buffers: make(map[string]*Buffer),
	}
}

// UpdateSpec updates the virtual device specs (e.g., from user config)
func (v *VRAMManager) UpdateSpec(spec VirtualDeviceSpec) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.Spec = spec
}

// GetSpec returns a copy of the current device specs
func (v *VRAMManager) GetSpec() VirtualDeviceSpec {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.Spec
}

// Allocate creates a new buffer in virtual VRAM
func (v *VRAMManager) Allocate(id string, size uint64, flags uint32, bufType BufferType) (*Buffer, error) {
	v.mu.Lock()
	defer v.mu.Unlock()

	// Check VRAM limit
	if v.TotalAllocated + size > v.Spec.VRAMTotal {
		return nil, fmt.Errorf("VRAM full: requested %d, available %d",
			size, v.Spec.VRAMTotal - v.TotalAllocated)
	}

	buf := &Buffer{
		ID:      id,
		Size:    size,
		Flags:   flags,
		BufType: bufType,
		HostPtr: make([]byte, size),
		Workers: make(map[string]bool),
	}

	v.Buffers[id] = buf
	v.TotalAllocated += size
	v.Spec.VRAMUsed = v.TotalAllocated

	return buf, nil
}

// Release frees a buffer from virtual VRAM
func (v *VRAMManager) Release(id string) error {
	v.mu.Lock()
	defer v.mu.Unlock()

	buf, ok := v.Buffers[id]
	if !ok {
		return fmt.Errorf("buffer %s not found", id)
	}

	v.TotalAllocated -= buf.Size
	v.Spec.VRAMUsed = v.TotalAllocated
	delete(v.Buffers, id)
	return nil
}

// Get returns a buffer by ID
func (v *VRAMManager) Get(id string) (*Buffer, error) {
	v.mu.RLock()
	defer v.mu.RUnlock()

	buf, ok := v.Buffers[id]
	if !ok {
		return nil, fmt.Errorf("buffer %s not found", id)
	}
	return buf, nil
}

// Write writes data to a buffer's local staging copy
func (v *VRAMManager) Write(id string, offset uint64, data []byte) error {
	buf, err := v.Get(id)
	if err != nil {
		return err
	}

	buf.mu.Lock()
	defer buf.mu.Unlock()

	if offset + uint64(len(data)) > buf.Size {
		return fmt.Errorf("write out of bounds: offset %d + %d > size %d",
			offset, len(data), buf.Size)
	}

	copy(buf.HostPtr[offset:], data)
	buf.Dirty = true
	return nil
}

// Read reads data from a buffer's local staging copy
func (v *VRAMManager) Read(id string, offset uint64, size uint64) ([]byte, error) {
	buf, err := v.Get(id)
	if err != nil {
		return nil, err
	}

	buf.mu.RLock()
	defer buf.mu.RUnlock()

	// size=0 means "read all remaining data"
	if size == 0 {
		size = buf.Size - offset
	}
	if offset + size > buf.Size {
		return nil, fmt.Errorf("read out of bounds: offset %d + %d > size %d",
			offset, size, buf.Size)
	}

	result := make([]byte, size)
	copy(result, buf.HostPtr[offset:offset+size])
	return result, nil
}

// MarkWorkerHasBuffer records that a worker has a copy of this buffer
func (v *VRAMManager) MarkWorkerHasBuffer(bufferID, workerID string) error {
	buf, err := v.Get(bufferID)
	if err != nil {
		return err
	}

	buf.mu.Lock()
	defer buf.mu.Unlock()
	buf.Workers[workerID] = true
	return nil
}

// WorkerHasBuffer checks if a worker already has a buffer
func (v *VRAMManager) WorkerHasBuffer(bufferID, workerID string) bool {
	buf, err := v.Get(bufferID)
	if err != nil {
		return false
	}

	buf.mu.RLock()
	defer buf.mu.RUnlock()
	return buf.Workers[workerID]
}

// GetDirtyBuffers returns all buffer IDs that need sync to workers
func (v *VRAMManager) GetDirtyBuffers() []string {
	v.mu.RLock()
	defer v.mu.RUnlock()

	var dirty []string
	for id, buf := range v.Buffers {
		buf.mu.RLock()
		if buf.Dirty {
			dirty = append(dirty, id)
		}
		buf.mu.RUnlock()
	}
	return dirty
}

// Stats returns memory usage statistics
func (v *VRAMManager) Stats() (total, used, buffers int) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return int(v.Spec.VRAMTotal), int(v.TotalAllocated), len(v.Buffers)
}
