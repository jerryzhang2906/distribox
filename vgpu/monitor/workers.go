/*
 * vgpu/monitor/workers.go — Worker health monitoring
 *
 * Tracks worker state, detects disconnections via heartbeat timeouts,
 * and triggers rescheduling when workers join or leave.
 */

package monitor

import (
	"log"
	"sync"
	"time"
)

const (
	HeartbeatTimeout = 10 * time.Second
	CheckInterval    = 5 * time.Second
)

// WorkerState tracks a worker's health
type WorkerState struct {
	ID            string
	Name          string
	LastHeartbeat time.Time
	Status        string // "online", "degraded", "offline"
	Connections   int    // Number of disconnections (for flapping detection)

	mu sync.RWMutex
}

// WorkerMonitor watches all connected workers
type WorkerMonitor struct {
	Workers map[string]*WorkerState
	OnWorkerLost func(workerID string) // Callback when worker times out
	OnWorkerReturn func(workerID string) // Callback when worker reconnects

	mu sync.RWMutex
}

func NewWorkerMonitor() *WorkerMonitor {
	return &WorkerMonitor{
		Workers: make(map[string]*WorkerState),
	}
}

// Register adds or updates a worker
func (m *WorkerMonitor) Register(workerID, name string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	if ws, ok := m.Workers[workerID]; ok {
		ws.mu.Lock()
		ws.LastHeartbeat = time.Now()
		ws.Status = "online"
		ws.Connections++
		ws.mu.Unlock()

		if m.OnWorkerReturn != nil {
			m.OnWorkerReturn(workerID)
		}
		log.Printf("Worker %s reconnected (connection #%d)", workerID, ws.Connections)
		return
	}

	m.Workers[workerID] = &WorkerState{
		ID:            workerID,
		Name:          name,
		LastHeartbeat: time.Now(),
		Status:        "online",
		Connections:   1,
	}
	log.Printf("Worker %s (%s) registered for monitoring", workerID, name)
}

// Heartbeat updates the last heartbeat time
func (m *WorkerMonitor) Heartbeat(workerID string) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	if ws, ok := m.Workers[workerID]; ok {
		ws.mu.Lock()
		ws.LastHeartbeat = time.Now()
		if ws.Status == "degraded" {
			ws.Status = "online"
			log.Printf("Worker %s recovered from degraded state", workerID)
		}
		ws.mu.Unlock()
	}
}

// Remove removes a worker (graceful disconnect)
func (m *WorkerMonitor) Remove(workerID string) {
	m.mu.Lock()
	defer m.mu.Unlock()

	delete(m.Workers, workerID)
	log.Printf("Worker %s removed from monitoring", workerID)
}

// Run starts the monitoring loop (should be called as a goroutine)
func (m *WorkerMonitor) Run() {
	ticker := time.NewTicker(CheckInterval)
	defer ticker.Stop()

	for range ticker.C {
		m.checkHeartbeats()
	}
}

func (m *WorkerMonitor) checkHeartbeats() {
	now := time.Now()
	var lost []string

	m.mu.RLock()
	for id, ws := range m.Workers {
		ws.mu.RLock()
		elapsed := now.Sub(ws.LastHeartbeat)
		if elapsed > HeartbeatTimeout && ws.Status == "online" {
			lost = append(lost, id)
		}
		ws.mu.RUnlock()
	}
	m.mu.RUnlock()

	for _, id := range lost {
		m.mu.RLock()
		ws := m.Workers[id]
		m.mu.RUnlock()

		if ws != nil {
			ws.mu.Lock()
			ws.Status = "degraded"
			ws.mu.Unlock()
			log.Printf("Worker %s appears to be degraded (no heartbeat for %v)", id, time.Since(ws.LastHeartbeat))

			if m.OnWorkerLost != nil {
				m.OnWorkerLost(id)
			}
		}
	}

	// Hard timeout: mark as offline after 2x heartbeat timeout
	m.mu.RLock()
	for id, ws := range m.Workers {
		ws.mu.RLock()
		elapsed := now.Sub(ws.LastHeartbeat)
		if elapsed > 2*HeartbeatTimeout && ws.Status == "degraded" {
			ws.Status = "offline"
			log.Printf("Worker %s is offline (last heartbeat %v ago)", id, elapsed)
			if m.OnWorkerLost != nil {
				m.OnWorkerLost(id)
			}
		}
		ws.mu.RUnlock()
	}
	m.mu.RUnlock()
}

// ActiveWorkers returns IDs of currently online workers
func (m *WorkerMonitor) ActiveWorkers() []string {
	var active []string
	m.mu.RLock()
	defer m.mu.RUnlock()

	for id, ws := range m.Workers {
		ws.mu.RLock()
		if ws.Status == "online" {
			active = append(active, id)
		}
		ws.mu.RUnlock()
	}
	return active
}
