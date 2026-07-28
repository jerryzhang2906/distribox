/*
 * vgpu/queue/cmdqueue.go — Command queue and event management
 *
 * Maps OpenCL command queues to internal tracking.
 * Each queue has an ordered list of pending/completed commands.
 */

package queue

import (
	"sync"
	"time"
)

// CommandType mirrors OpenCL command types
type CommandType int

const (
	CmdNDRangeKernel CommandType = iota
	CmdReadBuffer
	CmdWriteBuffer
	CmdCopyBuffer
	CmdFillBuffer
)

// CommandStatus tracks the lifecycle of a command
type CommandStatus int

const (
	StatusQueued    CommandStatus = iota
	StatusSubmitted
	StatusRunning
	StatusComplete
	StatusError
)

// Command represents one enqueued operation
type Command struct {
	ID        string
	QueueID   string
	CmdType   CommandType
	Status    CommandStatus
	TaskID    string       // For NDRange: the compute task ID
	CreatedAt time.Time

	// Events this command waits for
	WaitEvents []string
	// Event created by this command
	OutEventID string

	mu sync.RWMutex
}

// CommandQueue maps one cl_command_queue to its pending commands
type CommandQueue struct {
	ID       string
	Commands []*Command // Ordered list
	signal   chan struct{} // closed when queue becomes empty; recreated on new work

	mu sync.RWMutex
}

// CommandQueueManager manages all active command queues
type CommandQueueManager struct {
	Queues map[string]*CommandQueue
	mu     sync.RWMutex
}

func NewCommandQueueManager() *CommandQueueManager {
	return &CommandQueueManager{
		Queues: make(map[string]*CommandQueue),
	}
}

// GetOrCreate returns an existing queue or creates one
func (m *CommandQueueManager) GetOrCreate(queueID string) *CommandQueue {
	m.mu.Lock()
	defer m.mu.Unlock()

	if q, ok := m.Queues[queueID]; ok {
		return q
	}

	q := &CommandQueue{
		ID:     queueID,
		signal: make(chan struct{}),
	}
	close(q.signal) // initially empty → signaled
	m.Queues[queueID] = q
	return q
}

// Enqueue adds a command to a queue
func (m *CommandQueueManager) Enqueue(cmd *Command) {
	q := m.GetOrCreate(cmd.QueueID)
	q.mu.Lock()
	defer q.mu.Unlock()

	cmd.Status = StatusQueued
	cmd.CreatedAt = time.Now()
	q.Commands = append(q.Commands, cmd)

	// Reset signal: new work means queue is no longer empty
	select {
	case <-q.signal:
		// already closed, need new one
		q.signal = make(chan struct{})
	default:
		// not closed yet, keep existing
	}
}

// UpdateStatus changes a command's status
func (m *CommandQueueManager) UpdateStatus(cmdID string, status CommandStatus) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	for _, q := range m.Queues {
		q.mu.RLock()
		for _, cmd := range q.Commands {
			if cmd.ID == cmdID {
				cmd.mu.Lock()
				cmd.Status = status
				cmd.mu.Unlock()
				q.mu.RUnlock()
				return
			}
		}
		q.mu.RUnlock()
	}
}

// WaitForQueue blocks until all commands on the queue have completed.
func (m *CommandQueueManager) WaitForQueue(queueID string) {
	q := m.GetOrCreate(queueID)
	q.mu.RLock()
	sig := q.signal
	q.mu.RUnlock()
	<-sig // blocks until closed (queue empty)
}

// MarkComplete updates a command's status to complete and signals waiters if queue is empty.
func (m *CommandQueueManager) MarkComplete(queueID string, cmdID string) {
	q, ok := m.Queues[queueID]
	if !ok {
		return
	}
	q.mu.Lock()
	defer q.mu.Unlock()

	for _, cmd := range q.Commands {
		if cmd.ID == cmdID {
			cmd.mu.Lock()
			cmd.Status = StatusComplete
			cmd.mu.Unlock()
		}
	}

	// If queue is now empty, signal waiters
	allDone := true
	for _, cmd := range q.Commands {
		cmd.mu.RLock()
		if cmd.Status != StatusComplete && cmd.Status != StatusError {
			allDone = false
		}
		cmd.mu.RUnlock()
	}
	if allDone {
		select {
		case <-q.signal:
			// already closed
		default:
			close(q.signal)
		}
	}
}

// PendingCount returns the number of non-completed commands in a queue
func (m *CommandQueueManager) PendingCount(queueID string) int {
	q, ok := m.Queues[queueID]
	if !ok {
		return 0
	}

	q.mu.RLock()
	defer q.mu.RUnlock()

	count := 0
	for _, cmd := range q.Commands {
		cmd.mu.RLock()
		if cmd.Status != StatusComplete && cmd.Status != StatusError {
			count++
		}
		cmd.mu.RUnlock()
	}
	return count
}
