/*
 * cmd/worker/engine/engine.go — Common compute engine interface
 */

package engine

import "fmt"

// ── Interface types (used by callers) ─────────────────

// Engine is the compute engine. Callers use the concrete methods
// on GoEngine directly, or use the CGO bridge when available.
type Engine interface {
	BackendName() string
	DeviceInfo() string
	Close()
	RunMicroBenchmark() float64
}

// Buffer is a device memory buffer
type Buffer interface {
	Size() uint64
}

// Program is a compiled program
type Program interface {
	ID() string
	IsBuilt() bool
}

// Kernel is a kernel function
type Kernel interface {
	Name() string
}

// ── Method implementations for Go types ───────────────

func (b *GoBuffer) Size() uint64      { return b.size }
func (p *GoProgram) ID() string       { return p.id }
func (p *GoProgram) IsBuilt() bool    { return p.built }
func (k *GoKernel) Name() string      { return k.NameVal }

// ── Engine factory ────────────────────────────────────

type EngineType int

const (
	EngineGo EngineType = iota
)

// NewEngine creates the best available compute engine.
func NewEngine(engineType EngineType) (Engine, error) {
	engine, err := newEngineWithCGO(engineType)
	if err == nil {
		return engine, nil
	}
	fmt.Printf("CGO engine unavailable (%v), using Go fallback\n", err)
	return NewGoEngine(), nil
}
