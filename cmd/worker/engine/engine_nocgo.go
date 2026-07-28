//go:build !cgo

// cmd/worker/engine/engine_nocgo.go — Non-CGO engine factory
// Provides newEngineWithCGO stub when CGO is not available.

package engine

import "fmt"

func newEngineWithCGO(engineType EngineType) (Engine, error) {
	return nil, fmt.Errorf("CGO not available (no C compiler)")
}
