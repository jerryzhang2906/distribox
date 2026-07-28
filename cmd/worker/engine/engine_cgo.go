//go:build cgo

package engine

import "log"

func newEngineWithCGO(engineType EngineType) (Engine, error) {
	ocl, err := NewOCLEngine()
	if err != nil {
		log.Printf("OpenCL GPU unavailable: %v", err)
		return nil, err
	}
	log.Printf("REAL GPU ENGINE: %s", ocl.BackendName())
	return ocl, nil
}
