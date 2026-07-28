/*
 * vgpu/server/api.go — HTTP API for CLI and dashboard
 *
 * Endpoints:
 *   GET  /api/v1/status   — Virtual GPU status summary
 *   GET  /api/v1/workers   — Connected worker list
 *   POST /api/v1/device    — Configure virtual device specs
 */

package server

import (
	"encoding/json"
	"net/http"

	"github.com/distribox/vgpu/mem"
	"github.com/distribox/vgpu/scheduler"
	"github.com/distribox/vgpu/monitor"
)

type APIHandler struct {
	vram   *mem.VRAMManager
	sched  *scheduler.Scheduler
	workerMon *monitor.WorkerMonitor
}

func NewAPIHandler(vram *mem.VRAMManager, sched *scheduler.Scheduler,
	workerMon *monitor.WorkerMonitor) *APIHandler {
	return &APIHandler{vram: vram, sched: sched, workerMon: workerMon}
}

// HandleStatus returns the current virtual GPU status
func (h *APIHandler) HandleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", 405)
		return
	}

	spec := h.vram.GetSpec()
	total, used, buffers := h.vram.Stats()
	workers := h.sched.GetActiveWorkers()

	type WorkerSummary struct {
		ID    string  `json:"id"`
		Name  string  `json:"name"`
		Score float64 `json:"score"`
		Status string `json:"status"`
	}

	var ws []WorkerSummary
	for _, w := range workers {
		ws = append(ws, WorkerSummary{
			ID: w.ID, Name: w.Name, Score: w.CapabilityScore, Status: w.Status,
		})
	}

	resp := map[string]interface{}{
		"device": map[string]interface{}{
			"name":           spec.Name,
			"vram_total_mb":  total / (1024 * 1024),
			"vram_used_mb":   used / (1024 * 1024),
			"compute_units":  spec.ComputeUnits,
			"clock_mhz":      spec.MaxClockMHz,
			"buffer_count":   buffers,
		},
		"workers": ws,
		"active_workers": len(workers),
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// HandleWorkers returns detailed worker info
func (h *APIHandler) HandleWorkers(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", 405)
		return
	}

	type WorkerDetail struct {
		ID              string  `json:"id"`
		Name            string  `json:"name"`
		CapabilityScore float64 `json:"capability_score"`
		AvailableRAMMB  int     `json:"available_ram_mb"`
		HasGPU          bool    `json:"has_gpu"`
		Status          string  `json:"status"`
	}

	var workers []WorkerDetail
	for _, w := range h.sched.Workers {
		workers = append(workers, WorkerDetail{
			ID: w.ID, Name: w.Name,
			CapabilityScore: w.CapabilityScore,
			AvailableRAMMB:  int(w.AvailableRAM / (1024 * 1024)),
			HasGPU:          w.HasGPU,
			Status:          w.Status,
		})
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"workers": workers,
		"count":   len(workers),
	})
}

// HandleDevice configures virtual device specs
func (h *APIHandler) HandleDevice(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		spec := h.vram.GetSpec()
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(spec)

	case http.MethodPost:
		var spec mem.VirtualDeviceSpec
		if err := json.NewDecoder(r.Body).Decode(&spec); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		h.vram.UpdateSpec(spec)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]string{"status": "ok"})

	default:
		http.Error(w, "method not allowed", 405)
	}
}
