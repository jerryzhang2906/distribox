/*
 * vgpu/mem/kvcache.go — KV Cache for LLM inference
 *
 * Implements key-value caching for autoregressive transformer inference.
 * Each transformer layer has a pair of caches (K, V) storing past keys/values.
 *
 * Memory layout per layer:
 *   K cache: float32[num_heads * max_seq_len * head_dim]
 *   V cache: float32[num_heads * max_seq_len * head_dim]
 *
 * This avoids recomputing K,V for all previous tokens on each step,
 * reducing compute from O(n²) to O(n) for generation.
 */

package mem

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync"
)

// KVCacheConfig describes the cache dimensions
type KVCacheConfig struct {
	MaxSeqLen int // Maximum sequence length (context window)
	NumLayers int // Number of transformer layers
	NumHeads  int // Number of attention heads (for GQA: num_kv_heads)
	HeadDim   int // Dimension of each head
}

// KVCache stores past keys and values for fast autoregressive generation.
type KVCache struct {
	cfg KVCacheConfig

	// Flat byte storage: layer → head → (seq_pos, head_dim)
	kCache [][][]byte // [layer][head]key_data
	vCache [][][]byte // [layer][head]value_data

	// Current sequence length (how many tokens cached)
	seqLen int

	// Per-head byte size: head_dim * 4 (float32)
	headBytes int

	mu sync.RWMutex
}

// NewKVCache creates a pre-allocated KV cache for the given model config.
func NewKVCache(cfg KVCacheConfig) *KVCache {
	headBytes := cfg.HeadDim * 4 // float32

	kvc := &KVCache{
		cfg:       cfg,
		kCache:    make([][][]byte, cfg.NumLayers),
		vCache:    make([][][]byte, cfg.NumLayers),
		headBytes: headBytes,
	}

	for l := 0; l < cfg.NumLayers; l++ {
		kvc.kCache[l] = make([][]byte, cfg.NumHeads)
		kvc.vCache[l] = make([][]byte, cfg.NumHeads)
		for h := 0; h < cfg.NumHeads; h++ {
			size := cfg.MaxSeqLen * headBytes
			kvc.kCache[l][h] = make([]byte, size)
			kvc.vCache[l][h] = make([]byte, size)
		}
	}

	return kvc
}

// Store stores key and value vectors at a given position.
// pos is 0-indexed position in the sequence.
func (kvc *KVCache) Store(layer, head, pos int, key, value []float32) {
	if layer < 0 || layer >= kvc.cfg.NumLayers {
		return
	}
	if head < 0 || head >= kvc.cfg.NumHeads {
		return
	}
	if pos < 0 || pos >= kvc.cfg.MaxSeqLen {
		return
	}

	offset := pos * kvc.headBytes

	kvc.mu.Lock()
	floatsToBytes(key, kvc.kCache[layer][head][offset:])
	floatsToBytes(value, kvc.vCache[layer][head][offset:])
	if pos+1 > kvc.seqLen {
		kvc.seqLen = pos + 1
	}
	kvc.mu.Unlock()
}

// GetKeys returns all cached keys for a layer+head up to the current sequence length.
// Returns nil if nothing cached.
func (kvc *KVCache) GetKeys(layer, head int) []float32 {
	if layer < 0 || layer >= kvc.cfg.NumLayers {
		return nil
	}
	if head < 0 || head >= kvc.cfg.NumHeads {
		return nil
	}

	kvc.mu.RLock()
	length := kvc.seqLen
	kvc.mu.RUnlock()

	if length == 0 {
		return nil
	}

	size := length * kvc.cfg.HeadDim
	result := make([]float32, size)
	kvc.mu.RLock()
	bytesToFloats(kvc.kCache[layer][head][:length*kvc.headBytes], result)
	kvc.mu.RUnlock()
	return result
}

// GetValues returns all cached values for a layer+head.
func (kvc *KVCache) GetValues(layer, head int) []float32 {
	if layer < 0 || layer >= kvc.cfg.NumLayers {
		return nil
	}
	if head < 0 || head >= kvc.cfg.NumHeads {
		return nil
	}

	kvc.mu.RLock()
	length := kvc.seqLen
	kvc.mu.RUnlock()

	if length == 0 {
		return nil
	}

	size := length * kvc.cfg.HeadDim
	result := make([]float32, size)
	kvc.mu.RLock()
	bytesToFloats(kvc.vCache[layer][head][:length*kvc.headBytes], result)
	kvc.mu.RUnlock()
	return result
}

// SeqLen returns the current cached sequence length.
func (kvc *KVCache) SeqLen() int {
	kvc.mu.RLock()
	defer kvc.mu.RUnlock()
	return kvc.seqLen
}

// Clear resets the cache (e.g., for a new sequence).
func (kvc *KVCache) Clear() {
	kvc.mu.Lock()
	defer kvc.mu.Unlock()
	kvc.seqLen = 0
	for l := 0; l < kvc.cfg.NumLayers; l++ {
		for h := 0; h < kvc.cfg.NumHeads; h++ {
			for i := range kvc.kCache[l][h] {
				kvc.kCache[l][h][i] = 0
				kvc.vCache[l][h][i] = 0
			}
		}
	}
}

// TotalBytes returns the total memory allocated for the cache.
func (kvc *KVCache) TotalBytes() int64 {
	return int64(kvc.cfg.NumLayers) * int64(kvc.cfg.NumHeads) * int64(kvc.cfg.MaxSeqLen) * int64(kvc.headBytes) * 2
}

// Stats returns a human-readable summary.
func (kvc *KVCache) Stats() string {
	return fmt.Sprintf("KV Cache: %d layers × %d heads × %d tokens × %d dim = %s",
		kvc.cfg.NumLayers, kvc.cfg.NumHeads, kvc.cfg.MaxSeqLen, kvc.cfg.HeadDim,
		formatBytes(kvc.TotalBytes()))
}

// ── Float32 ↔ Bytes conversion ─────────────────────────

func floatsToBytes(src []float32, dst []byte) {
	for i, v := range src {
		if i*4+3 < len(dst) {
			bits := math.Float32bits(v)
			binary.LittleEndian.PutUint32(dst[i*4:], bits)
		}
	}
}

func bytesToFloats(src []byte, dst []float32) {
	for i := range dst {
		if i*4+3 < len(src) {
			bits := binary.LittleEndian.Uint32(src[i*4:])
			dst[i] = math.Float32frombits(bits)
		}
	}
}

func formatBytes(b int64) string {
	if b >= 1073741824 {
		return fmt.Sprintf("%.1f GB", float64(b)/1073741824.0)
	}
	if b >= 1048576 {
		return fmt.Sprintf("%.1f MB", float64(b)/1048576.0)
	}
	if b >= 1024 {
		return fmt.Sprintf("%.1f KB", float64(b)/1024.0)
	}
	return fmt.Sprintf("%d B", b)
}
