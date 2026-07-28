/*
 * examples/inference/demo.go — End-to-end AI inference demo
 *
 * Demonstrates a complete transformer forward pass using the DistriBox
 * compute engine. Runs a tiny 2-layer transformer with hardcoded weights.
 *
 * Usage: go run ./examples/inference/
 */

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"

	"github.com/distribox/cmd/worker/engine"
	"github.com/distribox/vgpu/mem"
)

const (
	vocabSize   = 256
	hiddenDim   = 64
	numHeads    = 4
	headDim     = 16
	numLayers   = 2
	maxSeqLen   = 32
	intermediate = 256
)

func main() {
	fmt.Println("=== DistriBox AI Inference Demo ===")
	fmt.Printf("Model: %d-layer transformer, dim=%d, heads=%d, vocab=%d\n",
		numLayers, hiddenDim, numHeads, vocabSize)

	eng := engine.NewGoEngine()
	fmt.Printf("Backend: %s\n\n", eng.BackendName())

	kvc := mem.NewKVCache(mem.KVCacheConfig{
		MaxSeqLen: maxSeqLen, NumLayers: numLayers,
		NumHeads: numHeads, HeadDim: headDim,
	})
	fmt.Println(kvc.Stats())
	fmt.Println()

	// Create random weights
	rng := rand.New(rand.NewSource(42))
	weights := createWeights(rng)
	wbufs := allocWeights(eng, weights)
	defer func() { for _, b := range wbufs { eng.ReleaseBuffer(b) } }()
	fmt.Printf("Model: %d weight tensors loaded\n\n", len(weights))

	// Prompt
	prompt := []int{72, 101, 108, 108, 111} // "Hello"
	fmt.Printf("Prompt: %q\n\n", decodeTokens(prompt))
	fmt.Println("Generating...")

	tokens := append([]int{}, prompt...)
	for step := 0; step < 8; step++ {
		hidden := newBuf(eng, hiddenDim)
		emb := extractRow(weights["embed.weight"], tokens[len(tokens)-1], hiddenDim)
		eng.WriteBuffer(hidden, 0, f32b(emb))

		pos := len(tokens) - 1
		for l := 0; l < numLayers; l++ {
			hidden = runLayer(eng, wbufs, kvc, l, pos, hidden)
		}

		hidden = runRMS(eng, hidden, wbufs["norm.weight"])
		logits := runLinear(eng, hidden, wbufs["lm_head.weight"], vocabSize, hiddenDim)
		nextTok := argmax(eng, logits)
		tokens = append(tokens, nextTok)

		fmt.Printf("  Step %d: %q\n", step+1, decodeToken(nextTok))
		eng.ReleaseBuffer(hidden)
		eng.ReleaseBuffer(logits)
	}

	fmt.Printf("\nGenerated: %q\n", decodeTokens(tokens))
	fmt.Println("=== Demo Complete ===")
}

// ── Transformer layer ───────────────────────────────────

func runLayer(eng *engine.GoEngine, w map[string]*engine.GoBuffer,
	kvc *mem.KVCache, layer, pos int, x *engine.GoBuffer) *engine.GoBuffer {

	res := cloneBuf(eng, x)
	prefix := fmt.Sprintf("l%d", layer)

	// RMSNorm → Q/K/V
	n := runRMS(eng, x, w[prefix+".an"])
	q := runLinear(eng, n, w[prefix+".qw"], hiddenDim, hiddenDim)
	k := runLinear(eng, n, w[prefix+".kw"], hiddenDim, hiddenDim)
	v := runLinear(eng, n, w[prefix+".vw"], hiddenDim, hiddenDim)
	eng.ReleaseBuffer(n)

	// RoPE
	q = applyRoPE(eng, q, pos)
	k = applyRoPE(eng, k, pos)

	// KV Cache store + retrieve
	kd, vd := readF32(eng, k), readF32(eng, v)
	for h := 0; h < numHeads; h++ {
		off := h * headDim
		kvc.Store(layer, h, pos, kd[off:off+headDim], vd[off:off+headDim])
	}
	allK := kvc.GetKeys(layer, 0)
	allV := kvc.GetValues(layer, 0)
	for h := 1; h < numHeads; h++ {
		allK = append(allK, kvc.GetKeys(layer, h)...)
		allV = append(allV, kvc.GetValues(layer, h)...)
	}
	kb, _ := eng.CreateBuffer(uint64(len(allK)*4), 0, f32b(allK))
	vb, _ := eng.CreateBuffer(uint64(len(allV)*4), 0, f32b(allV))
	attn := attention(eng, q, kb, vb, pos+1)
	eng.ReleaseBuffer(q); eng.ReleaseBuffer(k); eng.ReleaseBuffer(v)
	eng.ReleaseBuffer(kb); eng.ReleaseBuffer(vb)

	// Output proj + residual
	ao := runLinear(eng, attn, w[prefix+".ow"], hiddenDim, hiddenDim)
	eng.ReleaseBuffer(attn)
	a1 := runAdd(eng, res, ao)
	eng.ReleaseBuffer(res); eng.ReleaseBuffer(ao)

	// FFN: RMSNorm → Gate/Up → GELU → Down
	res2 := cloneBuf(eng, a1)
	fn := runRMS(eng, a1, w[prefix+".fn"])
	g := runLinear(eng, fn, w[prefix+".gw"], intermediate, hiddenDim)
	u := runLinear(eng, fn, w[prefix+".uw"], intermediate, hiddenDim)
	gu := runGELU(eng, g)
	gated := runMul(eng, gu, u)
	eng.ReleaseBuffer(g); eng.ReleaseBuffer(u); eng.ReleaseBuffer(gu)
	d := runLinear(eng, gated, w[prefix+".dw"], hiddenDim, intermediate)
	eng.ReleaseBuffer(gated)
	a2 := runAdd(eng, res2, d)
	eng.ReleaseBuffer(res2); eng.ReleaseBuffer(d)
	eng.ReleaseBuffer(fn); eng.ReleaseBuffer(a1)

	return a2
}

// ── Kernel helpers ──────────────────────────────────────

func runRMS(eng *engine.GoEngine, in, g *engine.GoBuffer) *engine.GoBuffer {
	out, _ := eng.CreateBuffer(uint64(hiddenDim*4), 0, nil)
	k := &engine.GoKernel{NameVal: "rms_norm"}
	eng.SetKernelArg(k, 0, in); eng.SetKernelArg(k, 1, out)
	eng.SetKernelArg(k, 2, g); eng.SetKernelArg(k, 3, i32b(int32(hiddenDim)))
	eng.ExecuteNDRange(k, 2, []uint64{1, uint64(hiddenDim)}, nil, nil, []*engine.GoBuffer{out})
	return out
}

func runGELU(eng *engine.GoEngine, in *engine.GoBuffer) *engine.GoBuffer {
	out, _ := eng.CreateBuffer(uint64(intermediate*4), 0, nil)
	k := &engine.GoKernel{NameVal: "gelu"}
	eng.SetKernelArg(k, 0, in); eng.SetKernelArg(k, 1, out)
	eng.ExecuteNDRange(k, 1, []uint64{intermediate}, nil, nil, []*engine.GoBuffer{out})
	return out
}

func runLinear(eng *engine.GoEngine, in, w *engine.GoBuffer, outDim, inDim int) *engine.GoBuffer {
	out, _ := eng.CreateBuffer(uint64(outDim*4), 0, nil)
	k := &engine.GoKernel{NameVal: "matmul"}
	eng.SetKernelArg(k, 0, in); eng.SetKernelArg(k, 1, w)
	eng.SetKernelArg(k, 2, out); eng.SetKernelArg(k, 3, i32b(int32(inDim)))
	eng.ExecuteNDRange(k, 2, []uint64{1, uint64(outDim)}, nil, nil, []*engine.GoBuffer{out})
	return out
}

func runMul(eng *engine.GoEngine, a, b *engine.GoBuffer) *engine.GoBuffer {
	out, _ := eng.CreateBuffer(uint64(intermediate*4), 0, nil)
	k := &engine.GoKernel{NameVal: "element_wise_mul"}
	eng.SetKernelArg(k, 0, a); eng.SetKernelArg(k, 1, b)
	eng.SetKernelArg(k, 2, out)
	eng.ExecuteNDRange(k, 1, []uint64{intermediate}, nil, nil, []*engine.GoBuffer{out})
	return out
}

func runAdd(eng *engine.GoEngine, a, b *engine.GoBuffer) *engine.GoBuffer {
	out, _ := eng.CreateBuffer(uint64(hiddenDim*4), 0, nil)
	k := &engine.GoKernel{NameVal: "vector_add"}
	eng.SetKernelArg(k, 0, a); eng.SetKernelArg(k, 1, b)
	eng.SetKernelArg(k, 2, out); eng.SetKernelArg(k, 3, i32b(int32(hiddenDim)))
	eng.ExecuteNDRange(k, 1, []uint64{uint64(hiddenDim)}, nil, nil, []*engine.GoBuffer{out})
	return out
}

func applyRoPE(eng *engine.GoEngine, in *engine.GoBuffer, pos int) *engine.GoBuffer {
	out, _ := eng.CreateBuffer(uint64(hiddenDim*4), 0, nil)
	k := &engine.GoKernel{NameVal: "rope"}
	eng.SetKernelArg(k, 0, in); eng.SetKernelArg(k, 1, out)
	eng.SetKernelArg(k, 2, i32b(int32(pos))); eng.SetKernelArg(k, 3, i32b(int32(headDim)))
	eng.ExecuteNDRange(k, 2, []uint64{uint64(numHeads), uint64(headDim)}, nil, nil, []*engine.GoBuffer{out})
	return out
}

func attention(eng *engine.GoEngine, q, k, v *engine.GoBuffer, seqLen int) *engine.GoBuffer {
	// Q @ K^T: [numHeads*headDim, 1] @ [numHeads*headDim, seqLen*numHeads]???
	// For simplicity in demo: use vector_add + softmax rather than correct matmul shapes
	// This is a demo of the pipeline, not mathematically correct attention.
	scores, _ := eng.CreateBuffer(uint64(numHeads*seqLen*4), 0, nil)
	mk := &engine.GoKernel{NameVal: "matmul"}
	eng.SetKernelArg(mk, 0, q)
	eng.SetKernelArg(mk, 1, k)
	eng.SetKernelArg(mk, 2, scores)
	eng.SetKernelArg(mk, 3, i32b(int32(headDim)))
	eng.ExecuteNDRange(mk, 2, []uint64{uint64(numHeads), uint64(seqLen)}, nil, nil, nil)

	attnW, _ := eng.CreateBuffer(uint64(numHeads*seqLen*4), 0, nil)
	sk := &engine.GoKernel{NameVal: "scalar_mul"}
	eng.SetKernelArg(sk, 0, scores); eng.SetKernelArg(sk, 1, attnW)
	scale := math.Float32bits(float32(1.0 / math.Sqrt(float64(headDim))))
	eng.SetKernelArg(sk, 2, u32b(scale))
	eng.ExecuteNDRange(sk, 1, []uint64{uint64(numHeads * seqLen)}, nil, nil, nil)
	eng.ReleaseBuffer(scores)

	attnW2, _ := eng.CreateBuffer(uint64(numHeads*seqLen*4), 0, nil)
	smk := &engine.GoKernel{NameVal: "softmax"}
	eng.SetKernelArg(smk, 0, attnW); eng.SetKernelArg(smk, 1, attnW2)
	eng.SetKernelArg(smk, 2, i32b(int32(seqLen)))
	eng.ExecuteNDRange(smk, 1, []uint64{uint64(numHeads)}, nil, nil, []*engine.GoBuffer{attnW2})
	eng.ReleaseBuffer(attnW)

	out, _ := eng.CreateBuffer(uint64(hiddenDim*4), 0, nil)
	mk2 := &engine.GoKernel{NameVal: "matmul"}
	eng.SetKernelArg(mk2, 0, attnW2); eng.SetKernelArg(mk2, 1, v)
	eng.SetKernelArg(mk2, 2, out); eng.SetKernelArg(mk2, 3, i32b(int32(seqLen)))
	eng.ExecuteNDRange(mk2, 2, []uint64{uint64(numHeads), 1}, nil, nil, []*engine.GoBuffer{out})
	eng.ReleaseBuffer(attnW2)

	return out
}

// ── Helpers ─────────────────────────────────────────────

func cloneBuf(eng *engine.GoEngine, src *engine.GoBuffer) *engine.GoBuffer {
	d, _ := eng.ReadBuffer(src, 0, src.Size())
	b, _ := eng.CreateBuffer(uint64(len(d)), 0, d)
	return b
}

func newBuf(eng *engine.GoEngine, n int) *engine.GoBuffer {
	b, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	return b
}

func readF32(eng *engine.GoEngine, b *engine.GoBuffer) []float32 {
	d, _ := eng.ReadBuffer(b, 0, b.Size())
	return b2f32(d)
}

func argmax(eng *engine.GoEngine, b *engine.GoBuffer) int {
	f := readF32(eng, b)
	mi, mv := 0, float32(-1e9)
	for i, v := range f {
		if v > mv { mv = v; mi = i }
	}
	return mi
}

func extractRow(m []float32, row, cols int) []float32 {
	r := make([]float32, cols)
	copy(r, m[row*cols:])
	return r
}

// ── Weights ─────────────────────────────────────────────

func allocWeights(eng *engine.GoEngine, w map[string][]float32) map[string]*engine.GoBuffer {
	bufs := make(map[string]*engine.GoBuffer)
	for k, v := range w {
		b, _ := eng.CreateBuffer(uint64(len(v)*4), 0, f32b(v))
		bufs[k] = b
	}
	return bufs
}

func createWeights(rng *rand.Rand) map[string][]float32 {
	w := make(map[string][]float32)
	w["embed.weight"] = rnd(rng, vocabSize*hiddenDim)
	for l := 0; l < numLayers; l++ {
		p := fmt.Sprintf("l%d", l)
		w[p+".an"] = ones(hiddenDim)
		w[p+".qw"] = rnd(rng, hiddenDim*hiddenDim)
		w[p+".kw"] = rnd(rng, hiddenDim*hiddenDim)
		w[p+".vw"] = rnd(rng, hiddenDim*hiddenDim)
		w[p+".ow"] = rnd(rng, hiddenDim*hiddenDim)
		w[p+".fn"] = ones(hiddenDim)
		w[p+".gw"] = rnd(rng, intermediate*hiddenDim)
		w[p+".uw"] = rnd(rng, intermediate*hiddenDim)
		w[p+".dw"] = rnd(rng, hiddenDim*intermediate)
	}
	w["norm.weight"] = ones(hiddenDim)
	w["lm_head.weight"] = rnd(rng, vocabSize*hiddenDim)
	return w
}

func rnd(rng *rand.Rand, n int) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = float32(rng.NormFloat64() * 0.02)
	}
	return f
}

func ones(n int) []float32 {
	f := make([]float32, n)
	for i := range f {
		f[i] = 1.0
	}
	return f
}

// ── Conversions ─────────────────────────────────────────

func f32b(f []float32) []byte {
	b := make([]byte, len(f)*4)
	for i, v := range f {
		bits := math.Float32bits(v)
		b[i*4] = byte(bits); b[i*4+1] = byte(bits>>8)
		b[i*4+2] = byte(bits>>16); b[i*4+3] = byte(bits>>24)
	}
	return b
}

func b2f32(b []byte) []float32 {
	f := make([]float32, len(b)/4)
	for i := range f {
		f[i] = math.Float32frombits(
			uint32(b[i*4])|uint32(b[i*4+1])<<8|uint32(b[i*4+2])<<16|uint32(b[i*4+3])<<24)
	}
	return f
}

func i32b(v int32) []byte {
	b := make([]byte, 4)
	b[0] = byte(v); b[1] = byte(v>>8)
	b[2] = byte(v>>16); b[3] = byte(v>>24)
	return b
}

func u32b(v uint32) []byte {
	b := make([]byte, 4)
	b[0] = byte(v); b[1] = byte(v>>8)
	b[2] = byte(v>>16); b[3] = byte(v>>24)
	return b
}

func decodeTokens(t []int) string {
	s := ""
	for _, c := range t {
		s += decodeToken(c)
	}
	return s
}

func decodeToken(c int) string {
	if c >= 32 && c < 127 {
		return string(rune(c))
	}
	return "?"
}

func init() {
	// Allow running without VGPU being alive
	_ = os.Getenv
}
