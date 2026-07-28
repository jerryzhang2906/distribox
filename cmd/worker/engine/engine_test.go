/*
 * cmd/worker/engine/engine_test.go — Unit tests for pure Go compute engine
 *
 * Tests all kernel implementations: vector_add, matmul, softmax, relu,
 * scalar_mul, element_wise_mul, transpose, reduce_sum, sigmoid.
 *
 * Run: go test ./cmd/worker/engine/ -v
 */

package engine

import (
	"math"
	"testing"
)

// ── Test helpers ────────────────────────────────────────

func newTestEngine(t *testing.T) *GoEngine {
	t.Helper()
	return NewGoEngine()
}

func float32Slice(n int, fill ...float32) []byte {
	val := float32(0)
	if len(fill) > 0 {
		val = fill[0]
	}
	data := make([]byte, n*4)
	sl := bytesAsFloat32Slice(data)
	for i := range sl {
		sl[i] = val
	}
	return data
}

func float32SliceRange(n int) []byte {
	data := make([]byte, n*4)
	sl := bytesAsFloat32Slice(data)
	for i := range sl {
		sl[i] = float32(i)
	}
	return data
}

func checkFloats(t *testing.T, name string, expected, actual []float32, tol float32) {
	t.Helper()
	if len(expected) != len(actual) {
		t.Fatalf("%s: length mismatch: expected %d, got %d", name, len(expected), len(actual))
	}
	for i := range expected {
		diff := actual[i] - expected[i]
		if diff < -tol || diff > tol {
			t.Errorf("%s[%d]: expected %.4f, got %.4f (diff=%.6f)", name, i, expected[i], actual[i], diff)
		}
	}
}

// ── vector_add tests ────────────────────────────────────

func TestVectorAdd(t *testing.T) {
	eng := newTestEngine(t)

	// Prepare buffers: A=[1,2,3,4], B=[5,6,7,8], C=A+B=[6,8,10,12]
	dataA := make([]byte, 16)
	bytesAsFloat32Slice(dataA)[0], bytesAsFloat32Slice(dataA)[1] = 1, 2
	bytesAsFloat32Slice(dataA)[2], bytesAsFloat32Slice(dataA)[3] = 3, 4

	dataB := make([]byte, 16)
	bytesAsFloat32Slice(dataB)[0], bytesAsFloat32Slice(dataB)[1] = 5, 6
	bytesAsFloat32Slice(dataB)[2], bytesAsFloat32Slice(dataB)[3] = 7, 8

	bufA, _ := eng.CreateBuffer(16, 0, dataA)
	bufB, _ := eng.CreateBuffer(16, 0, dataB)
	bufC, _ := eng.CreateBuffer(16, 0, nil)

	k := &GoKernel{NameVal: "vector_add"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufC)
	eng.SetKernelArg(k, 3, int32(4)) // n

	err := eng.ExecuteNDRange(k, 1, []uint64{4}, nil, nil, []*GoBuffer{bufC})
	if err != nil {
		t.Fatalf("vector_add failed: %v", err)
	}

	expected := []float32{6, 8, 10, 12}
	actual := bytesAsFloat32Slice(bufC.data)
	checkFloats(t, "vector_add", expected, actual, 0.001)
}

func TestVectorAddLarge(t *testing.T) {
	eng := newTestEngine(t)
	n := 1024

	dataA := float32SliceRange(n)
	dataB := float32SliceRange(n) // B = [0, 1, 2, ...]

	bufA, _ := eng.CreateBuffer(uint64(n*4), 0, dataA)
	bufB, _ := eng.CreateBuffer(uint64(n*4), 0, dataB)
	bufC, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "vector_add"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufC)
	eng.SetKernelArg(k, 3, int32(n))

	err := eng.ExecuteNDRange(k, 1, []uint64{1024}, nil, nil, []*GoBuffer{bufC})
	if err != nil {
		t.Fatalf("vector_add large failed: %v", err)
	}

	actual := bytesAsFloat32Slice(bufC.data)
	for i := 0; i < n; i++ {
		expected := float32(i * 2) // A[i]=i, B[i]=i, C[i]=2*i
		if math.Abs(float64(actual[i]-expected)) > 0.001 {
			t.Errorf("[%d]: expected %.1f, got %.1f", i, expected, actual[i])
		}
	}
}

// ── matmul tests ────────────────────────────────────────

func TestMatMul2x2(t *testing.T) {
	eng := newTestEngine(t)

	// A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
	// C = [[19, 22], [43, 50]]
	// M=2, K=2, N=2
	M, K, N := 2, 2, 2
	bufA, _ := eng.CreateBuffer(uint64(M*K*4), 0, nil)
	bufB, _ := eng.CreateBuffer(uint64(K*N*4), 0, nil)
	bufC, _ := eng.CreateBuffer(uint64(M*N*4), 0, nil)

	a := bytesAsFloat32Slice(bufA.data)
	a[0], a[1], a[2], a[3] = 1, 2, 3, 4
	b := bytesAsFloat32Slice(bufB.data)
	b[0], b[1], b[2], b[3] = 5, 6, 7, 8

	k := &GoKernel{NameVal: "matmul"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufC)
	eng.SetKernelArg(k, 3, int32(K))

	err := eng.ExecuteNDRange(k, 2, []uint64{2, 2}, nil, nil, []*GoBuffer{bufC})
	if err != nil {
		t.Fatalf("matmul failed: %v", err)
	}

	expected := []float32{19, 22, 43, 50}
	actual := bytesAsFloat32Slice(bufC.data)
	checkFloats(t, "matmul 2x2", expected, actual, 0.01)
}

func TestMatMulIdentity(t *testing.T) {
	eng := newTestEngine(t)

	M, K, N := 4, 4, 4
	// A = identity * 2
	bufA, _ := eng.CreateBuffer(uint64(M*K*4), 0, nil)
	a := bytesAsFloat32Slice(bufA.data)
	for i := 0; i < 4; i++ {
		a[i*4+i] = 2.0
	}

	// B = all 1s (so B acts as "sum rows" operator: each row is sum of A's row)
	bufB, _ := eng.CreateBuffer(uint64(K*N*4), 0, float32Slice(K*N, 1.0))
	bufC, _ := eng.CreateBuffer(uint64(M*N*4), 0, nil)

	k := &GoKernel{NameVal: "matmul"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufC)
	eng.SetKernelArg(k, 3, int32(K))

	err := eng.ExecuteNDRange(k, 2, []uint64{4, 4}, nil, nil, []*GoBuffer{bufC})
	if err != nil {
		t.Fatalf("matmul identity failed: %v", err)
	}

	// A=2*I, B=all ones → C=2*I (each column of C should be the same as each column of A?)
	// Actually C = A*B where A=2I, B=ones → C[i][j] = 2 if i==j else 0... no wait
	// B has all ones, so C[i][j] = sum_k A[i][k] * 1 = sum of row i of A
	// For A=2*I, row i sum = 2, so C = all 2s
	actual := bytesAsFloat32Slice(bufC.data)
	for i := 0; i < M*N; i++ {
		if math.Abs(float64(actual[i]-2.0)) > 0.01 {
			t.Errorf("matmul identity [%d]: expected 2.0, got %.4f", i, actual[i])
		}
	}
}

// ── softmax tests ───────────────────────────────────────

func TestSoftmax(t *testing.T) {
	eng := newTestEngine(t)

	// Input: [1, 2, 3, 4]
	n := 4
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2], in[3] = 1, 2, 3, 4

	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "softmax"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{4}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("softmax failed: %v", err)
	}

	actual := bytesAsFloat32Slice(bufOut.data)
	// Sum should be 1.0
	var sum float32
	for i := 0; i < n; i++ {
		sum += actual[i]
	}
	if math.Abs(float64(sum-1.0)) > 0.01 {
		t.Errorf("softmax sum: expected 1.0, got %.4f", sum)
	}
	// Values should be increasing (softmax preserves order)
	for i := 1; i < n; i++ {
		if actual[i] <= actual[i-1] {
			t.Errorf("softmax monotonicity violated at %d: %.4f <= %.4f", i, actual[i], actual[i-1])
		}
	}
}

// ── relu tests ──────────────────────────────────────────

func TestReLU(t *testing.T) {
	eng := newTestEngine(t)

	n := 6
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2], in[3], in[4], in[5] = -1, 2, -3, 4, 0, -0.5

	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "relu"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{6}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("relu failed: %v", err)
	}

	expected := []float32{0, 2, 0, 4, 0, 0}
	actual := bytesAsFloat32Slice(bufOut.data)
	checkFloats(t, "relu", expected, actual, 0.001)
}

func TestReLUAllPositive(t *testing.T) {
	eng := newTestEngine(t)

	n := 100
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, float32SliceRange(n)) // 0,1,2,...,99
	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "relu"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{100}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("relu all-pos failed: %v", err)
	}

	in := bytesAsFloat32Slice(bufIn.data)
	out := bytesAsFloat32Slice(bufOut.data)
	for i := 0; i < n; i++ {
		if out[i] != in[i] {
			t.Errorf("relu pos [%d]: expected %.1f, got %.1f", i, in[i], out[i])
		}
	}
}

// ── scalar_mul tests ────────────────────────────────────

func TestScalarMul(t *testing.T) {
	eng := newTestEngine(t)

	n := 5
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, float32SliceRange(n)) // 0,1,2,3,4
	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "scalar_mul"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, float32(2.5))

	err := eng.ExecuteNDRange(k, 1, []uint64{5}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("scalar_mul failed: %v", err)
	}

	expected := []float32{0, 2.5, 5.0, 7.5, 10.0}
	actual := bytesAsFloat32Slice(bufOut.data)
	checkFloats(t, "scalar_mul", expected, actual, 0.01)
}

// ── element_wise_mul tests ──────────────────────────────

func TestElementWiseMul(t *testing.T) {
	eng := newTestEngine(t)

	n := 4
	bufA, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	bufB, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	a := bytesAsFloat32Slice(bufA.data)
	b := bytesAsFloat32Slice(bufB.data)
	a[0], a[1], a[2], a[3] = 1, 2, 3, 4
	b[0], b[1], b[2], b[3] = 5, 6, 7, 8

	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "element_wise_mul"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{4}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("element_wise_mul failed: %v", err)
	}

	expected := []float32{5, 12, 21, 32}
	actual := bytesAsFloat32Slice(bufOut.data)
	checkFloats(t, "element_wise_mul", expected, actual, 0.01)
}

// ── sigmoid tests ───────────────────────────────────────

func TestSigmoid(t *testing.T) {
	eng := newTestEngine(t)

	n := 3
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2] = 0, 2, -2

	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "sigmoid"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{3}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("sigmoid failed: %v", err)
	}

	actual := bytesAsFloat32Slice(bufOut.data)
	// sigmoid(0) ≈ 0.5, sigmoid(2) ≈ 0.8808, sigmoid(-2) ≈ 0.1192
	expected := []float32{0.5, 0.8808, 0.1192}
	for i := 0; i < n; i++ {
		if math.Abs(float64(actual[i]-expected[i])) > 0.01 {
			t.Errorf("sigmoid[%d]: expected %.4f, got %.4f", i, expected[i], actual[i])
		}
	}
}

func TestSigmoidExtreme(t *testing.T) {
	eng := newTestEngine(t)

	// Test clamping behavior for large values
	bufIn, _ := eng.CreateBuffer(4*4, 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2], in[3] = 100, -100, 0, 0

	bufOut, _ := eng.CreateBuffer(4*4, 0, nil)

	k := &GoKernel{NameVal: "sigmoid"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{4}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("sigmoid extreme failed: %v", err)
	}

	actual := bytesAsFloat32Slice(bufOut.data)
	if actual[0] < 0.99 || actual[0] > 1.01 {
		t.Errorf("sigmoid(100): expected ~1.0, got %.6f", actual[0])
	}
	if actual[1] < -0.01 || actual[1] > 0.01 {
		t.Errorf("sigmoid(-100): expected ~0.0, got %.6f", actual[1])
	}
}

// ── reduce_sum tests ────────────────────────────────────

func TestReduceSum(t *testing.T) {
	eng := newTestEngine(t)

	n := 5
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, float32SliceRange(n)) // 0,1,2,3,4
	bufOut, _ := eng.CreateBuffer(4, 0, nil) // single float32

	k := &GoKernel{NameVal: "reduce_sum"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{5}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("reduce_sum failed: %v", err)
	}

	actual := bytesAsFloat32Slice(bufOut.data)
	expected := float32(10) // 0+1+2+3+4
	if math.Abs(float64(actual[0]-expected)) > 0.01 {
		t.Errorf("reduce_sum: expected %.1f, got %.1f", expected, actual[0])
	}
}

// ── transpose tests ─────────────────────────────────────

func TestTranspose2x3(t *testing.T) {
	eng := newTestEngine(t)

	// 2x3 → 3x2
	// Input:  [[1,2,3],
	//          [4,5,6]]
	// Output: [[1,4],
	//          [2,5],
	//          [3,6]]
	rows, cols := 2, 3
	bufIn, _ := eng.CreateBuffer(uint64(rows*cols*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2] = 1, 2, 3
	in[3], in[4], in[5] = 4, 5, 6

	bufOut, _ := eng.CreateBuffer(uint64(rows*cols*4), 0, nil)

	k := &GoKernel{NameVal: "transpose"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, int32(cols))

	err := eng.ExecuteNDRange(k, 2, []uint64{uint64(rows), uint64(cols)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("transpose failed: %v", err)
	}

	expected := []float32{1, 4, 2, 5, 3, 6}
	actual := bytesAsFloat32Slice(bufOut.data)
	checkFloats(t, "transpose 2x3", expected, actual, 0.001)
}

// ── gelu tests ──────────────────────────────────────────

func TestGELU(t *testing.T) {
	eng := newTestEngine(t)

	n := 5
	bufIn, _ := eng.CreateBuffer(uint64(n*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2], in[3], in[4] = 0, 1, -1, 2, -2

	bufOut, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "gelu"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 1, []uint64{5}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("gelu failed: %v", err)
	}

	actual := bytesAsFloat32Slice(bufOut.data)
	// GELU(0) ≈ 0, GELU(1) ≈ 0.8412, GELU(-1) ≈ -0.1588
	if math.Abs(float64(actual[0])) > 0.01 {
		t.Errorf("GELU(0): expected ~0, got %.4f", actual[0])
	}
	if math.Abs(float64(actual[1]-0.8412)) > 0.02 {
		t.Errorf("GELU(1): expected ~0.8412, got %.4f", actual[1])
	}
	if actual[1] <= 0 {
		t.Error("GELU(1) should be positive")
	}
	if actual[3] <= 0 {
		t.Error("GELU(2) should be positive")
	}
	// Negative values should be near zero but can be negative
	if actual[2] >= 0 || actual[2] < -0.2 {
		t.Errorf("GELU(-1): expected small negative, got %.4f", actual[2])
	}
}

// ── layer_norm tests ─────────────────────────────────────

func TestLayerNorm(t *testing.T) {
	eng := newTestEngine(t)

	// 2 rows, 4 dims
	// Row 0: [1, 2, 3, 4], mean=2.5
	// Row 1: [5, 6, 7, 8], mean=6.5
	rows, dim := 2, 4
	bufIn, _ := eng.CreateBuffer(uint64(rows*dim*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	for i := 0; i < rows*dim; i++ {
		in[i] = float32(i + 1) // 1,2,3,4,5,6,7,8
	}

	bufOut, _ := eng.CreateBuffer(uint64(rows*dim*4), 0, nil)

	k := &GoKernel{NameVal: "layer_norm"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)

	err := eng.ExecuteNDRange(k, 2, []uint64{uint64(rows), uint64(dim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("layer_norm failed: %v", err)
	}

	out := bytesAsFloat32Slice(bufOut.data)
	// Each row should have mean ~0 and variance ~1
	for row := 0; row < rows; row++ {
		var sum, varSum float64
		base := row * dim
		for j := 0; j < dim; j++ {
			sum += float64(out[base+j])
			varSum += float64(out[base+j]) * float64(out[base+j])
		}
		mean := sum / float64(dim)
		variance := varSum/float64(dim) - mean*mean

		if math.Abs(mean) > 0.01 {
			t.Errorf("layer_norm row %d mean: expected ~0, got %.4f", row, mean)
		}
		if math.Abs(variance-1.0) > 0.1 {
			t.Errorf("layer_norm row %d variance: expected ~1, got %.4f", row, variance)
		}
	}
}

func TestLayerNormWithGammaBeta(t *testing.T) {
	eng := newTestEngine(t)

	dim := 4
	bufIn, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2], in[3] = 2, 4, 6, 8 // mean=5, std=sqrt(5)=2.236

	bufOut, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)

	// gamma = [1, 2, 1, 2], beta = [0, 1, 0, 1]
	bufGamma, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)
	gamma := bytesAsFloat32Slice(bufGamma.data)
	gamma[0], gamma[1], gamma[2], gamma[3] = 1, 2, 1, 2

	bufBeta, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)
	beta := bytesAsFloat32Slice(bufBeta.data)
	beta[0], beta[1], beta[2], beta[3] = 0, 1, 0, 1

	k := &GoKernel{NameVal: "layer_norm"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, bufGamma)
	eng.SetKernelArg(k, 3, bufBeta)

	err := eng.ExecuteNDRange(k, 2, []uint64{1, uint64(dim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("layer_norm with gamma/beta failed: %v", err)
	}

	out := bytesAsFloat32Slice(bufOut.data)
	// Manual verification:
	// normalized = (x - 5) / sqrt(5) = (x-5)/2.236
	// out[j] = normalized * gamma[j] + beta[j]
	mean := float32(5.0)
	std := float32(math.Sqrt(5.0)) // ≈ 2.23607
	for j := 0; j < dim; j++ {
		normalized := (in[j] - mean) / std
		expected := normalized*gamma[j] + beta[j]
		if math.Abs(float64(out[j]-expected)) > 0.01 {
			t.Errorf("layer_norm[%d]: expected %.4f, got %.4f", j, expected, out[j])
		}
	}
}

// ── rms_norm tests ───────────────────────────────────────

func TestRMSNorm(t *testing.T) {
	eng := newTestEngine(t)

	dim := 4
	// Input: all 2s, so rms = sqrt(mean(4,4,4,4)) = sqrt(4) = 2
	bufIn, _ := eng.CreateBuffer(uint64(dim*4), 0, float32Slice(dim, 2.0))
	bufOut, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)

	k := &GoKernel{NameVal: "rms_norm"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, int32(dim))

	err := eng.ExecuteNDRange(k, 2, []uint64{1, uint64(dim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("rms_norm failed: %v", err)
	}

	out := bytesAsFloat32Slice(bufOut.data)
	// Each element = 2 / 2 = 1.0
	for i := 0; i < dim; i++ {
		if math.Abs(float64(out[i]-1.0)) > 0.01 {
			t.Errorf("rms_norm[%d]: expected 1.0, got %.4f", i, out[i])
		}
	}
}

func TestRMSNormWithGamma(t *testing.T) {
	eng := newTestEngine(t)

	dim := 4
	bufIn, _ := eng.CreateBuffer(uint64(dim*4), 0, float32Slice(dim, 2.0))
	bufOut, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)

	// gamma = [0.5, 1, 0.5, 1]
	bufGamma, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)
	gamma := bytesAsFloat32Slice(bufGamma.data)
	gamma[0], gamma[1], gamma[2], gamma[3] = 0.5, 1.0, 0.5, 1.0

	k := &GoKernel{NameVal: "rms_norm"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, bufGamma)
	eng.SetKernelArg(k, 3, int32(dim))

	err := eng.ExecuteNDRange(k, 2, []uint64{1, uint64(dim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("rms_norm with gamma failed: %v", err)
	}

	out := bytesAsFloat32Slice(bufOut.data)
	// rms = 2, so out[i] = 2/2 * gamma[i] = gamma[i]
	if math.Abs(float64(out[0]-0.5)) > 0.01 {
		t.Errorf("rms_norm gamma[0]=0.5: expected 0.5, got %.4f", out[0])
	}
	if math.Abs(float64(out[1]-1.0)) > 0.01 {
		t.Errorf("rms_norm gamma[1]=1.0: expected 1.0, got %.4f", out[1])
	}
}

// ── rope tests ───────────────────────────────────────────

func TestRoPE(t *testing.T) {
	eng := newTestEngine(t)

	headDim := 8
	// Single row of head_dim floats
	// For RoPE: first half and second half are rotated by position-dependent angles
	dataIn := float32SliceRange(headDim) // 0,1,2,3,4,5,6,7
	bufIn, _ := eng.CreateBuffer(uint64(headDim*4), 0, dataIn)
	bufOut, _ := eng.CreateBuffer(uint64(headDim*4), 0, nil)

	k := &GoKernel{NameVal: "rope"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, int32(0))       // position
	eng.SetKernelArg(k, 3, int32(headDim)) // head_dim

	err := eng.ExecuteNDRange(k, 2, []uint64{1, uint64(headDim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("rope failed: %v", err)
	}

	out := bytesAsFloat32Slice(bufOut.data)
	// At position 0, theta = 0 for all i, so cos=1, sin=0
	// Output should be identical to input
	for i := 0; i < headDim; i++ {
		expected := float32(i)
		if math.Abs(float64(out[i]-expected)) > 0.01 {
			t.Errorf("rope pos=0 [%d]: expected %.1f, got %.4f", i, expected, out[i])
		}
	}
}

func TestRoPEPositionRotation(t *testing.T) {
	eng := newTestEngine(t)

	headDim := 4
	// Input: [1, 0, 1, 0] — first pair (1,0), second pair (1,0)
	bufIn, _ := eng.CreateBuffer(uint64(headDim*4), 0, nil)
	in := bytesAsFloat32Slice(bufIn.data)
	in[0], in[1], in[2], in[3] = 1, 0, 1, 0

	bufOut, _ := eng.CreateBuffer(uint64(headDim*4), 0, nil)

	k := &GoKernel{NameVal: "rope"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, int32(1))       // position = 1
	eng.SetKernelArg(k, 3, int32(headDim))

	err := eng.ExecuteNDRange(k, 2, []uint64{1, uint64(headDim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("rope pos=1 failed: %v", err)
	}

	out := bytesAsFloat32Slice(bufOut.data)
	// At position 1, the output should be rotated (not identity)
	// First half should differ from input
	hasChanged := false
	for i := 0; i < headDim/2; i++ {
		if math.Abs(float64(out[i]-in[i])) > 0.001 {
			hasChanged = true
			break
		}
	}
	if !hasChanged {
		t.Error("RoPE at position 1 should rotate values (not identity)")
	}
	// Rotation should preserve magnitude: x^2 + y^2 per pair
	for i := 0; i < headDim/2; i++ {
		magSq := float64(out[i]*out[i]) + float64(out[i+headDim/2]*out[i+headDim/2])
		origSq := float64(in[i]*in[i]) + float64(in[i+headDim/2]*in[i+headDim/2])
		if math.Abs(magSq-origSq) > 0.01 {
			t.Errorf("rope pair %d: magnitude changed from %.4f to %.4f", i, origSq, magSq)
		}
	}
}

// ── add_bias tests ───────────────────────────────────────

func TestAddBias(t *testing.T) {
	eng := newTestEngine(t)

	rows, dim := 2, 3
	// Input: [[1,2,3], [4,5,6]]
	bufIn, _ := eng.CreateBuffer(uint64(rows*dim*4), 0, float32SliceRange(rows*dim))
	// Bias: [10, 20, 30]
	bufBias, _ := eng.CreateBuffer(uint64(dim*4), 0, nil)
	bias := bytesAsFloat32Slice(bufBias.data)
	bias[0], bias[1], bias[2] = 10, 20, 30

	bufOut, _ := eng.CreateBuffer(uint64(rows*dim*4), 0, nil)

	k := &GoKernel{NameVal: "add_bias"}
	eng.SetKernelArg(k, 0, bufIn)
	eng.SetKernelArg(k, 1, bufOut)
	eng.SetKernelArg(k, 2, bufBias)

	err := eng.ExecuteNDRange(k, 2, []uint64{uint64(rows), uint64(dim)}, nil, nil, []*GoBuffer{bufOut})
	if err != nil {
		t.Fatalf("add_bias failed: %v", err)
	}

	expected := []float32{10, 21, 32, 13, 24, 35}
	actual := bytesAsFloat32Slice(bufOut.data)
	checkFloats(t, "add_bias", expected, actual, 0.01)
}

// ── Engine lifecycle tests ──────────────────────────────

func TestEngineBackendName(t *testing.T) {
	eng := NewGoEngine()
	if eng.BackendName() != "Go-CPU" {
		t.Errorf("BackendName: expected Go-CPU, got %s", eng.BackendName())
	}
}

func TestBufferRelease(t *testing.T) {
	eng := newTestEngine(t)
	buf, _ := eng.CreateBuffer(16, 0, nil)
	eng.ReleaseBuffer(buf)

	// Buffer should no longer be in maps (release is a no-op on reads from Go map)
	// Verify we can create a new buffer with the same sequential ID
	buf2, _ := eng.CreateBuffer(32, 0, nil)
	if buf2.Size() != 32 {
		t.Errorf("Buffer size mismatch: expected 32, got %d", buf2.Size())
	}
}

func TestUnknownKernel(t *testing.T) {
	eng := newTestEngine(t)
	k := &GoKernel{NameVal: "nonexistent_kernel"}
	err := eng.ExecuteNDRange(k, 1, []uint64{4}, nil, nil, nil)
	if err == nil {
		t.Error("Expected error for unknown kernel, got nil")
	}
}

// ── Benchmark ────────────────────────────────────────────

func BenchmarkVectorAdd(b *testing.B) {
	eng := NewGoEngine()
	n := 1024 * 1024 // 1M elements

	bufA, _ := eng.CreateBuffer(uint64(n*4), 0, float32Slice(n, 1.0))
	bufB, _ := eng.CreateBuffer(uint64(n*4), 0, float32Slice(n, 2.0))
	bufC, _ := eng.CreateBuffer(uint64(n*4), 0, nil)

	k := &GoKernel{NameVal: "vector_add"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufC)
	eng.SetKernelArg(k, 3, int32(n))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eng.ExecuteNDRange(k, 1, []uint64{uint64(n)}, nil, nil, []*GoBuffer{bufC})
	}
}

func BenchmarkMatMul(b *testing.B) {
	eng := NewGoEngine()
	M, K, N := 128, 128, 128

	bufA, _ := eng.CreateBuffer(uint64(M*K*4), 0, float32SliceRange(M*K))
	bufB, _ := eng.CreateBuffer(uint64(K*N*4), 0, float32SliceRange(K*N))
	bufC, _ := eng.CreateBuffer(uint64(M*N*4), 0, nil)

	k := &GoKernel{NameVal: "matmul"}
	eng.SetKernelArg(k, 0, bufA)
	eng.SetKernelArg(k, 1, bufB)
	eng.SetKernelArg(k, 2, bufC)
	eng.SetKernelArg(k, 3, int32(K))

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		eng.ExecuteNDRange(k, 2, []uint64{uint64(M), uint64(N)}, nil, nil, []*GoBuffer{bufC})
	}
}
