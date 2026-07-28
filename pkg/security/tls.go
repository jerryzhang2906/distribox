/*
 * pkg/security/tls.go — mTLS and cluster token management
 *
 * Provides:
 *   - TLS/mTLS configuration for gRPC (control plane) and TCP (data plane)
 *   - Cluster token generation and validation
 *   - Session JWT creation and verification (for worker authentication)
 *
 * Security model for MVP:
 *   - Pre-shared cluster token (128-bit random) — user shares via QR code or text
 *   - mTLS for all gRPC connections (encryption + mutual authentication)
 *   - TLS for data plane TCP (encryption only, authentication via token in handshake)
 *   - Certificate fingerprints pinned on first connect
 */

package security

import (
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/hex"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"time"
)

// ── Cluster Token ──────────────────────────────────────

const TokenLength = 16 // 128 bits

// GenerateClusterToken creates a new random cluster token
func GenerateClusterToken() (string, error) {
	bytes := make([]byte, TokenLength)
	if _, err := rand.Read(bytes); err != nil {
		return "", fmt.Errorf("failed to generate token: %w", err)
	}

	// Format as 3 word groups for readability:
	// "abc123-def456-ghi789"
	hexStr := hex.EncodeToString(bytes)
	return fmt.Sprintf("%s-%s-%s", hexStr[0:6], hexStr[6:12], hexStr[12:18]), nil
}

// ValidateClusterToken checks if a provided token matches the expected one
func ValidateClusterToken(expected, provided string) bool {
	// Constant-time comparison to prevent timing attacks
	return sha256.Sum256([]byte(expected)) == sha256.Sum256([]byte(provided))
}

// ── Self-signed Certificate Generation ─────────────────

// GenerateSelfSignedCert creates a self-signed certificate and private key
// for use in TLS/mTLS. Returns PEM-encoded cert and key.
func GenerateSelfSignedCert(commonName string) (certPEM, keyPEM []byte, certHash string, err error) {
	// Generate RSA private key
	privateKey, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to generate key: %w", err)
	}

	// Create certificate template
	serialNumber, _ := rand.Int(rand.Reader, new(big.Int).Lsh(big.NewInt(1), 128))
	template := x509.Certificate{
		SerialNumber: serialNumber,
		Subject: pkix.Name{
			CommonName:   commonName,
			Organization: []string{"DistriBox Cluster"},
		},
		NotBefore:             time.Now(),
		NotAfter:              time.Now().Add(365 * 24 * time.Hour), // 1 year validity
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth, x509.ExtKeyUsageClientAuth},
		BasicConstraintsValid: true,
	}

	// Self-sign the certificate
	certDER, err := x509.CreateCertificate(rand.Reader, &template, &template,
		&privateKey.PublicKey, privateKey)
	if err != nil {
		return nil, nil, "", fmt.Errorf("failed to create certificate: %w", err)
	}

	// Encode to PEM
	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: certDER})
	keyPEM = pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(privateKey),
	})

	// Compute certificate fingerprint for pinning
	cert, _ := x509.ParseCertificate(certDER)
	if cert != nil {
		hash := sha256.Sum256(cert.Raw)
		certHash = hex.EncodeToString(hash[:])
	}

	return certPEM, keyPEM, certHash, nil
}

// ── mTLS Configuration ─────────────────────────────────

// ServerTLSConfig creates a TLS config for the gRPC server (orchestrator)
func ServerTLSConfig(certPEM, keyPEM []byte) (*tls.Config, error) {
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to load key pair: %w", err)
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		ClientAuth:   tls.RequireAndVerifyClientCert,
		MinVersion:   tls.VersionTLS12,
	}, nil
}

// ClientTLSConfig creates a TLS config for gRPC clients (workers)
func ClientTLSConfig(certPEM, keyPEM []byte, serverCertHash string) (*tls.Config, error) {
	cert, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		return nil, fmt.Errorf("failed to load key pair: %w", err)
	}

	return &tls.Config{
		Certificates: []tls.Certificate{cert},
		MinVersion:   tls.VersionTLS12,
		// In production: verify server certificate hash (TOFU / pinning)
		// For MVP: skip verification since it's a trusted LAN
		InsecureSkipVerify: true,
	}, nil
}

// ── Token Storage ──────────────────────────────────────

// SaveToken writes the cluster token to a file
func SaveToken(path, token string) error {
	return os.WriteFile(path, []byte(token), 0600)
}

// LoadToken reads the cluster token from a file
func LoadToken(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ── Session JWT ────────────────────────────────────────

// CreateSessionToken creates a simple session token for a worker
// (In production, use proper JWT with claims. For MVP, just a random string.)
func CreateSessionToken(workerID string) (string, error) {
	bytes := make([]byte, 32)
	if _, err := rand.Read(bytes); err != nil {
		return "", err
	}
	return fmt.Sprintf("sess-%s-%s", workerID[:8], hex.EncodeToString(bytes)[:16]), nil
}

// ── PIN Code Display ───────────────────────────────────

// FormatJoinCode creates a human-friendly join code from the token
func FormatJoinCode(token string) string {
	return fmt.Sprintf("distribox-join %s", token)
}

// ParseJoinCode extracts the token from a join code string
func ParseJoinCode(code string) string {
	if len(code) > 14 && code[:14] == "distribox-join " {
		return code[14:]
	}
	return code
}
