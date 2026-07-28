/*
 * pkg/discovery/mdns.go — mDNS/DNS-SD device discovery
 *
 * Pure-Go implementation using only the standard library.
 * No external dependencies required.
 *
 * Service type: _distribox._tcp.local.
 * TXT records carry device metadata for quick filtering.
 */

package discovery

import (
	"encoding/binary"
	"fmt"
	"log"
	"net"
	"strings"
	"sync"
	"time"
)

const (
	mdnsMulticastAddr = "224.0.0.251"
	mdnsPort          = 5353
	serviceType       = "_distribox._tcp.local."
	dnsClassIN        = 1
	dnsTypePTR        = 12
	dnsTypeSRV        = 33
	dnsTypeTXT        = 16
	dnsTypeA          = 1
	dnsFlagQRResponse = 0x8400
)

// ── Types ──────────────────────────────────────────────

type DeviceInfo struct {
	Name           string
	Host           string
	Port           int
	Role           string
	Arch           string
	OS             string
	HasGPU         bool
	TotalRAMMB     uint64
	ProtocolVersion string
	ClusterToken   string // Auto-generated cluster token for zero-config auth
}

type Discovery struct {
	serviceType string
	deviceInfo  DeviceInfo
	browseCh    chan DeviceInfo
	stopCh      chan struct{}
	mu          sync.RWMutex
	found       map[string]DeviceInfo
	conn        *net.UDPConn
}

func New(role string, info DeviceInfo) *Discovery {
	info.Role = role
	return &Discovery{
		serviceType: serviceType,
		deviceInfo:  info,
		browseCh:    make(chan DeviceInfo, 16),
		stopCh:      make(chan struct{}),
		found:       make(map[string]DeviceInfo),
	}
}

// ── Advertise ──────────────────────────────────────────

func (d *Discovery) Advertise(port int) error {
	addr := &net.UDPAddr{
		IP:   net.ParseIP(mdnsMulticastAddr),
		Port: mdnsPort,
	}

	conn, err := net.ListenMulticastUDP("udp4", nil, addr)
	if err != nil {
		return fmt.Errorf("mDNS advertise: %w", err)
	}
	d.conn = conn
	d.deviceInfo.Port = port

	log.Printf("mDNS: advertising as %s (role=%s) on port %d",
		d.deviceInfo.Name, d.deviceInfo.Role, port)

	go d.listenAndRespond(conn, port)

	return nil
}

func (d *Discovery) listenAndRespond(conn *net.UDPConn, port int) {
	buf := make([]byte, 1500)
	for {
		select {
		case <-d.stopCh:
			return
		default:
		}

		conn.SetReadDeadline(time.Now().Add(1 * time.Second))
		n, remote, err := conn.ReadFromUDP(buf)
		if err != nil {
			if netErr, ok := err.(net.Error); ok && netErr.Timeout() {
				continue
			}
			continue
		}

		// Only respond to queries (not responses)
		if n < 12 {
			continue
		}
		flags := binary.BigEndian.Uint16(buf[2:4])
		if flags&0x8000 != 0 {
			continue // This is a response, not a query
		}

		// Check if this is a query for our service
		query := string(buf[12:n])
		if !strings.Contains(query, d.serviceType) {
			continue
		}

		// Build and send response
		resp := d.buildResponse(port)
		conn.WriteTo(resp, remote)
	}
}

func (d *Discovery) buildResponse(port int) []byte {
	// Build a minimal mDNS response with PTR + SRV + TXT + A records
	d.mu.RLock()
	info := d.deviceInfo
	d.mu.RUnlock()

	hostname := info.Name
	if hostname == "" {
		hostname = "distribox"
	}
	hostname = strings.ReplaceAll(hostname, " ", "-")
	fullName := hostname + "." + serviceType

	// Get local IP
	localIP := getLocalAddr()
	ip := net.ParseIP(localIP)
	if ip == nil {
		ip = net.ParseIP("127.0.0.1")
	}
	ip4 := ip.To4()

	// Build TXT records
	txtStrs := []string{
		"ver=1.0",
		fmt.Sprintf("role=%s", info.Role),
		fmt.Sprintf("name=%s", info.Name),
		fmt.Sprintf("arch=%s", info.Arch),
		fmt.Sprintf("os=%s", info.OS),
		fmt.Sprintf("gpu=%s", boolStr(info.HasGPU)),
		fmt.Sprintf("mem=%d", info.TotalRAMMB),
	}
	if info.ClusterToken != "" {
		txtStrs = append(txtStrs, fmt.Sprintf("token=%s", info.ClusterToken))
	}

	// Encode DNS records
	nameBytes := encodeDNSName(fullName)
	svcName := encodeDNSName(serviceType)
	hostBytes := encodeDNSName(hostname + ".local.")

	txtBytes := encodeTXT(txtStrs)

	// Build the response packet
	var pkt []byte

	// Header: Transaction ID (0), Flags (response), Questions (0), Answers (4)
	header := make([]byte, 12)
	binary.BigEndian.PutUint16(header[0:2], 0)     // TXID
	binary.BigEndian.PutUint16(header[2:4], 0x8400) // Response + Authoritative
	binary.BigEndian.PutUint16(header[4:6], 0)      // Questions
	binary.BigEndian.PutUint16(header[6:8], 4)      // Answer RRs: PTR + SRV + TXT + A
	pkt = append(pkt, header...)

	// Answer 1: PTR record (service → instance)
	ptrRR := buildRR(svcName, dnsTypePTR, dnsClassIN, 120, nameBytes)
	pkt = append(pkt, ptrRR...)

	// Answer 2: SRV record (instance → host:port)
	srvRD := make([]byte, 6)
	binary.BigEndian.PutUint16(srvRD[0:2], 0)     // Priority
	binary.BigEndian.PutUint16(srvRD[2:4], 0)     // Weight
	binary.BigEndian.PutUint16(srvRD[4:6], uint16(port))
	srvRD = append(srvRD, hostBytes...)
	srvRR := buildRR(nameBytes, dnsTypeSRV, dnsClassIN, 120, srvRD)
	pkt = append(pkt, srvRR...)

	// Answer 3: TXT record
	txtRR := buildRR(nameBytes, dnsTypeTXT, dnsClassIN, 120, txtBytes)
	pkt = append(pkt, txtRR...)

	// Answer 4: A record
	aRD := make([]byte, 4)
	copy(aRD, ip4)
	aRR := buildRR(hostBytes, dnsTypeA, dnsClassIN, 120, aRD)
	pkt = append(pkt, aRR...)

	return pkt
}

func (d *Discovery) StopAdvertising() {
	close(d.stopCh)
	if d.conn != nil {
		d.conn.Close()
	}
	log.Println("mDNS: stopped advertising")
}

// ── Browse ─────────────────────────────────────────────

func (d *Discovery) Browse() (<-chan DeviceInfo, error) {
	log.Printf("mDNS: browsing for %s", d.serviceType)

	go d.browseLoop()

	return d.browseCh, nil
}

func (d *Discovery) browseLoop() {
	defer close(d.browseCh)

	// Join multicast group
	iface := getDefaultInterface()
	addr := &net.UDPAddr{
		IP:   net.ParseIP(mdnsMulticastAddr),
		Port: mdnsPort,
	}

	conn, err := net.ListenMulticastUDP("udp4", iface, addr)
	if err != nil {
		log.Printf("mDNS browse: %v (trying fallback)", err)
		conn, err = net.ListenUDP("udp4", &net.UDPAddr{Port: 0})
		if err != nil {
			return
		}
	}
	defer conn.Close()

	// Send query
	query := buildQuery(serviceType)
	dst := &net.UDPAddr{IP: net.ParseIP(mdnsMulticastAddr), Port: mdnsPort}

	// Listen for responses
	buf := make([]byte, 1500)

	// Query multiple times (mDNS is unreliable by nature)
	for attempt := 0; attempt < 5; attempt++ {
		select {
		case <-d.stopCh:
			return
		default:
		}

		conn.WriteTo(query, dst)

		conn.SetReadDeadline(time.Now().Add(2 * time.Second))
		for {
			n, _, err := conn.ReadFromUDP(buf)
			if err != nil {
				break // timeout, will retry
			}
			if n >= 12 {
				flags := binary.BigEndian.Uint16(buf[2:4])
				if flags&0x8000 != 0 {
					info := parseMDNSResponse(buf[12:n], d.serviceType)
					if info != nil {
						key := fmt.Sprintf("%s:%d", info.Host, info.Port)
						d.mu.Lock()
						if _, exists := d.found[key]; !exists {
							d.found[key] = *info
							d.mu.Unlock()
							log.Printf("mDNS: discovered %s (%s/%s) at %s:%d [role=%s]",
								info.Name, info.OS, info.Arch, info.Host, info.Port, info.Role)
							select {
							case d.browseCh <- *info:
							default:
							}
						} else {
							d.mu.Unlock()
						}
					}
				}
			}
		}
	}
}

func (d *Discovery) FoundDevices() map[string]DeviceInfo {
	d.mu.RLock()
	defer d.mu.RUnlock()
	result := make(map[string]DeviceInfo, len(d.found))
	for k, v := range d.found {
		result[k] = v
	}
	return result
}

// ── DNS encoding helpers ───────────────────────────────

func encodeDNSName(name string) []byte {
	var result []byte
	for _, part := range strings.Split(name, ".") {
		if part == "" {
			continue
		}
		result = append(result, byte(len(part)))
		result = append(result, []byte(part)...)
	}
	result = append(result, 0)
	return result
}

func encodeTXT(entries []string) []byte {
	var result []byte
	for _, e := range entries {
		if len(e) > 255 {
			e = e[:255]
		}
		result = append(result, byte(len(e)))
		result = append(result, []byte(e)...)
	}
	return result
}

func buildRR(name []byte, rtype, class uint16, ttl uint32, rdata []byte) []byte {
	rr := make([]byte, 10+len(name)+len(rdata))
	copy(rr, name)
	offset := len(name)
	binary.BigEndian.PutUint16(rr[offset:], rtype)
	binary.BigEndian.PutUint16(rr[offset+2:], class)
	binary.BigEndian.PutUint32(rr[offset+4:], ttl)
	binary.BigEndian.PutUint16(rr[offset+8:], uint16(len(rdata)))
	copy(rr[offset+10:], rdata)
	return rr
}

func buildQuery(service string) []byte {
	// DNS query header + question for PTR record
	header := make([]byte, 12)
	binary.BigEndian.PutUint16(header[0:2], 0)    // TXID
	binary.BigEndian.PutUint16(header[2:4], 0)    // Flags
	binary.BigEndian.PutUint16(header[4:6], 1)    // Questions
	name := encodeDNSName(service)
	q := make([]byte, len(name)+4)
	copy(q, name)
	binary.BigEndian.PutUint16(q[len(name):], dnsTypePTR)
	binary.BigEndian.PutUint16(q[len(name)+2:], dnsClassIN)
	return append(header, q...)
}

func parseMDNSResponse(data []byte, targetService string) *DeviceInfo {
	// Minimal parser: scan for TXT records containing _distribox metadata
	_ = encodeDNSName(targetService)

	// Find all TXT records by scanning the response
	offset := 0
	for offset < len(data) {
		_, consumed := parseDNSName(data, offset)
		if consumed == 0 {
			break
		}
		offset = consumed
		if offset+10 > len(data) {
			break
		}
		rtype := binary.BigEndian.Uint16(data[offset:])
		rdLen := binary.BigEndian.Uint16(data[offset+8:])
		offset += 10

		if int(offset)+int(rdLen) > len(data) {
			break
		}

		if rtype == dnsTypeTXT {
			txtMap := parseTXT(data[offset : offset+int(rdLen)])
			if txtMap["ver"] != "" {
				info := &DeviceInfo{
					ProtocolVersion: txtMap["ver"],
					Role:            txtMap["role"],
					Name:            txtMap["name"],
					Arch:            txtMap["arch"],
					OS:              txtMap["os"],
					HasGPU:          txtMap["gpu"] == "yes",
					ClusterToken:    txtMap["token"],
				}
				fmt.Sscanf(txtMap["mem"], "%d", &info.TotalRAMMB)
				if info.ProtocolVersion != "" && info.Name != "" {
					return info
				}
			}
		}
		offset += int(rdLen)
	}
	return nil
}

func parseDNSName(data []byte, offset int) ([]byte, int) {
	start := offset
	for offset < len(data) {
		length := int(data[offset])
		if length == 0 {
			return data[start : offset+1], offset + 1 // include null terminator
		}
		if length&0xC0 == 0xC0 {
			// Compressed name — skip
			return data[start : offset+2], offset + 2
		}
		offset += 1 + length
	}
	return nil, 0
}

func parseTXT(data []byte) map[string]string {
	result := make(map[string]string)
	offset := 0
	for offset < len(data) {
		length := int(data[offset])
		offset++
		if offset+length > len(data) {
			break
		}
		entry := string(data[offset : offset+length])
		offset += length
		parts := strings.SplitN(entry, "=", 2)
		if len(parts) == 2 {
			result[parts[0]] = parts[1]
		}
	}
	return result
}

// ── Network helpers ────────────────────────────────────

func getLocalAddr() string {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "127.0.0.1"
	}
	defer conn.Close()
	return conn.LocalAddr().(*net.UDPAddr).IP.String()
}

func getDefaultInterface() *net.Interface {
	ifaces, err := net.Interfaces()
	if err != nil {
		return nil
	}
	for _, iface := range ifaces {
		if iface.Flags&net.FlagUp != 0 && iface.Flags&net.FlagMulticast != 0 && iface.Flags&net.FlagLoopback == 0 {
			return &iface
		}
	}
	return nil
}

func GetLocalIP() (string, error) {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		return "", err
	}
	defer conn.Close()
	return conn.LocalAddr().(*net.UDPAddr).IP.String(), nil
}

func GetHostname() string {
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return "unknown"
	}
	for _, addr := range addrs {
		if ipnet, ok := addr.(*net.IPNet); ok && !ipnet.IP.IsLoopback() && ipnet.IP.To4() != nil {
			return ipnet.IP.String()
		}
	}
	return "unknown"
}

func boolStr(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

// ── Background discovery ──────────────────────────────

func (d *Discovery) DiscoverLoop() {
	ch, err := d.Browse()
	if err != nil {
		log.Printf("mDNS browse error: %v", err)
		return
	}

	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case device := <-ch:
			log.Printf("  → Found %s device: %s (%s/%s) at %s:%d",
				device.Role, device.Name, device.OS, device.Arch,
				device.Host, device.Port)

		case <-ticker.C:
			devices := d.FoundDevices()
			if len(devices) > 0 {
				log.Printf("mDNS: currently seeing %d device(s)", len(devices))
			}

		case <-d.stopCh:
			return
		}
	}
}
