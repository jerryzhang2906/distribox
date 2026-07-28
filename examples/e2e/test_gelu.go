package main
import ("encoding/hex";"encoding/json";"fmt";"math";"net";"time")
func main() {
    conn,_ := net.Dial("tcp","127.0.0.1:9876")
    defer conn.Close()
    send := func(m map[string]interface{}) { d,_:=json.Marshal(m);conn.Write(append(d,'\n')) }
    read := func() string { b:=make([]byte,65536);conn.SetReadDeadline(time.Now().Add(5*time.Second));n,_:=conn.Read(b);return string(b[:n]) }
    send(map[string]interface{}{"type":"hello","msg_id":"h","protocol":"1.0"});read()
    inVals := []float32{-2,-1,0,1,2}
    inHex := hex.EncodeToString(makeBytes(inVals))
    send(map[string]interface{}{"type":"buffer_create","msg_id":"c1","buffer_id":"g_in","size":uint64(len(inVals)*4),"flags":0,"buffer_type":"read_write"});read()
    send(map[string]interface{}{"type":"buffer_write","msg_id":"w1","buffer_id":"g_in","offset":0,"size":uint64(len(inVals)*4),"data_b64":inHex});read()
    send(map[string]interface{}{"type":"buffer_create","msg_id":"c2","buffer_id":"g_out","size":uint64(len(inVals)*4),"flags":0,"buffer_type":"read_write"});read()
    send(map[string]interface{}{"type":"ndrange","msg_id":"n","queue_id":"q","kernel_id":"k","kernel_name":"gelu","program_id":"p","work_dim":1,"global":[]int{len(inVals)},"args":[]map[string]interface{}{{"index":0,"type":"buffer","id":"g_in"},{"index":1,"type":"buffer","id":"g_out"}}});read()
    send(map[string]interface{}{"type":"queue_finish","msg_id":"qf","queue_id":"q"});read()
    send(map[string]interface{}{"type":"buffer_read","msg_id":"r","buffer_id":"g_out","offset":0,"size":uint64(len(inVals)*4)}); r:=read()
    var res map[string]interface{};json.Unmarshal([]byte(r),&res)
    outRaw,_:=hex.DecodeString(res["data_b64"].(string));out:=bytesToF(outRaw)
    fmt.Println("GELU distributed results:");allOK:=true
    for i,v:=range inVals{c:=v*(0.5+0.5*math.Erf(float64(v)/1.41421356237));ok:="OK";if math.Abs(float64(out[i])-c)>0.01{ok="FAIL";allOK=false};fmt.Printf("  GELU(%.0f) = %.4f (expected %.4f) %s\n",v,out[i],c,ok)}
    if allOK {fmt.Println("\n  ✅ ALL CORRECT!")} else {fmt.Println("\n  ❌ MISMATCH")}
}
func makeBytes(f []float32)[]byte{b:=make([]byte,len(f)*4);for i,v:=range f{bits:=math.Float32bits(v);b[i*4]=byte(bits);b[i*4+1]=byte(bits>>8);b[i*4+2]=byte(bits>>16);b[i*4+3]=byte(bits>>24)};return b}
func bytesToF(b []byte)[]float32{f:=make([]float32,len(b)/4);for i:=range f{f[i]=math.Float32frombits(uint32(b[i*4])|uint32(b[i*4+1])<<8|uint32(b[i*4+2])<<16|uint32(b[i*4+3])<<24)};return f}
