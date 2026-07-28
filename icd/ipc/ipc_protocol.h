/**
 * ipc/ipc_protocol.h — IPC message format between ICD and Virtual GPU Core
 *
 * Two transport mechanisms:
 *   1. Unix Domain Socket / Named Pipe — for commands and small responses
 *   2. Shared Memory — for large buffer transfers (> 1 MB)
 *
 * All messages use newline-delimited JSON over the socket.
 * Shared memory transfers use a handshake: the sender writes a {shm_name, size}
 * message over the socket, then the receiver reads from shared memory.
 */

#ifndef DISTRIBOX_IPC_PROTOCOL_H
#define DISTRIBOX_IPC_PROTOCOL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── TCP connection config ──────────────────────────────

#define IPC_TCP_PORT        9876
#define IPC_TCP_HOST        "127.0.0.1"

// ── Message types (sent over socket as JSON) ───────────

// All messages are JSON with a "type" field. Responses have a
// "request_id" field that echoes the request's message ID.

// REQUEST: ICD → VGPU Core
#define IPC_MSG_HELLO           "hello"             // Initial handshake
#define IPC_MSG_DEVICE_CONFIG   "device_config"     // Query virtual GPU specs
#define IPC_MSG_BUFFER_CREATE   "buffer_create"     // Allocate virtual VRAM
#define IPC_MSG_BUFFER_WRITE    "buffer_write"      // Write to buffer
#define IPC_MSG_BUFFER_READ     "buffer_read"       // Read from buffer
#define IPC_MSG_BUFFER_FILL     "buffer_fill"       // Fill buffer
#define IPC_MSG_BUFFER_COPY     "buffer_copy"       // Copy between buffers
#define IPC_MSG_BUFFER_RELEASE  "buffer_release"    // Free buffer
#define IPC_MSG_PROGRAM_BUILD   "program_build"     // Compile OpenCL program
#define IPC_MSG_NDRANGE         "ndrange"           // Execute NDRange kernel
#define IPC_MSG_QUEUE_FINISH    "queue_finish"      // Wait for queue completion
#define IPC_MSG_SHUTDOWN        "shutdown"          // ICD unloading

// RESPONSE: VGPU Core → ICD
#define IPC_RESP_OK             "ok"
#define IPC_RESP_ERROR          "error"
#define IPC_RESP_DEVICE_INFO    "device_info"       // Virtual GPU specs
#define IPC_RESP_EVENT          "event"             // Event completion

// ── Commonly used JSON message structures ──────────────

/* Hello message (ICD → VGPU Core on connect):
{
  "type": "hello",
  "msg_id": "msg-1",
  "protocol_version": "1.0",
  "pid": 12345
}
*/

/* Device config response (VGPU Core → ICD):
{
  "type": "device_info",
  "request_id": "msg-2",
  "device_name": "DistriBox Virtual GPU",
  "vram_bytes": 8589934592,
  "compute_units": 2048,
  "clock_mhz": 1500
}
*/

/* Buffer create (ICD → VGPU Core):
{
  "type": "buffer_create",
  "msg_id": "msg-3",
  "buffer_id": "buf-1",
  "size": 1048576,
  "flags": 1,
  "buffer_type": "read_only"
}
*/

/* NDRange (ICD → VGPU Core) — THE KEY MESSAGE:
{
  "type": "ndrange",
  "msg_id": "msg-4",
  "queue_id": "q-1",
  "kernel_id": "kern-1",
  "kernel_name": "matmul",
  "program_id": "prog-1",
  "work_dim": 2,
  "global": [4096, 320],
  "global_offset": [0, 0],
  "local": [32, 8],
  "args": [
    {"type": "buffer", "id": "buf-a", "size": 4194304},
    {"type": "buffer", "id": "buf-b", "size": 2097152},
    {"type": "scalar", "size": 8, "data_b64": "AAAAAEA="}
  ]
}
*/

/* Buffer write via shared memory (ICD → VGPU Core):
{
  "type": "buffer_write",
  "msg_id": "msg-5",
  "buffer_id": "buf-a",
  "offset": 0,
  "size": 4194304,
  "shm_name": "/distribox_buf_a_0001"
}
// Then ICD writes data to shared memory, signals VGPU Core
// VGPU Core reads from shared memory, sends ACK:
{"type": "ok", "request_id": "msg-5"}
*/

// ── Shared memory configuration ────────────────────────

#define SHM_MAX_SIZE        (256ULL * 1024 * 1024)  // 256 MB max per shm segment
#define SHM_ALIGNMENT       64                       // Cache line alignment

#ifdef __cplusplus
}
#endif

#endif // DISTRIBOX_IPC_PROTOCOL_H
