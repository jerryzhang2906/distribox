/**
 * ipc/ipc_client.c — TCP-based IPC client for ICD ↔ VGPU Core communication
 *
 * The ICD library connects to the local VGPU Core daemon via
 * TCP localhost (127.0.0.1). All commands and small responses
 * flow over this connection as newline-delimited JSON.
 */

#include "../icd.h"
#include "ipc_protocol.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
typedef SOCKET socket_t;
#define SOCKET_INVALID INVALID_SOCKET
#define sock_close closesocket
#define sock_errno WSAGetLastError()
#define sock_eagain WSAEWOULDBLOCK
#else
#include <sys/socket.h>
#include <sys/un.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
typedef int socket_t;
#define SOCKET_INVALID (-1)
#define sock_close close
#define sock_errno errno
#define sock_eagain EAGAIN
#endif

static socket_t g_sock = SOCKET_INVALID;
static uint64_t g_msg_counter = 0;

// ── TCP connection to VGPU Core ──────────────────────

int ipc_connect(void) {
    if (g_sock != SOCKET_INVALID) {
        return 0; // Already connected
    }

#ifdef _WIN32
    // Initialize Winsock
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        return -1;
    }
#endif

    g_sock = socket(AF_INET, SOCK_STREAM, 0);
    if (g_sock == SOCKET_INVALID) {
        return -1;
    }

    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(IPC_TCP_PORT);
    addr.sin_addr.s_addr = inet_addr("127.0.0.1");

    if (connect(g_sock, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        sock_close(g_sock);
        g_sock = SOCKET_INVALID;
        return -1;
    }

    g_ipc_fd = (int)(intptr_t)g_sock;
    g_ipc_connected = true;

    // ── Send hello message ──────────────────────────
    char hello[256];
    snprintf(hello, sizeof(hello),
        "{\"type\":\"%s\",\"msg_id\":\"msg-%llu\",\"protocol_version\":\"1.0\"}\n",
        IPC_MSG_HELLO, (unsigned long long)++g_msg_counter);

    ipc_send_command(hello, strlen(hello));

    // Read device config response
    char resp[4096];
    int n = ipc_recv_response(resp, sizeof(resp), 2000);
    if (n > 0) {
        // TODO: parse device config and update g_device specs
        (void)n;
    }

    return 0;
}

// ── Send a command (JSON string) ─────────────────────

int ipc_send_command(const char *json, uint64_t len) {
    if (g_sock == SOCKET_INVALID) {
        return -1;
    }

    uint64_t sent = 0;
    while (sent < len) {
        int n = send(g_sock, json + sent, (int)(len - sent), 0);
        if (n < 0) {
#ifdef _WIN32
            if (WSAGetLastError() == WSAEINTR) continue;
#else
            if (errno == EINTR) continue;
#endif
            return -1;
        }
        sent += n;
    }
    return 0;
}

// ── Receive a response (JSON string, newline-delimited) ─

int ipc_recv_response(char *buf, uint64_t max_len, int timeout_ms) {
    if (g_sock == SOCKET_INVALID) {
        return -1;
    }

    // Set socket timeout for recv
#ifdef _WIN32
    DWORD tv = (DWORD)timeout_ms;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVTIMEO, (const char *)&tv, sizeof(tv));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(g_sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
#endif

    uint64_t received = 0;

    while (received < max_len - 1) {
        int n = recv(g_sock, buf + received, (int)(max_len - received - 1), 0);
        if (n < 0) {
            break; // Timeout or error
        }
        if (n == 0) break; // Connection closed
        received += n;
        buf[received] = '\0';

        // Check for newline terminator
        if (buf[received - 1] == '\n') {
            buf[received - 1] = '\0'; // Strip newline
            break;
        }
    }

    return (int)received;
}

// ── Disconnect ────────────────────────────────────────

void ipc_disconnect(void) {
    if (g_sock != SOCKET_INVALID) {
        sock_close(g_sock);
        g_sock = SOCKET_INVALID;
    }
    g_ipc_fd = -1;
    g_ipc_connected = false;
#ifdef _WIN32
    WSACleanup();
#endif
}
