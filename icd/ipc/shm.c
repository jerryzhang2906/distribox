/**
 * ipc/shm.c — Shared memory transport for large buffer transfers
 *
 * When buffer data exceeds a threshold (default 1 MB), we use
 * shared memory instead of the socket to avoid serialization overhead.
 *
 * Linux: POSIX shm_open / mmap
 * Windows: CreateFileMapping / MapViewOfFile
 */

#include "../icd.h"
#include "ipc_protocol.h"
#include <string.h>

#ifdef _WIN32
#include <windows.h>

int ipc_shm_write(const char *shm_name, const void *data, uint64_t size) {
    if (size > SHM_MAX_SIZE) return -1;

    HANDLE hMap = CreateFileMappingA(
        INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE,
        (DWORD)(size >> 32), (DWORD)(size & 0xFFFFFFFF), shm_name
    );
    if (hMap == NULL) return -1;

    void *ptr = MapViewOfFile(hMap, FILE_MAP_WRITE, 0, 0, (SIZE_T)size);
    if (ptr == NULL) {
        CloseHandle(hMap);
        return -1;
    }

    memcpy(ptr, data, (size_t)size);
    UnmapViewOfFile(ptr);
    CloseHandle(hMap);
    return 0;
}

int ipc_shm_read(const char *shm_name, void *buf, uint64_t size, uint64_t offset) {
    HANDLE hMap = OpenFileMappingA(FILE_MAP_READ, FALSE, shm_name);
    if (hMap == NULL) return -1;

    void *ptr = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, (SIZE_T)(offset + size));
    if (ptr == NULL) {
        CloseHandle(hMap);
        return -1;
    }

    memcpy(buf, (char *)ptr + offset, (size_t)size);
    UnmapViewOfFile(ptr);
    CloseHandle(hMap);
    return 0;
}

void ipc_shm_unlink(const char *shm_name) {
    (void)shm_name;
    // Windows: memory is auto-released when all handles are closed
}

#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

int ipc_shm_write(const char *shm_name, const void *data, uint64_t size) {
    if (size > SHM_MAX_SIZE) return -1;

    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600);
    if (fd < 0) return -1;

    if (ftruncate(fd, (off_t)size) < 0) {
        close(fd);
        shm_unlink(shm_name);
        return -1;
    }

    void *ptr = mmap(NULL, (size_t)size, PROT_WRITE, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        shm_unlink(shm_name);
        return -1;
    }

    memcpy(ptr, data, (size_t)size);
    munmap(ptr, (size_t)size);
    close(fd);
    return 0;
}

int ipc_shm_read(const char *shm_name, void *buf, uint64_t size, uint64_t offset) {
    int fd = shm_open(shm_name, O_RDONLY, 0600);
    if (fd < 0) return -1;

    void *ptr = mmap(NULL, (size_t)(offset + size), PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        return -1;
    }

    memcpy(buf, (char *)ptr + offset, (size_t)size);
    munmap(ptr, (size_t)(offset + size));
    close(fd);
    return 0;
}

void ipc_shm_unlink(const char *shm_name) {
    shm_unlink(shm_name);
}
#endif
