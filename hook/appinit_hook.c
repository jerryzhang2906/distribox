/*
 * hook/appinit_hook.c — AppInit_DLLs injection hook
 *
 * Loaded by user32.dll into EVERY GUI process.
 * Hooks OpenGL wglSwapBuffers for FPS overlay and future distributed rendering.
 *
 * Install:
 *   reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows"
 *     /v AppInit_DLLs /t REG_SZ /d "C:\distribox_hook.dll"
 *   reg add "HKLM\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Windows"
 *     /v LoadAppInit_DLLs /t REG_DWORD /d 1
 *
 * Build: zig cc -shared -O2 appinit_hook.c -o distribox_hook.dll
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>

// ── Log ─────────────────────────────────────────────────

static FILE* g_log = NULL;
static void hlog(const char* fmt, ...) {
    if (!g_log) {
        // Try user-writable locations first
        const char* home = getenv("LOCALAPPDATA");
        char path[512];
        if (home) snprintf(path, sizeof(path), "%s\\DistriBox\\hook.log", home);
        else strcpy(path, "C:\\ProgramData\\DistriBox\\hook.log");
        // Ensure directory exists
        char dir[512]; strcpy(dir, path);
        char* slash = strrchr(dir, '\\');
        if (slash) { *slash = 0; CreateDirectoryA(dir, NULL); CreateDirectoryA("C:\\ProgramData\\DistriBox", NULL); }
        g_log = fopen(path, "a");
        if (!g_log) g_log = fopen("C:\\ProgramData\\DistriBox\\hook.log", "a");
        if (!g_log) return;
    }
    va_list a; va_start(a, fmt); vfprintf(g_log, fmt, a); fflush(g_log); va_end(a);
}

// ── OpenGL hook ─────────────────────────────────────────

typedef BOOL (WINAPI *PFN_wglSwapBuffers)(HDC);
static PFN_wglSwapBuffers real_wglSwapBuffers = NULL;

static int g_frames = 0;
static DWORD g_lastFPS = 0;

static BOOL WINAPI hook_wglSwapBuffers(HDC hdc) {
    g_frames++;
    DWORD now = GetTickCount();
    if (now - g_lastFPS > 2000 && g_lastFPS > 0) {
        float fps = g_frames * 1000.0f / (now - g_lastFPS);
        hlog("🎮 FPS: %.1f | frames: %d\n", fps, g_frames);
        g_frames = 0;
    }
    if (g_lastFPS == 0) g_lastFPS = now;

    // Call real wglSwapBuffers
    if (real_wglSwapBuffers) return real_wglSwapBuffers(hdc);
    return TRUE;
}

// ── DXGI hook ──────────────────────────────────────────

typedef HRESULT (WINAPI *PFN_DXGI_Present)(void*, UINT, UINT);
static PFN_DXGI_Present real_DXGI_Present = NULL;

// ── IAT hook: modify lwjgl_opengl.dll import table ──────

static BOOL iat_hook_wglSwapBuffers(void) {
    // Find lwjgl_opengl.dll
    HMODULE hLWJGL = GetModuleHandleA("lwjgl_opengl.dll");
    if (!hLWJGL) return FALSE;

    // Get opengl32's wglSwapBuffers address (our target)
    HMODULE hGL = GetModuleHandleA("opengl32.dll");
    if (!hGL) return FALSE;
    void* realFunc = GetProcAddress(hGL, "wglSwapBuffers");
    if (!realFunc) return FALSE;

    // Parse PE header
    PIMAGE_DOS_HEADER dos = (PIMAGE_DOS_HEADER)hLWJGL;
    PIMAGE_NT_HEADERS nt = (PIMAGE_NT_HEADERS)((BYTE*)hLWJGL + dos->e_lfanew);
    IMAGE_DATA_DIRECTORY importDir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (importDir.Size == 0) return FALSE;

    PIMAGE_IMPORT_DESCRIPTOR imp = (PIMAGE_IMPORT_DESCRIPTOR)((BYTE*)hLWJGL + importDir.VirtualAddress);

    // Walk import descriptors looking for opengl32.dll
    for (; imp->Name; imp++) {
        char* dllName = (char*)((BYTE*)hLWJGL + imp->Name);
        if (_stricmp(dllName, "opengl32.dll") != 0) continue;

        // Found opengl32.dll — walk its thunks
        PIMAGE_THUNK_DATA thunk = (PIMAGE_THUNK_DATA)((BYTE*)hLWJGL + imp->FirstThunk);
        for (; thunk->u1.Function; thunk++) {
            if (thunk->u1.Function == (ULONGLONG)realFunc) {
                // Found wglSwapBuffers! Replace with our hook
                DWORD oldProtect;
                VirtualProtect(&thunk->u1.Function, sizeof(void*), PAGE_READWRITE, &oldProtect);
                thunk->u1.Function = (ULONGLONG)hook_wglSwapBuffers;
                VirtualProtect(&thunk->u1.Function, sizeof(void*), oldProtect, &oldProtect);

                hlog("IAT HOOK: lwjgl_opengl!wglSwapBuffers @ %p -> FPS counter\n", realFunc);
                return TRUE;
            }
        }
    }
    return FALSE;
}

// ── DllMain ─────────────────────────────────────────────

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)hinst; (void)reserved;

    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinst);

        char procName[256];
        GetModuleFileNameA(NULL, procName, sizeof(procName));
        hlog(">>> [%lu] %s\n", GetCurrentProcessId(), procName);

        // Load OpenGL to check if this is a rendering process
        HMODULE hGL = LoadLibraryA("opengl32.dll");
        if (hGL) {
            real_wglSwapBuffers = (PFN_wglSwapBuffers)GetProcAddress(hGL, "wglSwapBuffers");
            if (!strstr(procName, "system32") && !strstr(procName, "SYSTEM32")) {
                hlog(">>> GAME: %s | wglSwapBuffers=%p\n", procName, real_wglSwapBuffers);

                // Try IAT hook on lwjgl_opengl.dll (safe — modifies data, not code)
                if (iat_hook_wglSwapBuffers()) {
                    hlog(">>> FPS counter ACTIVE! Every frame logged.\n");
                }
            }
        }
    }
    if (reason == DLL_PROCESS_DETACH) {
        hlog("<<< [%lu] unloaded\n", GetCurrentProcessId());
        if (g_log) fclose(g_log);
    }
    return TRUE;
}
