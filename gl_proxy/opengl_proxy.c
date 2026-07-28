/*
 * gl_proxy/opengl_proxy.c — OpenGL proxy DLL for Minecraft shader interception
 *
 * Intercepts OpenGL calls from Minecraft (Java + OptiFine/Iris/Sodium)
 * to enable FPS overlay, shader analysis, and distributed chunk rendering.
 *
 * Architecture:
 *   MC Java → opengl32.dll (our proxy) → opengl32_orig.dll (real driver)
 *
 * Install (admin):
 *   cd C:\Windows\System32
 *   ren opengl32.dll opengl32_orig.dll
 *   copy distri_opengl32.dll opengl32.dll
 *
 * Build (zig):
 *   zig cc -shared -O2 opengl_proxy.c -o distri_opengl32.dll
 */

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdarg.h>

// Minimal GL types
typedef unsigned int GLbitfield;
typedef unsigned int GLenum;
typedef unsigned int GLuint;
typedef int GLint;
typedef float GLfloat;
typedef double GLdouble;
typedef char GLchar;
typedef void GLvoid;
typedef struct HGLRC__* HGLRC;

// ── Logging ─────────────────────────────────────────────

static FILE* g_log = NULL;
static void glog(const char* fmt, ...) {
    if (!g_log) { g_log = fopen("distribox_gl.log", "a"); if (!g_log) return; }
    va_list a; va_start(a, fmt); vfprintf(g_log, fmt, a); fflush(g_log); va_end(a);
}

// ── Real OpenGL function pointers ──────────────────────

static HMODULE g_realGL = NULL;

typedef BOOL  (WINAPI *PFN_wglSwapBuffers)(HDC);
typedef BOOL  (WINAPI *PFN_wglSwapLayerBuffers)(HDC, UINT);
typedef HGLRC (WINAPI *PFN_wglCreateContext)(HDC);
typedef BOOL  (WINAPI *PFN_wglDeleteContext)(HGLRC);
typedef BOOL  (WINAPI *PFN_wglMakeCurrent)(HDC, HGLRC);
typedef PROC  (WINAPI *PFN_wglGetProcAddress)(LPCSTR);
typedef void  (WINAPI *PFN_glClear)(GLbitfield);
typedef void  (WINAPI *PFN_glFlush)(void);
typedef void  (WINAPI *PFN_glFinish)(void);

static PFN_wglSwapBuffers  real_wglSwapBuffers  = NULL;
static PFN_wglMakeCurrent  real_wglMakeCurrent  = NULL;
static PFN_wglCreateContext real_wglCreateContext = NULL;
static PFN_wglGetProcAddress real_wglGetProcAddress = NULL;

// ── FPS counter ────────────────────────────────────────

static int g_frames = 0;
static DWORD g_lastFPS = 0;

// ── Intercepted: wglSwapBuffers (every frame!) ──────────

BOOL WINAPI proxy_wglSwapBuffers(HDC hdc) {
    g_frames++;
    DWORD now = GetTickCount();
    if (now - g_lastFPS > 2000 && g_lastFPS > 0) {
        float fps = g_frames * 1000.0f / (now - g_lastFPS);
        glog("MC FPS: %.1f | frame %d\n", fps, g_frames);
        g_frames = 0;
    }
    if (g_lastFPS == 0) g_lastFPS = now;

    // Chain to real driver
    if (real_wglSwapBuffers) return real_wglSwapBuffers(hdc);
    return TRUE;
}

// ── Intercepted: wglMakeCurrent ────────────────────────

BOOL WINAPI proxy_wglMakeCurrent(HDC hdc, HGLRC hglrc) {
    glog("wglMakeCurrent: HDC=%p HGLRC=%p\n", hdc, hglrc);
    if (real_wglMakeCurrent) return real_wglMakeCurrent(hdc, hglrc);
    return TRUE;
}

// ── Intercepted: wglCreateContext ──────────────────────

HGLRC WINAPI proxy_wglCreateContext(HDC hdc) {
    glog("wglCreateContext: HDC=%p\n", hdc);
    if (real_wglCreateContext) return real_wglCreateContext(hdc);
    return NULL;
}

// ── Intercepted: wglGetProcAddress ──────────────────────

PROC WINAPI proxy_wglGetProcAddress(LPCSTR name) {
    if (name) glog("wglGetProcAddress: %s\n", name);
    if (real_wglGetProcAddress) return real_wglGetProcAddress(name);
    return NULL;
}

// ── Init: load real OpenGL ──────────────────────────────

static void load_real_gl(void) {
    static int done = 0;
    if (done) return; done = 1;

    g_realGL = LoadLibraryA("C:\\Windows\\System32\\opengl32.dll");
    if (!g_realGL) {
        glog("FATAL: cannot load opengl32_orig.dll\n");
        return;
    }

    real_wglSwapBuffers = (PFN_wglSwapBuffers)GetProcAddress(g_realGL, "wglSwapBuffers");
    real_wglMakeCurrent = (PFN_wglMakeCurrent)GetProcAddress(g_realGL, "wglMakeCurrent");
    real_wglCreateContext = (PFN_wglCreateContext)GetProcAddress(g_realGL, "wglCreateContext");
    real_wglGetProcAddress = (PFN_wglGetProcAddress)GetProcAddress(g_realGL, "wglGetProcAddress");

    glog("DistriBox GL Proxy loaded — real OpenGL: %p\n", g_realGL);
    glog("  wglSwapBuffers: %p\n", real_wglSwapBuffers);
    glog("  wglMakeCurrent: %p\n", real_wglMakeCurrent);
}

// ── DLL Entry ──────────────────────────────────────────

BOOL WINAPI DllMain(HINSTANCE hinst, DWORD reason, LPVOID reserved) {
    (void)hinst; (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        DisableThreadLibraryCalls(hinst);
        load_real_gl();
    }
    if (reason == DLL_PROCESS_DETACH) {
        glog("=== GL Proxy unloaded (PID=%lu) ===\n", GetCurrentProcessId());
        if (g_log) fclose(g_log);
    }
    return TRUE;
}

// ── Export ALL OpenGL/WGL functions ─────────────────────
// We forward these to the real driver via a .def file
// Only the intercepted functions are implemented here.
//
// For a complete proxy, we'd need a .def file with all exports.
// For Minecraft (which uses wglGetProcAddress for modern GL),
// intercepting the WGL functions is sufficient.
//
// The .def file forwards everything else to opengl32_orig.dll
