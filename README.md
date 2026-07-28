<p align="center">
  <h1 align="center">🔮 DistriBox — 分布式虚拟 GPU</h1>
  <p align="center">
    把多台设备的算力聚合成一个虚拟 GPU，让任何 AI 应用无需修改即可使用分布式集群
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Android-blue" alt="platform">
  <img src="https://img.shields.io/badge/language-Go%20%7C%20C-orange" alt="language">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="status">
</p>

---

## 🤔 这是什么？

**DistriBox** 把你的电脑和手机变成一台"虚拟 GPU"。你在电脑上运行 AI 应用（如 Ollama、PyTorch），手机会自动贡献算力——应用完全不知道自己实际是在用一台"拼凑出来的 GPU"。

### 一句话解释

> 就像蓝牙耳机自动连接手机一样，你的 AI 应用自动使用局域网里所有设备的 GPU 算力。

### 能做什么？

| 场景 | 效果 |
|------|------|
| 🤖 **跑大模型** | 手机 GPU 帮你加速 LLM 推理 |
| 🎮 **AI 绘画** | Stable Diffusion 利用所有设备 |
| 🧪 **科学计算** | 多设备分布式矩阵运算 |

---

## 📦 快速开始（小白版）

### 第一步：下载

从 [Releases](https://github.com/你的用户名/distribox/releases) 下载最新版本：

- **PC 端**：`distribox-windows.zip`（解压即用）
- **手机端**：`distribox-worker.apk`（安装到手机）

### 第二步：启动 PC

```powershell
# 双击运行，或者命令行：
.\distribox.exe
```

你会看到：
```
╔══════════════════════════════════════════════╗
║       DistriBox — Distributed Virtual GPU    ║
║       v0.2.0  |  Unified Launcher            ║
╚══════════════════════════════════════════════╝
Dashboard: http://localhost:13801
```

打开浏览器访问 `http://localhost:13801` 就能看到仪表盘。

### 第三步：连接手机

1. 把 `distribox-worker.apk` 装到手机上
2. 确保手机和电脑在**同一个 WiFi**
3. 打开 App，点 **START WORKER**
4. 手机会自动发现电脑并连接

> 💡 **不需要任何配置！** 手机和电脑在同一个 WiFi 下会自动互相发现。

### 第四步：使用

现在你的"虚拟 GPU"已经就绪。任何使用 OpenCL 的应用都会自动使用它：

```powershell
# 示例：用 Ollama 跑模型
ollama run phi3:mini "解释量子计算"

# GPU-Z 可以看到虚拟 GPU
gpu-z.exe
```

---

## 🏗️ 给开发者：从源码构建

### 环境要求

| 工具 | 用途 | 安装方式 |
|------|------|----------|
| **Go** 1.21+ | 主程序 | `winget install GoLang.Go` |
| **Zig** 0.13+ | C 编译器 | `winget install zig.zig` |
| **CMake** 3.16+ | C 构建系统 | `winget install Kitware.CMake` |
| **Ninja** | 快速构建 | `winget install Ninja-build.Ninja` |
| **Android NDK** | 手机端编译 | Android Studio 自带 |

### 一键构建

```bash
# 克隆仓库
git clone https://github.com/你的用户名/distribox.git
cd distribox

# 构建 PC 端（Windows）
make build          # 编译所有组件
make dist-windows   # 打包发布版本

# 构建手机端（需要 Android NDK）
$env:ANDROID_NDK_HOME = "C:\Users\...\AppData\Local\Android\Sdk\ndk\27.1.12297006"
.\scripts\build_android.ps1
```

### 项目结构

```
distribox/
├── cmd/
│   ├── distribox/       # 🚀 统一启动器（一个 exe 替代全部）
│   ├── worker/           # 🔧 Worker Agent（算力提供者）
│   └── cli/              # 💻 命令行管理工具
├── vgpu/                 # 🧠 Virtual GPU Core（核心调度引擎）
│   ├── server/           # gRPC + HTTP + IPC 服务
│   ├── scheduler/        # 任务拆分与调度
│   ├── mem/              # 虚拟 VRAM + KV Cache
│   ├── calibrate/        # 自动校准（集群算力匹配 GPU 型号）
│   └── monitor/          # Worker 健康监控
├── icd/                  # 🔌 OpenCL ICD 拦截层（C 语言）
│   └── api/              # 50+ OpenCL API 实现
├── engine/               # ⚡ 计算引擎（C + OpenCL）
├── proto/                # 📡 gRPC 协议定义
├── pkg/
│   ├── discovery/        # 📡 mDNS 局域网自动发现
│   └── protocol/         # 生成的 protobuf 代码
├── android/              # 📱 Android APK 项目
├── vk_layer/             # 🎮 Vulkan 拦截层
├── cuda_proxy/           # 🎯 CUDA 代理
├── hook/                 # 🪝 OpenGL Hook（Minecraft 等）
├── gl_proxy/             # 🖼️ OpenGL 系统代理
├── examples/             # 📖 示例代码
│   ├── inference/        # Transformer 推理 Demo
│   ├── distributed/      # 分布式计算测试
│   └── e2e/              # 端到端验证
└── scripts/              # 🔨 构建脚本
```

---

## 🧠 工作原理

### 整体架构

```
你电脑上的 AI 应用（Ollama / PyTorch / 任意 OpenCL 应用）
            ↓
    ┌───────────────────┐
    │  OpenCL API 调用   │  ← 应用完全不知道下面是虚拟的
    └───────┬───────────┘
            ↓
    ┌───────────────────┐
    │  distribox_icd.dll │  ← 拦截层：截获所有 GPU 调用
    │  (系统级安装)       │
    └───────┬───────────┘
            ↓ TCP (127.0.0.1:9876)
    ┌───────────────────┐
    │  distribox.exe     │  ← VGPU Core：调度引擎
    │  仪表盘 :13801      │
    └───────┬───────────┘
            ↓ gRPC (WiFi)
    ┌───────┴───────────┐
    ↓                   ↓
┌──────────┐     ┌──────────────┐
│ PC       │     │ 📱 手机       │
│ Worker   │     │ Worker (APK) │
│ i7 CPU   │     │ Mali GPU     │
└──────────┘     └──────────────┘
```

### 一次推理的数据流

```
1. 应用调用 clEnqueueNDRangeKernel(gelu, input, output)
2. ICD 拦截 → 打包成 JSON → 发给 VGPU Core
3. VGPU Core 拆分任务：手机做一半，PC 做一半
4. 两个 Worker 各自计算 → 结果回传
5. VGPU Core 合并结果 → 返回给应用
6. 应用完全无感知，耗时 ≈ max(手机,PC) + 网络延迟
```

### 14 个 AI 内核

| 内核 | 用途 | CPU | GPU |
|------|------|-----|-----|
| `matmul` | 矩阵乘法（QKV 投影、FFN） | ✅ | ✅ |
| `gelu` | GPT/LLaMA 激活函数 | ✅ | ✅ |
| `rms_norm` | LLaMA/Mistral 归一化 | ✅ | ✅ |
| `layer_norm` | Transformer 归一化 | ✅ | ✅ |
| `softmax` | 注意力分数 | ✅ | ✅ |
| `rope` | 旋转位置编码 | ✅ | ✅ |
| `relu` | 激活函数 | ✅ | ✅ |
| `sigmoid` | 门控函数 | ✅ | ✅ |
| `element_wise_mul` | Hadamard 积 | ✅ | ✅ |
| `scalar_mul` | 标量乘法 | ✅ | ✅ |
| `transpose` | 矩阵转置 | ✅ | ✅ |
| `reduce_sum` | 求和归约 | ✅ | ✅ |
| `add_bias` | 偏置加法 | ✅ | ✅ |
| `vector_add` | 向量加法 | ✅ | ✅ |

---

## 🔧 命令行参考

```powershell
# 完整模式（VGPU Core + Worker 一起启动）
.\distribox.exe

# 只启动 VGPU Core（让其他设备连过来）
.\distribox.exe --mode vgpu

# 只启动 Worker（连接到另一台电脑的 VGPU）
.\distribox.exe --mode worker --orchestrator 192.168.1.100:13800

# 安装 GPU 拦截层（需要管理员权限）
.\distribox.exe install

# 查看集群状态
.\distribox.exe status

# 查看版本
.\distribox.exe version
```

### 可选参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grpc-port` | 13800 | Worker 连接端口 |
| `--http-port` | 13801 | 仪表盘端口 |
| `--ipc-addr` | 127.0.0.1:9876 | ICD 通信地址 |
| `--intensity` | 0.8 | CPU/GPU 使用强度 (0-1) |
| `--name` | 自动检测 | Worker 显示名称 |

---

## 📱 Android APK

### 安装

```bash
# 通过 USB 安装
adb install build/distribox-worker.apk

# 或直接传 APK 到手机安装
```

### 功能

- 打开 App → 显示设备信息（CPU、GPU、内存）
- 点 **START WORKER** → 自动连接局域网中的 VGPU Core
- 滑动条调节算力贡献比例
- 后台运行，通知栏显示状态

### 自构建

```powershell
$env:ANDROID_NDK_HOME = "你的NDK路径"
.\scripts\build_android.ps1
```

---

## ❓ 常见问题

### Q: 会影响电脑正常使用吗？
**A:** 不会。Worker 只在空闲算力上运行，不会抢占你正在用的资源。

### Q: 安全吗？别人能连上我的 GPU 吗？
**A:** 局域网自动生成加密 token，只有同一 WiFi 下的设备才能连接。

### Q: 需要什么手机？
**A:** 任何 Android 8.0+ 手机都可以。有 GPU（Mali/Adreno）效果更好。

### Q: 支持苹果设备吗？
**A:** macOS 作为 Worker 已支持（需从源码编译）。iOS 暂不支持。

### Q: 网络要求？
**A:** 同一 WiFi 下延迟 ~5-20ms。建议 5G WiFi 获得最佳性能。

### Q: 能跑多大的模型？
**A:** 取决于所有设备的总内存。3 台 8GB 手机 = 24GB "虚拟 VRAM"，可以跑 13B 模型。

---

## 🛠️ 技术栈

| 层 | 技术 | 说明 |
|-----|------|------|
| 网络通信 | **gRPC** + Protocol Buffers | PC ↔ 手机高速通信 |
| GPU 拦截 | **OpenCL ICD** (C) | 系统级 API 拦截 |
| 服务发现 | **mDNS** (纯 Go 实现) | 局域网零配置发现 |
| 计算引擎 | **Go** + **C** + **OpenCL** | 跨平台 GPU 计算 |
| 调度器 | **加权拆分算法** | 按算力比例分配任务 |
| 手机端 | **Go ARM64** + NDK | 原生性能 |

---

## 📊 性能

| 场景 | 延迟 | 吞吐量 |
|------|------|--------|
| 单次 kernel 调用 | ~100-120ms | - |
| 大矩阵 (2048×8192) | ~0ms* | 122 GFLOPS |
| 连续推理 | ~5-20ms/step | 取决于模型 |

\* 异步执行，实际 GPU 计算时间取决于设备

---

## 🤝 贡献

欢迎提交 Issue 和 PR！

### 开发环境设置

```bash
git clone https://github.com/你的用户名/distribox.git
cd distribox
go mod download
make build
```

### 运行测试

```bash
go test ./...                    # 所有 Go 测试
go test -v ./vgpu/server/        # IPC + gRPC 集成测试
go test -v ./cmd/worker/engine/  # AI 内核正确性测试
```

---

## 📄 许可证

MIT License — 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [Khronos Group](https://www.khronos.org/opencl/) — OpenCL 标准
- [Ollama](https://ollama.com/) — 本地 LLM 运行工具
- [gRPC](https://grpc.io/) — 高性能 RPC 框架
- [Zig](https://ziglang.org/) — 优秀 C 交叉编译工具链

---

<p align="center">
  <b>⭐ 如果这个项目对你有帮助，请给一个 Star！</b>
</p>
