<p align="center">
  <br>
  <img src="https://img.shields.io/badge/DistriBox-v0.4.0-00d4ff?style=for-the-badge" alt="version">
  <br><br>
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&duration=3000&pause=1000&color=00D4FF&center=true&vCenter=true&width=600&lines=%F0%9F%94%AE+DistriBox;Distributed+Virtual+GPU;One+GPU.+Any+Device." alt="DistriBox" />
</p>

<p align="center">
  <b>把多台设备的算力聚合成一个虚拟 GPU</b><br>
  <sub>AI 应用无需修改，自动使用局域网内所有设备的 GPU 算力</sub>
</p>

<p align="center">
  <a href="https://github.com/jerryzhang2906/distribox/releases"><img src="https://img.shields.io/github/v/release/jerryzhang2906/distribox?color=00d4ff&style=flat-square" alt="release"></a>
  <a href="https://github.com/jerryzhang2906/distribox/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="license"></a>
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Android-blue?style=flat-square" alt="platform">
  <img src="https://img.shields.io/badge/language-Go%20%7C%20C-orange?style=flat-square" alt="language">
  <a href="https://github.com/jerryzhang2906/distribox/stargazers"><img src="https://img.shields.io/github/stars/jerryzhang2906/distribox?style=flat-square" alt="stars"></a>
</p>

<hr>

## 🤔 这是什么？

> **就像蓝牙耳机自动连接手机一样，你的 AI 应用自动使用局域网里所有设备的 GPU 算力。**

DistriBox 是一个**分布式虚拟 GPU 平台**。它把多台设备（PC、手机）的 GPU/CPU 算力通过 OpenCL ICD 技术聚合成一块"虚拟显卡"，让任何 OpenCL 应用（Ollama、PyTorch、Stable Diffusion）**无需修改代码**即可使用分布式集群。

```
你电脑上的 AI 应用（Ollama / PyTorch / 任意 OpenCL 应用）
            │
            ▼
   ┌────────────────────┐
   │  distribox_icd.dll  │  ← 系统级拦截：应用不知道下面是虚拟的
   │  (OpenCL ICD)        │
   └────────┬───────────┘
            │ TCP (127.0.0.1:9876)
            ▼
   ┌────────────────────┐
   │  distribox.exe      │  ← VGPU Core：调度 + 拆分 + 合并
   │  仪表盘 :13801       │
   └────────┬───────────┘
            │ gRPC (WiFi)
     ┌──────┴──────┐
     ▼              ▼
 ┌────────┐   ┌────────────┐
 │ PC     │   │ 📱 Android  │
 │ Worker │   │ Worker      │
 │ i7 CPU │   │ Mali GPU    │
 └────────┘   └────────────┘
```

### 能做什么？

| 场景 | 效果 |
|------|------|
| 🤖 **跑大模型** | 手机 GPU 加速 LLM 推理（Ollama, llama.cpp） |
| 🎨 **AI 绘画** | Stable Diffusion 并行利用所有设备 |
| 🧪 **科学计算** | 多设备分布式矩阵运算 |
| 🎮 **游戏渲染** | Vulkan/OpenGL 分布式渲染（实验性） |

---

## 📦 快速开始

### 第一步：下载

从 [Releases](https://github.com/jerryzhang2906/distribox/releases) 下载最新版本：

| 平台 | 文件 | 说明 |
|------|------|------|
| 🖥️ **PC (Windows)** | [`distribox-windows.zip`](https://github.com/jerryzhang2906/distribox/releases/latest) | 解压即用 |
| 📱 **Android** | [`distribox-worker.apk`](https://github.com/jerryzhang2906/distribox/releases/latest) | 安装到手机 |

### 第二步：启动 PC

```powershell
# 解压后双击 distribox.exe，自动弹出深色仪表盘窗口
.\distribox.exe

# 或纯控制台模式
.\distribox.exe --console
```

双击运行后，会弹出内置的赛博朋克风格深色仪表盘：

```
     ⚡ ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄ ⚡
     ▐   DISTRIBOX — Distributed Virtual GPU   ▐
     ▐  v0.4.0  |  Unified Launcher            ▐
     ▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▐
     ▐  One GPU. Any Device. Zero Config.      ▐
     ▐▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▐

  API → localhost:13801
```

### 第三步：连接手机

1. 安装 `distribox-worker.apk` 到 Android 手机
2. 确保手机和电脑在**同一 WiFi**
3. 打开 App → 点 **START WORKER**
4. 手机自动通过 **mDNS** 发现电脑并连接

> 💡 **零配置！** 同一 WiFi 下全自动互相发现。

### 第四步：使用

现在虚拟 GPU 已就绪，任何 OpenCL 应用都能使用：

```powershell
# 用 Ollama 跑模型 — 自动利用手机的 Mali GPU 加速
ollama run qwen2:7b "解释量子计算"

# GPU-Z 里可以看到虚拟 GPU
gpu-z.exe
```

---

## 🧠 工作原理

### 14 个 AI 内核

| 内核 | 用途 | CPU | GPU |
|------|------|:---:|:---:|
| `matmul` | 矩阵乘法（QKV 投影、FFN） | ✅ | ✅ |
| `gelu` | GPT/LLaMA 激活 | ✅ | ✅ |
| `rms_norm` | LLaMA/Mistral 归一化 | ✅ | ✅ |
| `layer_norm` | Transformer 归一化 | ✅ | ✅ |
| `softmax` | 注意力分数 | ✅ | ✅ |
| `rope` | 旋转位置编码 (RoPE) | ✅ | ✅ |
| `relu` | 激活函数 | ✅ | ✅ |
| `sigmoid` | 门控函数 | ✅ | ✅ |
| `element_wise_mul` | Hadamard 积 | ✅ | ✅ |
| `scalar_mul` | 标量乘法 | ✅ | ✅ |
| `transpose` | 矩阵转置 | ✅ | ✅ |
| `reduce_sum` | 求和归约 | ✅ | ✅ |
| `add_bias` | 偏置加法 | ✅ | ✅ |
| `vector_add` | 向量加法 | ✅ | ✅ |

### 一次推理的旅程

```
1. Ollama 调用 clEnqueueNDRangeKernel(gelu, input, output)
2. distribox_icd.dll 拦截 → JSON → VGPU Core
3. VGPU Core 拆分: PC 做 60%，手机做 40%
4. 并行执行 → 结果回传
5. VGPU Core 合并 → 返回给 Ollama
6. Ollama 完全无感知，延迟 ≈ max(PC, 手机) + 5ms 网络
```

---

## 🔧 命令行

```powershell
# GUI 模式（默认，双击弹出内置深色仪表盘）
.\distribox.exe

# 控制台模式（终端内 ANSI 实时面板）
.\distribox.exe --console

# 只启动 VGPU Core（让其他设备连过来）
.\distribox.exe --mode vgpu

# 只启动 Worker（连接到远程 VGPU）
.\distribox.exe --mode worker --orchestrator 192.168.1.100:13800

# 安装 GPU 拦截层（管理员权限）
.\distribox.exe install

# 查看集群状态
.\distribox.exe status

# 版本信息
.\distribox.exe version
```

### 参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--grpc-port` | 13800 | Worker gRPC 端口 |
| `--http-port` | 13801 | Dashboard 端口 |
| `--ipc-addr` | 127.0.0.1:9876 | ICD ↔ VGPU Core 通信 |
| `--intensity` | 0.8 | 算力使用强度 (0-1) |
| `--name` | 自动检测 | Worker 显示名称 |
| `--orchestrator` | — | 远程 VGPU Core 地址 |

---

## 📊 性能

| 场景 | 延迟 | 吞吐 |
|------|------|------|
| 单次 kernel 调用 | ~5-20ms | — |
| 大矩阵 (2048×8192) | ~0ms* | 122+ GFLOPS |
| 连续推理 | ~5-20ms/step | 取决于模型 |
| 3 设备集群 | ~3× 单设备 | 近线性扩展 |

*\*异步执行，实际耗时取决于慢设备*

### 对比

| | DistriBox | 单 GPU | 纯 CPU |
|--|-----------|--------|--------|
| 硬件要求 | 任何设备 | 需要独立显卡 | 任何 CPU |
| VRAM 池化 | ✅ 多设备叠加 | ❌ 单卡限制 | ❌ 系统内存 |
| 应用兼容性 | 100% (OpenCL 2.0) | 100% | 需重写 |
| 零配置 | ✅ mDNS | ✅ | ✅ |
| 手机加速 | ✅ | ❌ | ❌ |

---

## 🏗️ 开发者：从源码构建

| 工具 | 用途 | 安装 |
|------|------|------|
| **Go** 1.21+ | 主程序 | `winget install GoLang.Go` |
| **Zig** 0.13+ | C 编译 | `winget install zig.zig` |
| **CMake** 3.16+ | C 构建 | `winget install Kitware.CMake` |
| **Ninja** | 快速构建 | `winget install Ninja-build.Ninja` |

```bash
git clone https://github.com/jerryzhang2906/distribox.git
cd distribox
make build          # 编译所有组件
make dist-windows   # 打包 Windows 发布版
```

### 项目结构

```
distribox/
├── cmd/
│   ├── distribox/    🚀 统一启动器（一个 exe 替代全部）
│   ├── worker/       🔧 Worker Agent（算力提供者）
│   └── cli/          💻 命令行管理工具
├── vgpu/             🧠 Virtual GPU Core
│   ├── server/       gRPC + HTTP + IPC 服务
│   ├── scheduler/    任务拆分与调度
│   └── mem/          虚拟 VRAM + KV Cache
├── icd/              🔌 OpenCL ICD 拦截层（纯 C）
├── engine/           ⚡ 计算引擎（C + OpenCL）
├── proto/            📡 gRPC 协议 (Protobuf)
├── pkg/
│   ├── discovery/    📡 mDNS 零配置发现
│   ├── security/     🔒 加密 Token + mTLS
│   └── installer/    📦 Windows 一键安装
├── android/          📱 Android APK
├── vk_layer/         🎮 Vulkan 隐式拦截层
├── cuda_proxy/       🎯 CUDA 代理 DLL
└── examples/         📖 推理示例 + E2E 测试
```

---

## 📱 Android APK

```bash
# 安装
adb install build/distribox-worker.apk

# 构建（需要 Android NDK）
$env:ANDROID_NDK_HOME = "你的NDK路径"
.\scripts\build_android.ps1
```

**App 功能:**
- 🎨 Material Dark UI — 现代化深色界面，脉冲状态指示器
- ⚡ 一键连接 — 打开即用，mDNS 自动发现
- 📊 实时显示 CPU/GPU/内存状态
- 🎚️ 算力滑动条调节贡献比例
- 🔔 前台服务 + WakeLock 保活，通知栏状态
- 📋 滑动日志面板，实时查看连接状态

---

## 🛠️ 技术栈

| 层 | 技术 |
|-----|------|
| 网络 | gRPC + Protocol Buffers |
| GPU 拦截 | OpenCL ICD (纯 C, 50+ API) |
| 服务发现 | mDNS (纯 Go) |
| 计算引擎 | Go + C + OpenCL |
| 调度 | 加权拆分算法 |
| 安全 | Cluster Token + mTLS |
| 移动端 | Go ARM64 + NDK 交叉编译 |

---

## ❓ FAQ

<details>
<summary><b>Q: 安全吗？别人能连上我的 GPU 吗？</b></summary>
<b>A:</b> 局域网自动生成 128-bit 加密 Cluster Token，只有同一 WiFi 下的设备才能连接。支持 mTLS 双向认证。
</details>

<details>
<summary><b>Q: 需要什么手机？</b></summary>
<b>A:</b> Android 8.0+ 即可。有 GPU（Mali/Adreno）效果更好，纯 CPU 也能贡献算力。
</details>

<details>
<summary><b>Q: 能跑多大的模型？</b></summary>
<b>A:</b> 取决于集群总内存。3 台 8GB 设备 = 24GB "虚拟 VRAM"，可跑 13B+ 模型。
</details>

<details>
<summary><b>Q: 支持苹果设备吗？</b></summary>
<b>A:</b> macOS Worker 已支持（源码编译）。iOS 暂不支持。
</details>

<details>
<summary><b>Q: 延迟怎么样？</b></summary>
<b>A:</b> 同一 WiFi 下 5-20ms。建议 5GHz WiFi 获得最佳性能。
</details>

---

## 🤝 贡献

欢迎 Issue 和 PR！

```bash
go test ./...                     # 所有 Go 测试
go test -v ./vgpu/server/         # IPC + gRPC 集成测试
go test -v ./cmd/worker/engine/   # AI 内核正确性测试
```

---

## 📄 许可证

[MIT](LICENSE) © 2025-2026

---

## ⭐ Star History

<p align="center">
  <sub>如果这个项目对你有帮助，请给一个 ⭐ Star！</sub>
</p>

---

<p align="center">
  <sub>Built with ❤️ using Go, C, OpenCL, gRPC, and a lot of ☕</sub>
</p>
