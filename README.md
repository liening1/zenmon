<div align="center">

# ✦ ZENMON

**A serene, zero-dependency telemetry TUI for modern workstations & AI rigs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-black.svg)](https://www.python.org/)
[![Dependencies](https://img.shields.io/badge/dependencies-0%20(pure%20stdlib)-success.svg)](#zero-dependencies)
[![Architecture](https://img.shields.io/badge/arch-x86__64%20%7C%20aarch64-lightgrey.svg)](#universal-hardware)

*Low data-ink. Semantic grayscale. Zero visual noise. Instant launch.*

[Features](#-key-features) • [Installation](#-installation) • [Keybindings](#-keybindings) • [Comparison](#-why-zenmon) • [Architecture](#-universal-hardware-engine)

<br/><br/>

<img src="assets/demo.gif" alt="ZenMon Live Demo" width="95%" />

</div>

---

<details>
<summary><b>📄 Text Layout Preview (ASCII)</b></summary>

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ ZENMON ✦ quantum-node-1 ✦ AMD Ryzen 9 9950X (32T)                UP: 170h 56m  [LIVE]  │
│ PSU BUDGET [550W]  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░   182.4W  |  REMAIN: 367.6W │
│ ── COMPUTE [32T] ──────────────────────   ── ACCELERATOR [RTX 5060 Ti] ─────────────── │
│ Util   ██░░░░░░░░    12.4%                Util   █████████░    92.1%  SM:2647M Mem:13801M│
│ Temp   ████░░░░░░    58.2°C  C1:56° C2:52° Temp   █████░░░░░    62.0°C  Fan:45%        │
│ Pwr    ███░░░░░░░    65.4W                Pwr    ████████░░   117.0W / 180W            │
│ Load   1.24 1.15 0.98            Avg 5230M VRAM   ███████░░░    11.4G / 16.0G          │
│ Cores  ▂▃▄▅▆▇█ ▂▃▄▅▆▇█ ▂▃▄▅▆▇█ ▂▃▄▅▆▇█     PCIe   ▲   1.2M  ▼  18.4M                  │
│ ── MEMORY & STORAGE ───────────────────   ── MOTHERBOARD & SENSORS ─────────────────── │
│ RAM    █████░░░░░    17.0G / 30.5G C:8.4G NVMe   ███░░░░░░░    42.0°C  D2:38°C         │
│ SWP    ░░░░░░░░░░     0.0G /  2.0G        Mobo   temp1:38°  temp2:54°  temp3:87°       │
│ Disk/  ████░░░░░░   384G/ 953G (40%)      NET    ▲ 142.5 K/s  ▼   2.4 M/s              │
│ NVMeIO R:  0.2M W: 14.8M (320 IOPS)       Fans   fan1:1250RPM fan2:980RPM              │
│ ── TOP PROCESSES (Sorted by CPU) ──────── [✦ AI Workloads Tagged] ──────────────────── │
│    PID   USER       PRI  NI   VIRT    RES   %CPU  %MEM   COMMAND                           │
│  12842  quantum1   20   0  14.2G  11.4G  158.4  37.4   ✦[Ollama] ollama serve            │
│  24891  quantum1   20   0   4.8G   2.1G   42.5   6.8   ✦[vLLM] python -m vllm.entrypoints│
│   2471  quantum1   20   0   1.8G   890M   18.2   2.9   ✦[PyTorch] python train.py        │
│  GPU PROXY  PID:12842  ollama_llama_server  VRAM:11640MiB  ✦ [Ollama]                  │
│ [0:Main] 1:CPU 2:GPU 3:IO 4:Proc 5:Wave | [sudo]Root 1.0s [p]Snapshot [w]Wide [?]Help  │
└────────────────────────────────────────────────────────────────────────────────────────┘
```
</details>

---

## ⚡ Key Features

- **🎨 Claude Minimal Paradigm (Low Data-Ink)**
  No visual clutter, no garish 1990s rainbow blocks. Built on semantic grayscale with high-contrast functional colors reserved strictly for thermal and load excursions.
- **🪐 Balanced Dual-Axis Floating Island**
  Centered on your display with generous, balanced margins on all 4 sides. Seamlessly toggle between floating card and fullscreen wide mode with <kbd>w</kbd>.
- **🚀 Zero External Dependencies (Pure Python Standard Library)**
  No `pip install` required. Pure direct kernel `/proc` and `/sys` telemetry (<1ms sample latency). Uses dynamic `ctypes` directly for NVIDIA NVML—zero driver wrapper packages needed.
- **🤖 Built-in Local AI / LLM Workload Inspector**
  Automatically detects, tags, and profiles active AI workloads (`Ollama`, `vLLM`, `llama.cpp`, `PyTorch`, `ComfyUI`, `TensorFlow`, `DeepSpeed`). Displays exact VRAM allocations and compute footprints.
- **⚡ PSU Budget & Transient Power Guard**
  Visual real-time tracking of total power consumption against your power supply's capacity, calculating instantaneous headroom to catch transient power spikes before tripping.
- **📸 One-Key Shareable Telemetry Snapshots (<kbd>p</kbd>)**
  Press <kbd>p</kbd> at any moment to export clean Markdown and JSON hardware reports directly to `~/`, ready to paste into GitHub Issues, Reddit, or Discord.
- **🌊 60-Sample Braille Historical Waveforms**
  Smooth multi-channel sparklines tracking Power, Temperatures, Utilization, and Network I/O without degrading terminal FPS.

---

## 🆚 Why ZenMon?

| Metric | `btop` | `htop` | `glances` | **`zenmon`** |
| :--- | :---: | :---: | :---: | :---: |
| **Dependencies** | C++ Compiler / Heavy | C / ncurses | 20+ Python Packages | **0 (Python 3 Standard Library Only)** |
| **Startup Latency** | ~250ms | ~50ms | ~1800ms | **< 15ms** |
| **Runtime CPU Overhead**| 2.5% ~ 6.0% | 0.8% ~ 1.5% | 4.0% ~ 8.0% | **< 0.1%** |
| **NVIDIA GPU Telemetry**| Basic | None / Plugin | Optional pip pkg | **Native CTypes NVML (Zero pip)** |
| **AI / LLM Workload Tags**| ❌ No | ❌ No | ❌ No | **✅ Automatic (Ollama, vLLM, PyTorch)** |
| **PSU Headroom Guard**| ❌ No | ❌ No | ❌ No | **✅ Real-time Budget & Remainder** |
| **One-Key Markdown Export**| ❌ No | ❌ No | ❌ No | **✅ Instant `<p>` Snapshot** |
| **Visual Aesthetics** | Neon Cyberpunk | Retro Unix | Dense Web UI | **Claude Minimal (Low Data-Ink)** |

---

## 📦 Installation

### Option 1: Instant Single-Line Run (Zero Installation)
```bash
curl -sSL https://raw.githubusercontent.com/liening1/zenmon/main/standalone/zenmon.py | python3
```

### Option 2: Modern Python Tooling (`uvx` or `pipx`)
```bash
# Run ephemerally via uv:
uvx zenmon

# Or install globally via pipx:
pipx install git+https://github.com/liening1/zenmon.git
```

### Option 3: Manual Git Clone
```bash
git clone https://github.com/liening1/zenmon.git
cd zenmon
sudo make install
```

### Option 4: Recommended Alias (`~/.zshrc` or `~/.bashrc`)
```bash
alias zm="python3 /path/to/zenmon/standalone/zenmon.py"
alias zmr="sudo python3 /path/to/zenmon/standalone/zenmon.py" # For CPU RAPL package power
```

---

## ⌨️ Keybindings

| Key | Action |
| :--- | :--- |
| <kbd>0</kbd> ~ <kbd>5</kbd> | Switch View (`0:Main`, `1:CPU`, `2:GPU`, `3:IO`, `4:Proc`, `5:Wave`) |
| <kbd>w</kbd> / <kbd>W</kbd> | Toggle Layout: **Centered Floating Island** ⟷ **Fullscreen Wide** |
| <kbd>p</kbd> / <kbd>P</kbd> | **Export Snapshot**: Saves Markdown + JSON hardware report to `~/` |
| <kbd>s</kbd> / <kbd>S</kbd> | Cycle Process Sorting: `CPU%` → `MEM%` → `PID` → `NAME` |
| <kbd>/</kbd> | Interactive process filter/search (<kbd>Esc</kbd> to clear) |
| <kbd>Enter</kbd> | Inspect process details (threads, file descriptors, memory map) |
| <kbd>k</kbd> | Terminate selected process with `SIGTERM` (1) or `SIGKILL` (2) |
| <kbd>Space</kbd> | Pause / Resume live telemetry feed |
| <kbd>c</kbd> / <kbd>C</kbd> | Toggle 1:1 Pixel-Perfect Classic Mode |
| <kbd>?</kbd> / <kbd>h</kbd> | Toggle Help Modal |
| <kbd>q</kbd> / <kbd>Esc</kbd> | Quit |

---

## 🧩 Views Overview

1. **`0: Main Dashboard`**: Unified system telemetry card featuring CPU, GPU, Memory, NVMe, Motherboard sensors, Network, and Top Processes.
2. **`1: CPU Matrix`**: Micro-thermal and frequency grid scaling across Dual-CCD (AMD Zen) or multi-core topologies up to 128 threads.
3. **`2: GPU Deep-Dive`**: Clocks, fan speeds, PCIe throughput (MB/s), VRAM allocations, and active compute/graphics AI processes.
4. **`3: Storage & Network`**: Partition usage, IOPS, and real-time network interface bandwidth metrics.
5. **`4: Process Manager`**: Full htop-grade interactive process manager with AI framework tagging and signaling.
6. **`5: Waveforms`**: 60-sample historical braille charts for power, temperatures, and throughput.

---

## ⚙️ Universal Hardware Engine

ZenMon incorporates an automatic, zero-dependency Hardware Abstraction Layer (HAL):
- **CPU Detection**: Dynamically parses `/proc/cpuinfo` and `/sys/devices/system/cpu/`, supporting AMD Zen 1-5, Intel Core/Xeon, and ARM/Raspberry Pi.
- **GPU HAL**:
  - **NVIDIA**: Pure `ctypes` direct loader for `libnvidia-ml.so`.
  - **AMD Radeon**: Reads direct from `/sys/class/drm/card*/device/` and `amdgpu` hwmon.
  - **Headless / No GPU**: Automatically reorganizes the layout to display expanded filesystem and network telemetry with zero blank space.
- **RAPL Power**: Universal powercap domain scanner reading package energy counters; gracefully falls back to unprivileged telemetry when run without `sudo`.

---

## 📄 License

Distributed under the [MIT License](LICENSE). Built for developers, researchers, and minimalists.
