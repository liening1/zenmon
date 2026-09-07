#!/usr/bin/env python3
"""ZenMon: A serene, zero-dependency telemetry TUI for modern workstations & AI rigs.

GitHub: https://github.com/quantum1/zenmon
License: MIT
"""

# =====================================================================
# SECTION 1: AI WORKLOAD INSPECTOR
# =====================================================================
"""zenmon.ai_inspector: 本地大模型与深度学习工作流智能感知引擎.

自动识别当前系统中运行的 AI / ML / LLM 进程 (Ollama, vLLM, PyTorch, ComfyUI 等),
提取显存与算力画像, 专为现代 AI 开发者与算力工作站打造.
"""
import re

# AI / ML 签名库与对应的语义标签
AI_SIGNATURES = [
    (r"\bollama(_llama_server)?\b", "Ollama"),
    (r"\bvllm(\.entrypoints)?\b", "vLLM"),
    (r"\bllama(-server|\.cpp)?\b", "llama.cpp"),
    (r"\bcomfyui\b", "ComfyUI"),
    (r"\b(stable-diffusion|sd-webui|webui\.py)\b", "SD-WebUI"),
    (r"\bdeepspeed\b", "DeepSpeed"),
    (r"\baccelerate\b", "Accelerate"),
    (r"\btriton\b", "Triton"),
    (r"\btext-generation-launcher\b", "TGI"),
    (r"\btorchrun\b", "Torchrun"),
    (r"\b(python\d*.*[-_](train|infer|eval|finetune|generate|serve)\.py)\b", "ML-Train/Infer"),
    (r"\b(torch|tensorflow|jax|tensorrt)\b", "PyTorch/TF"),
    (r"\bjupyter-(lab|notebook)\b", "Jupyter"),
    (r"\bwhisper\b", "Whisper"),
]

COMPILED_SIGS = [(re.compile(pat, re.IGNORECASE), label) for pat, label in AI_SIGNATURES]

def detect_ai_tag(cmdline: str, comm: str = "") -> str:
    """检测进程命令行或名称是否匹配已知 AI 工作流, 返回标签名称 (如 'Ollama') 或 None."""
    full_text = f"{comm} {cmdline}"
    for regex, label in COMPILED_SIGS:
        if regex.search(full_text):
            return label
    return None

def analyze_ai_summary(procs: list, gpu_procs: list = None) -> dict:
    """聚合分析系统全局 AI 算力与显存消耗画像."""
    ai_procs = []
    total_ai_cpu = 0.0
    total_ai_mem_gb = 0.0
    detected_frameworks = set()

    for p in procs:
        tag = p.get("ai_tag")
        if tag:
            ai_procs.append(p)
            total_ai_cpu += p.get("cpu", 0.0)
            total_ai_mem_gb += (p.get("res", 0) / (1024 * 1024))
            detected_frameworks.add(tag)

    ai_vram_mb = 0.0
    if gpu_procs:
        for gp in gpu_procs:
            if gp.get("ai_tag"):
                ai_vram_mb += gp.get("mem_mb", 0.0)

    return {
        "count": len(ai_procs),
        "frameworks": sorted(list(detected_frameworks)),
        "total_cpu_pct": total_ai_cpu,
        "total_sys_mem_gb": total_ai_mem_gb,
        "total_vram_mb": ai_vram_mb,
    }


# =====================================================================
# SECTION 2: UNIVERSAL HARDWARE ABSTRACTION LAYER (HAL)
# =====================================================================
"""zenmon.hal: 零外部依赖通用硬件抽象层 (Universal Hardware Abstraction Layer).

纯 Python 3 标准库实现, 直接基于 Linux /proc, /sys 和内置 ctypes 机制访问底层硬件:
  - CPU: 通用多核与拓扑感知 (Intel / AMD Zen / ARM / RPi), 自适应核心矩阵
  - GPU: 纯 ctypes 直连 libnvidia-ml.so (免装 pynvml), 兼顾 AMD/Intel sysfs 探测
  - 传感器: 全动态 hwmon 拓扑遍历, 自适应识别 CPU Package/CCD/Core, NVMe, 主板与风扇
  - 功耗: 通用 RAPL powercap 能量域差分计算 (非 root 优雅降级)
  - 存储与网络: /proc/mounts + statvfs, /proc/diskstats, /proc/net/dev
"""
import ctypes
import glob
import os
import pwd
import time
# [Inlined ai_inspector]

# =====================================================================
# 1. 通用 CPU 拓扑与遥测
# =====================================================================
class UniversalCPU:
    def __init__(self):
        self.model = self._detect_model()
        self.vendor = self._detect_vendor()
        self.threads = os.cpu_count() or 1
        self.physical_cores = self._detect_physical_cores()
        self._prev_stat = None
        self._prev_cores_stat = {}

    def _detect_vendor(self) -> str:
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("vendor_id"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return "Unknown"

    def _detect_model(self) -> str:
        model = "Generic CPU"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                        break
        except Exception:
            pass
        for r in ["(R)", "(TM)", "Processor", "16-Core", "8-Core", "12-Core", "6-Core", "24-Core", "32-Core", "64-Core"]:
            model = model.replace(r, "")
        return " ".join(model.split()).strip()

    def _detect_physical_cores(self) -> int:
        cores = set()
        try:
            for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/core_id"):
                with open(p) as f:
                    cores.add(f.read().strip())
        except Exception:
            pass
        return len(cores) if cores else (self.threads // 2 or 1)

    def get_frequencies(self) -> list[int]:
        freqs = []
        paths = sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"),
                       key=lambda x: int(os.path.basename(os.path.dirname(os.path.dirname(x)))[3:]))
        if paths:
            for p in paths:
                try:
                    with open(p) as f:
                        freqs.append(int(f.read().strip()) // 1000)
                except Exception:
                    pass
        if not freqs:
            # Fallback to /proc/cpuinfo
            try:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.startswith("cpu MHz"):
                            freqs.append(int(float(line.split(":", 1)[1].strip())))
            except Exception:
                pass
        return freqs or [0] * self.threads

    def get_usage(self) -> tuple[float, list[float]]:
        """计算 CPU 总使用率以及每个逻辑核心使用率."""
        try:
            with open("/proc/stat") as f:
                lines = f.readlines()
        except Exception:
            return 0.0, [0.0] * self.threads

        # Total CPU
        tot_line = lines[0].split()[1:]
        tot_vals = list(map(int, tot_line))
        tot_idle = tot_vals[3] + (tot_vals[4] if len(tot_vals) > 4 else 0)
        tot_sum = sum(tot_vals)

        total_pct = 0.0
        if self._prev_stat:
            p_total, p_idle = self._prev_stat
            dt = tot_sum - p_total
            di = tot_idle - p_idle
            if dt > 0:
                total_pct = max(0.0, min(100.0, (dt - di) / dt * 100.0))
        self._prev_stat = (tot_sum, tot_idle)

        # Per core
        per_core = []
        for line in lines[1:]:
            parts = line.split()
            if not parts[0].startswith("cpu") or parts[0] == "cpu":
                continue
            core_id = parts[0]
            vals = list(map(int, parts[1:]))
            c_idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
            c_sum = sum(vals)

            pct = 0.0
            if core_id in self._prev_cores_stat:
                p_c_tot, p_c_idle = self._prev_cores_stat[core_id]
                dt = c_sum - p_c_tot
                di = c_idle - p_c_idle
                if dt > 0:
                    pct = max(0.0, min(100.0, (dt - di) / dt * 100.0))
            self._prev_cores_stat[core_id] = (c_sum, c_idle)
            per_core.append(pct)

        return total_pct, per_core or [total_pct] * self.threads


# =====================================================================
# 2. 通用 GPU 统一抽象 (Zero-Pip Dependency via CTypes NVML)
# =====================================================================
class CTypesNVML:
    """基于 ctypes 直接加载宿主机 libnvidia-ml.so, 彻底免去 pip install 依赖."""
    class ProcessInfo(ctypes.Structure):
        _fields_ = [("pid", ctypes.c_uint), ("usedGpuMemory", ctypes.c_ulonglong)]

    class UtilRates(ctypes.Structure):
        _fields_ = [("gpu", ctypes.c_uint), ("memory", ctypes.c_uint)]

    class MemInfo(ctypes.Structure):
        _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]

    def __init__(self):
        self.available = False
        self.handle = None
        self.lib = None
        self._init_nvml()

    def _init_nvml(self):
        sonames = ["libnvidia-ml.so.1", "libnvidia-ml.so", "nvml.dll"]
        for soname in sonames:
            try:
                self.lib = ctypes.CDLL(soname)
                init_fn = getattr(self.lib, "nvmlInit_v2", getattr(self.lib, "nvmlInit", None))
                if init_fn and init_fn() == 0:
                    self.available = True
                    h = ctypes.c_void_p()
                    get_handle = getattr(self.lib, "nvmlDeviceGetHandleByIndex_v2", self.lib.nvmlDeviceGetHandleByIndex)
                    if get_handle(0, ctypes.byref(h)) == 0:
                        self.handle = h
                    break
            except Exception:
                continue

    def query(self) -> dict | None:
        if not self.available or not self.handle:
            return None
        info = {
            "name": "NVIDIA GPU", "temp": 0.0, "power": 0.0, "power_limit": 0.0,
            "fan": 0.0, "sm_clk": 0, "mem_clk": 0, "util": 0.0, "mem_util": 0.0,
            "mem_total": 0.0, "mem_used": 0.0, "pcie_rx": 0.0, "pcie_tx": 0.0,
            "procs": []
        }
        try:
            # Name
            buf = ctypes.create_string_buffer(64)
            if self.lib.nvmlDeviceGetName(self.handle, buf, 64) == 0:
                raw_n = buf.value.decode("utf-8", errors="ignore")
                info["name"] = raw_n.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "").strip()

            # Temp
            t = ctypes.c_uint()
            if self.lib.nvmlDeviceGetTemperature(self.handle, 0, ctypes.byref(t)) == 0:
                info["temp"] = float(t.value)

            # Power
            p = ctypes.c_uint()
            if self.lib.nvmlDeviceGetPowerUsage(self.handle, ctypes.byref(p)) == 0:
                info["power"] = p.value / 1000.0

            plim = ctypes.c_uint()
            enforced = getattr(self.lib, "nvmlDeviceGetEnforcedPowerLimit", None)
            if enforced and enforced(self.handle, ctypes.byref(plim)) == 0:
                info["power_limit"] = plim.value / 1000.0

            # Fan
            f = ctypes.c_uint()
            if self.lib.nvmlDeviceGetFanSpeed(self.handle, ctypes.byref(f)) == 0:
                info["fan"] = float(f.value)

            # Clocks
            sm = ctypes.c_uint(); mem = ctypes.c_uint()
            self.lib.nvmlDeviceGetClockInfo(self.handle, 0, ctypes.byref(sm))
            self.lib.nvmlDeviceGetClockInfo(self.handle, 2, ctypes.byref(mem))
            info["sm_clk"] = sm.value
            info["mem_clk"] = mem.value

            # Util
            u = self.UtilRates()
            if self.lib.nvmlDeviceGetUtilizationRates(self.handle, ctypes.byref(u)) == 0:
                info["util"] = float(u.gpu)
                info["mem_util"] = float(u.memory)

            # VRAM
            m = self.MemInfo()
            if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(m)) == 0:
                info["mem_total"] = m.total / (1024 * 1024)
                info["mem_used"] = m.used / (1024 * 1024)

            # PCIe
            tx = ctypes.c_uint(); rx = ctypes.c_uint()
            if getattr(self.lib, "nvmlDeviceGetPcieThroughput", None):
                self.lib.nvmlDeviceGetPcieThroughput(self.handle, 0, ctypes.byref(tx))
                self.lib.nvmlDeviceGetPcieThroughput(self.handle, 1, ctypes.byref(rx))
                info["pcie_tx"] = tx.value / 1024.0
                info["pcie_rx"] = rx.value / 1024.0

            # GPU Processes
            gpu_pids = {}
            for fn_name in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
                fn = getattr(self.lib, fn_name, None)
                if not fn: continue
                cnt = ctypes.c_uint(32)
                p_buf = (self.ProcessInfo * 32)()
                if fn(self.handle, ctypes.byref(cnt), p_buf) == 0:
                    for i in range(cnt.value):
                        pid = p_buf[i].pid
                        mb = p_buf[i].usedGpuMemory / (1024 * 1024)
                        gpu_pids[pid] = max(gpu_pids.get(pid, 0.0), mb)

            # Resolve Process Names & AI Tags
            p_list = []
            for pid, mb in sorted(gpu_pids.items(), key=lambda x: -x[1]):
                p_name = f"PID {pid}"
                ai_tag = None
                try:
                    with open(f"/proc/{pid}/comm") as pf:
                        p_name = pf.read().strip()
                    with open(f"/proc/{pid}/cmdline") as cf:
                        cmdline = cf.read().replace("\x00", " ")
                        ai_tag = detect_ai_tag(cmdline, p_name)
                except Exception:
                    pass
                p_list.append({"pid": pid, "name": p_name, "mem_mb": mb, "ai_tag": ai_tag})
            info["procs"] = p_list
            return info
        except Exception:
            return None


class UniversalGPU:
    def __init__(self):
        self.nvml = CTypesNVML()
        self.has_nvml = self.nvml.available
        self.amdgpu_path = self._find_amdgpu()

    def _find_amdgpu(self) -> str | None:
        for card in sorted(glob.glob("/sys/class/drm/card[0-9]*")):
            dev = os.path.join(card, "device")
            uevent = os.path.join(dev, "uevent")
            if os.path.exists(uevent):
                try:
                    with open(uevent) as f:
                        for line in f:
                            if "DRIVER=amdgpu" in line:
                                return dev
                except Exception:
                    pass
        return None

    def query(self) -> dict | None:
        if self.has_nvml:
            info = self.nvml.query()
            if info: return info

        if self.amdgpu_path:
            try:
                busy = 0.0
                vused = 0.0
                vtotal = 0.0
                b_file = os.path.join(self.amdgpu_path, "gpu_busy_percent")
                if os.path.exists(b_file):
                    busy = float(open(b_file).read().strip())
                u_file = os.path.join(self.amdgpu_path, "mem_info_vram_used")
                if os.path.exists(u_file):
                    vused = float(open(u_file).read().strip()) / (1024 * 1024)
                t_file = os.path.join(self.amdgpu_path, "mem_info_vram_total")
                if os.path.exists(t_file):
                    vtotal = float(open(t_file).read().strip()) / (1024 * 1024)

                return {
                    "name": "AMD Radeon GPU", "temp": 0.0, "power": 0.0, "power_limit": 0.0,
                    "fan": 0.0, "sm_clk": 0, "mem_clk": 0, "util": busy, "mem_util": (vused/vtotal*100) if vtotal else 0.0,
                    "mem_total": vtotal, "mem_used": vused, "pcie_rx": 0.0, "pcie_tx": 0.0,
                    "procs": []
                }
            except Exception:
                pass
        return None


# =====================================================================
# 3. 通用传感器全域动态扫描 (hwmon)
# =====================================================================
class UniversalSensors:
    def scan(self) -> dict:
        res = {
            "cpu_temp": 0.0,
            "cpu_subtemps": {},
            "nvme_temps": [],
            "mobo_temps": [],
            "fan_speeds": [],
            "igpu_temp": 0.0,
        }
        for hpath in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            name_path = os.path.join(hpath, "name")
            if not os.path.exists(name_path):
                continue
            try:
                with open(name_path) as f:
                    drv = f.read().strip().lower()
            except Exception:
                continue

            # Fans
            for ff in glob.glob(os.path.join(hpath, "fan*_input")):
                try:
                    with open(ff) as f:
                        rpm = int(f.read().strip())
                        if rpm > 0:
                            lbl = os.path.basename(ff).split("_")[0]
                            res["fan_speeds"].append((lbl, rpm))
                except Exception:
                    pass

            # Temps
            for tf in glob.glob(os.path.join(hpath, "temp*_input")):
                base = os.path.basename(tf).split("_")[0]
                label_file = os.path.join(hpath, f"{base}_label")
                label = base
                if os.path.exists(label_file):
                    try:
                        with open(label_file) as lf:
                            label = lf.read().strip()
                    except Exception:
                        pass
                try:
                    with open(tf) as f:
                        val = float(f.read().strip()) / 1000.0
                        if val <= 0 or val > 125: continue
                except Exception:
                    continue

                if drv in ("k10temp", "zenpower"):
                    if label == "Tctl":
                        res["cpu_temp"] = max(res["cpu_temp"], val)
                    elif label.startswith("Tccd"):
                        res["cpu_subtemps"][label] = val
                    elif label == "Tdie" and not res["cpu_temp"]:
                        res["cpu_temp"] = val
                elif drv == "coretemp":
                    if "Package" in label:
                        res["cpu_temp"] = max(res["cpu_temp"], val)
                    elif "Core" in label:
                        res["cpu_subtemps"][label] = val
                elif drv in ("cpu_thermal", "soc_thermal"):
                    res["cpu_temp"] = max(res["cpu_temp"], val)
                elif drv.startswith("nvme"):
                    res["nvme_temps"].append((label, val))
                elif drv in ("amdgpu", "nouveau"):
                    res["igpu_temp"] = max(res["igpu_temp"], val)
                elif any(m in drv for m in ("wmi", "it8", "nct", "asus", "dell", "thinkpad", "acpitz")):
                    res["mobo_temps"].append((f"{drv}:{label}", val))
                else:
                    if not res["cpu_temp"] and "temp1" in base:
                        res["cpu_temp"] = val

        return res


# =====================================================================
# 4. 通用功耗传感器 (RAPL powercap)
# =====================================================================
class UniversalPower:
    def __init__(self):
        self.rapl_path = self._find_rapl()
        self._prev_rapl = None

    def _find_rapl(self) -> str | None:
        candidates = []
        for p in sorted(glob.glob("/sys/class/powercap/intel-rapl*") + glob.glob("/sys/class/powercap/amd_energy*")):
            ef = os.path.join(p, "energy_uj")
            nf = os.path.join(p, "name")
            if os.path.exists(ef):
                name = "unknown"
                if os.path.exists(nf):
                    try: name = open(nf).read().strip()
                    except Exception: pass
                candidates.append((p, name, ef))
        for p, name, ef in candidates:
            if name in ("package-0", "package") or p.endswith(":0"):
                return ef
        return candidates[0][2] if candidates else None

    def get_cpu_power(self) -> float | None:
        if not self.rapl_path or not os.access(self.rapl_path, os.R_OK):
            return None
        try:
            with open(self.rapl_path) as f:
                uj = int(f.read().strip())
            now = time.time()
            power = None
            if self._prev_rapl:
                last_uj, last_t = self._prev_rapl
                dt = now - last_t
                if dt > 0:
                    power = max(0.0, (uj - last_uj) / 1e6 / dt)
            self._prev_rapl = (uj, now)
            return power
        except Exception:
            return None


# =====================================================================
# 5. 通用内存、磁盘与网络
# =====================================================================
def get_memory_info() -> dict:
    try:
        data = {}
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":")
                data[parts[0].strip()] = int(parts[1].split()[0])
        total = data.get("MemTotal", 0) / 1024.0
        avail = data.get("MemAvailable", data.get("MemFree", 0)) / 1024.0
        used = total - avail
        cached = (data.get("Cached", 0) + data.get("Buffers", 0)) / 1024.0
        stot = data.get("SwapTotal", 0) / 1024.0
        sfree = data.get("SwapFree", 0) / 1024.0
        sused = stot - sfree
        return {
            "total_gb": total / 1024.0,
            "used_gb": used / 1024.0,
            "cached_gb": cached / 1024.0,
            "pct": (used / total * 100.0) if total else 0.0,
            "swap_total_gb": stot / 1024.0,
            "swap_used_gb": sused / 1024.0,
            "swap_pct": (sused / stot * 100.0) if stot else 0.0,
        }
    except Exception:
        return {"total_gb": 0.0, "used_gb": 0.0, "cached_gb": 0.0, "pct": 0.0,
                "swap_total_gb": 0.0, "swap_used_gb": 0.0, "swap_pct": 0.0}

def get_mounts() -> list[dict]:
    mounts = []
    seen = set()
    try:
        with open("/proc/mounts") as f:
            for line in f:
                p = line.split()
                dev, mnt, fstype = p[0], p[1], p[2]
                if fstype in ("ext4", "btrfs", "xfs", "zfs", "f2fs", "vfat", "ntfs", "fuseblk"):
                    if mnt in seen or not os.path.exists(mnt): continue
                    seen.add(mnt)
                    try:
                        st = os.statvfs(mnt)
                        tot = (st.f_blocks * st.f_frsize) / (1024**3)
                        free = (st.f_bavail * st.f_frsize) / (1024**3)
                        used = tot - free
                        if tot > 0:
                            mounts.append({
                                "mount": mnt, "dev": os.path.basename(dev),
                                "total_gb": tot, "used_gb": used, "pct": (used / tot * 100.0)
                            })
                    except Exception:
                        pass
    except Exception:
        pass
    return sorted(mounts, key=lambda x: x["mount"])

class UniversalDiskIO:
    def __init__(self):
        self._prev = {}

    def get_io(self) -> dict:
        now = time.time()
        r_bytes, w_bytes, ios = 0, 0, 0
        try:
            with open("/proc/diskstats") as f:
                for line in f:
                    p = line.split()
                    name = p[2]
                    if not (name.startswith("sd") or name.startswith("nvme") or name.startswith("vd")):
                        continue
                    if name[-1].isdigit() and not name.startswith("nvme"):
                        continue
                    if name.startswith("nvme") and "p" in name:
                        continue
                    r_bytes += int(p[5]) * 512
                    w_bytes += int(p[9]) * 512
                    ios += int(p[3]) + int(p[7])
        except Exception:
            pass

        r_rate, w_rate, iops = 0.0, 0.0, 0.0
        if self._prev:
            pr, pw, pio, pt = self._prev
            dt = now - pt
            if dt > 0:
                r_rate = max(0.0, (r_bytes - pr) / (1024*1024) / dt)
                w_rate = max(0.0, (w_bytes - pw) / (1024*1024) / dt)
                iops = max(0.0, (ios - pio) / dt)
        self._prev = (r_bytes, w_bytes, ios, now)
        return {"read_mb": r_rate, "write_mb": w_rate, "iops": iops}

class UniversalNetwork:
    def __init__(self):
        self._prev = {}

    def get_net(self) -> tuple[float, float, dict]:
        now = time.time()
        ifaces = {}
        tot_rx, tot_tx = 0, 0
        try:
            with open("/proc/net/dev") as f:
                for line in f:
                    if ":" not in line: continue
                    name, data = line.split(":", 1)
                    name = name.strip()
                    if name == "lo": continue
                    parts = data.split()
                    rx = int(parts[0])
                    tx = int(parts[8])
                    tot_rx += rx; tot_tx += tx

                    rx_r, tx_r = 0.0, 0.0
                    if name in self._prev:
                        prx, ptx, pt = self._prev[name]
                        dt = now - pt
                        if dt > 0:
                            rx_r = max(0.0, (rx - prx) / dt)
                            tx_r = max(0.0, (tx - ptx) / dt)
                    self._prev[name] = (rx, tx, now)
                    ifaces[name] = {"rx_rate": rx_r, "tx_rate": tx_r, "rx_total": rx, "tx_total": tx}
        except Exception:
            pass

        rx_rate = sum(v["rx_rate"] for v in ifaces.values())
        tx_rate = sum(v["tx_rate"] for v in ifaces.values())
        return rx_rate, tx_rate, ifaces


# =====================================================================
# 6. 通用进程管理器 (集成 AI 工作流标签)
# =====================================================================
class UniversalProcesses:
    def __init__(self):
        self._last_cpu_time = {}
        self._user_cache = {}

    def _get_user(self, uid: int) -> str:
        if uid not in self._user_cache:
            try: self._user_cache[uid] = pwd.getpwuid(uid).pw_name
            except Exception: self._user_cache[uid] = str(uid)
        return self._user_cache[uid]

    def get_procs(self, sort_by="cpu", filter_text="") -> list[dict]:
        now = time.time()
        procs = []
        clock_ticks = os.sysconf("SC_CLK_TCK") or 100
        mem_total_kb = 1
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_total_kb = int(line.split()[1])
                        break
        except Exception:
            pass

        for p_dir in glob.glob("/proc/[0-9]*"):
            pid = int(os.path.basename(p_dir))
            try:
                with open(f"{p_dir}/stat") as sf:
                    stat_str = sf.read()
                comm_start = stat_str.find("(")
                comm_end = stat_str.rfind(")")
                comm = stat_str[comm_start + 1:comm_end]
                fields = stat_str[comm_end + 2:].split()

                utime = int(fields[11])
                stime = int(fields[12])
                pri = fields[15]
                ni = fields[16]
                tot_ticks = utime + stime

                cpu_pct = 0.0
                if pid in self._last_cpu_time:
                    p_ticks, p_t = self._last_cpu_time[pid]
                    dt = now - p_t
                    if dt > 0:
                        cpu_pct = max(0.0, (tot_ticks - p_ticks) / clock_ticks / dt * 100.0)
                self._last_cpu_time[pid] = (tot_ticks, now)

                # Status for memory & user
                uid = 0
                res_kb = 0
                virt_kb = 0
                with open(f"{p_dir}/status") as stf:
                    for line in stf:
                        if line.startswith("Uid:"):
                            uid = int(line.split()[1])
                        elif line.startswith("VmRSS:"):
                            res_kb = int(line.split()[1])
                        elif line.startswith("VmSize:"):
                            virt_kb = int(line.split()[1])

                cmd = comm
                with open(f"{p_dir}/cmdline") as cf:
                    c_line = cf.read().replace("\x00", " ").strip()
                    if c_line: cmd = c_line

                ai_tag = detect_ai_tag(cmd, comm)
                user = self._get_user(uid)
                mem_pct = (res_kb / mem_total_kb * 100.0) if mem_total_kb else 0.0

                item = {
                    "pid": pid, "user": user, "pri": pri, "ni": ni,
                    "virt": virt_kb, "res": res_kb, "cpu": cpu_pct, "mem": mem_pct,
                    "cmd": cmd, "comm": comm, "ai_tag": ai_tag
                }

                if filter_text:
                    ft = filter_text.lower()
                    if ft not in cmd.lower() and ft not in user.lower() and ft != str(pid) and (not ai_tag or ft not in ai_tag.lower()):
                        continue

                procs.append(item)
            except Exception:
                continue

        # Sort
        if sort_by == "cpu":
            procs.sort(key=lambda x: -x["cpu"])
        elif sort_by == "mem":
            procs.sort(key=lambda x: -x["mem"])
        elif sort_by == "pid":
            procs.sort(key=lambda x: x["pid"])
        elif sort_by == "name":
            procs.sort(key=lambda x: x["comm"].lower())
        return procs


# =====================================================================
# SECTION 3: ZENMON TUI APPLICATION ENGINE
# =====================================================================
"""zenmon.app: 极简低墨水比交互式 TUI 监控引擎 (ZenMon TUI Application).

设计哲学 (Claude Minimal Paradigm):
  - 单色基底与高对比度灰阶, 颜色仅用于状态偏离 (正常灰蓝 / 警告脏橘 / 危险珊瑚红)
  - 彻底关闭震动特效, 零频闪, 绝对静止稳健
  - 默认采用双向居中悬浮卡片 (Balanced Floating Island), 充裕呼吸留白
  - 零外部依赖, 100% 纯 Python 3 标准库
"""
import collections
import curses
import json
import os
import signal
import sys
import time
# [Inlined hal]
# [Inlined ai_inspector]

# ===== 默认调优参数 =====
PSU_WATTS_DEFAULT = 550.0
REFRESH_DEFAULT   = 1.0
TEMP_WARN = 75.0
TEMP_CRIT = 88.0
UTIL_WARN = 65.0
UTIL_CRIT = 85.0

COMPACT_WIDTH  = 88
COMPACT_HEIGHT = 25

RAMP = "  ▂▃▄▅▆▇█"
MICRO_FILL  = "█"
MICRO_EMPTY = "░"

# Curses Color Pairs
P_VALUE        = 1
P_LABEL        = 2
P_DIM          = 3
P_NORMAL       = 4
P_WARN         = 5
P_CRIT         = 6
PAIR_DIM_TRACK = 7
PAIR_SEL       = 8
P_ACCENT       = 9

def temp_pair(t):
    if t is None: return P_DIM
    if t >= TEMP_CRIT: return P_CRIT
    if t >= TEMP_WARN: return P_WARN
    return P_NORMAL

def util_pair(u):
    if u >= UTIL_CRIT: return P_CRIT
    if u >= UTIL_WARN: return P_WARN
    return P_VALUE

def pow_pair(frac):
    if frac >= 0.90: return P_CRIT
    if frac >= 0.75: return P_WARN
    return P_NORMAL

def fmt_bytes(b):
    if b < 1024: return f"{b:.0f} B"
    if b < 1024**2: return f"{b/1024:.1f} KB"
    if b < 1024**3: return f"{b/(1024**2):.1f} MB"
    return f"{b/(1024**3):.2f} GB"

def fmt_rate(b):
    if b < 1024: return f"{b:4.0f} B"
    if b < 1024**2: return f"{b/1024:4.1f} K"
    if b < 1024**3: return f"{b/(1024**2):4.1f} M"
    return f"{b/(1024**3):4.2f} G"

def fmt_kb(kb):
    if kb < 1024: return f"{kb}K"
    if kb < 1024**2: return f"{kb/1024:.1f}M"
    return f"{kb/(1024**2):.1f}G"

def sparkline(values, width, vmin=0.0, vmax=1.0):
    if not values or width <= 0: return " " * max(0, width)
    vals = list(values)
    n = len(vals)
    out = []
    levels = len(RAMP) - 1
    for c in range(width):
        if n == 1: v = vals[0]
        else:
            pos = c / max(1, width - 1) * (n - 1)
            i0 = int(pos); i1 = min(i0 + 1, n - 1); f = pos - i0
            v = vals[i0] * (1 - f) + vals[i1] * f
        t = (v - vmin) / (vmax - vmin) if vmax > vmin else 0.0
        idx = min(levels, max(0, int(t * levels)))
        out.append(RAMP[idx])
    return "".join(out)

def make_micro_bar(value, maximum, length=10):
    if maximum <= 0: maximum = 1.0
    frac = max(0.0, min(1.0, value / maximum))
    full = int(round(frac * length))
    return (MICRO_FILL * full) + (MICRO_EMPTY * (length - full))

def safe_addstr(win, y, x, text, pair=P_LABEL, bold=False, max_w=None):
    try:
        my, mx = win.getmaxyx()
        if y < 0 or y >= my or x < 0 or x >= mx: return
        avail = mx - x
        if max_w is not None: avail = min(avail, max_w)
        if avail <= 0: return
        t = str(text)[:avail]
        attr = curses.color_pair(pair) | (curses.A_BOLD if bold else 0)
        win.addstr(y, x, t, attr)
    except curses.error:
        pass

def make_card_put(win, ox, oy, cw, ch):
    def put(y, x, text, pair=P_LABEL, bold=False, max_w=None):
        if y < 0 or y >= ch or x < 0 or x >= cw: return
        avail = cw - x
        if max_w is not None: avail = min(avail, max_w)
        safe_addstr(win, y + oy, x + ox, text, pair=pair, bold=bold, max_w=avail)
    return put

def render_split_header(put, y, lx, rx, col_w, t1, t2, cw):
    put(y, lx, "── ", P_DIM)
    put(y, lx + 3, t1, P_LABEL, bold=True)
    rem_l = max(0, col_w - len(t1) - 4)
    put(y, lx + 3 + len(t1) + 1, "─" * rem_l, P_DIM)

    put(y, rx, "── ", P_DIM)
    put(y, rx + 3, t2, P_LABEL, bold=True)
    rem_r = max(0, col_w - len(t2) - 4)
    put(y, rx + 3 + len(t2) + 1, "─" * rem_r, P_DIM)

def render_single_header(put, y, ox, cw, title):
    put(y, ox, "── ", P_DIM)
    put(y, ox + 3, title, P_LABEL, bold=True)
    rem = max(0, cw - len(title) - 4)
    put(y, ox + 3 + len(title) + 1, "─" * rem, P_DIM)


class ZenMonApp:
    def __init__(self, psu_watts=PSU_WATTS_DEFAULT, refresh=REFRESH_DEFAULT,
                 start_classic=False, initial_view=0, compact_mode=True):
        self.cpu = UniversalCPU()
        self.gpu = UniversalGPU()
        self.sensors = UniversalSensors()
        self.power = UniversalPower()
        self.disk_io = UniversalDiskIO()
        self.net = UniversalNetwork()
        self.processes = UniversalProcesses()

        self.psu_watts = psu_watts
        self.refresh = refresh
        self.is_classic = start_classic
        self.current_view = initial_view
        self.compact_mode = compact_mode
        self.paused = False
        self.sort_mode = "cpu"
        self.filter_text = ""
        self.filter_prompt = False
        self.filter_buffer = ""
        self.proc_selected_idx = 0
        self.proc_scroll = 0
        self.kill_prompt = False
        self.show_inspect = False
        self.inspect_pid = None
        self.show_help = False
        self.toast_msg = ""
        self.toast_time = 0.0

        self.history = {k: collections.deque(maxlen=60) for k in
                        ("tpow", "ctemp", "gtemp", "gpow", "cusg", "mem", "cpow", "gutil", "net_rx")}

    def set_toast(self, msg, duration=2.5):
        self.toast_msg = msg
        self.toast_time = time.time() + duration

    def update_telemetry(self) -> dict:
        cpu_u, cores = self.cpu.get_usage()
        freqs = self.cpu.get_frequencies()
        cpu_p = self.power.get_cpu_power()
        gpu_info = self.gpu.query()
        sens = self.sensors.scan()
        mem = get_memory_info()
        mounts = get_mounts()
        disk = self.disk_io.get_io()
        rx_r, tx_r, ifaces = self.net.get_net()

        total_pw = (gpu_info["power"] if gpu_info else 0.0) + (cpu_p or 0.0)

        # History tracking
        self.history["tpow"].append(total_pw)
        self.history["ctemp"].append(sens["cpu_temp"])
        self.history["gtemp"].append(gpu_info["temp"] if gpu_info else 0.0)
        self.history["gpow"].append(gpu_info["power"] if gpu_info else 0.0)
        self.history["cusg"].append(cpu_u)
        self.history["mem"].append(mem["pct"])
        self.history["cpow"].append(cpu_p or 0.0)
        self.history["gutil"].append(gpu_info["util"] if gpu_info else 0.0)
        self.history["net_rx"].append(rx_r)

        up_sec = 0.0
        try:
            with open("/proc/uptime") as f: up_sec = float(f.readline().split()[0])
        except Exception: pass
        hrs, rem = divmod(int(up_sec), 3600)
        mins, secs = divmod(rem, 60)
        up_str = f"{hrs}h {mins:02d}m {secs:02d}s"

        return {
            "hostname": os.uname().nodename, "uptime": up_str,
            "cpu_usage": cpu_u, "cores": cores, "freqs": freqs, "cpu_power": cpu_p,
            "gpu": gpu_info, "sensors": sens, "mem": mem, "mounts": mounts,
            "disk": disk, "net_rx": rx_r, "net_tx": tx_r, "net_ifaces": ifaces,
            "total_power": total_pw
        }

    def _get_layout_geometry(self, my, mx, target_w=COMPACT_WIDTH, target_h=COMPACT_HEIGHT):
        """计算完全双向居中悬浮卡片的几何坐标 (OX, OY, CW, CH)."""
        if self.compact_mode and mx >= target_w:
            cw = target_w
            ox = max(0, (mx - cw) // 2)
        else:
            cw = mx
            ox = 0

        if self.compact_mode and my >= target_h:
            ch = target_h
            oy = max(0, (my - ch) // 2)
        else:
            ch = my
            oy = 0
        return cw, ch, ox, oy

    # -----------------------------------------------------------------
    # 视图 0: 通用自适应主仪表盘 (PRO DASHBOARD)
    # -----------------------------------------------------------------
    def render_dashboard(self, win, snap, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, COMPACT_HEIGHT)
        put = make_card_put(win, ox, oy, cw, ch)

        cpu_t = snap["sensors"]["cpu_temp"]
        gpu = snap["gpu"]
        total_pw = snap["total_power"]
        is_alert = (total_pw > self.psu_watts * 0.9) or (cpu_t > 85.0) or (gpu and gpu.get("temp", 0) > 85.0)

        # 1. 顶栏
        host = snap["hostname"]
        up = snap["uptime"]
        status_txt = "PAUSED" if self.paused else ("ALERT" if is_alert else "LIVE")
        left_h = f"ZENMON ✦ {host} ✦ {self.cpu.model[:24]} ({self.cpu.threads}T)"
        right_h = f"UP: {up}  [{status_txt}]"
        put(0, 0, left_h, P_LABEL, bold=True)
        put(0, cw - len(right_h), right_h, P_CRIT if is_alert else P_VALUE)

        # 2. PSU BUDGET BAR
        frac = (total_pw / self.psu_watts) if self.psu_watts else 0.0
        frac = max(0.0, min(1.0, frac))
        remain = max(0.0, self.psu_watts - total_pw)
        p_note = "" if snap["cpu_power"] is not None else " [GPU only]"
        hpre = f"PSU BUDGET [{self.psu_watts:.0f}W]{p_note} "
        hsuf = f"  {total_pw:5.1f}W  |  REMAIN: {remain:5.1f}W"
        bar_w = max(8, cw - len(hpre) - len(hsuf))
        full = int(round(frac * bar_w))
        empty = max(0, bar_w - full)
        dyn_color = pow_pair(frac)
        put(1, 0, hpre, PAIR_DIM_TRACK, bold=True)
        put(1, len(hpre), MICRO_FILL * full, dyn_color)
        put(1, len(hpre) + full, MICRO_EMPTY * empty, PAIR_DIM_TRACK)
        put(1, len(hpre) + full + empty, hsuf, dyn_color)

        # 3. COMPUTE & ACCELERATOR / STORAGE
        col_w = min(42, (cw - 4) // 2)
        lx = 0
        rx = col_w + 4

        if gpu:
            gname = gpu["name"][:col_w - 15]
            render_split_header(put, 2, lx, rx, col_w, f"COMPUTE [{self.cpu.threads}T]", f"ACCELERATOR [{gname}]", cw)
        else:
            render_split_header(put, 2, lx, rx, col_w, f"COMPUTE [{self.cpu.threads}T]", "STORAGE & FILE SYSTEMS", cw)

        # Row 3
        u_bar = make_micro_bar(snap["cpu_usage"], 100.0)
        put(3, lx, "Util  ", P_LABEL)
        put(3, lx + 7, u_bar, util_pair(snap["cpu_usage"]))
        put(3, lx + 18, f"{snap['cpu_usage']:5.1f}%", util_pair(snap["cpu_usage"]))
        if gpu:
            gu_bar = make_micro_bar(gpu["util"], 100.0)
            put(3, rx, "Util  ", P_LABEL)
            put(3, rx + 7, gu_bar, util_pair(gpu["util"]))
            put(3, rx + 18, f"{gpu['util']:5.1f}%", util_pair(gpu["util"]))
            if gpu.get("sm_clk"):
                put(3, rx + 26, f"SM:{gpu['sm_clk']}M Mem:{gpu['mem_clk']}M", P_DIM)
        else:
            root_m = next((mnt for mnt in snap["mounts"] if mnt["mount"] == "/"), None)
            if root_m:
                put(3, rx, "Disk/ ", P_LABEL)
                put(3, rx + 7, make_micro_bar(root_m["pct"], 100.0), P_VALUE)
                put(3, rx + 18, f"{root_m['used_gb']:.0f}G/{root_m['total_gb']:.0f}G ({root_m['pct']:.0f}%)", P_VALUE)

        # Row 4 (Temp)
        t_bar = make_micro_bar(cpu_t, 100.0)
        put(4, lx, "Temp  ", P_LABEL)
        put(4, lx + 7, t_bar, temp_pair(cpu_t))
        put(4, lx + 18, f"{cpu_t:5.1f}°C", temp_pair(cpu_t))
        subtemps = snap["sensors"]["cpu_subtemps"]
        if "Tccd1" in subtemps and "Tccd2" in subtemps:
            put(4, lx + 26, f"C1:{subtemps['Tccd1']:.0f}° C2:{subtemps['Tccd2']:.0f}°", P_DIM)
        elif subtemps:
            first_sub = list(subtemps.values())[0]
            put(4, lx + 26, f"Core:{first_sub:.0f}°C", P_DIM)

        if gpu:
            gt_bar = make_micro_bar(gpu["temp"], 100.0)
            put(4, rx, "Temp  ", P_LABEL)
            put(4, rx + 7, gt_bar, temp_pair(gpu["temp"]))
            put(4, rx + 18, f"{gpu['temp']:5.1f}°C", temp_pair(gpu["temp"]))
            fan_txt = f"Fan:{gpu['fan']:.0f}%" if gpu["fan"] > 0 else "Fan:0%"
            put(4, rx + 26, fan_txt, P_DIM)
        else:
            disk = snap["disk"]
            put(4, rx, "DiskIO", P_LABEL)
            put(4, rx + 7, f"R:{disk['read_mb']:4.1f}M W:{disk['write_mb']:4.1f}M ({disk['iops']:.0f} IOPS)", P_VALUE)

        # Row 5 (Pwr)
        cpu_pw = snap["cpu_power"]
        put(5, lx, "Pwr   ", P_LABEL)
        if cpu_pw is not None:
            p_bar = make_micro_bar(cpu_pw, 200.0)
            put(5, lx + 7, p_bar, pow_pair(cpu_pw / 200.0))
            put(5, lx + 18, f"{cpu_pw:5.1f}W", pow_pair(cpu_pw / 200.0))
        else:
            put(5, lx + 7, "Restricted (Use sudo)", P_DIM)

        if gpu:
            gp = gpu["power"]
            glim = gpu["power_limit"] or 180.0
            gp_bar = make_micro_bar(gp, glim)
            put(5, rx, "Pwr   ", P_LABEL)
            put(5, rx + 7, gp_bar, pow_pair(gp / glim))
            put(5, rx + 18, f"{gp:5.1f}W / {glim:.0f}W", P_VALUE)
        else:
            rx_r, tx_r = snap["net_rx"], snap["net_tx"]
            put(5, rx, "NET   ", P_LABEL)
            put(5, rx + 7, f"▲ {fmt_rate(rx_r)}/s  ▼ {fmt_rate(tx_r)}/s", P_VALUE)

        # Row 6 (Load / VRAM)
        load = os.getloadavg()
        freqs = snap["freqs"]
        avg_f = sum(freqs) // max(1, len(freqs)) if freqs else 0
        put(6, lx, "Load  ", P_LABEL)
        put(6, lx + 7, f"{load[0]:.2f} {load[1]:.2f} {load[2]:.2f}", P_VALUE)
        put(6, lx + 26, f"Avg {avg_f}MHz", P_DIM)

        if gpu:
            mu, mt = gpu["mem_used"], gpu["mem_total"]
            v_bar = make_micro_bar(mu, mt)
            put(6, rx, "VRAM  ", P_LABEL)
            put(6, rx + 7, v_bar, P_VALUE)
            put(6, rx + 18, f"{mu/1024:.1f}G / {mt/1024:.1f}G", P_VALUE)
        else:
            if len(snap["mounts"]) > 1:
                m2 = snap["mounts"][1]
                put(6, rx, f"{m2['mount'][:5]:<5s} ", P_LABEL)
                put(6, rx + 7, f"{m2['used_gb']:.0f}G/{m2['total_gb']:.0f}G ({m2['pct']:.0f}%)", P_VALUE)

        # Row 7 (Cores / PCIe)
        put(7, lx, "Cores ", P_LABEL)
        c_spark = "".join(RAMP[min(len(RAMP)-1, int(c / 12.5))] for c in snap["cores"][:32])
        put(7, lx + 7, c_spark, P_ACCENT)
        if gpu:
            put(7, rx, "PCIe  ", P_LABEL)
            put(7, rx + 7, f"▲ {gpu['pcie_rx']:5.1f}M  ▼ {gpu['pcie_tx']:5.1f}M", P_VALUE)
            if snap["sensors"]["igpu_temp"] > 0:
                put(7, rx + 26, f"iGPU:{snap['sensors']['igpu_temp']:.0f}°C", P_DIM)

        # 4. 分栏 2: MEMORY & STORAGE | MOTHERBOARD & SENSORS
        render_split_header(put, 8, lx, rx, col_w, "MEMORY & STORAGE", "MOTHERBOARD & SENSORS", cw)

        # RAM / NVMe
        m = snap["mem"]
        r_bar = make_micro_bar(m["pct"], 100.0)
        put(9, lx, "RAM   ", P_LABEL)
        put(9, lx + 7, r_bar, util_pair(m["pct"]))
        put(9, lx + 18, f"{m['used_gb']:.1f}G / {m['total_gb']:.1f}G", P_VALUE)
        put(9, lx + 31, f"C:{m['cached_gb']:.1f}G", P_DIM)

        nv_list = snap["sensors"]["nvme_temps"]
        if nv_list:
            nv_val = nv_list[0][1]
            put(9, rx, "NVMe  ", P_LABEL)
            put(9, rx + 7, make_micro_bar(nv_val, 100.0), temp_pair(nv_val))
            put(9, rx + 18, f"{nv_val:5.1f}°C", temp_pair(nv_val))
            if len(nv_list) > 1:
                put(9, rx + 26, f"D2:{nv_list[1][1]:.0f}°C", P_DIM)

        # Swap / Mobo
        s_bar = make_micro_bar(m["swap_pct"], 100.0)
        put(10, lx, "SWP   ", P_LABEL)
        put(10, lx + 7, s_bar, P_WARN if m["swap_pct"] > 50 else P_DIM)
        put(10, lx + 18, f"{m['swap_used_gb']:.1f}G / {m['swap_total_gb']:.1f}G", P_VALUE)

        mobo = snap["sensors"]["mobo_temps"]
        if mobo:
            put(10, rx, "Mobo  ", P_LABEL)
            t_str = "  ".join(f"{item[0].split(':')[-1]}:{item[1]:.0f}°" for item in mobo[:3])
            put(10, rx + 7, t_str, P_VALUE)

        # Disk / Net
        root_m = next((mnt for mnt in snap["mounts"] if mnt["mount"] == "/"), None)
        if root_m:
            put(11, lx, "Disk/ ", P_LABEL)
            d_bar = make_micro_bar(root_m["pct"], 100.0)
            put(11, lx + 7, d_bar, P_VALUE)
            put(11, lx + 18, f"{root_m['used_gb']:.0f}G/{root_m['total_gb']:.0f}G ({root_m['pct']:.0f}%)", P_VALUE)

        rx_r, tx_r = snap["net_rx"], snap["net_tx"]
        n_bar = make_micro_bar((rx_r + tx_r) / (50 * 1024 * 1024) * 100.0, 100.0)
        put(11, rx, "NET   ", P_LABEL)
        put(11, rx + 7, n_bar, P_DIM)
        put(11, rx + 18, f"▲ {fmt_rate(rx_r)}/s  ▼ {fmt_rate(tx_r)}/s", P_VALUE)

        # NVMeIO / Fans
        disk = snap["disk"]
        put(12, lx, "NVMeIO", P_LABEL)
        put(12, lx + 7, f"R:{disk['read_mb']:4.1f}M W:{disk['write_mb']:4.1f}M ({disk['iops']:.0f} IOPS)", P_VALUE)

        fans = snap["sensors"]["fan_speeds"]
        if fans:
            put(12, rx, "Fans  ", P_LABEL)
            f_str = " ".join(f"{f[0]}:{f[1]}RPM" for f in fans[:2])
            put(12, rx + 7, f_str, P_DIM)
        else:
            active_ifaces = [k for k, v in snap["net_ifaces"].items() if v["rx_total"] > 1024*1024]
            if active_ifaces:
                put(12, rx, "Ifaces", P_LABEL)
                put(12, rx + 7, " ".join(active_ifaces[:3]), P_DIM)

        # 5. 进程列表分栏
        proc_title = f"TOP PROCESSES (Sorted by {self.sort_mode.upper()})"
        if self.filter_text: proc_title += f" [Filter: '{self.filter_text}']"
        render_single_header(put, 13, 0, cw, proc_title)

        hdr = "   PID   USER       PRI  NI   VIRT    RES   %CPU  %MEM   COMMAND"
        put(14, 0, hdr, P_LABEL)

        gpu_procs = gpu.get("procs", []) if gpu else []
        has_gpu_proxy = bool(gpu_procs and ch >= 25)
        proc_end_row = ch - (3 if has_gpu_proxy else 2)

        procs = self.processes.get_procs(self.sort_mode, self.filter_text)
        row_idx = 0
        for y in range(15, proc_end_row + 1):
            if row_idx >= len(procs): break
            p = procs[row_idx]
            tag_s = f"✦[{p['ai_tag']}] " if p.get("ai_tag") else ""
            line = f" {p['pid']:5d}  {p['user'][:9]:<9s} {p['pri']:>3s} {p['ni']:>3s} {fmt_kb(p['virt']):>6s} {fmt_kb(p['res']):>6s} {p['cpu']:5.1f} {p['mem']:5.1f}   {tag_s}{p['cmd']}"
            pair = P_ACCENT if p.get("ai_tag") else P_VALUE
            put(y, 0, line, pair, max_w=cw)
            row_idx += 1

        if has_gpu_proxy:
            gp = gpu_procs[0]
            ai_badge = f"✦ [{gp['ai_tag']}]" if gp.get("ai_tag") else ""
            gp_line = f" GPU PROXY  PID:{gp['pid']}  {gp['name'][:18]:<18s}  VRAM:{gp['mem_mb']:.0f}MiB  {ai_badge}"
            put(ch - 2, 0, gp_line, P_ACCENT)

        # 6. 底栏
        self._render_card_footer(put, ch - 1, cw)

    def _render_card_footer(self, put, y, cw):
        tabs = ["0:Main", "1:CPU", "2:GPU", "3:IO", "4:Proc", "5:Wave"]
        tabs[self.current_view] = f"[{tabs[self.current_view]}]"
        tab_bar = " ".join(tabs)
        elevated = "[sudo]Root" if os.geteuid() == 0 else "[user]"
        p_state = "[PAUSED]" if self.paused else ""
        footer = f"{tab_bar} | {p_state} {elevated} {self.refresh:.1f}s [p]Snapshot [w]Wide [?]Help [q]Quit"
        put(y, 0, footer, P_LABEL, max_w=cw)

    # -----------------------------------------------------------------
    # 视图 1: 自适应多核 CPU 拓扑矩阵 (Zen / Core / Xeon / RPi)
    # -----------------------------------------------------------------
    def render_cpu_matrix(self, win, snap, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, target_h=23)
        put = make_card_put(win, ox, oy, cw, ch)

        put(0, 0, f"CPU MATRIX ✦ {self.cpu.model} ({self.cpu.threads} Threads)", P_LABEL, bold=True)
        cpu_t = snap["sensors"]["cpu_temp"]
        subtemps = snap["sensors"]["cpu_subtemps"]
        cpu_pw = snap["cpu_power"]
        pw_str = f"{cpu_pw:.1f}W" if cpu_pw is not None else "Restricted"

        t_detail = f"Temp: {cpu_t:.1f}°C"
        if subtemps:
            t_detail += " (" + ", ".join(f"{k}:{v:.0f}°C" for k, v in subtemps.items()) + ")"
        summary = f"Total: {snap['cpu_usage']:.1f}%  |  {t_detail}  |  Power: {pw_str}"
        put(1, 0, summary, P_VALUE)

        col_w = min(42, (cw - 4) // 2)
        half = (self.cpu.threads + 1) // 2

        if "Tccd1" in subtemps and "Tccd2" in subtemps:
            render_split_header(put, 2, 0, col_w + 4, col_w, f"CCD 1 [C0-7] {subtemps['Tccd1']:.0f}°C", f"CCD 2 [C8-15] {subtemps['Tccd2']:.0f}°C", cw)
        else:
            render_split_header(put, 2, 0, col_w + 4, col_w, f"CORES [T0-{half-1}]", f"CORES [T{half}-{self.cpu.threads-1}]", cw)

        cores_u = snap["cores"]
        freqs = snap["freqs"]

        rows_avail = min(16, ch - 4)
        for i in range(rows_avail):
            row_y = 3 + i
            if row_y >= ch - 1: break
            # Left thread
            t1 = i
            if t1 < len(cores_u):
                u1 = cores_u[t1]
                f1 = freqs[t1] if t1 < len(freqs) else 0
                b1 = make_micro_bar(u1, 100.0, length=8)
                line1 = f"T{t1:02d} {b1} {u1:5.1f}%  {f1:4d}M"
                put(row_y, 0, line1, util_pair(u1))

            # Right thread
            t2 = i + half
            if t2 < len(cores_u):
                u2 = cores_u[t2]
                f2 = freqs[t2] if t2 < len(freqs) else 0
                b2 = make_micro_bar(u2, 100.0, length=8)
                line2 = f"T{t2:02d} {b2} {u2:5.1f}%  {f2:4d}M"
                put(row_y, col_w + 4, line2, util_pair(u2))

        self._render_card_footer(put, ch - 1, cw)

    # -----------------------------------------------------------------
    # 视图 2: 通用 GPU 深度遥测与 AI 显存透视 (GPU DEEP DIVE)
    # -----------------------------------------------------------------
    def render_gpu_deepdive(self, win, snap, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, target_h=23)
        put = make_card_put(win, ox, oy, cw, ch)

        gpu = snap["gpu"]
        put(0, 0, f"ACCELERATOR DEEP-DIVE ✦ {gpu['name'] if gpu else 'No Dedicated GPU'}", P_LABEL, bold=True)
        render_single_header(put, 1, 0, cw, "HARDWARE TELEMETRY & CLOCKS")

        if not gpu:
            put(3, 2, "No supported discrete NVIDIA or AMD GPU detected.", P_WARN)
            put(4, 2, "CPU Integrated graphics or software rendering is active.", P_DIM)
            self._render_card_footer(put, ch - 1, cw)
            return

        put(2, 2, f"GPU Utilization   : {gpu['util']:5.1f}%   {make_micro_bar(gpu['util'], 100.0, 15)}", util_pair(gpu['util']))
        put(3, 2, f"Memory Controller : {gpu['mem_util']:5.1f}%   {make_micro_bar(gpu['mem_util'], 100.0, 15)}", util_pair(gpu['mem_util']))
        put(4, 2, f"GPU Core Temp     : {gpu['temp']:5.1f}°C  {make_micro_bar(gpu['temp'], 100.0, 15)}", temp_pair(gpu['temp']))
        plim = gpu["power_limit"] or 180.0
        put(5, 2, f"Power Draw / TDP  : {gpu['power']:5.1f}W / {plim:.0f}W   {make_micro_bar(gpu['power'], plim, 15)}", pow_pair(gpu['power']/plim))
        put(6, 2, f"Fan Speed         : {gpu['fan']:.0f}%", P_VALUE)
        put(7, 2, f"Clocks            : SM Core {gpu['sm_clk']} MHz  |  VRAM Memory {gpu['mem_clk']} MHz", P_ACCENT)
        put(8, 2, f"PCIe Throughput   : RX {gpu['pcie_rx']:.1f} MB/s  |  TX {gpu['pcie_tx']:.1f} MB/s", P_VALUE)
        pct = (gpu['mem_used']/gpu['mem_total']*100) if gpu['mem_total'] else 0.0
        put(9, 2, f"VRAM Allocation   : {gpu['mem_used']:.0f} MiB / {gpu['mem_total']:.0f} MiB ({pct:.1f}%)", P_VALUE)

        render_single_header(put, 11, 0, cw, "ACTIVE GPU PROCESSES & AI INFERENCE")
        put(12, 2, "   PID    NAME                  VRAM (MiB)   WORKLOAD", P_LABEL)
        procs = gpu.get("procs", [])
        if procs:
            for idx, p in enumerate(procs[:max(2, ch - 15)]):
                tag = f"✦ [{p['ai_tag']}]" if p.get("ai_tag") else ""
                put(13 + idx, 2, f"  {p['pid']:<7d} {p['name'][:20]:<20s}  {p['mem_mb']:>9.1f} MiB   {tag}", P_ACCENT if tag else P_VALUE)
        else:
            put(13, 4, "(No active compute/graphics processes detected)", P_DIM)

        self._render_card_footer(put, ch - 1, cw)

    # -----------------------------------------------------------------
    # 视图 3: 存储与网络全域监控
    # -----------------------------------------------------------------
    def render_storage_net(self, win, snap, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, target_h=21)
        put = make_card_put(win, ox, oy, cw, ch)

        put(0, 0, "STORAGE & NETWORK TELEMETRY", P_LABEL, bold=True)
        render_single_header(put, 1, 0, cw, "FILESYSTEM PARTITIONS & USAGE")

        put(2, 2, "  MOUNT POINT       DEVICE               USED / TOTAL        PERCENT   BAR", P_LABEL)
        y = 3
        for m in snap["mounts"][:4]:
            b = make_micro_bar(m["pct"], 100.0, 10)
            line = f"  {m['mount']:<16s}  {m['dev']:<18s}   {m['used_gb']:>5.1f}G / {m['total_gb']:>5.1f}G   {m['pct']:>5.1f}%   {b}"
            put(y, 2, line, util_pair(m["pct"]))
            y += 1

        render_single_header(put, y, 0, cw, "NETWORK INTERFACES & BANDWIDTH")
        y += 1
        put(y, 2, "  INTERFACE       RX RATE         TX RATE           TOTAL RX       TOTAL TX", P_LABEL)
        y += 1
        for iface, data in snap["net_ifaces"].items():
            if y >= ch - 2: break
            line = f"  {iface:<14s}  {fmt_rate(data['rx_rate']):>9s}/s  {fmt_rate(data['tx_rate']):>9s}/s   {fmt_bytes(data['rx_total']):>12s}   {fmt_bytes(data['tx_total']):>12s}"
            put(y, 2, line, P_VALUE)
            y += 1

        self._render_card_footer(put, ch - 1, cw)

    # -----------------------------------------------------------------
    # 视图 4: 完整交互式进程管理器 (PRO PROCESS MANAGER)
    # -----------------------------------------------------------------
    def render_proc_manager(self, win, snap, my, mx):
        target_h = min(my, max(22, my - 4)) if self.compact_mode else my
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, max(COMPACT_WIDTH, min(96, mx)), target_h=target_h)
        put = make_card_put(win, ox, oy, cw, ch)

        procs = self.processes.get_procs(self.sort_mode, self.filter_text)
        ai_stat = analyze_ai_summary(procs, snap["gpu"].get("procs") if snap["gpu"] else None)

        title = f"PROCESS MANAGER ✦ {self.sort_mode.upper()} ✦ Count: {len(procs)}"
        if ai_stat["count"] > 0:
            fw_str = ",".join(ai_stat["frameworks"][:3])
            title += f" ✦ AI Workloads: {ai_stat['count']} ({fw_str})"
        if self.filter_text: title += f" ✦ Filter: '{self.filter_text}'"
        put(0, 0, title, P_LABEL, bold=True)
        render_single_header(put, 1, 0, cw, "ACTIVE SYSTEM PROCESSES")

        hdr = "   PID   USER       PRI  NI   VIRT    RES   %CPU  %MEM   COMMAND"
        put(2, 0, hdr, P_LABEL, bold=True)

        visible_rows = max(1, ch - 5)
        if self.proc_selected_idx >= len(procs):
            self.proc_selected_idx = max(0, len(procs) - 1)
        if self.proc_selected_idx < self.proc_scroll:
            self.proc_scroll = self.proc_selected_idx
        elif self.proc_selected_idx >= self.proc_scroll + visible_rows:
            self.proc_scroll = self.proc_selected_idx - visible_rows + 1

        for i in range(visible_rows):
            idx = self.proc_scroll + i
            if idx >= len(procs): break
            p = procs[idx]
            row_y = 3 + i
            is_sel = (idx == self.proc_selected_idx)
            pair = PAIR_SEL if is_sel else (P_ACCENT if p.get("ai_tag") else P_VALUE)
            tag_s = f"✦[{p['ai_tag']}] " if p.get("ai_tag") else ""
            line = f" {p['pid']:5d}  {p['user'][:9]:<9s} {p['pri']:>3s} {p['ni']:>3s} {fmt_kb(p['virt']):>6s} {fmt_kb(p['res']):>6s} {p['cpu']:5.1f} {p['mem']:5.1f}   {tag_s}{p['cmd']}"
            put(row_y, 0, line, pair, bold=is_sel, max_w=cw)

        hint = "↑/↓: 导航 | Enter: 详情 | k: 终止 | s: 排序 | /: 搜索 | p: 导出快照"
        put(ch - 2, 2, hint, P_DIM)
        self._render_card_footer(put, ch - 1, cw)

    # -----------------------------------------------------------------
    # 视图 5: 历史遥测波形 (HISTORICAL WAVEFORMS)
    # -----------------------------------------------------------------
    def render_waveforms(self, win, snap, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, target_h=21)
        put = make_card_put(win, ox, oy, cw, ch)

        put(0, 0, "REAL-TIME TELEMETRY WAVEFORMS (Last 60 Samples)", P_LABEL, bold=True)
        render_single_header(put, 1, 0, cw, "HISTORICAL METRIC WAVEFORMS")

        w = max(16, cw - 32)
        h = self.history

        # 1. Total Power
        put(2, 2, "PSU Total Power (W) :", P_LABEL)
        sp_tpow = sparkline(list(h["tpow"]), w, 0.0, self.psu_watts)
        put(2, 24, sp_tpow, pow_pair(snap["total_power"]/self.psu_watts))
        put(2, 24 + w + 1, f"{snap['total_power']:5.1f}W", P_VALUE)

        # 2. CPU Power
        put(4, 2, "CPU Power (W)       :", P_LABEL)
        sp_cpow = sparkline(list(h["cpow"]), w, 0.0, 150.0)
        put(4, 24, sp_cpow, P_NORMAL)
        put(4, 24 + w + 1, f"{snap['cpu_power'] or 0.0:5.1f}W", P_VALUE)

        # 3. GPU Power
        put(6, 2, "GPU Power (W)       :", P_LABEL)
        gpw = snap["gpu"]["power"] if snap["gpu"] else 0.0
        sp_gpow = sparkline(list(h["gpow"]), w, 0.0, 180.0)
        put(6, 24, sp_gpow, P_NORMAL)
        put(6, 24 + w + 1, f"{gpw:5.1f}W", P_VALUE)

        # 4. CPU Util
        put(8, 2, "CPU Utilization (%) :", P_LABEL)
        sp_cusg = sparkline(list(h["cusg"]), w, 0.0, 100.0)
        put(8, 24, sp_cusg, util_pair(snap["cpu_usage"]))
        put(8, 24 + w + 1, f"{snap['cpu_usage']:5.1f}%", P_VALUE)

        # 5. CPU Temp
        put(10, 2, "CPU Temperature (°C):", P_LABEL)
        sp_ctemp = sparkline(list(h["ctemp"]), w, 30.0, 100.0)
        put(10, 24, sp_ctemp, temp_pair(snap["sensors"]["cpu_temp"]))
        put(10, 24 + w + 1, f"{snap['sensors']['cpu_temp']:5.1f}°C", P_VALUE)

        # 6. GPU Temp
        put(12, 2, "GPU Temp (°C)       :", P_LABEL)
        gt = snap["gpu"]["temp"] if snap["gpu"] else 0.0
        sp_gtemp = sparkline(list(h["gtemp"]), w, 30.0, 100.0)
        put(12, 24, sp_gtemp, temp_pair(gt))
        put(12, 24 + w + 1, f"{gt:5.1f}°C", P_VALUE)

        # 7. RAM Usage %
        put(14, 2, "RAM Utilization (%) :", P_LABEL)
        sp_mem = sparkline(list(h["mem"]), w, 0.0, 100.0)
        put(14, 24, sp_mem, P_VALUE)
        put(14, 24 + w + 1, f"{snap['mem']['pct']:5.1f}%", P_VALUE)

        # 8. Network RX
        put(16, 2, "Network RX Rate     :", P_LABEL)
        sp_rx = sparkline(list(h["net_rx"]), w, 0.0, max(1.0, max(list(h["net_rx"]) or [1.0])))
        put(16, 24, sp_rx, P_ACCENT)
        put(16, 24 + w + 1, f"{fmt_rate(snap['net_rx'])}/s", P_VALUE)

        self._render_card_footer(put, ch - 1, cw)

    # -----------------------------------------------------------------
    # 快照导出功能 (Markdown & JSON)
    # -----------------------------------------------------------------
    def export_snapshot(self, snap) -> str:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.expanduser("~")
        md_path = os.path.join(out_dir, f"zenmon_snapshot_{timestamp}.md")
        json_path = os.path.join(out_dir, f"zenmon_snapshot_{timestamp}.json")

        procs = self.processes.get_procs()
        ai_stat = analyze_ai_summary(procs, snap["gpu"].get("procs") if snap["gpu"] else None)

        # JSON
        data = {
            "timestamp": time.time(),
            "datetime": time.ctime(),
            "host": snap["hostname"],
            "uptime": snap["uptime"],
            "cpu": {
                "model": self.cpu.model, "vendor": self.cpu.vendor, "threads": self.cpu.threads,
                "usage_pct": snap["cpu_usage"], "power_watts": snap["cpu_power"],
                "temp_c": snap["sensors"]["cpu_temp"], "subtemps": snap["sensors"]["cpu_subtemps"]
            },
            "gpu": snap["gpu"],
            "memory": snap["mem"],
            "storage": {"mounts": snap["mounts"], "disk_io": snap["disk"]},
            "network": {"rx_bytes_sec": snap["net_rx"], "tx_bytes_sec": snap["net_tx"]},
            "ai_workload_summary": ai_stat,
            "top_processes": procs[:20]
        }
        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

            # Markdown
            with open(md_path, "w") as f:
                f.write(f"# ZenMon Hardware Telemetry Snapshot\n\n")
                f.write(f"- **Captured At**: `{time.ctime()}`\n")
                f.write(f"- **Host**: `{snap['hostname']}` | **Uptime**: `{snap['uptime']}`\n")
                f.write(f"- **CPU**: `{self.cpu.model}` ({self.cpu.threads}T) - **Util**: `{snap['cpu_usage']:.1f}%` | **Temp**: `{snap['sensors']['cpu_temp']:.1f}°C`\n")
                if snap["gpu"]:
                    f.write(f"- **GPU**: `{snap['gpu']['name']}` - **Util**: `{snap['gpu']['util']:.1f}%` | **VRAM**: `{snap['gpu']['mem_used']:.0f}/{snap['gpu']['mem_total']:.0f} MiB`\n")
                f.write(f"- **Memory**: `{snap['mem']['used_gb']:.1f} GB` / `{snap['mem']['total_gb']:.1f} GB` (`{snap['mem']['pct']:.1f}%`)\n")
                f.write(f"- **PSU Power**: `{snap['total_power']:.1f} W` / `{self.psu_watts:.0f} W`\n\n")
                if ai_stat["count"] > 0:
                    f.write(f"### ✦ Active AI/ML Workloads ({ai_stat['count']})\n")
                    f.write(f"- Frameworks: `{', '.join(ai_stat['frameworks'])}`\n")
                    f.write(f"- Total AI VRAM: `{ai_stat['total_vram_mb']:.1f} MiB` | System RAM: `{ai_stat['total_sys_mem_gb']:.2f} GB`\n\n")
                f.write("### Top Processes\n\n")
                f.write("| PID | User | %CPU | %MEM | AI Tag | Command |\n")
                f.write("| --- | --- | --- | --- | --- | --- |\n")
                for p in procs[:15]:
                    tag = p.get("ai_tag") or "-"
                    f.write(f"| {p['pid']} | {p['user']} | {p['cpu']:.1f}% | {p['mem']:.1f}% | {tag} | `{p['cmd'][:50]}` |\n")

            return f"Snapshot saved to ~/{os.path.basename(md_path)}"
        except Exception as e:
            return f"Export error: {e}"

    # -----------------------------------------------------------------
    # 弹窗渲染: 帮助模态框 & 进程详情
    # -----------------------------------------------------------------
    def render_help_modal(self, win, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, COMPACT_HEIGHT)
        mw, mh = min(74, cw - 4), min(21, ch - 2)
        px = ox + max(0, (cw - mw) // 2)
        py = oy + max(0, (ch - mh) // 2)

        for y in range(mh):
            safe_addstr(win, py + y, px, " " * mw, PAIR_SEL)

        safe_addstr(win, py + 1, px + 2, "ZENMON ✦ KEYBINDINGS & CAPABILITIES", P_LABEL, bold=True)
        safe_addstr(win, py + 2, px + 2, "─" * (mw - 4), P_DIM)

        help_items = [
            ("0 ~ 5", "Switch View: 0:Main | 1:CPU | 2:GPU | 3:IO | 4:Proc | 5:Wave"),
            ("w / W", "Toggle Layout: Centered Floating Island <-> Fullscreen Wide"),
            ("p / P", "Export Telemetry Snapshot (Markdown + JSON to ~/)"),
            ("c / C", "Toggle 1:1 Pixel-Perfect Classic Mode"),
            ("Space", "Pause / Resume live telemetry feed"),
            ("s / S", "Cycle Process Sort: CPU% -> MEM% -> PID -> NAME"),
            ("/    ", "Interactive Process Search / Filter (Esc to clear)"),
            ("Enter", "Inspect Selected Process (threads, memory maps, fds)"),
            ("k    ", "Terminate Selected Process (SIGTERM / SIGKILL)"),
            ("? / h", "Toggle this Help Modal"),
            ("q/Esc", "Quit ZenMon"),
        ]

        for i, (k, desc) in enumerate(help_items):
            if 3 + i >= mh - 2: break
            safe_addstr(win, py + 3 + i, px + 4, f"{k:<6s}", P_ACCENT, bold=True)
            safe_addstr(win, py + 3 + i, px + 12, desc, P_VALUE)

        safe_addstr(win, py + mh - 2, px + 2, "Press [Esc], [?], or [q] to close help", P_DIM)

    def render_inspect_modal(self, win, my, mx):
        cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, COMPACT_HEIGHT)
        mw, mh = min(76, cw - 4), min(20, ch - 2)
        px = ox + max(0, (cw - mw) // 2)
        py = oy + max(0, (ch - mh) // 2)

        for y in range(mh): safe_addstr(win, py + y, px, " " * mw, PAIR_SEL)
        safe_addstr(win, py + 1, px + 2, f"PROCESS INSPECTOR ✦ PID {self.inspect_pid}", P_LABEL, bold=True)
        safe_addstr(win, py + 2, px + 2, "─" * (mw - 4), P_DIM)

        p_dir = f"/proc/{self.inspect_pid}"
        if not os.path.exists(p_dir):
            safe_addstr(win, py + 4, px + 4, "Process has terminated.", P_WARN)
            safe_addstr(win, py + mh - 2, px + 2, "Press [Esc] to return", P_DIM)
            return

        cmd = ""
        try: cmd = open(f"{p_dir}/cmdline").read().replace("\x00", " ").strip()
        except Exception: pass
        safe_addstr(win, py + 3, px + 2, f"CMD: {cmd[:mw-8]}", P_VALUE)

        threads = len(glob.glob(f"{p_dir}/task/*"))
        fds = len(glob.glob(f"{p_dir}/fd/*"))
        safe_addstr(win, py + 5, px + 2, f"Threads: {threads}  |  Open FDs: {fds}", P_VALUE)
        safe_addstr(win, py + mh - 2, px + 2, "Press [Esc] or [Enter] to return", P_DIM)

    # -----------------------------------------------------------------
    # 主运行循环
    # -----------------------------------------------------------------
    def run(self, stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        curses.use_default_colors()

        curses.init_pair(P_VALUE, 252, -1)
        curses.init_pair(P_LABEL, 245, -1)
        curses.init_pair(P_DIM, 238, -1)
        curses.init_pair(P_NORMAL, 67, -1)
        curses.init_pair(P_WARN, 173, -1)
        curses.init_pair(P_CRIT, 203, -1)
        curses.init_pair(PAIR_DIM_TRACK, 237, -1)
        curses.init_pair(PAIR_SEL, 255, 236)
        curses.init_pair(P_ACCENT, 110, -1)

        last_update = 0.0
        snap = self.update_telemetry()

        while True:
            now = time.time()
            my, mx = stdscr.getmaxyx()

            if not self.paused and (now - last_update >= self.refresh):
                snap = self.update_telemetry()
                last_update = now

            stdscr.erase()

            if self.current_view == 0:
                self.render_dashboard(stdscr, snap, my, mx)
            elif self.current_view == 1:
                self.render_cpu_matrix(stdscr, snap, my, mx)
            elif self.current_view == 2:
                self.render_gpu_deepdive(stdscr, snap, my, mx)
            elif self.current_view == 3:
                self.render_storage_net(stdscr, snap, my, mx)
            elif self.current_view == 4:
                self.render_proc_manager(stdscr, snap, my, mx)
            elif self.current_view == 5:
                self.render_waveforms(stdscr, snap, my, mx)

            # 浮层与交互弹窗
            cw, ch, ox, oy = self._get_layout_geometry(my, mx, COMPACT_WIDTH, COMPACT_HEIGHT)
            if self.show_help:
                self.render_help_modal(stdscr, my, mx)
            elif self.show_inspect and self.inspect_pid:
                self.render_inspect_modal(stdscr, my, mx)
            elif self.filter_prompt:
                safe_addstr(stdscr, oy + ch - 1, ox, f" Filter Processes: {self.filter_buffer}_ (Enter to apply, Esc to clear) ", PAIR_SEL, max_w=cw)
            elif self.kill_prompt:
                procs = self.processes.get_procs(self.sort_mode, self.filter_text)
                if procs and self.proc_selected_idx < len(procs):
                    kp = procs[self.proc_selected_idx]
                    safe_addstr(stdscr, oy + ch - 1, ox, f" Terminate PID {kp['pid']} ({kp['cmd'][:20]})?  [1] SIGTERM  [2] SIGKILL  [Esc] Cancel ", P_CRIT, bold=True, max_w=cw)

            if self.toast_msg and time.time() < self.toast_time:
                safe_addstr(stdscr, oy + ch - 2, ox + 2, f" ✦ {self.toast_msg} ", P_ACCENT, bold=True, max_w=cw - 4)

            stdscr.refresh()

            try:
                ch_key = stdscr.getch()
            except curses.error:
                ch_key = -1

            if ch_key == -1:
                time.sleep(0.04)
                continue

            if self.filter_prompt:
                if ch_key in (10, 13):
                    self.filter_text = self.filter_buffer
                    self.filter_prompt = False
                    self.proc_selected_idx = 0
                elif ch_key == 27:
                    self.filter_buffer = ""
                    self.filter_text = ""
                    self.filter_prompt = False
                elif ch_key in (127, 8, curses.KEY_BACKSPACE):
                    self.filter_buffer = self.filter_buffer[:-1]
                elif 32 <= ch_key <= 126:
                    self.filter_buffer += chr(ch_key)
                continue

            if self.kill_prompt:
                procs = self.processes.get_procs(self.sort_mode, self.filter_text)
                if procs and self.proc_selected_idx < len(procs):
                    kp = procs[self.proc_selected_idx]
                    if ch_key == ord("1"):
                        try:
                            os.kill(kp["pid"], signal.SIGTERM)
                            self.set_toast(f"SIGTERM sent to PID {kp['pid']}")
                        except Exception as e:
                            self.set_toast(f"Kill failed: {e}")
                        self.kill_prompt = False
                    elif ch_key == ord("2"):
                        try:
                            os.kill(kp["pid"], signal.SIGKILL)
                            self.set_toast(f"SIGKILL sent to PID {kp['pid']}")
                        except Exception as e:
                            self.set_toast(f"Kill failed: {e}")
                        self.kill_prompt = False
                    elif ch_key in (27, ord("q"), ord("Q")):
                        self.kill_prompt = False
                else:
                    self.kill_prompt = False
                continue

            if self.show_inspect:
                if ch_key in (27, ord("q"), ord("Q"), 10, 13):
                    self.show_inspect = False
                continue

            if self.show_help:
                if ch_key in (27, ord("?"), ord("h"), ord("H"), ord("q"), ord("Q"), 10):
                    self.show_help = False
                continue

            if ch_key in (ord("q"), ord("Q"), 27):
                break
            elif ch_key in (ord("?"), ord("h"), ord("H")):
                self.show_help = True
            elif ch_key in (ord("w"), ord("W")):
                self.compact_mode = not self.compact_mode
                self.set_toast("Card: Centered Floating" if self.compact_mode else "Card: Fullscreen Wide")
            elif ch_key in (ord("p"), ord("P")):
                msg = self.export_snapshot(snap)
                self.set_toast(msg, duration=3.5)
            elif ch_key == ord(" "):
                self.paused = not self.paused
                self.set_toast("Telemetry Paused" if self.paused else "Telemetry Resumed")
            elif ch_key in (ord("s"), ord("S")):
                cycle = ["cpu", "mem", "pid", "name"]
                self.sort_mode = cycle[(cycle.index(self.sort_mode) + 1) % len(cycle)]
                self.set_toast(f"Sorted by: {self.sort_mode.upper()}")
            elif ch_key == ord("/"):
                self.filter_prompt = True
                self.filter_buffer = self.filter_text
            elif ch_key in (ord("0"), ord("1"), ord("2"), ord("3"), ord("4"), ord("5")):
                self.current_view = int(chr(ch_key))
            elif ch_key == curses.KEY_UP:
                self.proc_selected_idx = max(0, self.proc_selected_idx - 1)
            elif ch_key == curses.KEY_DOWN:
                self.proc_selected_idx += 1
            elif ch_key in (10, 13):
                procs = self.processes.get_procs(self.sort_mode, self.filter_text)
                if procs and self.proc_selected_idx < len(procs):
                    self.inspect_pid = procs[self.proc_selected_idx]["pid"]
                    self.show_inspect = True
            elif ch_key in (ord("k"), ord("K")):
                self.kill_prompt = True


# =====================================================================
# SECTION 4: CLI ENTRYPOINT
# =====================================================================
"""zenmon CLI 启动入口与参数解析器."""
import argparse
import curses
import json
import os
import sys
# [Inlined app]
__version__ = "2.0.0"

def load_config() -> dict:
    cfg_path = os.path.expanduser("~/.config/zenmon/config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def main():
    cfg = load_config()

    parser = argparse.ArgumentParser(
        prog="zenmon",
        description="ZenMon: A serene, zero-dependency telemetry TUI for modern workstations & AI rigs."
    )
    parser.add_argument("-p", "--psu", type=float, default=cfg.get("psu_watts", PSU_WATTS_DEFAULT),
                        help=f"Total PSU power budget in Watts (default: {PSU_WATTS_DEFAULT})")
    parser.add_argument("-r", "--refresh", type=float, default=cfg.get("refresh", REFRESH_DEFAULT),
                        help=f"Telemetry refresh interval in seconds (default: {REFRESH_DEFAULT})")
    parser.add_argument("-w", "--wide", action="store_true", default=cfg.get("wide_mode", False),
                        help="Start in fullscreen wide mode instead of centered card")
    parser.add_argument("-c", "--classic", action="store_true", default=False,
                        help="Start in 1:1 pixel-perfect classic mode")
    parser.add_argument("-v", "--view", type=int, choices=[0, 1, 2, 3, 4, 5], default=cfg.get("default_view", 0),
                        help="Initial view index (0:Main, 1:CPU, 2:GPU, 3:IO, 4:Proc, 5:Wave)")
    parser.add_argument("--snapshot", action="store_true", default=False,
                        help="Capture a headless telemetry snapshot to ~/ and exit immediately")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    app = ZenMonApp(
        psu_watts=args.psu,
        refresh=args.refresh,
        start_classic=args.classic,
        initial_view=args.view,
        compact_mode=not args.wide
    )

    if args.snapshot:
        snap = app.update_telemetry()
        res = app.export_snapshot(snap)
        print(f"✦ {res}")
        sys.exit(0)

    try:
        curses.wrapper(app.run)
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()

