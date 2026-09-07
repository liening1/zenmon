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
from .ai_inspector import detect_ai_tag

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
