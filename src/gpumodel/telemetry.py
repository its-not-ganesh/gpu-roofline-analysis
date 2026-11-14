import csv
import platform
import os
import threading
import subprocess
import time
from pathlib import Path

import psutil
import torch

try:
    import pynvml
except ImportError:
    pynvml = None


def safe_call(fn):
    try:
        return fn()
    except Exception:
        return None


def gpu_metadata(index=0):
    meta = {
        "python": platform.python_version(), "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpu_count": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "torch": torch.__version__, "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if os.name == "nt":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            meta["cpu_model"] = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        battery = safe_call(psutil.sensors_battery)
        meta["ac_power_connected"] = battery.power_plugged if battery else None
    if torch.cuda.is_available():
        meta.update({
            "gpu": torch.cuda.get_device_name(index),
            "gpu_uuid": str(getattr(torch.cuda.get_device_properties(index), "uuid", "")),
            "gpu_compute_capability": str(torch.cuda.get_device_capability(index)),
            "gpu_total_memory_gb": round(torch.cuda.get_device_properties(index).total_memory / 1e9, 2),
            "cudnn_version": torch.backends.cudnn.version(),
        })
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2)
        if git.returncode == 0:
            meta["git_commit"] = git.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    if pynvml:
        try:
            pynvml.nvmlInit()
            meta["nvidia_driver"] = pynvml.nvmlSystemGetDriverVersion()
            handle = pynvml.nvmlDeviceGetHandleByUUID(meta["gpu_uuid"] if meta["gpu_uuid"].startswith("GPU-")
                                                       else "GPU-" + meta["gpu_uuid"]) if meta.get("gpu_uuid") else None
            if handle is not None:
                meta["gpu_power_state"] = safe_call(lambda: pynvml.nvmlDeviceGetPerformanceState(handle))
                limit = safe_call(lambda: pynvml.nvmlDeviceGetPowerManagementLimit(handle))
                meta["gpu_power_limit_w"] = limit / 1000 if limit is not None else None
        except Exception:
            pass
        finally:
            safe_call(pynvml.nvmlShutdown)
    return meta


class Monitor:
    def __init__(self, interval=0.2, gpu_index=0):
        self.interval = interval
        self.gpu_index = gpu_index
        self.samples = []
        self._stop = threading.Event()
        self._thread = None
        self._handle = None
        self._enabled = False
        self.process = psutil.Process()

    def __enter__(self):
        self.process.cpu_percent(interval=None)
        if pynvml:
            try:
                pynvml.nvmlInit()
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
                self._enabled = True
            except Exception:
                self._enabled = False
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def _sample(self):
        while not self._stop.is_set():
            row = {
                "seconds": time.monotonic(),
                "cpu_pct_process": safe_call(lambda: self.process.cpu_percent(interval=None)),
                "rss_mb": safe_call(lambda: self.process.memory_info().rss / 1e6),
                "gpu_util_pct": None, "gpu_mem_controller_util_pct": None,
                "gpu_vram_mb": None, "power_w": None, "temp_c": None,
                "sm_clock_mhz": None, "memory_clock_mhz": None,
            }
            if self._enabled:
                h = self._handle
                util = safe_call(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
                if util is not None:
                    row["gpu_util_pct"] = util.gpu
                    row["gpu_mem_controller_util_pct"] = util.memory
                mem = safe_call(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
                if mem is not None:
                    row["gpu_vram_mb"] = mem.used / 1e6
                p = safe_call(lambda: pynvml.nvmlDeviceGetPowerUsage(h))
                row["power_w"] = p / 1000 if p is not None else None
                row["temp_c"] = safe_call(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                row["sm_clock_mhz"] = safe_call(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
                row["memory_clock_mhz"] = safe_call(lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
            self.samples.append(row)
            self._stop.wait(self.interval)

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=5)
        if self._enabled:
            pynvml.nvmlShutdown()

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.samples:
            return
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.samples[0].keys())
            writer.writeheader()
            writer.writerows(self.samples)

    def summarize(self):
        out = {"samples": len(self.samples)}
        fields = ["gpu_util_pct", "gpu_mem_controller_util_pct", "gpu_vram_mb", "power_w", "temp_c", "sm_clock_mhz", "cpu_pct_process", "rss_mb"]
        for k in fields:
            vals = [float(r[k]) for r in self.samples if r[k] is not None]
            out[k + "_mean"] = sum(vals) / len(vals) if vals else None
            out[k + "_max"] = max(vals) if vals else None
        pairs = [(r["seconds"], r["power_w"]) for r in self.samples if r["power_w"] is not None]
        out["sampled_board_energy_j"] = sum(
            (b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(pairs[:-1], pairs[1:])
        ) if len(pairs) >= 2 else None
        return out
