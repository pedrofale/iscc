"""Wait for memory before starting a heavy task.

The paper pipeline runs on laptops that are already near their memory limit, where a second large
process pushes everything into swap and turns minutes into hours. :func:`wait_for_memory` blocks
until the machine has room (available RAM and free swap above thresholds) or a timeout passes, and
the pools in this directory call it before every tumour or tool run. On a cluster node with plenty
of memory it returns at once.
"""
import os
import re
import subprocess
import sys
import time


def _darwin():
    """macOS grows its swap on demand, so free swap says little; use the kernel's own pressure signal.
    Returns (available RAM GB, a headroom figure that is 0 under critical pressure or <25% free)."""
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", out)}
    avail = sum(pages.values()) * page / 1e9

    def sysctl(name):
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip())
    level, free_pct = sysctl("kern.memorystatus_vm_pressure_level"), sysctl("kern.memorystatus_level")
    headroom = float("inf") if (level < 4 and free_pct >= 25) else 0.0
    return avail, headroom


def _linux():
    info = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":")
            info[k] = int(v.split()[0]) / 1e6       # kB -> GB
    return info.get("MemAvailable", 0.0), info.get("SwapFree", float("inf"))


def memory():
    """(available RAM in GB, swap or pressure headroom in GB) of this machine."""
    try:
        return _darwin() if sys.platform == "darwin" else _linux()
    except Exception:
        return float("inf"), float("inf")           # cannot tell: do not block


def wait_for_memory(min_avail_gb=1.5, min_swap_free_gb=1.5, timeout_s=1800, poll_s=15, label=""):
    """Block until available RAM and free swap are both above their thresholds.

    Returns True when there is room, False if ``timeout_s`` passed first (the caller decides whether
    to go ahead anyway). Thresholds can be relaxed with ``ISCC_MEMGUARD=off``."""
    if os.environ.get("ISCC_MEMGUARD", "on") == "off":
        return True
    t0 = time.time()
    warned = False
    while True:
        avail, swap_free = memory()
        if avail >= min_avail_gb and swap_free >= min_swap_free_gb:
            return True
        if time.time() - t0 > timeout_s:
            print(f"  [memguard] {label}: still short after {timeout_s}s "
                  f"(available {avail:.1f} GB, swap free {swap_free:.1f} GB)", flush=True)
            return False
        if not warned:
            print(f"  [memguard] {label}: waiting for memory "
                  f"(available {avail:.1f} GB, swap free {swap_free:.1f} GB)", flush=True)
            warned = True
        time.sleep(poll_s)


if __name__ == "__main__":
    a, s = memory()
    print(f"available {a:.2f} GB, swap free {s:.2f} GB")
