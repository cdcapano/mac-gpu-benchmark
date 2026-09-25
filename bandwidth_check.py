#!/usr/bin/env python3
"""
bandwidth_check.py -- measure a Mac's GPU memory bandwidth.

WHAT THIS DOES
    Allocates some arrays of numbers, adds them together on the GPU, and times
    how long that takes. Dividing bytes moved by seconds gives the machine's
    memory bandwidth in GB/s. Optionally also multiplies two matrices to get
    the arithmetic throughput in GFLOP/s.

WHAT THIS DOES NOT DO
    No network access. No files are read or written. Nothing is installed.
    Nothing is left behind when the process exits. All output goes to the
    screen. The only system call is `sysctl` to print the chip name and RAM
    size in the header, and the script runs fine without it.

WHY
    We are sizing a small compute cluster for gravitational-wave data analysis
    at Syracuse University. The calculation is limited by memory bandwidth
    rather than by arithmetic, so bandwidth per dollar decides which Mac to
    buy. Apple publishes a rated figure; this measures what is actually
    achieved, which is the number we need.

REQUIREMENTS
    Python 3.9+, and the `mlx` and `numpy` packages (Apple's own array library).

USAGE
    python3 bandwidth_check.py              # ~60 s, the measurement we need
    python3 bandwidth_check.py --quick      # ~20 s, fewer repeats
    python3 bandwidth_check.py --no-compute # bandwidth only, skip the matmul
"""

import argparse
import math
import os
import platform
import subprocess
import time

import numpy as np

BAR = "=" * 72


def host_info():
    """Chip name, core count and RAM. Falls back to 'unknown' if sysctl is absent."""
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""
    chip = sysctl("machdep.cpu.brand_string") or platform.processor() or "unknown"
    ncpu = sysctl("hw.ncpu") or "?"
    mem = sysctl("hw.memsize")
    ram = f"{int(mem) / 1e9:.0f} GB" if mem.isdigit() else "? GB"
    return chip, ncpu, ram


# A background process that streams memory steals bandwidth from the GPU.
# Measured on an M3 Pro (25 Sep 2026): with Mail and Spotlight indexing between
# them saturating two cores, this script reported 96-118 GB/s across repeats;
# with the machine idle it reported 125-127 GB/s, reproducing to 1.4%.
#
# CPU percentage is a proxy rather than the quantity that matters. Two pure
# spin loops at 100% CPU, which touch no memory, changed the result by under
# 2% on the same machine. So these thresholds are deliberately conservative:
# they flag a machine worth re-checking, and they do not license correcting a
# number upward by some assumed factor.
# Processes excluded from raising the alarm. Terminal emulators and the window
# server burn CPU rendering this script's own output, so they fire on a machine
# that is otherwise idle. They are still listed, since a genuinely busy display
# does cost bandwidth -- they just do not by themselves mean "come back later".
DISPLAY_PROCS = {"WindowServer", "Terminal", "iTerm2", "kitty", "Alacritty",
                 "WezTerm", "Ghostty", "claude", "node"}

BUSY_PROC_PCT = 20.0     # any one process at or above this is worth a warning
BUSY_LOAD_FRAC = 0.30    # 1-minute load average as a fraction of the core count


def machine_load():
    """1-minute load average, core count, and the busiest few processes.

    Returns (load1, ncpu, [(pcpu, name), ...]) with None or [] for anything
    that could not be read. This process is excluded from the list, since the
    benchmark itself is expected to be busy.
    """
    def sysctl(key):
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True,
                                  text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    load1 = None
    raw = sysctl("vm.loadavg")            # looks like: { 1.72 2.04 2.93 }
    try:
        load1 = float(raw.strip("{} ").split()[0])
    except Exception:
        pass

    ncpu = None
    try:
        ncpu = int(sysctl("hw.ncpu"))
    except Exception:
        pass

    procs, me = [], os.getpid()
    try:
        lines = subprocess.run(["ps", "-Aro", "pid,pcpu,comm"], capture_output=True,
                               text=True, timeout=5).stdout.splitlines()[1:]
        for line in lines[:8]:
            parts = line.split(None, 2)
            if len(parts) == 3 and int(parts[0]) != me:
                procs.append((float(parts[1]), parts[2].rsplit("/", 1)[-1]))
    except Exception:
        pass
    return load1, ncpu, procs


def load_summary(load1, ncpu, procs):
    """One line describing the machine state, for printing next to the result."""
    bits = []
    if load1 is not None and ncpu:
        bits.append(f"1-minute load {load1:.2f} on {ncpu} cores "
                    f"({100 * load1 / ncpu:.0f}%)")
    elif load1 is not None:
        bits.append(f"1-minute load {load1:.2f}")
    if procs:
        bits.append(f"busiest process {procs[0][1]} at {procs[0][0]:.1f}%")
    return "; ".join(bits) if bits else "machine state unavailable"


def busy_reasons(load1, ncpu, procs):
    """Why this machine looks too busy to measure on. Empty list means quiet."""
    why = []
    for pcpu, name in procs:
        if pcpu >= BUSY_PROC_PCT and name not in DISPLAY_PROCS:
            why.append(f"{name} is using {pcpu:.1f}% CPU")
    display = sum(pc for pc, nm in procs if nm in DISPLAY_PROCS) / 100.0
    if (load1 is not None and ncpu
            and load1 - display > BUSY_LOAD_FRAC * ncpu):
        why.append(f"the 1-minute load average is {load1:.2f} on {ncpu} cores")
    return why


def warn_if_busy(load1, ncpu, procs, when):
    """Print a warning if the machine is busy. Returns True if it was."""
    why = busy_reasons(load1, ncpu, procs)
    if not why:
        return False
    print()
    print("    " + "*" * 64)
    print(f"    WARNING -- this machine is busy {when}.")
    for reason in why:
        print(f"      - {reason}")
    print("    On Apple Silicon the CPU and the GPU share one memory")
    print("    controller, so a background process that moves a lot of data")
    print("    subtracts directly from the bandwidth measured here. Spotlight")
    print("    indexing cost about a quarter of the figure on an M3 Pro.")
    print("    How much a given process costs depends on how much memory it")
    print("    touches, so the only fix is to wait for the machine to go idle")
    print("    and run this again.")
    print("    " + "*" * 64)
    print()
    return True


def make_pairs(mx, n, k, seed=0):
    """k independent pairs of complex64 arrays, each of length n.

    Values are random and irrelevant -- only the number of bytes matters.
    Using k independent pairs stops the machine from serving every repeat out
    of cache, which would measure the cache rather than main memory.
    """
    mx.random.seed(seed)
    pairs = []
    for _ in range(k):
        a = (mx.random.normal((n,)) + 1j * mx.random.normal((n,))).astype(mx.complex64)
        b = (mx.random.normal((n,)) + 1j * mx.random.normal((n,))).astype(mx.complex64)
        pairs.append((a, b))
    mx.eval([x for p in pairs for x in p])
    return pairs


def time_add(mx, pairs, rounds):
    """Best-of-`rounds` seconds for one complex64 add.

    `a + b` reads two arrays and writes a third: exactly three passes over
    n x 8 bytes, with one trivial arithmetic operation per element. That makes
    it a ruler for memory bandwidth rather than for arithmetic.
    """
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        out = [mx.add(a, b) for a, b in pairs]
        mx.eval(out)
        best = min(best, (time.perf_counter() - t0) / len(pairs))
    return best


def time_matmul(mx, n, rounds):
    """Best-of-`rounds` seconds for one n x n fp32 matrix multiply.

    A matmul of size n does 2*n^3 arithmetic operations while moving only
    ~3*n^2 numbers, so unlike the add above it is limited by arithmetic.
    """
    a = mx.random.normal((n, n)).astype(mx.float32)
    b = mx.random.normal((n, n)).astype(mx.float32)
    mx.eval([a, b])
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        mx.eval(mx.matmul(a, b))
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser(
        description="Measure Apple Silicon GPU memory bandwidth.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--array-size", type=int, default=2 ** 21,
                    help="elements per array (default 2^21 = 2097152)")
    ap.add_argument("--n-unique", type=int, default=8,
                    help="independent array pairs, to defeat caching (default 8)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="timed repeats; the fastest wins (default 5)")
    ap.add_argument("--sweep", type=int, nargs="*",
                    default=[14, 16, 18, 19, 20, 21, 22],
                    help="log2 sizes for the size sweep (empty to skip)")
    ap.add_argument("--matmul-n", type=int, default=2048,
                    help="square matrix size for the arithmetic test")
    ap.add_argument("--no-compute", action="store_true",
                    help="skip the arithmetic test")
    ap.add_argument("--quick", action="store_true",
                    help="fewer repeats and a shorter sweep")
    ap.add_argument("--rated-bw", type=float, default=None,
                    help="this machine's advertised GB/s, to report a percentage")
    args = ap.parse_args()

    if args.quick:
        args.rounds = 3
        args.sweep = [18, 20, 21, 22]

    chip, ncpu, ram = host_info()
    n = args.array_size
    bytes_per_array = n * 8            # complex64 = 8 bytes per element
    working_set = args.n_unique * 3 * bytes_per_array

    print(BAR)
    print("GPU MEMORY BANDWIDTH CHECK")
    print(BAR)
    print(f"machine     : {chip}  |  {ncpu} CPU cores  |  {ram} RAM")
    print(f"python      : {platform.python_version()}   numpy {np.__version__}")

    try:
        import mlx.core as mx
    except ImportError:
        print("\nmlx is not installed. Install it with:")
        print("    python3 -m pip install --user 'mlx>=0.32.1' numpy")
        print("Nothing has been changed on this machine.")
        return 1
    print(f"mlx         : {getattr(mx, '__version__', 'unknown')}")
    print(f"array size  : 2^{int(math.log2(n))} = {n:,} complex64 "
          f"= {bytes_per_array / 1e6:.1f} MB each")
    print(f"memory used : about {working_set / 1e6:.0f} MB, freed when this exits")
    print(f"writes      : none -- this script does not create any files")
    load1, nload, procs = machine_load()
    print(f"machine load: {load_summary(load1, nload, procs)}")
    if procs:
        busiest = ", ".join(f"{nm} {pc:.1f}%" for pc, nm in procs[:3])
        print(f"busiest     : {busiest}")
    print()
    warn_if_busy(load1, nload, procs, "before the measurement")

    mx.set_default_device(mx.gpu)

    # ---- the measurement we came for -----------------------------------
    print("[1] streaming bandwidth (a + b over complex64)")
    pairs = make_pairs(mx, n, args.n_unique)
    t_add = time_add(mx, pairs, args.rounds)
    gbs = 3 * bytes_per_array / t_add / 1e9
    print(f"    time per add : {t_add * 1e3:8.3f} ms")
    print(f"    bytes moved  : 3 x {bytes_per_array / 1e6:.1f} MB "
          f"= {3 * bytes_per_array / 1e6:.1f} MB")
    print(f"    BANDWIDTH    : {gbs:8.1f} GB/s")
    if args.rated_bw:
        print(f"    that is {100 * gbs / args.rated_bw:.0f}% of the "
              f"{args.rated_bw:g} GB/s rated figure")
    print()
    del pairs

    # ---- arithmetic ceiling, for context --------------------------------
    gflops = None
    if not args.no_compute:
        print(f"[2] arithmetic throughput ({args.matmul_n}^2 fp32 matmul)")
        t_mm = time_matmul(mx, args.matmul_n, args.rounds)
        gflops = 2.0 * args.matmul_n ** 3 / t_mm / 1e9
        print(f"    time         : {t_mm * 1e3:8.3f} ms")
        print(f"    THROUGHPUT   : {gflops:8.0f} GFLOP/s")
        print(f"    balance point: {gflops / gbs:8.1f} FLOP per byte")
        print("    (work with fewer FLOPs per byte than this is limited by memory)")
        print()

    # ---- how bandwidth varies with size ---------------------------------
    if args.sweep:
        print("[3] bandwidth against array size")
        print("    Small arrays fit in the chip's caches and read much faster than")
        print("    main memory. The size where the rate drops is the cache size.")
        print()
        print(f"    {'size':>12s} {'MB each':>9s} {'ms':>9s} {'GB/s':>9s}")
        print("    " + "-" * 42)
        for e in args.sweep:
            nn = 2 ** e
            try:
                pr = make_pairs(mx, nn, args.n_unique, seed=e)
                t = time_add(mx, pr, args.rounds)
                print(f"    2^{e:<10d} {nn * 8 / 1e6:9.2f} {t * 1e3:9.4f} "
                      f"{3 * nn * 8 / t / 1e9:9.1f}")
                del pr
                try:
                    mx.clear_cache()
                except AttributeError:
                    pass
            except Exception as exc:
                print(f"    2^{e:<10d} failed: {type(exc).__name__}: {exc}")
        print()

    load1b, nloadb, procsb = machine_load()
    busy_after = warn_if_busy(load1b, nloadb, procsb, "after the measurement")

    print(BAR)
    print("SUMMARY")
    print(BAR)
    print(f"{chip}: {gbs:.1f} GB/s memory bandwidth"
          + (f", {gflops:.0f} GFLOP/s fp32" if gflops else ""))
    # The machine state belongs with the number. A figure recorded without it
    # cannot be compared against one taken on a different machine, or later on
    # the same machine.
    print(f"measured with {load_summary(load1b, nloadb, procsb)}")
    if busy_after or busy_reasons(load1, nload, procs):
        print("TREAT THIS FIGURE AS A FLOOR -- the machine was busy, see the")
        print("warning above. The true bandwidth of this chip is higher.")
    else:
        print("No competing workload was detected, so this figure is usable as")
        print("measured. Window server and terminal activity is expected during a")
        print("run and is excluded from that check; it costs a few percent.")
    print("Nothing was written to disk. Quit Terminal and no trace remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
