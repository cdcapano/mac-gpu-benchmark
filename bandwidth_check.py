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
    print()

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

    print(BAR)
    print("SUMMARY")
    print(BAR)
    print(f"{chip}: {gbs:.1f} GB/s memory bandwidth"
          + (f", {gflops:.0f} GFLOP/s fp32" if gflops else ""))
    print("Nothing was written to disk. Quit Terminal and no trace remains.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
