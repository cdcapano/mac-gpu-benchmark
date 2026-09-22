# hardware_sizing_benchmark.py
#
# Settles the one open question in the Mac-cluster purchasing decision:
# is the matched-filter pipeline limited by GPU memory bandwidth, or by
# CPU-side waveform generation?
#
# Measures, on THIS machine:
#   1. t_filter  -- MLX GPU matched filter, batched, at --array_size  (the GPU side)
#   2. t_cpu_1   -- MLX CPU matched filter, single thread, complex64  (fair CPU baseline)
#   3. T_wf      -- single-core frequency-domain waveform generation at df = 1/seglen
#
# Then reports, for each candidate 2026 Mac, the minimum number of segments a
# template must be reused across (n_seg) before that machine's CPUs can keep its
# GPU fed. If your actual n_seg is above the number printed, that machine is
# GPU-bound and you should buy bandwidth. If below, it is CPU-bound and you
# should buy cores.
#
# Usage (inside the `mlx` conda env):
#     conda activate mlx
#     python hardware_sizing_benchmark.py                # sizing table (wants pycbc)
#     python hardware_sizing_benchmark.py --decompose    # byte-count measurement (no deps)
#     python hardware_sizing_benchmark.py --surrogate    # PE waveform (wants jax-nrsur7dq4-nn)
#
# --decompose answers a different question. It needs nothing but mlx+numpy, and
# it measures four things:
#   * this GPU's streaming bandwidth ceiling (mx.add, known byte count)
#   * this GPU's compute ceiling (FMA chain + matmul), replacing the ESTIMATED
#     ALU-count figure -- Apple publishes no FLOPS number for these chips
#   * the real per-kernel byte traffic, replacing the ESTIMATED "9 array passes"
#   * the CPU's ceilings in both regimes, bounding GPU-vs-one-core speedup
# Run it first.
#
# --surrogate answers the PE-side question. A search reuses each template across
# many segments so generation cost amortises; PE generates a fresh waveform per
# likelihood call and amortises nothing. It times jax-nrsur7dq4-nn across vmap
# batch sizes, finds where throughput plateaus, and turns that into waveforms/s
# per candidate machine. Run it twice, once per precision -- jax-mps has no
# float64, so the fp64-vs-fp32 mismatch decides whether the Metal path is open
# at all. The second run finds the first run's saved waveform and compares.
#
# Useful flags:
#     --array_size 2097152    filter length (default 2**21 = 512 s @ 4096 Hz)
#     --seglen 512            segment length in seconds -> df = 1/seglen
#     --f-lower 20 --f-final 1024
#     --approximant IMRPhenomD
#     --n-wf 20               waveforms to time (default 20)
#     --list-size 200         filter ops per batch (default 200)
#     --host-bw 100           this machine's peak memory bandwidth, GB/s (M2 = 100)
#     --core-speedup 1.6      assumed M5 big-core speed vs this machine's P-core
#
# --surrogate flags:
#     --surrogate-precision f64|f32   f64 is the package default; run both
#     --surrogate-weight-f32  round the weights to 32 bit but keep float64
#                             arithmetic -- separates what the NETWORK needs
#                             from what the reference-frequency solvers need
#     --surrogate-batches 1 4 16 64 256
#     --surrogate-threads 1   XLA CPU threads (0 = all cores)
#     --cpu-bw 78             CPU streaming GB/s, for the weight-streaming floor
#     --twf-ms 20.339         measured IMRPhenomD time to compare against

import os

# Single-threaded CPU timing must be pinned BEFORE numpy is imported.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import math
import platform
import subprocess
import time

import numpy as np


# ----------------------------------------------------------------------------
# Candidate hardware. (chip, education price USD, peak GB/s, big CPU cores)
# Apple US education store, September 2026.
# ----------------------------------------------------------------------------
CANDIDATES = [
    ("mini  M5 Pro 15c/16c  24GB",  1599,  307, 15),
    ("mini  M5 Pro 18c/20c  24GB",  1779,  307, 18),
    ("Studio M5 Max 18c/32c 36GB",  2299,  460, 18),
    ("Studio M5 Max 18c/40c 48GB",  2839,  614, 18),
    ("Studio M5 Ultra 30c/64c 96GB", 5099, 1200, 30),
]
BUDGET = 14500.0


def hostinfo():
    out = {"python": platform.python_version(), "machine": platform.machine()}
    for key, cmd in (("chip", ["sysctl", "-n", "machdep.cpu.brand_string"]),
                     ("ncpu", ["sysctl", "-n", "hw.ncpu"]),
                     ("nperf", ["sysctl", "-n", "hw.perflevel0.logicalcpu"]),
                     ("ram", ["sysctl", "-n", "hw.memsize"])):
        try:
            out[key] = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            out[key] = "?"
    try:
        out["ram_gb"] = round(int(out["ram"]) / 2**30)
    except Exception:
        out["ram_gb"] = "?"
    return out


# ----------------------------------------------------------------------------
# 1 + 2. Matched filter, MLX, batched -- mirrors matched_filter_benchmark-mlx-single.py
# ----------------------------------------------------------------------------
def time_filter(mx, ifft, device, n, list_size, n_unique=20):
    """Return seconds per matched-filter op on `device`."""
    mx.set_default_device(device)
    rng = np.random.default_rng(0)
    k = min(n_unique, list_size)
    pairs = []
    for _ in range(k):
        a = mx.array((rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64))
        b = mx.array((rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64))
        pairs.append((a, b))

    def one(a, b):
        return mx.max(mx.abs(ifft(mx.multiply(a.conj(), b))))

    mx.eval(one(*pairs[0]))                       # warm up: compile Metal shaders
    batch = [one(*pairs[i % k]) for i in range(list_size)]
    t0 = time.perf_counter()
    mx.eval(batch)
    return (time.perf_counter() - t0) / list_size


# ----------------------------------------------------------------------------
# 3. Waveform generation, single core, at df = 1/seglen
# ----------------------------------------------------------------------------
def time_waveform(df, f_lower, f_final, approximant, n_wf):
    """Return (seconds per waveform, backend name, n_freq_points)."""
    rng = np.random.default_rng(1)
    # A spread of masses: generation cost scales with the number of frequency
    # points below merger, so a single mass pair would misrepresent a real bank.
    masses = [(float(m1), float(m2)) for m1, m2 in
              zip(rng.uniform(1.2, 40.0, n_wf), rng.uniform(1.2, 40.0, n_wf))]

    # --- preferred: PyCBC -----------------------------------------------
    try:
        from pycbc.waveform import get_fd_waveform
        hp, _ = get_fd_waveform(approximant=approximant, mass1=30.0, mass2=30.0,
                                delta_f=df, f_lower=f_lower, f_final=f_final)
        npts = len(hp)
        t0 = time.perf_counter()
        for m1, m2 in masses:
            get_fd_waveform(approximant=approximant, mass1=max(m1, m2), mass2=min(m1, m2),
                            delta_f=df, f_lower=f_lower, f_final=f_final)
        return (time.perf_counter() - t0) / n_wf, f"pycbc/{approximant}", npts
    except Exception as exc:
        print(f"    [pycbc unavailable or failed: {type(exc).__name__}: {exc}]")

    # --- fallback: LALSimulation ----------------------------------------
    try:
        import lal
        import lalsimulation as lalsim
        appr = lalsim.SimInspiralGetApproximantFromString(approximant)

        def gen(m1, m2):
            return lalsim.SimInspiralChooseFDWaveform(
                m1 * lal.MSUN_SI, m2 * lal.MSUN_SI, 0., 0., 0., 0., 0., 0.,
                1e6 * lal.PC_SI, 0., 0., 0., 0., 0.,
                df, f_lower, f_final, f_lower, None, appr)

        hp, _ = gen(30.0, 30.0)
        npts = hp.data.length
        t0 = time.perf_counter()
        for m1, m2 in masses:
            gen(max(m1, m2), min(m1, m2))
        return (time.perf_counter() - t0) / n_wf, f"lalsimulation/{approximant}", npts
    except Exception as exc:
        print(f"    [lalsimulation unavailable or failed: {type(exc).__name__}: {exc}]")

    # --- last resort: pure-numpy TaylorF2 (3.5PN phase) ------------------
    # LOWER BOUND ONLY. PhenomD/XPHM cost noticeably more than this.
    print("    [falling back to a pure-numpy TaylorF2 proxy -- treat T_wf as a LOWER BOUND]")
    npts = int(f_final / df) + 1
    f = np.arange(npts) * df
    band = f >= f_lower
    fb = f[band]
    MTSUN = 4.925491025543576e-06

    def gen(m1, m2):
        mt = (m1 + m2) * MTSUN
        eta = m1 * m2 / (m1 + m2) ** 2
        v = (np.pi * mt * fb) ** (1.0 / 3.0)
        v2 = v * v
        psi = (3.0 / (128.0 * eta * v ** 5)) * (
            1.0
            + (3715.0 / 756.0 + 55.0 * eta / 9.0) * v2
            - 16.0 * np.pi * v2 * v
            + (15293365.0 / 508032.0 + 27145.0 * eta / 504.0 + 3085.0 * eta * eta / 72.0) * v2 * v2
        )
        amp = (fb ** (-7.0 / 6.0)) * np.sqrt(5.0 * eta / 24.0) / (np.pi ** (2.0 / 3.0))
        h = np.zeros(npts, dtype=np.complex128)
        h[band] = amp * np.exp(-1j * psi)
        return h

    gen(30.0, 30.0)
    t0 = time.perf_counter()
    for m1, m2 in masses:
        gen(max(m1, m2), min(m1, m2))
    return (time.perf_counter() - t0) / n_wf, "numpy/TaylorF2-proxy (LOWER BOUND)", npts


# ----------------------------------------------------------------------------
# --decompose: replace the estimated byte count with a measurement
#
# The headline claim "the matched filter is memory-bandwidth-bound" rested on an
# ESTIMATE that the kernel chain moves 9 array-sized passes over memory:
#     conj-multiply 3  +  two-pass ifft 4  +  abs 1.5  +  max-reduce 0.5  =  9
# The ifft term was a guess. This mode removes the guess.
#
# Method: mx.add(a, b) over complex64 moves exactly 3 units (read a, read b,
# write c) and does negligible arithmetic, so it runs at the GPU's real
# streaming bandwidth. Timing every other kernel against it gives
#
#     implied_units(K) = 3 * t_K / t_add
#
# with no need to know the absolute bandwidth. Because a strided kernel can only
# be SLOWER than pure streaming, implied_units is an UPPER BOUND on the true
# byte traffic: if the ifft's implied units come out at or below 4, the two-pass
# assumption was right and the kernel is running at near-ceiling bandwidth.
#
# The sweep is the assumption-free test. t_full / t_add roughly CONSTANT as N
# grows => the chain scales exactly like a streaming kernel => bandwidth-bound.
# That ratio growing like log N => arithmetic is starting to intrude.
# ----------------------------------------------------------------------------
UNIT_LABELS = {
    "add        (a + b)":            ("ceiling: exactly 3 units, ~0 FLOP", 3.0),
    "mul_conj   (conj(a) * b)":      ("3 if conj fuses, 4 if it does not", 3.0),
    "ifft       (ifft(p))":          ("THE UNKNOWN: 4 if two-pass", 4.0),
    "absmax     (max(abs(p)))":      ("1.5 + 0.5 = 2 if unfused", 2.0),
    "full chain (the benchmark)":    ("my estimate was 9", 9.0),
}


# ----------------------------------------------------------------------------
# Measured compute ceiling.
#
# The roofline's horizontal ceiling was previously a DERIVED number:
# 1280 ALUs x 2 FLOP/cycle x 1.398 GHz = 3.58 TFLOP/s. Apple publishes GPU core
# counts for the M2, not FLOPS, so that figure is a community calculation rather
# than a vendor specification. These two kernels measure it instead.
#
# Two ceilings, because they are not the same number:
#
#   fma_chain -- a compiled chain of `x = x*a + b` over a cache-resident array.
#                Arithmetic intensity grows with the chain depth, so sweeping
#                depth walks the kernel up the memory roof and onto the compute
#                roof; the plateau is the general-purpose fp32 ALU throughput.
#                THIS is the right ceiling for an FFT roofline.
#
#   matmul    -- 2n^3 FLOPs on 3n^2 elements, intensity ~n/6, heavily optimised
#                by Apple. On M5 and later the GPU cores carry Neural
#                Accelerators that speed up matrix work specifically, so this
#                can exceed what ordinary elementwise arithmetic reaches. Useful
#                context; misleading as an FFT ceiling.
#
# mx.compile is what makes the first one possible: without kernel fusion every
# elementwise op round-trips through DRAM and the chain stays bandwidth-bound.
# ----------------------------------------------------------------------------
def bench_compute_fma(mx, n, depth, rounds):
    """Compiled FMA chain. Returns (best seconds, GFLOP/s, FLOP/byte)."""
    rng = np.random.default_rng(7)
    x = mx.array(rng.random(n, dtype=np.float32))
    # arrays, not scalars, so the chain cannot collapse to a single affine op
    a = mx.array((1.0 + 1e-7 * rng.random(n)).astype(np.float32))
    b = mx.array((1e-7 * rng.random(n)).astype(np.float32))
    mx.eval(x, a, b)

    def chain(x, a, b):
        for _ in range(depth):
            x = x * a + b
        return x

    try:
        fn = mx.compile(chain)
    except Exception:
        fn = chain
    mx.eval(fn(x, a, b))                      # trace + compile, outside timing

    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        mx.eval(fn(x, a, b))
        best = min(best, time.perf_counter() - t0)

    flops = 2.0 * depth * n                   # one multiply + one add per step
    byts = 4.0 * n * 4                        # read x,a,b + write result (fused)
    return best, flops / best / 1e9, flops / byts


def bench_compute_matmul(mx, n, rounds):
    """Square fp32 matmul. Returns (best seconds, GFLOP/s)."""
    rng = np.random.default_rng(8)
    A = mx.array(rng.random((n, n), dtype=np.float32))
    B = mx.array(rng.random((n, n), dtype=np.float32))
    mx.eval(A, B)
    mx.eval(mx.matmul(A, B))
    best = float("inf")
    for _ in range(rounds):
        t0 = time.perf_counter()
        mx.eval(mx.matmul(A, B))
        best = min(best, time.perf_counter() - t0)
    return best, 2.0 * n**3 / best / 1e9


def run_compute_peak(mx, args):
    """Sweep FMA-chain depth to find the compute roof. Returns measured GFLOP/s."""
    print("=" * 78)
    print("MEASURED COMPUTE CEILING")
    print("=" * 78)
    print(f"fp32 FMA chain over {args.peak_n} elements ({args.peak_n*4/1e6:.1f} MB/array, "
          f"cache-resident), best of {args.rounds}.")
    print("Deeper chain -> higher intensity. The plateau is the compute roof.\n")
    print(f"{'depth':>6s} {'ms':>9s} {'GFLOP/s':>10s} {'FLOP/byte':>10s}")
    print("-" * 40)

    best_gflops, curve = 0.0, []
    for d in args.peak_depths:
        try:
            dt, gf, ai = bench_compute_fma(mx, args.peak_n, d, args.rounds)
            curve.append(dict(depth=d, seconds=dt, gflops=gf, intensity=ai))
            best_gflops = max(best_gflops, gf)
            print(f"{d:6d} {dt*1e3:9.3f} {gf:10.1f} {ai:10.1f}")
        except Exception as exc:
            print(f"{d:6d}   FAILED: {type(exc).__name__}: {exc}")

    mm = None
    try:
        dt, gf = bench_compute_matmul(mx, args.matmul_n, max(3, args.rounds - 2))
        mm = dict(n=args.matmul_n, seconds=dt, gflops=gf)
        print(f"\nmatmul {args.matmul_n}x{args.matmul_n} fp32: {dt*1e3:.2f} ms -> {gf:.0f} GFLOP/s")
        print("  (includes any matrix-specific acceleration; not the FFT ceiling)")
    except Exception as exc:
        print(f"\nmatmul FAILED: {type(exc).__name__}: {exc}")

    derived = 1280 * 2 * 1.398        # GFLOP/s, M2: ALUs x FLOP/cycle x GHz
    mm_g = mm["gflops"] if mm else 0.0

    # Diagnose the FMA chain. If mx.compile fused it, time is roughly flat in
    # depth. Time growing linearly with depth means every step became its own
    # kernel with its own memory round trip, so the sweep measured cache
    # bandwidth and tells us nothing about ALU throughput.
    fused = None
    if len(curve) >= 2:
        lo, hi = curve[0], curve[-1]
        growth = (hi["seconds"] / lo["seconds"]) / (hi["depth"] / lo["depth"])
        fused = growth < 0.5      # well below linear => real fusion
        print(f"\n  chain time grew {hi['seconds']/lo['seconds']:.0f}x over a "
              f"{hi['depth']//lo['depth']}x depth increase "
              f"-> {'FUSED' if fused else 'NOT fused (per-step kernel launches)'}")
        if not fused:
            print("  the FMA sweep therefore measured cache bandwidth, not ALU throughput.")

    # Pick the ceiling. Whichever kernel got furthest is the better lower bound.
    best = max(best_gflops, mm_g)
    which = "fma_chain" if best_gflops >= mm_g else f"matmul {args.matmul_n}^2"
    print(f"\n  fma chain      : {best_gflops:7.0f} GFLOP/s")
    print(f"  matmul         : {mm_g:7.0f} GFLOP/s")
    print(f"  measured peak  : {best:7.0f} GFLOP/s  (from {which})")
    print(f"  derived peak   : {derived:7.0f} GFLOP/s  (1280 ALU x 2 x 1.398 GHz; "
          f"Apple does not publish a FLOPS figure)")
    print(f"  measured/derived: {100*best/derived:6.0f}%")
    if best == mm_g and mm_g > 0:
        print("  NOTE: matmul is plain fp32 ALU work on M1-M4. From M5 on, the GPU cores")
        print("        carry Neural Accelerators that speed up matrix work specifically,")
        print("        so on those chips matmul overstates the elementwise ceiling.")
    print()
    return best, curve, mm


# ----------------------------------------------------------------------------
# CPU ceilings, for the GPU-vs-one-core comparison.
#
# The GPU's advantage over a CPU core is bounded differently in each regime:
#   bandwidth-bound -> ratio of achievable GB/s
#   compute-bound   -> ratio of achievable GFLOP/s
#
# The bandwidth row is the one worth measuring. Apple's big cores have unusually
# deep memory-level parallelism -- a single M1 Firestorm core was measured at
# roughly the whole chip's bandwidth -- so the GPU's edge on a streaming kernel
# could be anywhere from ~1.5x to ~3.6x. That is a 2.4x spread on a number that
# decides how to read any single-threaded CPU baseline.
#
# CAVEAT: MLX's CPU backend does its own threading and does not necessarily obey
# OMP_NUM_THREADS. Watch core occupancy (Activity Monitor, or macmon) while this
# runs. The original matched-filter benchmark showed P-CPU pinned near 25% on a
# 4-P-core M2, i.e. one core, so MLX CPU elementwise work appears single-
# threaded -- but confirm rather than assume, and treat the compute row as an
# upper bound on one core if more than one core lights up.
# ----------------------------------------------------------------------------
def run_cpu_baseline(mx, ifft, args, gpu_bw_gbs, gpu_peak_gflops):
    print("=" * 78)
    print("CPU CEILINGS  (GPU vs one core)")
    print("=" * 78)
    n, unit = args.array_size, args.array_size * 8
    rounds = max(3, args.rounds - 2)
    out = {}

    mx.set_default_device(mx.cpu)
    try:
        k = min(4, args.n_unique)
        pairs, prods = _make_inputs(mx, n, k)
        kern = _kernels(mx, ifft, pairs, prods)
        t_add = bench_kernel(mx, kern["add        (a + b)"], k, rounds)
        cpu_bw = 3 * unit / t_add / 1e9
        out["cpu_stream_gbs"] = cpu_bw
        print(f"  streaming (mx.add, 3 units)  : {cpu_bw:7.1f} GB/s")
        del pairs, prods, kern
    except Exception as exc:
        cpu_bw = 0.0
        print(f"  streaming FAILED: {type(exc).__name__}: {exc}")

    try:
        _, cpu_mm = bench_compute_matmul(mx, args.matmul_n, rounds)
        out["cpu_matmul_gflops"] = cpu_mm
        print(f"  matmul {args.matmul_n}^2 fp32          : {cpu_mm:7.0f} GFLOP/s   "
              f"(Accelerate may dispatch this to AMX)")
    except Exception as exc:
        cpu_mm = 0.0
        print(f"  matmul FAILED: {type(exc).__name__}: {exc}")

    mx.set_default_device(mx.gpu)

    # one M2 P-core, from microarchitecture: 4 NEON pipes x 4 fp32 lanes x 2 x GHz
    core_peak = 4 * 4 * 2 * args.core_ghz
    print(f"\n  one P-core fp32 peak (derived): {core_peak:7.0f} GFLOP/s   "
          f"(4 NEON pipes x 4 lanes x 2 FMA x {args.core_ghz:g} GHz)")
    print()
    print(f"{'regime':34s} {'CPU':>10s} {'GPU':>10s} {'GPU advantage':>15s}")
    print("-" * 74)
    if cpu_bw > 0:
        print(f"{'bandwidth-bound (GB/s)':34s} {cpu_bw:10.1f} {gpu_bw_gbs:10.1f} "
              f"{gpu_bw_gbs/cpu_bw:14.1f}x")
        out["ratio_bandwidth"] = gpu_bw_gbs / cpu_bw
    if cpu_mm > 0:
        print(f"{'compute, matmul/AMX (GFLOP/s)':34s} {cpu_mm:10.0f} {gpu_peak_gflops:10.0f} "
              f"{gpu_peak_gflops/cpu_mm:14.1f}x")
        out["ratio_compute_matmul"] = gpu_peak_gflops / cpu_mm
    print(f"{'compute, NEON only (GFLOP/s)':34s} {core_peak:10.0f} {gpu_peak_gflops:10.0f} "
          f"{gpu_peak_gflops/core_peak:14.1f}x")
    out["ratio_compute_neon"] = gpu_peak_gflops / core_peak
    out["core_peak_gflops_derived"] = core_peak
    print()
    print("  AMX is per-CLUSTER, so one thread can drive much of it -- the matmul row")
    print("  is the right bound for GEMM-shaped work. Transcendentals and polynomials")
    print("  (waveform generation) get no AMX, so the NEON row bounds those.")
    print()
    return out


def bench_kernel(mx, build, n_ops, rounds=5):
    """build(i) -> a lazy mlx array. Returns the best seconds/op over `rounds`."""
    mx.eval(build(0))                                   # warm up: compile shaders
    best = float("inf")
    for _ in range(rounds):
        batch = [build(i) for i in range(n_ops)]        # distinct inputs: no CSE
        t0 = time.perf_counter()
        mx.eval(batch)
        best = min(best, (time.perf_counter() - t0) / n_ops)
        del batch
        try:
            mx.clear_cache()
        except AttributeError:
            pass
    return best


def _make_inputs(mx, n, k, seed=0):
    """k pre-evaluated complex64 pairs plus their conj-products."""
    rng = np.random.default_rng(seed)
    pairs, prods = [], []
    for _ in range(k):
        a = mx.array((rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64))
        b = mx.array((rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64))
        mx.eval(a, b)
        pairs.append((a, b))
        p = mx.multiply(a.conj(), b)
        mx.eval(p)
        prods.append(p)
    return pairs, prods


def _kernels(mx, ifft, pairs, prods):
    k = len(pairs)
    return {
        "add        (a + b)":         lambda i: mx.add(*pairs[i % k]),
        "mul_conj   (conj(a) * b)":   lambda i: mx.multiply(pairs[i % k][0].conj(), pairs[i % k][1]),
        "ifft       (ifft(p))":       lambda i: ifft(prods[i % k]),
        "absmax     (max(abs(p)))":   lambda i: mx.max(mx.abs(prods[i % k])),
        "full chain (the benchmark)": lambda i: mx.max(mx.abs(ifft(
            mx.multiply(pairs[i % k][0].conj(), pairs[i % k][1])))),
    }


def run_decompose(mx, ifft, args):
    n = args.array_size
    unit = n * 8
    k = args.n_unique
    flops = 5 * n * math.log2(n) + 9 * n

    # measure the compute roof before anything else, so the rest can use it
    peak, peak_curve, peak_mm = (0.0, [], None)
    if not args.skip_peak:
        mx.set_default_device(mx.gpu)
        peak, peak_curve, peak_mm = run_compute_peak(mx, args)
    peak_src = "measured"
    if args.skip_peak or not peak_curve:
        peak, peak_src = args.peak_flops, "assumed"
    elif peak < 0.10 * args.peak_flops:
        # A plausible measurement lands within a factor of a few of the derived
        # figure. Far below that means the chain never fused (mx.compile fell
        # back), so the kernel stayed bandwidth-bound and measured nothing.
        print(f"  !! measured peak {peak:.0f} GFLOP/s is under 10% of the derived "
              f"{args.peak_flops:.0f} -- the FMA chain probably did not fuse.")
        print(f"  !! falling back to the derived figure for the arithmetic below.\n")
        peak, peak_src = args.peak_flops, "assumed (measurement rejected)"

    print("=" * 78)
    print("DECOMPOSITION -- measuring the byte count instead of estimating it")
    print("=" * 78)
    print(f"N = 2^{int(math.log2(n))} = {n}   1 unit = N x 8 B = {unit / 1e6:.2f} MB")
    print(f"{k} distinct input sets, best of {args.rounds} rounds\n")

    mx.set_default_device(mx.gpu)
    pairs, prods = _make_inputs(mx, n, k)
    kern = _kernels(mx, ifft, pairs, prods)

    t = {}
    for name, build in kern.items():
        t[name] = bench_kernel(mx, build, k, args.rounds)

    t_add = t["add        (a + b)"]
    ceiling = 3 * unit / t_add / 1e9

    t_full_ = t["full chain (the benchmark)"]
    print(f"{'kernel':30s} {'ms/op':>8s} {'%oftotal':>9s} {'implied':>8s} {'assumed':>8s}  note")
    print("-" * 97)
    implied = {}
    for name, dt in t.items():
        u = 3.0 * dt / t_add
        implied[name] = u
        note, assumed = UNIT_LABELS[name]
        share = "-" if name.startswith(("add", "full")) else f"{100 * dt / t_full_:8.0f}%"
        print(f"{name:30s} {dt * 1e3:8.3f} {share:>9s} {u:8.2f} {assumed:8.1f}  {note}")
    print("\nimplied units = 3 x t_kernel / t_add. It is an UPPER BOUND on true byte")
    print("traffic: a strided kernel can only run slower than pure streaming, so a")
    print("kernel that looks like 4 units is moving at most 4 units.")

    u_full = implied["full chain (the benchmark)"]
    u_fft = implied["ifft       (ifft(p))"]
    parts = sum(implied[n_] for n_ in kern if not n_.startswith(("add", "full")))

    print()
    print("=" * 78)
    print("WHAT THIS SETTLES")
    print("=" * 78)
    print(f"  measured streaming ceiling      {ceiling:7.1f} GB/s   "
          f"(vs {args.host_bw:g} GB/s quoted peak = {100 * ceiling / args.host_bw:.0f}%)")
    print(f"  ifft implied units              {u_fft:7.2f}       "
          f"-> at most {u_fft / 2:.2f} memory passes (I assumed 2)")
    print(f"  full chain implied units        {u_full:7.2f}       -> I estimated 9")
    print(f"  sum of parts                    {parts:7.2f}       "
          f"({'fusion: whole < parts' if u_full < parts * 0.95 else 'no strong fusion'})")
    print()
    bw_full = u_full * unit / t["full chain (the benchmark)"] / 1e9
    gflops = flops / t["full chain (the benchmark)"] / 1e9
    print(f"  full chain achieves             {bw_full:7.1f} GB/s  "
          f"= {100 * bw_full / ceiling:.0f}% of the measured ceiling")
    print(f"  full chain achieves             {gflops:7.1f} GFLOP/s"
          f" = {100 * gflops / peak:.1f}% of {peak / 1000:.2f} TFLOP/s ({peak_src})")
    print(f"  arithmetic intensity            {flops / (u_full * unit):7.2f} FLOP/byte"
          f"  (machine balance {peak / ceiling:.0f})")
    print(f"  peak would have to fall below   {flops / (u_full * unit) * ceiling:7.0f} GFLOP/s"
          f" for this kernel to be compute-bound"
          f"  ({peak / (flops / (u_full * unit) * ceiling):.0f}x lower)")
    print()
    if 100 * bw_full / ceiling > 50 and 100 * gflops / peak < 15:
        print("  VERDICT: bandwidth-bound. Buy GB/s, not GPU cores.")
    elif 100 * gflops / peak > 40:
        print("  VERDICT: compute-bound -- this contradicts the M2 analysis. Re-check.")
    else:
        print("  VERDICT: mixed / inconclusive. Look at the sweep below.")

    # ---------------- CPU ceilings, for GPU-vs-one-core ----------------
    cpu = {}
    if not args.skip_cpu:
        print()
        cpu = run_cpu_baseline(mx, ifft, args, ceiling, peak)

    # ---------------- assumption-free scaling test ----------------
    print()
    print("=" * 78)
    print("SWEEP -- does the chain scale like a streaming kernel?")
    print("=" * 78)
    print("t_full/t_add flat across N  => pure bandwidth.  Growing like log N => compute.")
    print()
    print(f"{'N':>12s} {'t_add/ms':>9s} {'t_full/ms':>10s} {'t_full/t_add':>13s} "
          f"{'t_full/N (ns)':>14s} {'log2 N':>7s}")
    print("-" * 74)
    for e in args.sweep:
        nn = 2 ** e
        try:
            kk = max(4, min(k, int(k * n / nn)))        # cap memory at large N
            pp, qq = _make_inputs(mx, nn, kk, seed=e)
            kern2 = _kernels(mx, ifft, pp, qq)
            ta = bench_kernel(mx, kern2["add        (a + b)"], kk, max(3, args.rounds - 2))
            tf = bench_kernel(mx, kern2["full chain (the benchmark)"], kk, max(3, args.rounds - 2))
            print(f"{nn:12d} {ta * 1e3:9.3f} {tf * 1e3:10.3f} {tf / ta:13.2f} "
                  f"{tf / nn * 1e9:14.3f} {e:7d}")
            del pp, qq, kern2
            try:
                mx.clear_cache()
            except AttributeError:
                pass
        except Exception as exc:
            print(f"{nn:12d}   FAILED: {type(exc).__name__}: {exc}")
    print("\nIgnore sizes below ~2^19: at those the arrays fit closer to cache and the")
    print("time is dominated by kernel dispatch, not memory traffic.")

    out = dict(array_size=n, unit_bytes=unit, n_unique=k, rounds=args.rounds,
               times_s=t, implied_units=implied, ceiling_gbs=ceiling,
               full_chain_gbs=bw_full, full_chain_gflops=gflops,
               compute_peak_gflops=peak, compute_peak_source=peak_src,
               compute_peak_derived_gflops=1280 * 2 * 1.398,
               compute_peak_curve=peak_curve, matmul=peak_mm,
               machine_balance_flop_per_byte=peak / ceiling,
               cpu_ceilings=cpu)
    with open(args.decompose_output, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {args.decompose_output}")


# ----------------------------------------------------------------------------
# --surrogate: time the JAX NRSur7dq4 neural-network surrogate
#
# The PE side of the fleet needs a different number from the search side. A
# search reuses one template across many data segments, so generation cost
# amortises away. PE generates a fresh waveform for every likelihood call and
# amortises nothing, so there T_wf IS the wall clock.
#
# Two questions this settles:
#
#   1. Where the batching crossover sits. Evaluated one waveform at a time, the
#      network streams its whole weight matrix out of DRAM per call, which is
#      the same bandwidth-bound regime the matched filter is in. Evaluated on a
#      batch of B, the weights are read once and reused B times, and the work
#      turns into matmul. The measured curve says where that transition
#      actually happens on this chip -- which tells you whether a PE sampler
#      has to keep many walkers in flight to get the hardware's throughput.
#
#   2. Whether float32 is usable. The package README enables jax_enable_x64,
#      and jax-mps does NOT support float64 -- so if this is ever to run on the
#      Mac GPU, fp32 has to be accurate enough. Precision is a process-wide JAX
#      setting, so run this mode twice:
#          python hardware_sizing_benchmark.py --surrogate --surrogate-precision f64
#          python hardware_sizing_benchmark.py --surrogate --surrogate-precision f32
#      The second run finds the first run's saved reference waveform and prints
#      the mismatch between them.
# ----------------------------------------------------------------------------
def _white_match(h1, h2):
    """Match between two complex strains, white noise, maximised over time and phase.

    Zero-padded to twice the length so the circular correlation equals the
    linear one over the shifts that matter.
    """
    n = 1 << int(math.ceil(math.log2(2 * max(len(h1), len(h2)))))
    a = np.zeros(n, dtype=np.complex128)
    b = np.zeros(n, dtype=np.complex128)
    a[:len(h1)] = h1
    b[:len(h2)] = h2
    corr = np.fft.ifft(np.fft.fft(a) * np.conj(np.fft.fft(b)))
    norm = math.sqrt(float(np.vdot(a, a).real) * float(np.vdot(b, b).real))
    if not norm > 0.0:
        return float("nan")          # a zero waveform has no match to report
    return float(np.max(np.abs(corr)) / norm)


def _surrogate_precision_failure(where, exc, x64):
    """A float32 crash IS the precision answer. Report it as one."""
    msg = str(exc).strip().splitlines()
    print()
    print("=" * 78)
    print(f"FAILED during {where}")
    print("=" * 78)
    print(f"{type(exc).__name__}: " + (msg[0] if msg else repr(exc)))
    for line in msg[1:6]:
        print(f"    {line}")
    if x64:
        print("\nThis is the float64 run, so this is a real failure, not a precision")
        print("result. Check the weights file and the package install.")
        return
    print()
    print("This is the float32 run, and this failure IS the answer to the precision")
    print("question. The package README says so up front: enable double precision")
    print("before creating a model, because physical strain amplitudes are ~1e-21")
    print("and the reference-frequency solvers assume float64.")
    print()
    print("The traceback points at the solver, not the network. The README's")
    print("find_input_spins_at_fref_Hz inverts spin trajectories to match the")
    print("requested values at f_ref, and there are two ways float32 breaks that:")
    print("  * tolerance   -- float32 eps is 1.2e-7. A solve iterating to ~1e-10")
    print("                   can never converge, the step diverges, and the next")
    print("                   linear_solve gets non-finite input.")
    print("  * dynamic range -- a strain squared is (1e-21)^2 = 1e-42, below the")
    print("                   1.18e-38 float32 normal minimum, so any norm or inner")
    print("                   product of strain-scale quantities flushes to zero.")
    print("Both are fixable by isolating that stage; neither implicates the network.")
    print()
    print("WHAT THIS MEANS FOR A METAL PORT")
    print("  The reference-frequency solve is a handful of small linear systems and")
    print("  costs a negligible share of the per-waveform time. Running it in")
    print("  float64 on the CPU while the network runs in float32 on the GPU is a")
    print("  legitimate design, so float64 alone does not close the Metal path.")
    print("  Check the dtype breakdown printed above: any complex arrays are the")
    print("  harder blocker, because jax-metal has no complex support at all. MLX")
    print("  does, which is another reason an MLX port beats a jax-metal one.")
    print()
    print("To measure whether the NETWORK itself needs float64, run")
    print("    python hardware_sizing_benchmark.py --surrogate --surrogate-weight-f32")
    print("which keeps the solvers in float64 and rounds only the weights.")


def _round_trip_f32(jax, model):
    """Model with every float64/complex128 leaf rounded through 32-bit and back.

    Arithmetic stays in float64, so the solvers still work; only the stored
    values lose the low bits. That isolates how much precision the weights
    need from how much dynamic range the solvers need.
    """
    import jax.numpy as jnp
    n_cast = [0]

    def f(x):
        if not (hasattr(x, "dtype") and hasattr(x, "shape") and getattr(x, "ndim", 0) > 0):
            return x
        if x.dtype == jnp.float64:
            n_cast[0] += 1
            return x.astype(jnp.float32).astype(jnp.float64)
        if x.dtype == jnp.complex128:
            n_cast[0] += 1
            return x.astype(jnp.complex64).astype(jnp.complex128)
        return x

    return jax.tree_util.tree_map(f, model), n_cast[0]


def _surrogate_params(rng, b):
    """A batch of (q, chiA, chiB) inside the model's stated validity region."""
    p = np.empty((b, 7), dtype=np.float64)
    p[:, 0] = rng.uniform(1.0, 4.0, b)          # q in [1, 4]
    p[:, 1:] = rng.uniform(-0.6, 0.6, (b, 6))   # spins, inside [-0.8, 0.8]
    return p


def run_surrogate(args):
    prec = args.surrogate_precision
    x64 = (prec == "f64")

    # Both of these are process-wide and must be set before JAX starts.
    if args.surrogate_threads == 1:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   " --xla_cpu_multi_thread_eigen=false"
                                   " intra_op_parallelism_threads=1").strip()
        thread_note = "1  (single core, matching how T_wf was measured)"
    elif args.surrogate_threads > 1:
        os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") +
                                   f" intra_op_parallelism_threads={args.surrogate_threads}").strip()
        thread_note = f"{args.surrogate_threads}"
    else:
        thread_note = "XLA default (all cores)"
    if args.surrogate_platform:
        os.environ["JAX_PLATFORMS"] = args.surrogate_platform

    try:
        import jax
    except ImportError:
        print("jax is not installed. In the benchmark env:")
        print("    pip install jax-nrsur7dq4-nn")
        return
    jax.config.update("jax_enable_x64", x64)
    import jax.numpy as jnp

    try:
        from jax_nrsur7dq4_nn import NRSur7dq4NNModel
    except ImportError as exc:
        print(f"jax_nrsur7dq4_nn is not importable: {exc}")
        print("    pip install jax-nrsur7dq4-nn")
        print("and put NRSur7dq4_NN.h5 (Zenodo 10.5281/zenodo.21177182) at")
        print("    ~/.jax_nrsur7dq4_nn/NRSur7dq4_NN.h5   or point JAX_NRSUR7DQ4_NN_H5 at it")
        return

    h5 = os.environ.get("JAX_NRSUR7DQ4_NN_H5") or \
        os.path.expanduser("~/.jax_nrsur7dq4_nn/NRSur7dq4_NN.h5")
    h5_mb = os.path.getsize(h5) / 1e6 if os.path.exists(h5) else None

    print("=" * 78)
    print("SURROGATE -- jax-nrsur7dq4-nn, the waveform the PE side will use")
    print("=" * 78)
    print(f"precision     : float{'64' if x64 else '32'}  (jax_enable_x64={x64})")
    print(f"jax           : {jax.__version__}   devices: {jax.local_devices()}")
    print(f"XLA threads   : {thread_note}")
    print(f"weights file  : {h5}" + (f"   ({h5_mb:.1f} MB on disk)" if h5_mb else "   [NOT FOUND]"))

    try:
        t0 = time.perf_counter()
        model = NRSur7dq4NNModel(M_total_Msun=args.surrogate_mtotal,
                                 dist_mpc=args.surrogate_dist)
        t_build = time.perf_counter() - t0
    except Exception as exc:
        _surrogate_precision_failure("model construction", exc, x64)
        return

    leaves = [l for l in jax.tree_util.tree_leaves(model)
              if hasattr(l, "shape") and hasattr(l, "nbytes") and l.ndim > 0]
    n_par = sum(int(np.prod(l.shape)) for l in leaves)
    w_bytes = sum(int(l.nbytes) for l in leaves)
    dts = sorted({str(l.dtype) for l in leaves})
    by_dtype = {}
    for l in leaves:
        k = str(l.dtype)
        n_, b_ = by_dtype.get(k, (0, 0))
        by_dtype[k] = (n_ + 1, b_ + int(l.nbytes))
    print(f"model build   : {t_build:.2f} s")
    print(f"parameters    : {n_par:,} in {len(leaves)} arrays -> {w_bytes / 1e6:.1f} MB resident")
    for k in sorted(by_dtype, key=lambda z: -by_dtype[z][1]):
        n_, b_ = by_dtype[k]
        flag = ""
        if k.startswith("complex"):
            flag = "   <- jax-metal has no complex support at all"
        elif k == "float64":
            flag = "   <- jax-mps has no float64"
        print(f"  {k:12s} {n_:4d} arrays  {b_ / 1e6:8.1f} MB{flag}")

    p_ref = jnp.asarray(np.array([1.5, 0.1, 0.0, 0.3, -0.1, 0.0, 0.2]))
    try:
        hp_ref, hc_ref = jax.block_until_ready(model(p_ref))
    except Exception as exc:
        _surrogate_precision_failure("the first waveform evaluation", exc, x64)
        return
    n_t = int(np.asarray(hp_ref).shape[-1])
    itemsize = np.asarray(hp_ref).dtype.itemsize
    out_bytes = 2 * n_t * itemsize
    print(f"output        : hp, hc of {n_t:,} samples each -> {out_bytes / 1e6:.2f} MB per waveform")
    print()

    # ---- the two callables -------------------------------------------------
    try:
        import equinox as eqx
        call_1 = eqx.filter_jit(model)
        call_b = eqx.filter_jit(eqx.filter_vmap(model, in_axes=(0,)))
        wrap = "equinox filter_jit / filter_vmap"
    except ImportError:
        eqx = None
        call_1 = jax.jit(lambda p: model(p))
        call_b = jax.jit(jax.vmap(lambda p: model(p)))
        wrap = "jax.jit / jax.vmap (equinox not installed)"
    print(f"wrapping      : {wrap}")

    def timeit(fn, x, rounds):
        """(compile+first-call s, best s, median s). JIT is warmed outside the timer."""
        tc0 = time.perf_counter()
        jax.block_until_ready(fn(x))
        t_compile = time.perf_counter() - tc0
        ts = []
        for _ in range(rounds):
            tb0 = time.perf_counter()
            jax.block_until_ready(fn(x))
            ts.append(time.perf_counter() - tb0)
        return t_compile, min(ts), float(np.median(ts))

    rng = np.random.default_rng(7)
    rows = []

    # unbatched, no vmap: the latency a serial sampler actually sees
    p1 = jnp.asarray(_surrogate_params(rng, 1)[0])
    try:
        tc, tb, tm = timeit(call_1, p1, args.surrogate_rounds)
    except Exception as exc:
        _surrogate_precision_failure("the unbatched timing call", exc, x64)
        return
    rows.append(dict(batch=1, vmap=False, compile_s=tc, best_s=tb, median_s=tm))

    for b in args.surrogate_batches:
        need_gb = b * out_bytes / 1e9
        if need_gb > args.surrogate_max_gb:
            print(f"  [skipping batch {b}: {need_gb * 1e3:.0f} MB of output exceeds "
                  f"--surrogate-max-gb {args.surrogate_max_gb:g}]")
            continue
        pb = jnp.asarray(_surrogate_params(rng, b))
        try:
            tc, tb, tm = timeit(call_b, pb, args.surrogate_rounds)
        except Exception as exc:
            print(f"  [batch {b} FAILED: {type(exc).__name__}: {exc}]")
            continue
        rows.append(dict(batch=b, vmap=True, compile_s=tc, best_s=tb, median_s=tm))

    # ---- the table ---------------------------------------------------------
    print()
    print(f"{'batch':>6s} {'compile':>8s} {'ms/call':>10s} {'ms/wf':>9s} {'wf/s':>8s} "
          f"{'vs B=1':>7s} {'wtGB/s':>7s} {'GFLOP/s':>8s} {'FLOP/B':>7s}")
    print("-" * 78)
    base = rows[0]["best_s"]
    for r in rows:
        b = r["batch"]
        t = r["best_s"]
        per = t / b
        flops = 2.0 * n_par * b                      # MLP weight FLOPs: a LOWER bound
        inten = flops / (w_bytes + b * out_bytes)
        tag = f"{b}" + ("" if r["vmap"] else "*")
        r.update(per_wf_s=per, gflops=flops / t / 1e9, intensity=inten,
                 weights_gbs=w_bytes / t / 1e9, speedup=base / per)
        print(f"{tag:>6s} {r['compile_s']:7.2f}s {t * 1e3:10.3f} {per * 1e3:9.3f} "
              f"{1 / per:8.1f} {base / per:6.2f}x {w_bytes / t / 1e9:7.1f} "
              f"{flops / t / 1e9:8.1f} {inten:7.2f}")
    print("* = plain call, no vmap. GFLOP/s and FLOP/B count only the weight")
    print("  multiply-accumulates (2 x parameters x batch), so both are LOWER bounds:")
    print("  activations, the spline/basis layer and the output assembly are not in it.")
    print("wtGB/s = weight bytes / call time -- the DRAM rate the weights alone demand.")
    print("  It falls with batch because the weights are read once per CALL, not once")
    print("  per waveform; that fall is the whole point of batching.")

    # ---- what the curve means ---------------------------------------------
    batched = [r for r in rows if r["vmap"]]
    plateau = batched[-1] if batched else rows[0]
    for a_, b_ in zip(batched, batched[1:]):
        if b_["per_wf_s"] > 0.95 * a_["per_wf_s"]:
            plateau = a_
            break
    t_floor = w_bytes / (args.cpu_bw * 1e9)
    t1 = rows[0]["best_s"]

    print()
    print("=" * 78)
    print("WHERE THE CROSSOVER IS")
    print("=" * 78)
    print(f"weight-streaming floor : {t_floor * 1e3:.3f} ms  "
          f"({w_bytes / 1e6:.0f} MB at {args.cpu_bw:g} GB/s, the CPU STREAM figure)")
    print(f"measured single call   : {t1 * 1e3:.3f} ms  = {t1 / t_floor:.1f}x that floor")
    if t1 / t_floor < 2.0:
        print("  -> unbatched evaluation is essentially weight streaming: bandwidth-bound,")
        print("     exactly like the matched filter. Batching is the only lever.")
    else:
        print("  -> unbatched evaluation is well above the streaming floor, so it is NOT")
        print("     bandwidth-bound: the cost is small-matmul inefficiency and per-call")
        print("     overhead. Batching should recover most of the gap.")
    print(f"throughput plateaus at : batch {plateau['batch']}, "
          f"{plateau['per_wf_s'] * 1e3:.3f} ms/waveform "
          f"({plateau['speedup']:.1f}x the unbatched call)")
    if plateau["batch"] > 1:
        print(f"  A PE sampler has to keep >= {plateau['batch']} waveforms in flight to see")
        print(f"  this. Serial likelihood evaluation gets "
              f"{rows[0]['per_wf_s'] * 1e3:.1f} ms instead -- "
              f"{plateau['speedup']:.1f}x worse.")
    else:
        print("  Batching buys nothing here: one waveform at a time already saturates")
        print("  whatever the limit is, so a serial sampler loses nothing.")
    print()
    print(f"vs the search waveform : IMRPhenomD measured at {args.twf_ms:g} ms/template "
          f"(one core, PyCBC)")
    print(f"  surrogate, unbatched : {t1 * 1e3 / args.twf_ms:6.2f}x IMRPhenomD")
    print(f"  surrogate, batch {plateau['batch']:<4d}: "
          f"{plateau['per_wf_s'] * 1e3 / args.twf_ms:6.2f}x IMRPhenomD")

    # ---- what that means for the purchase ----------------------------------
    print()
    print("=" * 78)
    print(f"PE THROUGHPUT PER MACHINE  ({args.pe_evals:,.0f} likelihood evaluations)")
    print("=" * 78)
    print(f"Assumes one independent chain per big core, each batching {plateau['batch']} "
          f"waveforms,\nand an M5 big core {args.core_speedup:g}x this machine's P-core. "
          "Generation only --\nlikelihood overhead (FFT, inner products) is on top.")
    print()
    print("Every core streams the whole weight matrix per call. The cores share one")
    print(f"copy in memory, but at {w_bytes / 1e6:.0f} MB it will not sit in any cache, so the "
          "traffic\nis effectively per core. 'want' is the DRAM rate all cores together would")
    print("demand; where that exceeds the machine's rated bandwidth the cores cannot")
    print("all be fed, and wf/s is scaled down to what bandwidth allows (marked !).")
    print()
    print(f"{'machine':30s} {'$edu':>6s} {'cores':>5s} {'GB/s':>5s} {'want':>6s} "
          f"{'wf/s':>9s} {'units':>6s} {'fleet wf/s':>11s} {'hours':>8s}")
    print("-" * 96)
    pe_rows = []
    for name, price, bw_t, cores in CANDIDATES:
        per_core = args.core_speedup / plateau["per_wf_s"]
        machine = per_core * cores
        # one weight sweep per call; a call covers `batch` waveforms
        want = machine * w_bytes / plateau["batch"] / 1e9
        capped = want > bw_t
        if capped:
            machine *= bw_t / want
        units = int(BUDGET // price)
        fleet = machine * units
        hours = args.pe_evals / fleet / 3600.0
        pe_rows.append(dict(machine=name, price=price, cores=cores, peak_bw=bw_t,
                            want_gbs=want, bandwidth_capped=capped,
                            wf_per_s=machine, units=units, fleet_wf_per_s=fleet,
                            hours=hours))
        print(f"{name:30s} {price:6d} {cores:5d} {bw_t:5d} {want:6.0f} "
              f"{machine:8.1f}{'!' if capped else ' '} {units:6d} "
              f"{fleet:11.1f} {hours:8.1f}")
    print()
    if any(r["bandwidth_capped"] for r in pe_rows):
        print("Some rows are bandwidth-capped, so PE is bandwidth-bound here too and")
        print("ranks machines the same way the search does. Raising the batch size is")
        print("the fix: it cuts 'want' proportionally, at the cost of needing that many")
        print("waveforms in flight per chain.")
    else:
        print("No row is bandwidth-capped, so PE really does rank machines by core")
        print("count while the search ranks them by memory bandwidth. The two orderings")
        print("disagree, which is what makes a mixed fleet worth considering.")

    # ---- precision cross-check --------------------------------------------
    # One waveform could agree by luck, so check a fixed spread of the parameter
    # space. The seed is fixed, so the f64 and f32 runs evaluate the SAME points.
    check_rng = np.random.default_rng(20260921)
    p_check = np.vstack([np.asarray(p_ref, dtype=np.float64)[None, :],
                         _surrogate_params(check_rng, args.surrogate_n_check)])
    hs = []
    for row in p_check:
        a, b = jax.block_until_ready(call_1(jnp.asarray(row)))
        hs.append(np.asarray(a, dtype=np.float64) - 1j * np.asarray(b, dtype=np.float64))
    h_here = np.array(hs)

    # ---- how much precision do the WEIGHTS need? ---------------------------
    # Separate question from the one above. float32 fails outright because the
    # solvers need the dynamic range; this asks whether the network's stored
    # values need the extra bits. Arithmetic stays float64, only the weights
    # are rounded through 32-bit and back.
    weight_probe = None
    if args.surrogate_weight_f32:
        if not x64:
            print("\n  [--surrogate-weight-f32 is only meaningful under float64; skipping]")
        else:
            try:
                model32, n_cast = _round_trip_f32(jax, model)
                call_1_32 = (eqx.filter_jit(model32) if eqx is not None
                             else jax.jit(lambda q: model32(q)))
                mis32, l2_32 = [], []
                for row, h_a in zip(p_check, h_here):
                    a, b = jax.block_until_ready(call_1_32(jnp.asarray(row)))
                    h_b = (np.asarray(a, dtype=np.float64)
                           - 1j * np.asarray(b, dtype=np.float64))
                    mis32.append(1.0 - _white_match(h_a, h_b))
                    nrm = float(np.linalg.norm(h_a))
                    l2_32.append(float(np.linalg.norm(h_a - h_b) / nrm) if nrm else float("nan"))
                worst32 = max(mis32)
                if not np.isfinite(worst32):
                    raise ValueError("non-finite mismatch: one of the waveforms "
                                     "is zero or NaN")
                weight_probe = dict(n_arrays_rounded=n_cast, mismatch_worst=worst32,
                                    mismatch_median=float(np.median(mis32)),
                                    rel_l2_worst=max(l2_32), mismatch=mis32)
                print()
                print("=" * 78)
                print("WEIGHT-PRECISION PROBE  float64 arithmetic, 32-bit weights")
                print("=" * 78)
                b64 = sum(v[1] for k, v in by_dtype.items()
                          if k in ("float64", "complex128"))
                print(f"rounded {n_cast} of {len(leaves)} arrays through 32-bit and back: "
                      f"{b64 / 1e6:.2f} MB of {w_bytes / 1e6:.1f} MB "
                      f"({100 * b64 / w_bytes:.2f}%) was 64-bit to begin with")
                if b64 < 0.02 * w_bytes:
                    print("  The bulk of the model is ALREADY 32-bit, so this probe only tests")
                    print("  the few genuinely-64-bit arrays -- which is the sharper question.")
                print(f"mismatch, worst / median : {worst32:.3e} / {np.median(mis32):.3e}")
                print(f"relative L2, worst       : {max(l2_32):.3e}")
                if worst32 <= 0.0:
                    print("The weights are bit-identical at 32 bits: they were never float64.")
                else:
                    rho32 = 1.0 / math.sqrt(2.0 * worst32)
                    weight_probe["indistinguishable_below_snr"] = rho32
                    print(f"-> indistinguishable below SNR {rho32:,.0f}")
                    if rho32 > 200:
                        print("   The NETWORK does not need float64. What float32 broke is the")
                        print("   solver's dynamic range, which is a fixable scaling problem, so")
                        print("   a Metal port is not blocked on accuracy.")
                    else:
                        print("   The network itself loses too much at 32 bits, so a Metal port")
                        print("   would degrade the waveform even after the solvers are fixed.")
            except Exception as exc:
                print(f"\n  [weight-precision probe failed: {type(exc).__name__}: {exc}]")

    ref_path = f"{args.surrogate_prefix}_ref_{prec}.npz"
    np.savez(ref_path, h=h_here, params=p_check)
    print(f"\nSaved {len(h_here)} reference waveforms -> {ref_path}")

    other = "f32" if x64 else "f64"
    other_path = f"{args.surrogate_prefix}_ref_{other}.npz"
    comparison = None
    if not os.path.exists(other_path):
        print(f"  Run again with --surrogate-precision {other} to get the mismatch.")
    else:
        d = np.load(other_path)
        h_there = d["h"]
        if h_there.shape != h_here.shape:
            print(f"  [{other_path} holds {h_there.shape} waveforms, this run has "
                  f"{h_here.shape}: not comparable. Delete it and rerun both.]")
        elif not np.allclose(d["params"], p_check):
            print(f"  [{other_path} was generated at different parameter points: "
                  "not comparable. Delete it and rerun both.]")
        else:
            mis, l2s, amaxs = [], [], []
            for a, b in zip(h_here, h_there):
                mis.append(1.0 - _white_match(a, b))
                nrm = float(np.linalg.norm(a))
                l2s.append(float(np.linalg.norm(a - b) / nrm) if nrm else float("nan"))
                amaxs.append(float(np.max(np.abs(a - b)) / np.max(np.abs(a))))
            worst = max(mis)
            comparison = dict(against=other_path, n_points=len(mis),
                              mismatch_worst=worst, mismatch_median=float(np.median(mis)),
                              rel_l2_worst=max(l2s), rel_max_abs_worst=max(amaxs),
                              mismatch=mis)
            print()
            print("=" * 78)
            print(f"PRECISION CROSS-CHECK  float{'64' if x64 else '32'} vs {other}, "
                  f"{len(mis)} parameter points")
            print("=" * 78)
            print(f"mismatch, worst / median : {worst:.3e} / {np.median(mis):.3e}")
            print(f"relative L2, worst       : {max(l2s):.3e}")
            print(f"relative max pointwise   : {max(amaxs):.3e}")
            print("White-noise match, maximised over time and phase. A PSD-weighted")
            print("match would differ, but not by orders of magnitude.")
            if worst <= 0.0:
                print("The two runs agree to double precision -- nothing to choose between.")
            else:
                rho = 1.0 / math.sqrt(2.0 * worst)
                comparison["indistinguishable_below_snr"] = rho
                print(f"-> indistinguishable below SNR {rho:,.0f}  "
                      "(criterion: mismatch < 1/(2 rho^2))")
                if rho > 200:
                    print("   Loud O4/O5 BBH reach SNR 30-100, so fp32 costs nothing here")
                    print("   and the jax-mps / Metal path stays open on accuracy grounds.")
                elif rho > 50:
                    print("   That covers typical events but not the loudest. Usable with")
                    print("   care; check again against the events you actually target.")
                else:
                    print("   Too tight: fp32 would bias PE on ordinary events, so float64")
                    print("   is required and jax-mps is ruled out for this model.")

    payload = dict(host=hostinfo(), precision=prec, jax_version=jax.__version__,
                   devices=[str(d) for d in jax.local_devices()],
                   xla_threads=args.surrogate_threads, wrapping=wrap,
                   weights_h5=h5, weights_h5_mb=h5_mb,
                   n_parameters=n_par, weight_bytes=w_bytes, weight_dtypes=dts,
                   build_s=t_build, n_time_samples=n_t, output_bytes=out_bytes,
                   m_total_msun=args.surrogate_mtotal, dist_mpc=args.surrogate_dist,
                   rounds=args.surrogate_rounds, rows=rows,
                   n_check_points=int(len(p_check)),
                   streaming_floor_s=t_floor, cpu_bw_gbs=args.cpu_bw,
                   plateau_batch=plateau["batch"], plateau_s_per_wf=plateau["per_wf_s"],
                   twf_imrphenomd_ms=args.twf_ms, pe_evals=args.pe_evals,
                   pe_rows=pe_rows, precision_comparison=comparison,
                   weight_dtype_breakdown={k: dict(n=v[0], bytes=v[1])
                                           for k, v in by_dtype.items()},
                   weight_precision_probe=weight_probe)
    out = f"{args.surrogate_prefix}_results_{prec}.json"
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--array_size", type=int, default=2 ** 21)
    p.add_argument("--list-size", type=int, default=200)
    p.add_argument("--n-unique", type=int, default=20)
    p.add_argument("--seglen", type=float, default=512.0, help="segment length, s -> df = 1/seglen")
    p.add_argument("--f-lower", type=float, default=20.0)
    p.add_argument("--f-final", type=float, default=1024.0)
    p.add_argument("--approximant", type=str, default="IMRPhenomD")
    p.add_argument("--n-wf", type=int, default=20)
    p.add_argument("--host-bw", type=float, default=100.0, help="this machine's peak GB/s (M2 = 100)")
    p.add_argument("--core-speedup", type=float, default=1.6, help="M5 big core vs this machine's P-core")
    p.add_argument("--skip-cpu", action="store_true",
                   help="skip the CPU baseline (the sizing table's single-thread filter, "
                        "and --decompose's CPU ceilings)")
    p.add_argument("--output", type=str, default="hardware_sizing_results.json")
    # --decompose mode
    p.add_argument("--decompose", action="store_true",
                   help="measure the per-kernel byte traffic instead of running the sizing "
                        "table; needs no GW packages, takes ~1 min")
    p.add_argument("--rounds", type=int, default=5, help="timed rounds per kernel (best wins)")
    p.add_argument("--sweep", type=int, nargs="*", default=[18, 19, 20, 21, 22],
                   help="log2 sizes for the scaling test (empty to skip)")
    p.add_argument("--peak-flops", type=float, default=3578.0,
                   help="fallback fp32 GFLOP/s if the peak is not measured "
                        "(M2 derived: 1280 ALU x 2 x 1.398 GHz = 3578)")
    p.add_argument("--skip-peak", action="store_true",
                   help="skip the measured compute ceiling and use --peak-flops")
    p.add_argument("--peak-n", type=int, default=2 ** 18,
                   help="elements in the FMA-chain arrays; keep it cache-resident")
    p.add_argument("--peak-depths", type=int, nargs="*",
                   default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
                   help="FMA-chain depths to sweep; the plateau is the compute roof")
    p.add_argument("--units", type=float, default=11.5,
                   help="array-sized memory passes per matched filter; --decompose "
                        "measures this (M2: ~11.5, bounded 9-14)")
    p.add_argument("--core-ghz", type=float, default=3.49,
                   help="P-core clock in GHz for the derived single-core NEON peak "
                        "(M2 Avalanche = 3.49)")
    p.add_argument("--matmul-n", type=int, default=2048,
                   help="square fp32 matmul size for the second ceiling")
    p.add_argument("--decompose-output", type=str, default="decompose_results.json")
    # --surrogate mode
    p.add_argument("--surrogate", action="store_true",
                   help="time the jax-nrsur7dq4-nn neural-network surrogate (the PE "
                        "waveform) instead of running the sizing table")
    p.add_argument("--surrogate-batches", type=int, nargs="*",
                   default=[1, 4, 16, 64, 256],
                   help="vmap batch sizes to sweep")
    p.add_argument("--surrogate-precision", choices=["f64", "f32"], default="f64",
                   help="f64 matches the package README; f32 is what jax-mps would "
                        "have to use. Run both -- the second run prints the mismatch.")
    p.add_argument("--surrogate-rounds", type=int, default=5,
                   help="timed calls per batch size (best wins)")
    p.add_argument("--surrogate-threads", type=int, default=1,
                   help="XLA CPU threads: 1 = single core (comparable to T_wf), "
                        "0 = XLA default (all cores)")
    p.add_argument("--surrogate-platform", type=str, default=None,
                   help="force a JAX platform, e.g. cpu or metal (sets JAX_PLATFORMS)")
    p.add_argument("--surrogate-mtotal", type=float, default=60.0,
                   help="total mass in Msun for the model constructor")
    p.add_argument("--surrogate-dist", type=float, default=400.0,
                   help="distance in Mpc for the model constructor")
    p.add_argument("--surrogate-max-gb", type=float, default=8.0,
                   help="skip a batch size whose output would exceed this many GB")
    p.add_argument("--surrogate-prefix", type=str, default="surrogate",
                   help="prefix for the saved reference waveform and results JSON")
    p.add_argument("--surrogate-weight-f32", action="store_true",
                   help="round the model's weights through 32-bit while keeping the "
                        "arithmetic in float64, and report the mismatch -- isolates "
                        "what the NETWORK needs from what the solvers need")
    p.add_argument("--surrogate-n-check", type=int, default=8,
                   help="extra parameter points (beyond the README's) used for the "
                        "f64-vs-f32 mismatch check")
    p.add_argument("--cpu-bw", type=float, default=78.0,
                   help="CPU streaming bandwidth, GB/s, for the weight-streaming floor "
                        "(M2 published STREAM = 78)")
    p.add_argument("--twf-ms", type=float, default=20.339,
                   help="measured IMRPhenomD generation time, ms/template, to compare "
                        "the surrogate against")
    p.add_argument("--pe-evals", type=float, default=1e7,
                   help="likelihood evaluations in the PE throughput table")
    args = p.parse_args()

    n = args.array_size
    df = 1.0 / args.seglen

    hi = hostinfo()
    print("=" * 78)
    print("HARDWARE SIZING BENCHMARK")
    print("=" * 78)
    print(f"host          : {hi['chip']}  |  {hi['ncpu']} cores ({hi['nperf']} perf)  |  {hi['ram_gb']} GB")
    print(f"filter length : 2^{int(np.log2(n))} = {n}  ({n * 8 / 1e6:.1f} MB per complex64 array)")
    if not (args.decompose or args.surrogate):
        print(f"waveforms     : df = 1/{args.seglen:g} s = {df:g} Hz, "
              f"f in [{args.f_lower:g}, {args.f_final:g}] Hz")
    print()

    if args.surrogate:
        run_surrogate(args)
        return

    import mlx.core as mx
    from mlx.core.fft import ifft
    print(f"mlx {getattr(mx, '__version__', '?')}   numpy {np.__version__}")
    print()

    if args.decompose:
        run_decompose(mx, ifft, args)
        return

    print("[1/3] MLX GPU matched filter (batched)...")
    t_gpu = time_filter(mx, ifft, mx.gpu, n, args.list_size, args.n_unique)
    # Byte traffic per filter. 9 units was the original ESTIMATE; --decompose
    # measured ~11.5 on an M2 (bounded 9-14). Override with --units.
    bytes_per_op = args.units * n * 8
    bw = bytes_per_op / t_gpu / 1e9
    print(f"      t_filter   = {t_gpu * 1e3:8.3f} ms/op")
    print(f"      throughput = {1 / t_gpu:8.0f} filters/s")
    print(f"      effective  = {bw:8.1f} GB/s  ({100 * bw / args.host_bw:.0f}% of {args.host_bw:g} GB/s peak, assuming {args.units:g} array passes)")
    print()

    t_cpu = None
    if not args.skip_cpu:
        print("[2/3] MLX CPU matched filter, single thread, complex64...")
        t_cpu = time_filter(mx, ifft, mx.cpu, n, max(20, args.list_size // 10), args.n_unique)
        print(f"      t_cpu_1    = {t_cpu * 1e3:8.3f} ms/op   -> GPU is {t_cpu / t_gpu:.2f}x one core")
        print()

    print("[3/3] Waveform generation, single core...")
    t_wf, backend, npts = time_waveform(df, args.f_lower, args.f_final, args.approximant, args.n_wf)
    print(f"      backend    = {backend}")
    print(f"      npts       = {npts}")
    print(f"      T_wf       = {t_wf * 1e3:8.3f} ms per template (one core)")
    print()

    # ---------------- decision table ----------------
    print("=" * 78)
    print("MINIMUM n_seg FOR EACH MACHINE TO BE GPU-BOUND")
    print("=" * 78)
    print("n_seg = number of data segments each generated template is filtered against.")
    print(f"Assumes M5 big core is {args.core_speedup:g}x this machine's core, and that the")
    print("filter time scales inversely with peak memory bandwidth.")
    print()
    print(f"{'machine':30s} {'$edu':>6s} {'GB/s':>5s} {'cores':>5s} "
          f"{'t_filt/ms':>9s} {'min n_seg':>10s} {'units':>6s} {'agg GB/s':>9s}")
    print("-" * 92)

    rows = []
    for name, price, bw_t, cores in CANDIDATES:
        t_filt = t_gpu * args.host_bw / bw_t
        min_nseg = t_wf / (args.core_speedup * cores * t_filt)
        units = int(BUDGET // price)
        rows.append(dict(machine=name, price=price, peak_bw=bw_t, cores=cores,
                         t_filter_s=t_filt, min_nseg=min_nseg,
                         units=units, agg_bw=units * bw_t, agg_cores=units * cores))
        print(f"{name:30s} {price:6d} {bw_t:5d} {cores:5d} "
              f"{t_filt * 1e3:9.3f} {min_nseg:10.1f} {units:6d} {units * bw_t:9d}")

    print()
    print("HOW TO READ THIS")
    print("  Your n_seg ABOVE a row's 'min n_seg'  -> that machine is GPU-bound.")
    print("  Your n_seg BELOW it                   -> that machine's GPU starves; buy cores.")
    print()
    print("  If every row is comfortably GPU-bound for your n_seg, buy peak bandwidth:")
    print("     5 x Studio M5 Max 18c/40c 48GB = $14,195 -> 3,070 GB/s aggregate")
    print("  If the Studio/Ultra rows starve but the mini rows do not, buy the minis:")
    print("     9 x mini M5 Pro 15c/16c 24GB   = $14,291 -> 2,763 GB/s, 135 cores")
    print("  If even the mini rows starve, buy the most cores you can:")
    print("     8 x mini M5 Pro 18c/20c 24GB   = $14,232 -> 2,456 GB/s, 144 cores")

    payload = dict(host=hi, array_size=n, seglen=args.seglen, df=df,
                   f_lower=args.f_lower, f_final=args.f_final,
                   approximant=args.approximant, waveform_backend=backend,
                   waveform_npts=npts,
                   t_filter_gpu_s=t_gpu, t_filter_cpu1_s=t_cpu, t_waveform_s=t_wf,
                   effective_bw_gbs=bw, host_peak_bw_gbs=args.host_bw,
                   core_speedup=args.core_speedup, candidates=rows)
    with open(args.output, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
