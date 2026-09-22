# mac-gpu-benchmark

Measures the memory bandwidth and arithmetic throughput of the GPU in an Apple
Silicon Mac.

We are sizing a small compute cluster for gravitational-wave data analysis at
Syracuse University. The calculation at the centre of it — a matched filter
against a bank of waveform templates — turns out to be limited by how fast the
chip can move data in and out of memory, not by how fast it can do arithmetic.
So memory bandwidth per dollar decides which Mac to buy. Apple publishes a
rated bandwidth figure for each chip; these scripts measure what is actually
achieved, which is the number the decision needs.

**If you are here because someone asked to run this on a machine you are
responsible for, read [STORE_VISIT.md](STORE_VISIT.md).** It is one page and
says exactly what the script touches.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python bandwidth_check.py
```

Takes about a minute. Everything it produces goes to the screen; it writes no
files. Deleting the folder removes every trace, including the virtual
environment.

To compare against the chip's advertised figure, pass it in:

```bash
python bandwidth_check.py --rated-bw 614     # M5 Max, 40-core GPU
python bandwidth_check.py --quick            # ~20 s
```

## The two scripts

**`bandwidth_check.py`** — self-contained, about 200 lines, needs only `mlx`
and `numpy`. Measures streaming memory bandwidth, fp32 matmul throughput, and
how bandwidth varies with array size (which reveals the cache sizes). This is
the one to run on a machine you are just visiting.

**`hardware_sizing_benchmark.py`** — the full instrument used for the cluster
study. Same bandwidth measurement plus per-kernel memory-traffic decomposition,
a CPU baseline, waveform-generation timing, and a neural-network surrogate
benchmark. Some modes need PyCBC or JAX installed, so it is not a good fit for
a borrowed machine. `--decompose` needs nothing beyond `mlx` and `numpy`:

```bash
python hardware_sizing_benchmark.py --decompose --host-bw 150
python hardware_sizing_benchmark.py --help
```

## Reading the numbers

**Bandwidth.** `a + b` over complex64 reads two arrays and writes one: exactly
three passes over the data, with one trivial arithmetic operation per element.
Bytes moved divided by elapsed time is the machine's streaming bandwidth. On an
M2 Mac mini this gives 89.4 GB/s against a rated 100, which agrees to within 2%
with the published Metal STREAM figure for that machine ([Hübner et al.,
arXiv:2502.05317](https://arxiv.org/abs/2502.05317)).

**Balance point.** Dividing the matmul throughput by the bandwidth gives the
arithmetic intensity at which a workload stops being limited by memory and
starts being limited by arithmetic — 28 FLOP per byte on the M2. Our matched
filter runs at 1.24 FLOP per byte, a factor of 22 below it, which is the whole
reason bandwidth is the figure of merit.

**The size sweep.** Small arrays fit in the chip's caches and read far faster
than main memory. The size at which the rate collapses is the cache capacity.
On an M2 the drop appears between 2^19 and 2^20 complex64 elements, consistent
with its 8 MB system-level cache.

## Results so far

| Machine | Rated | Measured | Achieved | fp32 matmul |
|---|---|---|---|---|
| M2 Mac mini, 24 GB | 100 GB/s | 89.4 GB/s | 89% | 2,476 GFLOP/s |

The open question is whether the larger chips hold that ~89% ratio. If they do,
the aggregate bandwidth of a candidate fleet can be predicted from Apple's
published figures. If they do not, the predictions are optimistic and the
purchase plan needs a correction factor. That is what the store measurement is
for.

## Requirements

Python 3.9 or newer, `numpy`, and `mlx` 0.32.1 or newer. **The version floor is
not cosmetic**: earlier MLX returned silently incorrect FFT results for
transforms above 2^20 elements, with no error raised.

## Licence

MIT. See [LICENSE](LICENSE).
