# What this script does, for whoever is watching

I am an astrophysics professor at Syracuse University. I am deciding which Mac
to buy with a research grant, and I need to measure how fast a particular model
actually moves data in and out of memory. Apple publishes a rated figure; I need
the achieved one. This takes about a minute per machine.

Thank you for letting me run it. Here is exactly what happens.

## What the script does

It creates some arrays of random numbers in memory, adds them together on the
GPU, and times how long that takes. Dividing the number of bytes by the elapsed
time gives the memory bandwidth. It then multiplies two matrices to measure
arithmetic speed, and repeats the first test at several array sizes.

That is the whole program. It is about 200 lines of Python and you are welcome
to read it — it is the file `bandwidth_check.py`, and every function has a
comment saying what it is for.

## What it does not do

- **No network access.** The benchmark makes no connections of any kind.
- **No files written.** Every result is printed to the screen. The script
  creates no documents, caches, preferences or logs.
- **No personal data.** It reads nothing from the machine except the chip name,
  the core count and the amount of RAM, which it prints in its header. It runs
  fine without even that.
- **No permanent changes.** Nothing is installed outside a single folder, and
  deleting that folder removes everything.
- **No administrator rights needed.** If it asks for a password, something is
  wrong — stop it.

## What gets put on the machine, and how to remove it

Two things, both inside one folder in `/tmp`, which macOS clears on its own:

1. The script itself, copied from a public repository.
2. A Python virtual environment containing the `mlx` and `numpy` libraries.
   `mlx` is Apple's own open-source array library, published by Apple at
   <https://github.com/ml-explore/mlx>. It is downloaded from PyPI, the standard
   Python package index. This download is the only network activity involved,
   and it happens during setup rather than during the measurement.

To remove all of it:

```bash
rm -rf /tmp/mac-gpu-benchmark
```

## The commands, in order

```bash
cd /tmp
git clone https://github.com/USER/mac-gpu-benchmark.git
cd mac-gpu-benchmark
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python bandwidth_check.py --rated-bw 614
```

The last line is the measurement. Everything before it is setup. I will
photograph the output and then delete the folder.

## One thing I may need to ask about

This needs a working `python3`. On a Mac that has never had developer tools
installed, typing `python3` pops up a dialog offering to install Apple's Command
Line Tools, which is a large download and a change to the machine. **I do not
want to do that without your say-so.** If these machines already have Xcode or
the Command Line Tools, everything above works as written. If not, please just
tell me and I will find another way — I would rather leave without the
measurement than install something you have not agreed to.

## Who to contact

Collin Capano, Department of Physics, Syracuse University — cdcapano@syr.edu
