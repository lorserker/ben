A faster double dummy solver for BEN and for PIMC, faster dealing and
bidding from Randal Burns' performance work, and ONNX bidding models brought
up to date.

## bridge-solver as the double dummy solver

- **BEN now uses bridge-solver** for double dummy analysis on Windows x64,
  Linux x64, Linux arm64 and macOS arm64 (`bin/bsolver`). It is called
  through the DDS-compatible C API (`src/ddsolver/bsolver.py`).
  DDS 3.1.0 (`dds3`) is still there, and is used where bridge-solver is not
  built (Windows arm64, Intel Macs).
- **About twice as fast per core.** On a recording of every double dummy call
  from a full set of boards, one thread took 78 s with bridge-solver and
  162 s with DDS 3.1.0. With 32 threads, DDS 3.1.0 used 203 CPU-seconds and
  bridge-solver 77-85.
- **The same results.** Every recorded call, par scores included, gave the
  same tricks with both solvers, checked on each platform's build.
  Where several cards are equal, bridge-solver can list them in another
  order, so BEN may play a different spot card of the same value.
- **PIMC uses it too.** The `dds` library next to BGADLL is now
  bridge-solver, and PIMC uses it by default (`BGA_DDS_BACKEND=haglund`),
  also on Windows x64, where BGADLL preferred bcalcdds: on 8 boards PIMC took
  15.3 s with bridge-solver against 27.9 s with bcalcdds, with the same cards
  played. `BGA_DDS_BACKEND=bcalcdds` brings bcalcdds back.
- To choose the solver yourself, set `BEN_DDS=dds3` or `BEN_DDS=bsolver`;
  `BEN_BSOLVER_LIB` points at another bridge-solver library.
  `preflight.py` shows which solver is in use.
  Build instructions are in `src/ddsolver/README.md`.

## Faster dealing, claims and bidding

From Randal Burns' performance work
([randalburns/ben](https://github.com/randalburns/ben), PR #202):

- **Dealing** for the samples runs in compiled numba loops
  (`src/sample_jit.py`), and BEN falls back to the Python version if numba
  is missing.
- **Claim checks and rollouts** build their positions with less repeated
  work.
- **Bidding** repeats less work: network scores are cached per board, each
  distinct hand is scored once, double dummy results are reused for each
  bid, networks are loaded once, and the auction encoding and the equal-card
  scan are faster.
- **`BEN_HEADLESS=1`** skips the pauses `game.py` makes for redrawing.

## PIMC

- **The 50 ms wait after each PIMC evaluation is gone.** BGADLL 0.9.9.0 and
  later already wait for every worker thread before returning.

## ONNX

- **`bidder.onnx` and `bidinfo.onnx`** are converted from the GIB-BBO 8744
  models that `default.conf` uses (they were from 8730). On 40 test deals
  `onnx.conf` bid exactly as `default.conf`.
- **`onnx.conf`** is now `default.conf` with the ONNX models. Before it
  stopped PIMC after trick 13, not 8, and had no `[ace]` settings and no
  `play_nn_only`.
- Keras is still the default: on the i9 the ONNX bidding networks were
  slower than Keras.

## Packaging

- The Windows packages bundle bridge-solver for BEN and for PIMC.
- numba is left out of the frozen packages, which use the Python dealing
  loops.
- `Dockerfile.play` copies `sample_jit.py` and the Linux bridge-solver
  libraries.

## Requirements

- numba 0.60 or later, new in `requirements.txt` (optional: without it the
  dealing loops run in Python).
- TensorFlow 2.21 and Keras 3.15, as in `requirements.txt`.
- On Linux, glibc 2.34 or later (Ubuntu 22.04, Debian 12, RHEL 9 or newer).
