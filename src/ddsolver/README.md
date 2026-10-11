# DDS solver (`ddsolver`)

BEN's double-dummy solver wrapper. As of the DDS **3.0.0** upgrade this uses the
`dds3` Python extension instead of the old `ctypes`-loaded `dds.dll` / `libdds.so`.

## What changed

| Before (DDS 2.9.x)                         | Now (DDS 3.0.0)                             |
|--------------------------------------------|---------------------------------------------|
| `ctypes.cdll.LoadLibrary("dds.dll")`       | `import dds3` (compiled pybind11 extension)  |
| `dds.py`, `ddss.py` ctypes wrappers        | removed                                      |
| `DDSSolver` (ddss fast-fork) + fallback    | removed — single `DDSolver` class            |
| `bin/ddss.dll` / `.exp` / `.lib`           | removed (ddss fast-fork dropped)              |

`DDSolver` keeps the same public API (`solve`, `calculatepar`, `version`,
`expected_tricks_dds`, ...), so callers did not change.

> **Note:** `bin/dds.dll`, `bin/libdds.so` and `bin/darwin/libdds.2.9.0.dylib`
> are **kept** — they are not used by `ddsolver` anymore, but the BGADLL / PIMC
> native engine (`src/pimc/BGADLL_Native.py`) still P/Invokes the DDS C ABI and
> preloads them. See [The DDS C library used by PIMC](#the-dds-c-library-used-by-pimc-bgadll).

**Concurrency.** `DDSolver` runs a Python `ThreadPoolExecutor` and issues one
`dds3.solve_board_pbn(..., context=ctx)` per board, keeping **one `SolverContext`
per worker thread** — the modern DDS 3.x model (internal batch threading was
removed). `solve_board_pbn` releases the GIL during the search, so the pool runs
truly in parallel and each thread's context keeps a warm transposition table.
The deprecated `set_max_threads` / `solve_all_boards` bindings are **not** used
(and no longer exist in current `dds3`).

## Installing the `dds3` extension

`dds3` is a compiled extension and must be built per platform from the DDS
repository (https://github.com/dds-bridge/dds). It is **not** on PyPI.

### Build it

```bash
# The DDS repo uses bazelisk (the `bazel` launcher); `bazel` may not be on PATH.
bazelisk build -c opt //python:_dds3            # -> bazel-bin/python/_dds3.{so,pyd}
bazelisk build -c opt //python:dds3_wheel_dist  # -> a wheel under bazel-bin/python/dist/
```

> **Match the Python version.** A compiled extension is locked to one CPython
> version (it links e.g. `python312.dll`). BEN's `ben` conda env is **Python
> 3.12**, so build the extension for 3.12, not the DDS repo's default 3.14:
> ```bash
> bazelisk build -c opt \
>   --@rules_python//python/config_settings:python_version=3.12 //python:_dds3
> ```
> This requires the 3.12 toolchain to be registered in the DDS repo's
> `MODULE.bazel` (the default only registers 3.14). Verify the result with
> `grep -a python3 _dds3.pyd` — it must reference `python312.dll`. A 3.14-built
> extension fails to import under 3.12 with a confusing `ImportError`.

See `docs/python_interface.md` in the DDS repo for per-OS prerequisites
(Windows: MSVC; Linux: GCC/clang; macOS: bazelisk).

> **Linux: build against an old glibc for portability.** A `.so` records the
> highest glibc symbol version it uses and will not load on an older glibc
> (`version 'GLIBC_2.XX' not found`). Build the Linux extension on the **oldest**
> glibc you need to support — a [manylinux](https://github.com/pypa/manylinux)
> container (`manylinux_2_28` = glibc 2.28, ships CPython 3.12) is ideal: it runs
> on every current distro **and** on the Ubuntu 24.04 Docker image. Building on
> Ubuntu 24.04 (glibc 2.39) instead yields a `.so` that needs glibc ≥ 2.38, so it
> will **not** run on Ubuntu 22.04 / Debian 12. The vendored `bin/dds3-linux/`
> binary in this repo is currently a 24.04 build (glibc ≥ 2.38) — fine for Docker
> and Ubuntu 24.04+, but a native install on an older distro needs a rebuild.

> **macOS: always set a low deployment target.** Without it the extension is
> stamped with the build machine's macOS version and links libc++ symbols that
> older systems lack — the `_dds3.so` shipped before 0.8.8.8 had `minos 26.5` and
> failed on anything older with
> `Symbol not found: __ZNSt3__113__hash_memoryEPKvm`. Bazel is not needed; the
> extension is two binding files plus the library sources:
> ```bash
> # in a DDS checkout, with a Python 3.12 venv that has pybind11 installed
> mkdir -p inc && ln -sfn "$PWD/library/src" inc/dds     # for #include <dds/...>
> SDK=$(xcrun --show-sdk-path)      # see the note below about CLT 27
> MACOSX_DEPLOYMENT_TARGET=11.0 clang++ -isysroot "$SDK" -std=c++20 -O3 -flto=thin \
>   -fPIC -shared -undefined dynamic_lookup -mmacosx-version-min=11.0 -arch arm64 \
>   -Iinc -Ilibrary/src -Ilibrary/src/api -Ipython/src $(python -m pybind11 --includes) \
>   python/src/bindings.cpp python/src/converters.cpp \
>   $(find library/src -name '*.cpp' | grep -viE 'test|bench|_main|fuzz') \
>   -o _dds3.so
> otool -l _dds3.so | grep -A4 LC_BUILD_VERSION   # must say minos 11.0
> ```
> The macOS 27 Command Line Tools ship without libc++ headers (`'string' file not
> found`); point at the 26.5 SDK instead:
> `SDK=/Library/Developer/CommandLineTools/SDKs/MacOSX26.5.sdk` and add
> `-nostdinc++ -isystem "$SDK/usr/include/c++/v1"`.
>
> Build from the same DDS commit as the other platforms. DDS commit `dbffab90`
> (2026-08-17) renamed `initialise_static_memory` to `initialize_static_memory`,
> and the vendored `__init__.py` imports the old name. The current binaries are
> built from `671a4aa4`, the commit just before the rename.

### Make it importable — two options

`ddsolver.py` imports `dds3` and finds it either way:

1. **pip install (recommended)** — install the built wheel into BEN's Python
   environment:
   ```bash
   pip install dds3-1.0.0-*.whl
   ```

2. **Vendored in `bin/`** — copy the built `dds3` package into the
   platform-specific folder so it is shipped with BEN:
   ```
   bin/dds3-win/dds3/__init__.py     + _dds3.pyd     (Windows)
   bin/dds3-linux/dds3/__init__.py   + _dds3.so      (Linux)
   bin/dds3-darwin/dds3/__init__.py  + _dds3.so      (macOS)
   ```
   `ddsolver._load_dds3()` adds the right folder to `sys.path` automatically.

The Docker images expect option 2 (`bin/dds3-linux/`). PyInstaller builds
expect option 1 (`dds3` installed in the build environment; the `.spec` files
list it under `hiddenimports`).

## The DDS C library used by PIMC (BGADLL)

BGADLL (PIMC / PIMCDef) does not use `dds3`. On every platform except Windows
x64 (which uses `libbcalcdds.dll`) it calls the DDS **C ABI** (`SolveBoardPBN`)
from a native library next to it:

| Platform       | File                                   | DDS   |
|----------------|----------------------------------------|-------|
| Windows x64    | `bin/BGA/windows/x64/dds.dll` (fallback only) | 2.9.0 |
| Windows ARM64  | `bin/BGA/windows/arm64/dds.dll`        | 3.1.0 (`671a4aa4`) |
| Linux x64      | `bin/BGA/linux/x64/libdds.so` (+ identical `dds.dll`, `bin/libdds.so`) | 3.1.0 (`671a4aa4`) |
| Linux ARM64    | `bin/BGA/linux/arm64/libdds.so`        | 3.1.0 (`671a4aa4`) |
| macOS ARM64    | `bin/BGA/macos/arm64/libdds.2.9.0.dylib` | 2.9.0 |

> **The thread index must be accepted.** BGADLL spreads its solves over thread
> indexes 0–7. DDS builds from before commit `15eb56fd` (2026-05-22) only accept
> index 0 and return `-15` (`RETURN_THREAD_INDEX`) for the rest, which BGADLL
> scored as 0 tricks: from 0.8.8.0 to 0.8.8.8 PIMC on Linux x64 decided on about
> 1 in 8 of its samples. BGADLL 0.9.9.2+ prints a `BGADLL DDS warning` line on
> stderr the first time a solve fails, so check the log after changing any of
> these files.
>
> **Check the architecture.** Before 0.8.8.8 the Linux ARM64 and Windows ARM64
> folders held x86-64 copies, which cannot load next to an ARM64 BGADLL.

Build the Linux libraries in a manylinux container (glibc 2.28, static C++
runtime, so only libc/libm/libpthread are needed):

```bash
# from a DDS checkout; use quay.io/pypa/manylinux_2_28_aarch64 for ARM64
docker run --rm -v "$PWD:/dds:ro" -v "$PWD/out:/out" quay.io/pypa/manylinux_2_28_x86_64 bash -c '
  mkdir -p /inc && ln -s /dds/library/src /inc/dds
  g++ -std=c++20 -O3 -fPIC -shared -pthread -static-libstdc++ -static-libgcc \
      -Wl,-soname,libdds.so -I/inc -I/dds/library/src -I/dds/library/src/api \
      -o /out/libdds.so $(find /dds/library/src -name "*.cpp" | grep -viE "test|bench|_main|fuzz")
  strip --strip-unneeded /out/libdds.so'
```

The Windows ARM64 `dds.dll` is the same sources compiled with MSVC from an
`x64_arm64` developer prompt: `cl /std:c++20 /O2 /EHsc /MT /LD ... /link /MACHINE:ARM64`
(`/MT` so it needs no VC++ redistributable).

## Benchmarking: record a real session, replay it

`ddsrecorder.py` + `ddsreplay.py` turn a played session into a repeatable DDS
performance test. Every DDS call BEN makes is written to a JSON Lines file with
its **full input and output**; the replayer re-issues those calls, times them,
and checks the answers still match.

Benchmarking this way rather than on generated deals matters because DDS runtime
is dominated by *which* calls BEN makes — the trick-1 `solutions=1` batches for
bidding cost far more than the trick-11 ones for card play, and their mix is not
easy to guess. In a sample 16-deal session, trick 1 was **81 % of all DDS time**.

### Record

Recording is off unless you ask for it with `--ddsrecord`, accepted by
`game.py`, `gameserver.py`, `gameapi.py` and `table_manager_client.py`. The
value is a file or a directory (a directory gets one file per process, so
several BEN processes can record at once); a `.gz` suffix compresses.

```bash
python game.py --boards mysession.pbn --auto True --ddsrecord d:/tmp/dds-rec.jsonl
```

Where there is no command line to reach — Docker, frozen builds — the
`BEN_DDS_RECORD` environment variable does the same thing. `--ddsrecord` wins if
both are given.

The entry points arm the recorder right after parsing arguments, before any
`DDSolver` is constructed, so no call is missed. The header record is written
lazily with the first DDS call, which is how it can carry the DDS version, mode
and thread count that only become known later.

Calls are tagged with the board being played in `game.py` (so also
`gameserver.py`) and `table_manager_client.py`. `gameapi.py` is **not** tagged —
it serves requests concurrently under gevent, so a single "current board" would
mis-attribute interleaved requests. Its records still carry the complete input
and replay fine, just with `board: null`.

Records are flushed per line, so a session ended with Ctrl-C still yields a
usable file; the replayer skips a truncated final line.

Expect roughly 4 MB per 800 calls (200-sample batches dominate). Recording adds
the cost of serialising those batches, so **the recorded `ms` are not a clean
benchmark** — they are there for reference and appear in the report as
`rec ms/call`. The replay timings are the ones to compare.

### Replay

```bash
# straight benchmark, best of 3
python src/ddsolver/ddsreplay.py d:/tmp/dds-rec/dds-12345.jsonl --repeat 3

# what does the recording contain? (does not run DDS)
python src/ddsolver/ddsreplay.py rec.jsonl --list

# tune dds_max_threads for this machine
python src/ddsolver/ddsreplay.py rec.jsonl --threads 8 --threads 16 --threads 32

# is a DDS rebuild / a dds_mode change faster, and still correct?
python src/ddsolver/ddsreplay.py rec.jsonl --json new.json --baseline old.json
python src/ddsolver/ddsreplay.py rec.jsonl --dds-mode 2
```

The report breaks time down by `purpose` (`bid`, `lead`, `play`, `claimcheck`,
`claimtricks`, `contract`, `par`) and, with `--tricks`, by trick number. Other
useful filters: `--purpose`, `--board`, `--solutions`, `--min-trick`,
`--max-trick`, `--limit`, `--no-par`.

`--json` writes a machine-readable result that a later run can `--baseline`
against; the exit status is non-zero if any call returned something other than
what was recorded, so it can be used as a CI check that a DDS change is
behaviour-preserving.

Replay calls `solve_helper` / `_calculatepar_impl` rather than the public
`solve` / `calculatepar`, so it measures DDS itself and never re-triggers the
recorder.

**On repeatability:** with `dds_mode=1` DDS reuses a transposition table across
consecutive solves on the same `SolverContext`, and the thread pool assigns
boards to contexts nondeterministically, so a single run carries a few percent
of noise. Use `--repeat` (the best run is reported) and `--warmup`.
