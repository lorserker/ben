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
> preloads them. Do not delete them unless BGADLL is rebuilt against DDS 3.0.0.

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
> and Ubuntu 24.04+, but a native install on an older distro needs a rebuild. For
> macOS, set a low `MACOSX_DEPLOYMENT_TARGET` for the same reason.

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
