"""Pre-flight checks for BEN's native dependencies.

Run this before starting a server to get a one-line diagnosis instead of a
chained traceback:

    python preflight.py                          # generic checks
    python preflight.py --config config/GIB-BBO.conf   # also checks what that config needs

Exit code 0 means the server should start. 1 means it will not, or will crash
partway through a deal.

Deliberately dependency-light: no TensorFlow, no model loading. It imports only
what is needed to prove the native libraries resolve, so it stays useful in the
broken environments it is meant to diagnose.
"""
import argparse
import ctypes
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

OK, FAIL, WARN, INFO = "[ ok ]", "[FAIL]", "[warn]", "[info]"

_problems = []


def report(status, title, *details):
    print(f"{status} {title}")
    for line in details:
        print(f"       {line}")
    if status == FAIL:
        _problems.append(title)


def check_python():
    """Report the interpreter. Not a pass/fail on its own - but the single most
    common cause of the failures below is having started the wrong one."""
    report(INFO, f"Python {sys.version.split()[0]}", sys.executable)

    if sys.platform == "darwin" and "/Library/Frameworks/Python.framework" in sys.executable:
        report(
            WARN,
            "This is a python.org framework build",
            "It runs under the hardened runtime, which restricts how dyld resolves",
            "libraries loaded via dlopen. BEN's PIMC backend is affected (see below).",
            "Prefer the repo venv: ../.venv/bin/python",
        )


def check_dds3():
    """The dds3 extension is a compiled CPython extension, locked to one Python
    version. This is the check that catches 'ran it with the wrong python3'."""
    try:
        from ddsolver.ddsolver import DDSolver
    except ImportError as ex:
        report(
            FAIL,
            "dds3 (double dummy solver) will not import",
            f"Running: {sys.executable} ({sys.version.split()[0]})",
            "A compiled extension only loads under the Python version it was built for.",
            "Either start BEN with the matching interpreter (the repo venv is the",
            "intended one) or rebuild the extension - see src/ddsolver/README.md.",
            "",
            f"Underlying error: {ex}".split("\n")[0],
        )
        return None

    solver = DDSolver(max_threads=1)
    report(OK, f"dds3 {solver.version()}")
    return solver


def check_pimc_dds(required):
    """BGADLL resolves its DDS backend through a P/Invoke on 'dds'. When that
    fails the server still starts and bids, then aborts partway through the
    first trick - so catching it here is worth a lot."""
    try:
        import pimc.BGADLL_Native as native
    except ImportError:
        report(INFO, "PIMC/BGADLL not available for this platform - skipping")
        return

    try:
        lib = native._get_lib()
    except Exception as ex:
        report(FAIL if required else WARN, "BGADLL failed to load", str(ex))
        return

    if not hasattr(lib, "bga_dds_backend"):
        report(WARN, "BGADLL has no bga_dds_backend export - cannot verify its DDS backend")
        return

    lib.bga_dds_backend.argtypes = []
    lib.bga_dds_backend.restype = ctypes.c_void_p
    backend = ctypes.string_at(lib.bga_dds_backend()).decode("utf-8", "replace")

    if not backend.startswith("error"):
        report(OK, f"PIMC DDS backend: {backend}")
        return

    details = [
        "BGADLL loaded, but its DDS backend did not. The server would start and",
        "bid normally, then abort during the first trick PIMC is asked to play.",
        "",
        "This depends on which Python started the process, not on the platform:",
        "a hardened-runtime build (python.org installer) cannot resolve the",
        "preloaded library, while Homebrew/conda builds can.",
        "",
        "Fix: start BEN with the repo venv interpreter (../.venv/bin/python).",
    ]
    if sys.platform == "darwin":
        lib_dir = os.path.join(os.path.dirname(HERE), "bin", "BGA", "macos",
                               "arm64" if os.uname().machine == "arm64" else "x64")
        details += [
            "Workaround if you must use this interpreter:",
            f'  export DYLD_LIBRARY_PATH="{lib_dir}"',
            "  (must be set before the process starts; setting it in Python is too late)",
        ]
    report(FAIL if required else WARN, "PIMC DDS backend failed to load", *details)


def pimc_required(config_path):
    """True if the given config actually turns PIMC on. Without a config we
    cannot know, so the PIMC check downgrades to a warning."""
    if not config_path:
        return False
    try:
        import conf
        configuration = conf.load(config_path)
        return (configuration.getboolean('pimc', 'pimc_use_declaring', fallback=False)
                or configuration.getboolean('pimc', 'pimc_use_defending', fallback=False))
    except Exception as ex:
        report(WARN, f"Could not read {config_path}", str(ex))
        return False


def main():
    parser = argparse.ArgumentParser(description="Check BEN's native dependencies before starting a server")
    parser.add_argument("--config", default=None,
                        help="Config the server will use. Makes the PIMC check fatal when that config enables PIMC.")
    args = parser.parse_args()

    print("BEN pre-flight")
    print("-" * 60)

    check_python()
    check_dds3()
    check_pimc_dds(required=pimc_required(args.config))

    print("-" * 60)
    if _problems:
        print(f"{FAIL} {len(_problems)} problem(s) - not starting.")
        return 1
    print(f"{OK} All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
