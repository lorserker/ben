"""Standalone SuitC worker process.

Runs a single SuitC `call_suitc` in an isolated child process so the parent can
enforce a timeout and SIGKILL a runaway native call (e.g. the merge_quirk /
check_quirk loop) without freezing the gevent server. See SuitC.py and the
project memory note `suitc-hang-freezes-gameapi`.

Protocol:
  - stdin  : the SuitC input string (UTF-8), read in full.
  - argv[1]: absolute path to the SuitC shared library.
  - stdout : a single JSON object:
                {"rc": <int>, "output": <str>, "details": <str>}
             or, on failure:
                {"rc": -1, "error": <str>}

The native library prints its analysis tree to fd 1 regardless of flags; that
flood is redirected to devnull around the call so stdout carries only the JSON.
The JSON result comes from the output buffer, not stdout, so this is safe.

This file is intentionally self-contained (no imports from the rest of BEN) so
that launching it is cheap and free of side effects.
"""
import sys
import os
import json
import ctypes
from ctypes import c_wchar_p, c_int, POINTER

# Must match the buffer capacity used by SuitC.py.
BUFFER_CAPACITY = 8 * 32768


def main():
    if len(sys.argv) < 2:
        sys.stdout.write(json.dumps({"rc": -1, "error": "missing library path argument"}))
        return
    lib_path = sys.argv[1]
    input_str = sys.stdin.buffer.read().decode("utf-8")

    try:
        suitc = ctypes.CDLL(lib_path)
        suitc.call_suitc.argtypes = [POINTER(c_wchar_p), c_int,
                                     POINTER(c_wchar_p), POINTER(c_int),
                                     POINTER(c_wchar_p), POINTER(c_int)]
        suitc.call_suitc.restype = c_int

        input_length = len(input_str)
        c_input_str_ptr = c_wchar_p(input_str)
        pp_input_str = ctypes.byref(c_input_str_ptr)

        output_buffer = ctypes.create_unicode_buffer(BUFFER_CAPACITY)
        c_output_ptr = c_wchar_p(ctypes.addressof(output_buffer))
        pp_output = ctypes.byref(c_output_ptr)
        output_size = c_int()

        details_buffer = ctypes.create_unicode_buffer(BUFFER_CAPACITY)
        c_details_ptr = c_wchar_p(ctypes.addressof(details_buffer))
        pp_details = ctypes.byref(c_details_ptr)
        details_size = c_int()

        # Silence the native lib's fd-1 flood around the call.
        sys.stdout.flush()
        saved_fd = os.dup(1)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 1)
            rc = suitc.call_suitc(
                pp_input_str,
                input_length,
                pp_output,
                ctypes.byref(output_size),
                pp_details,
                ctypes.byref(details_size),
            )
        finally:
            os.dup2(saved_fd, 1)
            os.close(devnull_fd)
            os.close(saved_fd)

        result = {
            "rc": int(rc),
            "output": output_buffer.value[:output_size.value],
            "details": details_buffer.value[:details_size.value],
        }
    except Exception as ex:  # report cleanly to the parent rather than crashing
        result = {"rc": -1, "error": f"{type(ex).__name__}: {ex}"}

    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
