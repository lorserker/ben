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

The native library prints its analysis tree to fd 1 regardless of flags. fd 1 is
permanently redirected to devnull for the whole worker lifetime and the JSON is
written to a saved copy of the real stdout, so the native flood (including the
C stdio buffer flushed at process exit) can never corrupt the JSON. The JSON
result comes from the output buffer, not stdout, so this is safe.

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
        os.write(1, json.dumps({"rc": -1, "error": "missing library path argument"}).encode("utf-8"))
        return
    lib_path = sys.argv[1]
    input_str = sys.stdin.buffer.read().decode("utf-8")

    # Pin fd 1 to devnull for the *entire* worker lifetime, then write our JSON
    # result to a saved copy of the real stdout. The native lib prints its
    # analysis tree to fd 1 using C stdio buffering: redirecting only around the
    # call and then restoring fd 1 would let the residual buffer (and the
    # exit-time flush) spill onto the real stdout *after* our JSON, producing
    # "<valid JSON><native fragment>" that the parent cannot json.loads. Keeping
    # fd 1 on devnull for good — including process exit — guarantees the saved
    # fd carries only our JSON.
    sys.stdout.flush()
    real_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    os.close(devnull_fd)

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

        # fd 1 is already pinned to devnull (see above), so the native lib's
        # tree flood — during the call and at exit — is discarded.
        rc = suitc.call_suitc(
            pp_input_str,
            input_length,
            pp_output,
            ctypes.byref(output_size),
            pp_details,
            ctypes.byref(details_size),
        )

        result = {
            "rc": int(rc),
            "output": output_buffer.value[:output_size.value],
            "details": details_buffer.value[:details_size.value],
        }
    except Exception as ex:  # report cleanly to the parent rather than crashing
        result = {"rc": -1, "error": f"{type(ex).__name__}: {ex}"}

    # Write to the saved real stdout, NOT sys.stdout (which now points at
    # devnull). os.write is an unbuffered syscall, so there is nothing to flush.
    os.write(real_stdout_fd, json.dumps(result).encode("utf-8"))
    os.close(real_stdout_fd)


if __name__ == "__main__":
    main()
