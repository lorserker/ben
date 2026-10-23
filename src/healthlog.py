"""
Health log for the gevent API server: request timing, memory and hang reports.

Two problems cannot be diagnosed from the normal request log:

- A request that never finishes. A native call (PIMC/BGADLL, DDS, SuitC) blocks
  the gevent loop, so the stuck greenlet can log nothing and neither can any
  other request. The watchdog here runs on a real OS thread, so it keeps
  running while the loop is blocked.
- Memory that grows over days. Each request logs the process RSS and how much
  it changed, and a heartbeat logs RSS while the server is idle.

Everything goes to logs/gameapi-<host>-<port>-health-<date>.log, hang reports
also to stderr. The file is written with plain open()/write() and not with the
logging module: its locks are gevent locks once monkey patched, and the watchdog
thread must not touch them.

Settings (environment):
  BEN_HANG_SECONDS       report a request running longer than this (default 120)
  BEN_HANG_REPEAT        repeat the report while it is still stuck (default 60)
  BEN_HEARTBEAT_MINUTES  log memory this often, also when idle (default 10, 0 = off)
"""
import datetime
import faulthandler
import itertools
import os
import sys

import psutil

try:
    from gevent.monkey import get_original
    _start_thread = get_original('_thread', 'start_new_thread')
    _sleep = get_original('time', 'sleep')
    _time = get_original('time', 'time')
except ImportError:
    import _thread
    import time
    _start_thread = _thread.start_new_thread
    _sleep = time.sleep
    _time = time.time

HANG_SECONDS = float(os.getenv('BEN_HANG_SECONDS', '120'))
HANG_REPEAT = float(os.getenv('BEN_HANG_REPEAT', '60'))
HEARTBEAT_MINUTES = float(os.getenv('BEN_HEARTBEAT_MINUTES', '10'))
POLL_SECONDS = 5

_MB = 1024 ** 2
_process = psutil.Process()
_ids = itertools.count(1)
# id -> [start, method, url, rss_at_start, last_report]. Only the server's own
# thread adds and removes entries; the watchdog only reads a copy.
_in_flight = {}
_served = 0
_prefix = 'logs/gameapi'
_started = False


def _filename():
    date_str = datetime.datetime.now().strftime('%Y-%m-%d')
    return f"{_prefix}-health-{date_str}.log"


def _write(text, to_stderr=False):
    stamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    lines = ''.join(f"{stamp} {line}\n" for line in text.rstrip('\n').split('\n'))
    try:
        with open(_filename(), 'a', encoding='utf-8') as f:
            f.write(lines)
    except OSError:
        pass
    if to_stderr:
        print(lines, end='', file=sys.__stderr__, flush=True)


def _rss_mb():
    return _process.memory_info().rss / _MB


def _memory_line():
    mem = _process.memory_info()
    available = psutil.virtual_memory().available / _MB
    return (f"rss {mem.rss / _MB:.0f} MB vms {mem.vms / _MB:.0f} MB threads {_process.num_threads()} "
            f"system available {available:.0f} MB served {_served} in flight {len(_in_flight)}")


def request_started(method, url):
    """Call when a request starts; returns the id for request_finished."""
    request_id = next(_ids)
    _in_flight[request_id] = [_time(), method, url, _rss_mb(), None]
    return request_id


def request_finished(request_id, status=None):
    """Call when the request is done (also on errors). Returns a summary line for the request log."""
    global _served
    entry = _in_flight.pop(request_id, None)
    if entry is None:
        return None
    _served += 1
    start, method, url, rss_start, reported = entry
    elapsed = _time() - start
    rss = _rss_mb()
    path = url.split('?', 1)[0]
    line = f"Done {method} {path} {status or ''} {elapsed:.2f}s rss {rss:.0f} MB ({rss - rss_start:+.0f})"
    _write(line)
    if reported is not None:
        _write(f"Request {request_id} finished after {elapsed:.0f}s, after being reported as hanging", to_stderr=True)
    return line


def _thread_cpu():
    try:
        return {t.id: t.user_time + t.system_time for t in _process.threads()}
    except (psutil.Error, OSError):
        return {}


def _report_hang(request_id, entry, now, cpu_before, cpu_now, interval):
    start, method, url, rss_start, reported = entry
    lines = [f"HANG request {request_id} running {now - start:.0f}s: {method} {url}",
             f"pid {os.getpid()} {_memory_line()} (rss at request start {rss_start:.0f} MB)"]
    # Which native threads burned CPU since the last poll: a thread at ~100 % is
    # the one spinning. On Linux the id is the LWP that gdb shows.
    busy = sorted(((cpu_now[tid] - cpu_before.get(tid, 0.0), tid) for tid in cpu_now), reverse=True)
    busy = [f"{tid}:{100 * used / interval:.0f}%" for used, tid in busy[:8] if used > 0.01]
    lines.append(f"busy threads over the last {interval:.0f}s (thread id:cpu) {' '.join(busy) or 'none'}")
    if reported is None:
        lines.append(f"native stacks: gdb -p {os.getpid()} -batch -ex 'thread apply all bt' "
                     f"or py-spy dump --native --pid {os.getpid()}")
    _write('\n'.join(lines), to_stderr=True)
    if reported is None:
        # Python stacks of every OS thread; the main thread shows the stuck greenlet
        try:
            with open(_filename(), 'a', encoding='utf-8') as f:
                f.write("Python stacks:\n")
                f.flush()
                faulthandler.dump_traceback(file=f, all_threads=True)
        except OSError:
            pass


def _watchdog():
    cpu_before, last_poll = _thread_cpu(), _time()
    next_heartbeat = _time() + HEARTBEAT_MINUTES * 60
    while True:
        _sleep(POLL_SECONDS)
        now = _time()
        cpu_now = _thread_cpu()
        try:
            for request_id, entry in list(_in_flight.items()):
                start, reported = entry[0], entry[4]
                due = start + HANG_SECONDS if reported is None else reported + HANG_REPEAT
                if now >= due:
                    _report_hang(request_id, entry, now, cpu_before, cpu_now, now - last_poll)
                    entry[4] = now
            if HEARTBEAT_MINUTES > 0 and now >= next_heartbeat:
                _write(f"Heartbeat {_memory_line()}")
                next_heartbeat = now + HEARTBEAT_MINUTES * 60
        except Exception as ex:
            _write(f"Watchdog error: {ex!r}")
        cpu_before, last_poll = cpu_now, now


def start(prefix):
    """Start the watchdog thread once. prefix like 'logs/gameapi-<host>-<port>'."""
    global _prefix, _started
    if _started:
        return
    _prefix = prefix
    _started = True
    _write(f"Started pid {os.getpid()} hang report after {HANG_SECONDS:.0f}s, "
           f"heartbeat every {HEARTBEAT_MINUTES:g} min; {_memory_line()}")
    _start_thread(_watchdog, ())
