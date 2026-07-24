"""Record every DDS call BEN makes, so it can be replayed as a benchmark.

The point is to benchmark DDS on *BEN's actual workload* instead of on synthetic
deals: the mix of trick depths, batch sizes, `solutions` values and trump suits
that a real board produces is what determines DDS runtime, and it is not easy to
guess. Play a session with recording on, then replay the file against a new DDS
build / thread count / `dds_mode` and compare.

Recording is **off** unless asked for, and when off the hooks in `ddsolver.py`
cost one attribute lookup per call. Ask for it with `--ddsrecord` (accepted by
game.py, gameserver.py, gameapi.py and table_manager_client.py) or, where there
is no command line to reach — Docker, frozen builds — the `BEN_DDS_RECORD`
environment variable.

    # record (path may be a file or a directory; .gz is compressed)
    python game.py --boards mysession.pbn --ddsrecord d:/tmp/dds-rec.jsonl

    # replay
    python src/ddsolver/ddsreplay.py d:/tmp/dds-rec.jsonl

File format is JSON Lines, one record per line:

    {"t":"meta",  ...}                     once, first line
    {"t":"board", "board":3, "deal":"..."}  whenever the driver starts a board
    {"t":"solve", "purpose":"play", "strain_i":1, ..., "result":{...}, "ms":12.3}
    {"t":"par",   "hand":"...", "vuln":[false,false], "result":420, "ms":4.5}

`solve` records hold the full input (`strain_i`, `leader_i`, `current_trick`,
`hands_pbn`, `solutions`) and the full output, so the replayer can both re-issue
the call and verify that the answer did not change.
"""

import atexit
import gzip
import io
import json
import os
import socket
import sys
import threading
import time


def _coerce_cards(cards):
    """current_trick may arrive as numpy ints; JSON cannot serialise those."""
    return [int(c) for c in cards]


def _coerce_result(card_results):
    """Normalise a solve() result to JSON-safe {str_key: [int, ...]}.

    Keys are card52 ints for solutions>=2 and the strings "max"/"min" for
    solutions==1, so they are stringified either way and re-read as strings.
    """
    if card_results is None:
        return None
    return {str(k): [int(v) for v in values] for k, values in card_results.items()}


class DDSRecorder:
    """Process-wide sink for DDS call records. All methods are classmethods."""

    enabled = False
    path = None

    _fh = None
    _lock = threading.Lock()
    _board = None
    _deal = None
    _seq = 0
    _dropped = 0
    _meta = None
    _header_written = False

    @classmethod
    def set_meta(cls, **kw):
        """Add fields to the header record.

        The header is written lazily, with the first real record, so that
        callers can contribute to it in any order: the entry point knows the
        file to record into at argument-parsing time, while `DDSolver` only
        knows the DDS version and thread count once it is constructed.
        """
        if cls._meta is not None:
            cls._meta.update({k: v for k, v in kw.items() if v not in (None, "")})

    @classmethod
    def configure(cls, path=None, **meta):
        """Open the recording file, or add metadata if it is already open.

        `path` comes from `--ddsrecord`, falling back to $BEN_DDS_RECORD. A
        directory (or a path with no suffix) gets a per-process file created
        inside it, so several BEN processes can record concurrently without
        clobbering each other.
        """
        if cls._fh is not None:
            cls.set_meta(**meta)
            return

        path = path or os.getenv("BEN_DDS_RECORD")
        if not path:
            return

        if os.path.isdir(path) or not os.path.splitext(path)[1]:
            os.makedirs(path, exist_ok=True)
            path = os.path.join(path, f"dds-{os.getpid()}.jsonl")

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        try:
            if path.endswith(".gz"):
                cls._fh = gzip.open(path, "wt", encoding="utf-8")
            else:
                cls._fh = io.open(path, "w", encoding="utf-8")
        except OSError as ex:
            sys.stderr.write(f"DDSRecorder: cannot write {path}: {ex}\n")
            return

        cls.path = path
        cls.enabled = True
        cls._meta = {
            "t": "meta",
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "host": socket.gethostname(),
            "platform": sys.platform,
            "python": sys.version.split()[0],
            "argv": sys.argv,
        }
        cls.set_meta(**meta)
        atexit.register(cls.close)
        sys.stderr.write(f"DDSRecorder: recording DDS calls to {path}\n")

    @classmethod
    def _write(cls, record):
        with cls._lock:
            if cls._fh is None:
                return
            if not cls._header_written:
                cls._header_written = True
                try:
                    cls._fh.write(json.dumps(cls._meta, separators=(",", ":")) + "\n")
                except (TypeError, ValueError, OSError):
                    pass
            try:
                cls._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
                # Sessions usually end with Ctrl-C or a kill, so flush every
                # record rather than lose the tail. The replayer tolerates a
                # truncated final line either way.
                cls._fh.flush()
            except (TypeError, ValueError, OSError) as ex:
                # A bad record must never take the game down.
                cls._dropped += 1
                if cls._dropped <= 3:
                    sys.stderr.write(f"DDSRecorder: dropped a record: {ex}\n")

    @classmethod
    def start_board(cls, board_number=None, deal=None, **meta):
        """Tag the calls that follow as belonging to this board."""
        if not cls.enabled:
            return
        cls._board = board_number
        cls._deal = deal
        record = {"t": "board", "board": board_number, "deal": deal}
        record.update(meta)
        cls._write(record)

    @classmethod
    def record_solve(cls, strain_i, leader_i, current_trick, hands_pbn, solutions,
                     purpose, trick, result, elapsed_ms):
        if not cls.enabled:
            return
        cls._seq += 1
        cls._write({
            "t": "solve",
            "seq": cls._seq,
            "board": cls._board,
            "purpose": purpose,
            "trick": trick,
            "strain_i": int(strain_i),
            "leader_i": int(leader_i),
            "current_trick": _coerce_cards(current_trick),
            "solutions": int(solutions),
            "boards": len(hands_pbn),
            "hands_pbn": list(hands_pbn),
            "result": _coerce_result(result),
            "ms": round(elapsed_ms, 3),
        })

    @classmethod
    def record_par(cls, hand, vuln, result, elapsed_ms):
        if not cls.enabled:
            return
        cls._seq += 1
        cls._write({
            "t": "par",
            "seq": cls._seq,
            "board": cls._board,
            "hand": hand,
            "vuln": [bool(v) for v in vuln],
            "result": None if result is None else int(result),
            "ms": round(elapsed_ms, 3),
        })

    @classmethod
    def close(cls):
        with cls._lock:
            if cls._fh is None:
                return
            try:
                # A session that made no DDS calls would otherwise leave a
                # zero-byte file; write the header so it is still a valid
                # (if empty) recording.
                if not cls._header_written:
                    cls._header_written = True
                    cls._fh.write(json.dumps(cls._meta, separators=(",", ":")) + "\n")
                cls._fh.close()
            except (TypeError, ValueError, OSError):
                pass
            cls._fh = None
            cls.enabled = False
