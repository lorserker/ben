"""Replay a recorded DDS workload as a benchmark.

Reads a file produced by `DDSRecorder` (see ddsrecorder.py), re-issues every DDS
call in the order BEN made it, times them, and checks that the answers still
match what was recorded. Because the calls are BEN's own, the result reflects
the real mix of trick depths, batch sizes and `solutions` values rather than a
guess at what a representative deal looks like.

    # record a session
    BEN_DDS_RECORD=d:/tmp/dds-rec python game.py --boards mysession.pbn

    # benchmark it
    python src/ddsolver/ddsreplay.py d:/tmp/dds-rec/dds-12345.jsonl --repeat 3

    # find the best thread count for this machine
    python src/ddsolver/ddsreplay.py rec.jsonl --threads 8 --threads 16 --threads 32

    # compare a DDS rebuild against a saved run
    python src/ddsolver/ddsreplay.py rec.jsonl --json new.json --baseline old.json

Calls go through `DDSolver.solve_helper` / `_calculatepar_impl` rather than the
public `solve` / `calculatepar`, so the numbers measure DDS itself and replaying
never re-triggers the recorder.

Note on repeatability: with `dds_mode=1` DDS reuses a transposition table across
consecutive solves on the same `SolverContext`, and the thread pool assigns
boards to contexts nondeterministically, so a single run has a few percent of
noise. Use `--repeat` and compare the best run.
"""

import argparse
import gzip
import io
import json
import os
import sys
import time
from collections import OrderedDict

# Allow running as `python src/ddsolver/ddsreplay.py` — BEN's modules assume
# src/ is on sys.path (ddsolver.py does `from objects import Card`).
#
# Dropping this script's own directory matters: it puts `ddsolver.py` on the
# path as a top-level module, which shadows the `ddsolver` package (there is no
# __init__.py) and breaks `from ddsolver.ddsrecorder import ...` inside it.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(_HERE)
sys.path[:] = [p for p in sys.path
               if os.path.abspath(p or os.getcwd()) != _HERE]
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def load_records(path):
    """Yield the records of a .jsonl / .jsonl.gz recording."""
    opener = gzip.open if path.endswith(".gz") else io.open
    with opener(path, "rt", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as ex:
                # A run killed mid-write leaves a truncated last line; that is
                # normal and should not cost you the whole recording.
                sys.stderr.write(f"{path}:{line_no}: skipping unparseable line ({ex})\n")


def normalise_result(result):
    """Canonical form for comparing a recorded result with a replayed one.

    Recorded results come back from JSON with string keys; live results have
    int keys (or the strings "max"/"min" when solutions==1).
    """
    if result is None:
        return None
    return {str(k): [int(v) for v in values] for k, values in result.items()}


def describe_mismatch(expected, actual):
    """One short line saying how two results differ."""
    if actual is None:
        return "replay returned None (DDS error)"
    exp, act = normalise_result(expected), normalise_result(actual)
    missing = sorted(set(exp) - set(act))
    extra = sorted(set(act) - set(exp))
    if missing or extra:
        return f"keys differ (missing {missing[:4]}, extra {extra[:4]})"
    for k in exp:
        if exp[k] != act[k]:
            return f"key {k}: recorded {exp[k][:6]} vs replayed {act[k][:6]}"
    return "unknown difference"


class Bucket:
    __slots__ = ("calls", "boards", "seconds", "recorded_ms")

    def __init__(self):
        self.calls = 0
        self.boards = 0
        self.seconds = 0.0
        self.recorded_ms = 0.0

    def add(self, boards, seconds, recorded_ms):
        self.calls += 1
        self.boards += boards
        self.seconds += seconds
        self.recorded_ms += recorded_ms or 0.0


def select(records, args):
    """Apply the --purpose / --board / --trick / --limit filters.

    Returns (meta, n_deals, calls). n_deals counts the boards the *selected*
    calls belong to, not the board markers in the file, so it stays honest
    under filtering.
    """
    meta, calls = {}, []
    for rec in records:
        kind = rec.get("t")
        if kind == "meta":
            meta = rec
            continue
        if kind == "board":
            continue
        if kind == "par":
            # --purpose names the buckets the report uses, and par is one of
            # them, so --purpose bid must exclude par rather than silently
            # leave it in.
            if args.no_par or (args.purpose and "par" not in args.purpose):
                continue
        elif kind == "solve":
            if args.purpose and (rec.get("purpose") or "") not in args.purpose:
                continue
            trick = rec.get("trick") or 0
            if args.min_trick and trick < args.min_trick:
                continue
            if args.max_trick and trick > args.max_trick:
                continue
            if args.solutions and rec.get("solutions") not in args.solutions:
                continue
        else:
            continue
        if args.board and rec.get("board") not in args.board:
            continue
        calls.append(rec)
        if args.limit and len(calls) >= args.limit:
            break
    return meta, len({c.get("board") for c in calls}), calls


def replay_once(dds, calls, verify):
    """Issue every recorded call once. Returns (buckets, total_s, mismatches)."""
    by_purpose = OrderedDict()
    by_trick = OrderedDict()
    mismatches = []
    total = 0.0

    for rec in calls:
        if rec["t"] == "par":
            t0 = time.perf_counter()
            result = dds._calculatepar_impl(rec["hand"], rec["vuln"], False)
            elapsed = time.perf_counter() - t0
            boards, purpose, trick = 1, "par", 0
            if verify and result != rec.get("result"):
                mismatches.append((rec.get("seq"), "par",
                                   f"recorded {rec.get('result')} vs replayed {result}"))
        else:
            t0 = time.perf_counter()
            result = dds.solve_helper(rec["strain_i"], rec["leader_i"],
                                      rec["current_trick"], rec["hands_pbn"],
                                      rec["solutions"])
            elapsed = time.perf_counter() - t0
            boards = len(rec["hands_pbn"])
            purpose = rec.get("purpose") or "(none)"
            trick = rec.get("trick") or 0
            if verify and normalise_result(result) != normalise_result(rec.get("result")):
                mismatches.append((rec.get("seq"), purpose,
                                   describe_mismatch(rec.get("result"), result)))

        total += elapsed
        by_purpose.setdefault(purpose, Bucket()).add(boards, elapsed, rec.get("ms"))
        by_trick.setdefault(trick, Bucket()).add(boards, elapsed, rec.get("ms"))

    return by_purpose, by_trick, total, mismatches


def _row(name, b, width=12):
    ms_call = b.seconds * 1000 / b.calls if b.calls else 0.0
    ms_board = b.seconds * 1000 / b.boards if b.boards else 0.0
    rec_call = b.recorded_ms / b.calls if b.calls else 0.0
    return (f"{name:<{width}}{b.calls:>8}{b.boards:>10}{b.seconds:>10.3f}"
            f"{ms_call:>11.2f}{ms_board:>11.3f}{rec_call:>13.2f}")


def _header(width=12):
    return (f"{'':<{width}}{'calls':>8}{'boards':>10}{'seconds':>10}"
            f"{'ms/call':>11}{'ms/board':>11}{'rec ms/call':>13}")


def print_report(meta, n_deals, calls, by_purpose, by_trick, total, mismatches,
                 run_totals, threads, dds_mode, dds_version, show_tricks):
    print()
    print(f"  recorded : {meta.get('created', '?')} on {meta.get('host', '?')} "
          f"({meta.get('platform', '?')})  DDS {meta.get('dds_version', '?')}  "
          f"mode {meta.get('dds_mode', '?')}  {meta.get('threads', '?')} threads")
    print(f"  replaying: DDS {dds_version}  mode {dds_mode}  {threads} threads  "
          f"python {sys.version.split()[0]}")
    n_solve = sum(1 for c in calls if c["t"] == "solve")
    n_boards = sum(len(c["hands_pbn"]) for c in calls if c["t"] == "solve")
    print(f"  workload : {n_solve} solve calls ({n_boards} boards) + "
          f"{len(calls) - n_solve} par, over {n_deals} deals")
    print()

    print(_header())
    print("-" * 75)
    for purpose, bucket in sorted(by_purpose.items(), key=lambda kv: -kv[1].seconds):
        print(_row(purpose, bucket))
    print("-" * 75)
    grand = Bucket()
    for bucket in by_purpose.values():
        grand.calls += bucket.calls
        grand.boards += bucket.boards
        grand.seconds += bucket.seconds
        grand.recorded_ms += bucket.recorded_ms
    print(_row("TOTAL", grand))

    if show_tricks:
        print()
        print("by trick")
        print(_header())
        print("-" * 75)
        for trick in sorted(by_trick):
            label = "par" if trick == 0 else f"t{trick:02d}"
            print(_row(label, by_trick[trick]))

    print()
    if len(run_totals) > 1:
        runs = "  ".join(f"{t:.3f}" for t in run_totals)
        print(f"DDS time: {total:.3f} s   (best of {len(run_totals)}: {runs})")
    else:
        print(f"DDS time: {total:.3f} s")

    if mismatches:
        print(f"VERIFY: {len(mismatches)} of {len(calls)} calls returned a "
              f"different result than recorded")
        for seq, purpose, why in mismatches[:10]:
            print(f"  seq {seq} ({purpose}): {why}")
        if len(mismatches) > 10:
            print(f"  ... and {len(mismatches) - 10} more")
    else:
        print(f"VERIFY: all {len(calls)} calls returned the recorded result")


def print_baseline_diff(baseline_path, by_purpose, total):
    try:
        with io.open(baseline_path, "r", encoding="utf-8") as fh:
            base = json.load(fh)
    except (OSError, ValueError) as ex:
        sys.stderr.write(f"Cannot read baseline {baseline_path}: {ex}\n")
        return

    print()
    print(f"vs baseline {baseline_path} "
          f"({base.get('threads', '?')} threads, DDS {base.get('dds_version', '?')})")
    print(f"{'':<12}{'baseline s':>12}{'now s':>10}{'change':>10}")
    print("-" * 44)
    base_purposes = base.get("by_purpose", {})
    for purpose in sorted(set(base_purposes) | set(by_purpose)):
        b = base_purposes.get(purpose, {}).get("seconds", 0.0)
        n = by_purpose[purpose].seconds if purpose in by_purpose else 0.0
        change = f"{(n / b - 1) * 100:+.1f}%" if b else "n/a"
        print(f"{purpose:<12}{b:>12.3f}{n:>10.3f}{change:>10}")
    print("-" * 44)
    bt = base.get("total_seconds", 0.0)
    change = f"{(total / bt - 1) * 100:+.1f}%" if bt else "n/a"
    print(f"{'TOTAL':<12}{bt:>12.3f}{total:>10.3f}{change:>10}")
    if bt:
        print(f"speedup: {bt / total:.2f}x" if total else "")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay a recorded DDS workload as a benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Record one with:  BEN_DDS_RECORD=<dir> python game.py ...")
    parser.add_argument("recording", help="a .jsonl / .jsonl.gz file from DDSRecorder")
    parser.add_argument("--threads", type=int, action="append", default=None,
                        metavar="N",
                        help="DDSolver thread count; repeat the flag to sweep "
                             "several (default: the recorded value)")
    parser.add_argument("--dds-mode", type=int, default=None,
                        help="dds_mode to replay with (default: the recorded value)")
    parser.add_argument("--repeat", type=int, default=1, metavar="N",
                        help="run the workload N times and report the best")
    parser.add_argument("--warmup", action="store_true",
                        help="do one untimed pass first (warms the thread pool "
                             "and the OS file cache)")
    parser.add_argument("--purpose", action="append", default=None,
                        help="only replay this purpose (bid/lead/play/claimcheck/"
                             "claimtricks/contract); repeatable")
    parser.add_argument("--board", type=int, action="append", default=None,
                        help="only replay this board number; repeatable")
    parser.add_argument("--solutions", type=int, action="append", default=None,
                        help="only replay calls with this solutions value; repeatable")
    parser.add_argument("--min-trick", type=int, default=None)
    parser.add_argument("--max-trick", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N calls")
    parser.add_argument("--no-par", action="store_true",
                        help="skip the par calculations")
    parser.add_argument("--no-verify", action="store_true",
                        help="do not compare results against the recording")
    parser.add_argument("--tricks", action="store_true",
                        help="also break the report down by trick number")
    parser.add_argument("--json", dest="json_out", metavar="FILE",
                        help="write the results as JSON for later --baseline use")
    parser.add_argument("--baseline", metavar="FILE",
                        help="compare against a previous --json run")
    parser.add_argument("--list", action="store_true",
                        help="summarise the recording and exit without running DDS")
    args = parser.parse_args(argv)

    if not os.path.exists(args.recording):
        parser.error(f"no such recording: {args.recording}")

    meta, n_deals, calls = select(load_records(args.recording), args)
    if not calls:
        parser.error("no calls left after filtering")

    if args.list:
        n_solve = sum(1 for c in calls if c["t"] == "solve")
        n_boards = sum(len(c["hands_pbn"]) for c in calls if c["t"] == "solve")
        rec_s = sum(c.get("ms") or 0.0 for c in calls) / 1000
        print(f"{args.recording}")
        print(f"  recorded {meta.get('created', '?')} on {meta.get('host', '?')} "
              f"with DDS {meta.get('dds_version', '?')} mode {meta.get('dds_mode', '?')} "
              f"{meta.get('threads', '?')} threads")
        print(f"  {n_deals} deals, {n_solve} solve calls ({n_boards} boards), "
              f"{len(calls) - n_solve} par, {rec_s:.1f} s of DDS when recorded")
        counts = OrderedDict()
        for c in calls:
            key = c.get("purpose") or c["t"]
            entry = counts.setdefault(key, [0, 0, 0.0])
            entry[0] += 1
            entry[1] += len(c.get("hands_pbn") or [1])
            entry[2] += (c.get("ms") or 0.0) / 1000
        print(f"  {'purpose':<14}{'calls':>8}{'boards':>10}{'rec s':>10}")
        for key, (n, b, s) in sorted(counts.items(), key=lambda kv: -kv[1][2]):
            print(f"  {key:<14}{n:>8}{b:>10}{s:>10.3f}")
        return 0

    from ddsolver.ddsolver import DDSolver

    dds_mode = args.dds_mode if args.dds_mode is not None else meta.get("dds_mode", 1)
    thread_counts = args.threads or [meta.get("threads") or 0]

    last = None
    for threads in thread_counts:
        dds = DDSolver(dds_mode=dds_mode, max_threads=threads)
        actual_threads = dds._workers

        if args.warmup:
            replay_once(dds, calls, verify=False)

        best = None
        run_totals = []
        for _ in range(max(1, args.repeat)):
            by_purpose, by_trick, total, mismatches = replay_once(
                dds, calls, verify=not args.no_verify)
            run_totals.append(total)
            if best is None or total < best[2]:
                best = (by_purpose, by_trick, total, mismatches)

        by_purpose, by_trick, total, mismatches = best
        print()
        print("=" * 75)
        print(f"DDS replay — {args.recording}")
        print("=" * 75)
        print_report(meta, n_deals, calls, by_purpose, by_trick, total, mismatches,
                     run_totals, actual_threads, dds_mode, dds.version(), args.tricks)
        last = (by_purpose, total, actual_threads, dds.version(), mismatches)

    by_purpose, total, actual_threads, dds_version, mismatches = last

    if args.baseline:
        print_baseline_diff(args.baseline, by_purpose, total)

    if args.json_out:
        payload = {
            "recording": os.path.abspath(args.recording),
            "recorded_meta": meta,
            "dds_version": dds_version,
            "dds_mode": dds_mode,
            "threads": actual_threads,
            "repeat": args.repeat,
            "total_seconds": total,
            "mismatches": len(mismatches),
            "by_purpose": {p: {"calls": b.calls, "boards": b.boards,
                               "seconds": b.seconds}
                           for p, b in by_purpose.items()},
        }
        with io.open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_out}")

    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
