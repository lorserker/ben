#!/usr/bin/env python3
"""Smoke test for the native BBA (EPBot) integration.

Since BEN dropped the .NET EPBot64/EPBot86 builds, every platform loads the
native NativeAOT EPBot library (``bin/BBA/<os>/<arch>/...``) via ctypes. This
script verifies that path end-to-end:

  1. loads ``BBABotBid`` through the native library and exercises the whole
     public API surface — ``bid`` / ``explain_last_bid`` / ``explain_auction`` /
     ``find_info`` / ``find_aces`` / ``list_bids`` / ``get_sample`` / ``bid_hand``;
  2. optionally (``--challenge``) runs a short bidding challenge through
     ``game.py`` (which consults BBA) and checks the running score against an
     expected value, so a BBA/library regression shows up as a test failure.

Usage::

    python test_bba_native.py                          # API smoke test only
    python test_bba_native.py --challenge              # + GIB-BBO CTC challenge
    python test_bba_native.py --challenge --expected-score 109
    python test_bba_native.py --challenge --boards ../Challenges/pav_first120.pbn \
        --expected-score -123 --config ./config/BEN-21GF.conf

Exit code is 0 on success, 1 on any failure.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))      # .../ben/src
REPO_ROOT = os.path.dirname(HERE)                       # .../ben
CC_DIR = os.path.join(REPO_ROOT, "BBA", "CC")

sys.path.insert(0, HERE)


def _cc(name):
    path = os.path.join(CC_DIR, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Convention-card file not found: {path}")
    return path


def smoke_test_bba_api(verbose=False):
    """Exercise the whole BBABotBid API through the native EPBot library."""
    from bba.BBA import BBABotBid, EPBot_NATIVE_PATH, EPBotNative  # noqa: E402

    print(f"Native EPBot library: {EPBot_NATIVE_PATH}")
    if not os.path.isfile(EPBot_NATIVE_PATH):
        raise AssertionError(f"native EPBot library is missing: {EPBot_NATIVE_PATH}")
    assert BBABotBid.get_dll()["EPBot"] is EPBotNative, "BBABotBid is not using the native EPBot wrapper"

    version = BBABotBid(None, None, None, None, None, None, None, verbose).version()
    print(f"EPBot version: {version}")
    assert isinstance(version, int) and version > 0, f"unexpected EPBot version: {version!r}"

    gib = _cc("GIB-BBO.bbsa")
    ben = _cc("BEN-21GF.bbsa")

    # --- bid ---------------------------------------------------------------
    bot = BBABotBid(gib, gib, 2, "AT762.K74.Q8.654", [True, True], 0, False, verbose)
    resp = bot.bid(["PASS", "1H"])
    print(f"bid(['PASS','1H']) -> {resp.bid}  | {resp.explanation}")
    assert resp.bid and resp.bid.strip(), "bid() returned an empty bid"

    # --- explain_last_bid / explain_auction --------------------------------
    bot = BBABotBid(ben, ben, 1, "QJT85.Q6.KT764.2", [False, False], 2, False, verbose)
    expl, alert = bot.explain_last_bid(["PASS", "2S"])
    print(f"explain_last_bid(['PASS','2S']) -> {expl!r} alert={alert}")
    assert isinstance(alert, bool), f"explain_last_bid alert should be bool, got {type(alert)}"
    assert expl, "explain_last_bid returned an empty explanation"

    bot = BBABotBid(ben, ben, 1, "QJT85.Q6.KT764.2", [False, False], 3, False, verbose)
    meanings, controlled, preempted = bot.explain_auction(
        ["1H", "PASS", "2N", "PASS", "3H", "PASS", "3N", "PASS",
         "4C", "PASS", "4D", "PASS", "4N", "PASS", "5H", "PASS", "PASS", "PASS"])
    print(f"explain_auction(...) -> {len(meanings)} bids, bba_controlled={controlled}, preempted={preempted}")
    assert len(meanings) == 18, f"expected 18 bid meanings, got {len(meanings)}"
    assert controlled, "expected BBA to flag the 4N keycard auction as bba_controlled"

    # --- find_info / find_aces ---------------------------------------------
    bot = BBABotBid(gib, gib, 1, "T85.JT8642.3.765", [True, True], 2, False, verbose)
    info = bot.find_info(["1S", "PASS", "2C", "PASS", "2S", "PASS", "4H", "PASS", "4N", "PASS", "5C", "PASS"])
    print(f"find_info(...) -> trump={info.get('trump')}, asker={info.get('asker')}, players={sum(1 for k in info if isinstance(k, int))}")
    assert info["trump"] != 4, "expected a trump suit to be identified in the keycard auction"
    assert all(isinstance(info[i], dict) for i in range(8)), "find_info should describe 8 (real+mirror) hands"

    bot = BBABotBid(ben, ben, 2, "T942.AKT972.Q4.7", [True, True], 0, False, verbose)
    aces = bot.find_aces(["2N", "PASS", "3D", "PASS", "3N", "PASS", "4C", "PASS", "4H", "PASS", "4N", "PASS", "5D", "PASS"])
    print(f"find_aces(...) -> {aces}")
    assert {"LHO", "Partner", "RHO"} <= set(aces), f"find_aces missing seats: {aces}"

    # --- list_bids ---------------------------------------------------------
    bot = BBABotBid(gib, gib, 1, "AT762.K74.Q8.654", [True, True], 0, False, verbose)
    bids = bot.list_bids(["1D", "PASS", "2H", "PASS", "3D", "PASS"])
    print(f"list_bids(...) -> {len(bids)} legal bids, first={bids[0]['bid']}, last={bids[-1]['bid']}")
    assert len(bids) > 1, "list_bids returned too few bids"
    assert all("m" in b and "Alert" in b for b in bids), "list_bids entries are missing 'm'/'Alert'"

    # --- get_sample --------------------------------------------------------
    bot = BBABotBid(gib, gib, 3, "K87652.J9872.Q.Q", [True, True], 0, False, verbose)
    suits = bot.get_sample(["1D", "PASS", "2N", "PASS", "3N", "PASS", "PASS", "PASS"])
    print(f"get_sample(...) -> {len(suits)} suit strings")
    assert len(suits) >= 16, f"get_sample should return >=16 suit strings, got {len(suits)}"

    # --- bid_hand ----------------------------------------------------------
    bot = BBABotBid(gib, gib, 2, "AT762.K74.Q8.654", [True, True], 0, False, verbose)
    deal = "N:5.AQ8652.AK.AQT7 AT762.K74.Q8.654 QJ43.T9.J9732.92 K98.J3.T654.KJ83"
    finished = bot.bid_hand(["1H", "1S", "PASS", "PASS"], deal)
    print(f"bid_hand(...) -> auction of {len(finished)} calls, ends {finished[-3:]}")
    assert finished[-3:] == ["PASS", "PASS", "PASS"], f"bid_hand did not terminate the auction: {finished}"

    print("\nBBA native API smoke test: PASS\n")


_SCORE_RE = re.compile(r"Running score:\s*(-?\d+)", re.IGNORECASE)


def run_bidding_challenge(config, boards, expected_score, timeout):
    """Run `game.py --biddingonly NS` over a PBN and check the running score.

    `game.py` consults BBA (for ace-counting / rollouts / explanations depending
    on the config), so this is an end-to-end regression guard for the native
    EPBot path inside the real game pipeline.
    """
    out_fd, out_path = tempfile.mkstemp(prefix="ben_challenge_", suffix=".pbn")
    os.close(out_fd)
    cmd = [
        sys.executable, "game.py",
        "--config", config,
        "--auto", "True",
        "--biddingonly", "NS",
        "--boards", boards,
        "--facit", "True",
        "--outputpbn", out_path,
    ]
    print("Running:", " ".join(cmd))
    print(f"(cwd={HERE})")
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        print(f"ERROR: challenge timed out after {timeout}s")
        _cleanup(out_path)
        return False

    stdout = proc.stdout or ""
    # echo the tail so failures are diagnosable
    tail = "\n".join(stdout.splitlines()[-25:])
    print("--- game.py output (tail) ---")
    print(tail)
    print("-----------------------------")
    if proc.returncode != 0:
        print(f"ERROR: game.py exited with code {proc.returncode}")
        if proc.stderr:
            print(proc.stderr[-2000:])
        _cleanup(out_path)
        return False

    scores = [int(m) for m in _SCORE_RE.findall(stdout)]
    if not scores:
        print("ERROR: could not find a 'Running score:' line in game.py output")
        _cleanup(out_path)
        return False
    final = scores[-1]
    _cleanup(out_path)

    if expected_score is None:
        print(f"Bidding challenge finished. Running score = {final} (no expected value supplied — informational only).")
        return True
    if final == expected_score:
        print(f"Bidding challenge: PASS (running score = {final}, as expected).")
        return True
    print(f"Bidding challenge: FAIL — running score = {final}, expected {expected_score}.")
    print("  (If a BBA/library update legitimately changed bidding, re-run with the new --expected-score.)")
    return False


def _cleanup(path):
    try:
        os.remove(path)
    except OSError:
        pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Native BBA / EPBot smoke test.")
    parser.add_argument("--challenge", action="store_true",
                        help="also run a bidding challenge through game.py (slower; loads NN models)")
    parser.add_argument("--config", default="./config/GIB-BBO.conf",
                        help="game.py config for the challenge (default: ./config/GIB-BBO.conf)")
    parser.add_argument("--boards", default="../Challenges/CTC_mayjun20.pbn",
                        help="PBN file for the challenge (default: ../Challenges/CTC_mayjun20.pbn)")
    parser.add_argument("--expected-score", type=int, default=109,
                        help="expected running score for the challenge (default: 109 for the CTC/GIB-BBO baseline; "
                             "pass a new value when a BBA update changes bidding)")
    parser.add_argument("--no-expected-score", action="store_true",
                        help="don't assert on the challenge score, just report it")
    parser.add_argument("--challenge-timeout", type=int, default=900,
                        help="seconds to allow the challenge to run (default: 900)")
    parser.add_argument("--verbose", action="store_true", help="verbose BBA output")
    args = parser.parse_args(argv)

    ok = True
    try:
        smoke_test_bba_api(verbose=args.verbose)
    except Exception as ex:
        ok = False
        print(f"BBA native API smoke test: FAIL — {ex}")
        import traceback
        traceback.print_exc()

    if args.challenge:
        expected = None if args.no_expected_score else args.expected_score
        ok = run_bidding_challenge(args.config, args.boards, expected, args.challenge_timeout) and ok

    print("\n==> OVERALL:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
