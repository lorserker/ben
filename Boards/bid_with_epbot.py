"""Bid every deal in a PBN file with EPBot (the native library BEN ships in bin/BBA)
and write the boards in the PBN format Bridge Bidding Analyser archives, so
Split_PBN and bidding_binary_keras.py can read them.

The same convention card is used at all four seats. The result of each contract
is the double-dummy number of tricks, taken from DatumScores.pkl (with DD tables,
as in the Bridge-Robot-Utilities release); deals not found there are written
without Result/Score and end up in -missing-DD.pbn when Split_PBN runs.

Usage:
    python bid_with_epbot.py input.pbn CC output.pbn [--datum DatumScores.pkl] [--workers N] [--limit N]

    CC is the name of the card in BBA/CC, e.g. BEN-21GF
    Repeated boards in the input are bid only once.
"""
import argparse
import datetime
import multiprocessing as mp
import os
import pickle
import sys
import time

BEN_HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
os.environ.setdefault('BBA_BIN', os.path.join(BEN_HOME, 'bin'))
sys.path.insert(0, os.path.join(BEN_HOME, 'src'))

from bba.BBA import EPBotNative, BBABotBid  # noqa: E402
from scoring import score  # noqa: E402

SEATS = 'NESW'
STRAINS = 'CDHSN'
# DatumScores DD table: 20 bytes, declarer N/E/S/W x strain S/H/D/C/NT
DD_STRAIN_INDEX = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'N': 4}
VULN = {'None': (False, False), 'NS': (True, False), 'EW': (False, True), 'All': (True, True), 'Both': (True, True),
        'Love': (False, False), '-': (False, False)}

# Worker state
_players = None


def load_card(path):
    system, conventions = None, {}
    with open(path) as f:
        for i, line in enumerate(f):
            key, value = line.strip().split(' = ')
            if i == 0 and key == "System type":
                system = int(value)
            else:
                conventions[key] = bool(int(value))
    return system, conventions


def create_players(card_path):
    system, conventions = load_card(card_path)
    players = []
    for _ in range(4):
        p = EPBotNative()
        # Same card for both sides
        for side in (BBABotBid.C_NS, BBABotBid.C_WE):
            p.set_system_type(side, system)
            for name, selected in conventions.items():
                p.set_conventions(side, name, selected)
        p.scoring = BBABotBid.SCORING_IMP
        players.append(p)
    return players


def init_worker(card_path):
    global _players
    BBABotBid.get_dll()
    _players = create_players(card_path)


def read_boards(path, limit=None):
    """Yield (dealer, vulnerable, hands N-E-S-W) for every board, skipping repeated boards
    (same cards, dealer and vulnerability)"""
    seen = set()
    dealer = vulnerable = None
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            if line.startswith('[Dealer '):
                dealer = line.split('"')[1]
            elif line.startswith('[Vulnerable '):
                vulnerable = line.split('"')[1]
            elif line.startswith('[Deal '):
                first, hands = line.split('"')[1].split(':')
                hands = hands.split()
                i = SEATS.index(first)
                nesw = tuple(hands[(j - i) % 4] for j in range(4))
                key = (dealer, vulnerable, nesw)
                if key in seen:
                    continue
                seen.add(key)
                yield dealer, vulnerable, nesw
                if limit and len(seen) >= limit:
                    return


def call_name(code):
    if code == 0:
        return 'Pass'
    if code == 1:
        return 'X'
    if code == 2:
        return 'XX'
    level, strain = (code - 5) // 5 + 1, STRAINS[(code - 5) % 5]
    return f"{level}{'NT' if strain == 'N' else strain}"


def bid_board(board):
    dealer, vulnerable, nesw = board
    dealer_i = SEATS.index(dealer)
    vuln_ns, vuln_ew = VULN[vulnerable]
    bba_vuln = vuln_ew + vuln_ns * 2
    for seat in range(4):
        suits = nesw[seat].split('.')
        suits.reverse()
        _players[seat].new_hand(seat, suits, dealer_i, bba_vuln)

    calls, alerts = [], []
    passes, bid_made = 0, False
    position = dealer_i
    # Three passes end the auction after a bid, four passes when nobody has bid
    while passes < 3 or (not bid_made and passes < 4):
        code = _players[position].get_bid()
        for p in _players:
            p.set_bid(position, code)
        # The alert as the opponents see it
        lho = _players[(position + 1) % 4]
        alert = lho.get_info_alerting(position) if code > 2 else False
        alerts.append(lho.get_info_meaning(position) if alert else None)
        calls.append(code)
        if code == 0:
            passes += 1
        else:
            passes, bid_made = 0, True
        position = (position + 1) % 4
    return dealer, vulnerable, nesw, calls, alerts


def final_contract(dealer_i, calls):
    """Return (contract like '4SX', declarer index) or (None, None) when passed out"""
    last_bid_i = max((i for i, c in enumerate(calls) if c >= 5), default=None)
    if last_bid_i is None:
        return None, None
    bid = calls[last_bid_i]
    strain = STRAINS[(bid - 5) % 5]
    level = (bid - 5) // 5 + 1
    doubled = ''
    for c in calls[last_bid_i + 1:]:
        if c == 1:
            doubled = 'X'
        elif c == 2:
            doubled = 'XX'
    side = (dealer_i + last_bid_i) % 2
    for i, c in enumerate(calls):
        if c >= 5 and STRAINS[(c - 5) % 5] == strain and (dealer_i + i) % 2 == side:
            return f"{level}{strain}{doubled}", (dealer_i + i) % 4


def lookup_datum(datums, nesw):
    """DatumScores.pkl is keyed by the [Deal] text as it was written, which can start at any seat"""
    for first in range(4):
        key = f'[Deal "{SEATS[first]}:{" ".join(nesw[(first + j) % 4] for j in range(4))}"]'
        if key in datums:
            return datums[key]
    return None


def format_board(number, result, date, system_name, datums):
    dealer, vulnerable, nesw, calls, alerts = result
    dealer_i = SEATS.index(dealer)
    deal = 'N:' + ' '.join(nesw)
    contract, declarer = final_contract(dealer_i, calls)

    datum = lookup_datum(datums, nesw)
    tricks = score_str = None
    if contract and datum is not None and len(datum) > 3:
        tricks = datum[3][declarer * 5 + DD_STRAIN_INDEX[contract[1]]]
        vuln_ns, vuln_ew = VULN[vulnerable]
        points = score(contract, vuln_ns if declarer % 2 == 0 else vuln_ew, tricks)
        if declarer % 2 == 1:
            points = -points
        score_str = f"NS {points}"

    lines = ['[Event ""]', '[Site ""]', f'[Date "{date}"]', f'[Board "{number}"]',
             '[West "EPBot4"]', '[North "EPBot1"]', '[East "EPBot2"]', '[South "EPBot3"]',
             f'[Dealer "{dealer}"]', f'[Vulnerable "{vulnerable}"]', f'[Deal "{deal}"]', '[Scoring "IMP"]']
    if contract:
        pbn_contract = contract[0] + ('NT' if contract[1] == 'N' else contract[1]) + contract[2:]
        lines += [f'[Declarer "{SEATS[declarer]}"]', f'[Contract "{pbn_contract}"]',
                  f'[Result "{tricks if tricks is not None else ""}"]']
    else:
        lines += ['[Declarer ""]', '[Contract "Pass"]', '[Result ""]']
    lines.append(f'[Auction "{dealer}"]')
    notes, row = [], []
    for code, meaning in zip(calls, alerts):
        call = call_name(code)
        if meaning is not None:
            notes.append(meaning)
            call += f' ={len(notes)}='
        row.append(call)
        if len(row) == 4:
            lines.append(' '.join(row))
            row = []
    if row:
        lines.append(' '.join(row))
    lines += [f'[Note "{i}:{m}"]' for i, m in enumerate(notes, 1)]
    lines += ['[Room "Open"]', f'[Score "{score_str or ""}"]',
              f'[BidSystemEW "{system_name}"]', f'[BidSystemNS "{system_name}"]']
    if datum is not None:
        lines += [datum[0], datum[1]]
        if len(datum) > 3:
            # Columns separated by ';' as in the PBN standard, so endplay (Split_PBN) keeps all three
            lines.append('[OptimumResultTable "Declarer;Denomination\\2R;Result\\2R"]')
            lines += [f'{seat} {"NT" if strain == "N" else strain} {datum[3][s * 5 + DD_STRAIN_INDEX[strain]]}'
                      for s, seat in enumerate(SEATS) for strain in 'SHDCN']
    return '\n'.join(lines) + '\n\n'


def work(boards):
    return [bid_board(b) for b in boards]


def chunks(boards, size):
    batch = []
    for b in boards:
        batch.append(b)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input')
    parser.add_argument('cc')
    parser.add_argument('output')
    parser.add_argument('--datum', help='DatumScores.pkl with double-dummy tables')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 2))
    parser.add_argument('--limit', type=int, help='only the first N deals (for testing)')
    args = parser.parse_args()

    card = os.path.join(BEN_HOME, 'BBA', 'CC', args.cc + '.bbsa')
    if not os.path.exists(card):
        sys.exit(f"Convention card {card} not found")
    BBABotBid.get_dll()
    version = EPBotNative().version()
    system_name = create_players(card)[0].system_name(BBABotBid.C_NS)
    date = datetime.date.today().strftime('%Y.%m.%d')
    datums = {}
    if args.datum:
        with open(args.datum, 'rb') as f:
            datums = pickle.load(f)

    t_start = time.time()
    n = missing = 0
    with mp.Pool(args.workers, initializer=init_worker, initargs=(card,)) as pool, \
            open(args.output, 'w', encoding='utf-8', newline='\n') as out:
        out.write('% PBN 2.1\n% EXPORT\n')
        out.write(f'% Bid by EPBot {version} with {args.cc} at all four seats (Boards/bid_with_epbot.py)\n')
        out.write('% Result = double-dummy tricks from DatumScores.pkl\n\n')
        for results in pool.imap(work, chunks(read_boards(args.input, args.limit), 500)):
            for result in results:
                n += 1
                if lookup_datum(datums, result[2]) is None:
                    missing += 1
                out.write(format_board(n, result, date, system_name, datums))
            if n % 50000 < 500:
                print(f"{n} boards, {n / (time.time() - t_start):.0f} boards/s", flush=True)
    print(f"Done: {n} boards in {time.time() - t_start:.0f} s, EPBot {version}, card {args.cc} -> {args.output}")
    if missing:
        print(f"{missing} deals are not in the datum file and have no result")


if __name__ == '__main__':
    main()
