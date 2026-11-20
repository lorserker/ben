# COPY of Bridge-Robot-Utilities/src/datumscore.py -- that file is the master; change it there and
# copy it here (bid_with_epbot.py reads DatumScores pickles the same way the utilities do).

"""The datum-score cache, DatumScores.pkl: one double-dummy table per deal, nothing else.

    key   = '[Deal "N:..."]'           the deal's exact PBN tag line
    value = 20 bytes                    double-dummy tricks, one per (declarer, denomination)
                                        in the order N/E/S/W x S/H/D/C/NT

Only TRICKS are stored, because only tricks belong to the deal alone. Everything scored
from them -- [OptimumScore] and [ParContract] (depend on vulnerability and dealer), the
double-dummy [Score] of a contract (depends on the contract and vulnerability) -- is
computed when it is needed, from the board being annotated. That is what lets one entry
serve every copy of a deal: the same hands at another vulnerability, with another
dealer, or played in another contract.

The cache used to store the par tags themselves, keyed by the deal, and so gave every
copy of a deal the par of whichever copy was stored: 3,310 of TrainingDeals.pbn's
boards carried the par of their twin at another vulnerability. Older pickles are still
read -- a (OptimumScore, ParContract, vul, dd_bytes) tuple contributes its dd_bytes and
its stored par is ignored -- and a tuple without dd_bytes counts as absent: a stored
par cannot be trusted for any particular board.

Used by AddDatumScoreToPBN, UpdateScore, Split_PBN and CheckDatumScores (readers) and
by ComputeDatumScores, ComputeDDTables and ExtractDatumScore (writers).
"""
import os
import pickle
import shutil

# Packing order of the 20 bytes. build_ort in AddDatumScoreToPBN and every writer use it.
PLAYER_ORDER = ('N', 'E', 'S', 'W')
DENOM_ORDER = ('S', 'H', 'D', 'C', 'NT')
TABLE_SIZE = len(PLAYER_ORDER) * len(DENOM_ORDER)

# PBN [Vulnerable] values -> endplay Vul member NAMES (endplay numbers them none/both/ns/ew,
# not in DDS's order, so never map to integers).
_VUL = {'none': 'none', 'love': 'none', '-': 'none', '': 'none',
        'ns': 'ns', 'n-s': 'ns', 'ew': 'ew', 'e-w': 'ew', 'all': 'both', 'both': 'both'}


def load(pickle_path):
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def save(data, pickle_path):
    """Write the pickle atomically, keeping a one-time backup of what was there."""
    backup = pickle_path + '.bak'
    if os.path.exists(pickle_path) and not os.path.exists(backup):
        shutil.copy2(pickle_path, backup)
        print(f"backed up original to {backup}", flush=True)
    tmp = pickle_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, pickle_path)


def deal_key(deal):
    """The pickle key for a deal string ('N:...') or a whole [Deal "..."] line."""
    deal = deal.strip()
    if deal.startswith('['):
        deal = deal.split('"')[1]
    return f'[Deal "{deal}"]'


def tricks_of(value):
    """The 20 trick counts held by a pickle value, in either format, or None."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value) if len(value) == TABLE_SIZE else None
    if isinstance(value, tuple) and len(value) > 3 and value[3] and len(value[3]) == TABLE_SIZE:
        return bytes(value[3])
    return None


def normalise(data):
    """Rewrite every old-format value that carries tricks as plain bytes, in place.

    Values without tricks are left alone (a writer solves them); returns how many
    values were converted."""
    converted = 0
    for k, v in data.items():
        if not isinstance(v, (bytes, bytearray)):
            t = tricks_of(v)
            if t is not None:
                data[k] = t
                converted += 1
    return converted


def tricks(data, deal_line):
    """Look a deal up: the exact line first, then endplay's canonical form of the deal,
    so a differently formatted deal still matches. Returns 20 bytes or None."""
    value = data.get(deal_line.strip())
    if value is None:
        try:
            from endplay.types import Deal
            value = data.get(f'[Deal "{Deal(deal_key(deal_line).split(chr(34))[1]).to_pbn()}"]')
        except Exception:
            value = None
    return None if value is None else tricks_of(value)


def tricks_from_ort(rows):
    """Pack an [OptimumResultTable]'s rows ({(declarer, denom): tricks}) into 20 bytes,
    or None unless all 20 are present and in range."""
    try:
        packed = bytes(rows[(p, d)] for p in PLAYER_ORDER for d in DENOM_ORDER)
    except (KeyError, ValueError):
        return None
    return packed if all(0 <= t <= 13 for t in packed) else None


def dd_table(dd_bytes):
    """An endplay DDTable built from packed tricks (no solving)."""
    from endplay import _dds
    from endplay.dds.ddtable import DDTable
    from endplay.types import Player, Denom
    players = [Player.north, Player.east, Player.south, Player.west]
    denoms = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs, Denom.nt]
    res = _dds.ddTableResults()
    i = 0
    for p in players:
        for d in denoms:
            res.resTable[int(d)][int(p)] = dd_bytes[i]
            i += 1
    return DDTable(res)


# DDS Par() vulnerability codes, and the DD table index of each seat / denomination letter.
_DDS_VUL = {'none': 0, 'both': 1, 'ns': 2, 'ew': 3}
_SEAT_INDEX = {s: i for i, s in enumerate(PLAYER_ORDER)}
_DENOM_INDEX = {'S': 0, 'H': 1, 'D': 2, 'C': 3, 'N': 4, 'NT': 4}
_DENOM_RANK = {'C': 0, 'D': 1, 'H': 2, 'S': 3, 'N': 4}


def _dds_par(dd_bytes, vulnerable):
    """DDS Par() on the stored tricks: ([score per side], [contract string per side]),
    side 0 seen from N/S, side 1 from E/W -- e.g. ['NS 110', 'EW -110'],
    ['NS:NS 2H,NS 3C', 'EW:NS 2H,NS 3C']."""
    import ctypes
    from endplay import _dds
    res = _dds.parResults()
    vul = _VUL.get((vulnerable or '').strip().lower(), 'none') if isinstance(vulnerable, str) or vulnerable is None         else str(getattr(vulnerable, 'name', 'none'))
    _dds.Par(dd_table(dd_bytes)._data, res, _DDS_VUL[vul])
    # DDS declares char parScore[2][16] / parContractsString[2][128]; read the raw bytes
    # (endplay's ctypes struct has the dimensions transposed, the layout is the same).
    sc = ctypes.string_at(ctypes.addressof(res.parScore), 32)
    pc = ctypes.string_at(ctypes.addressof(res.parContractsString), 256)
    scores = [sc[k * 16:(k + 1) * 16].split(b'\0')[0].decode().strip() for k in (0, 1)]
    contracts = [pc[k * 128:(k + 1) * 128].split(b'\0')[0].decode().strip() for k in (0, 1)]
    return scores, contracts


def _format_contract(token, dd_bytes):
    """One DDS par contract ('W 123H', 'EW 2Sx') in tag notation ('W 1H+2', 'EW 2SX-1'):
    the cheapest listed level, its result from the DD table, an uppercase X. A token
    that does not parse is passed through, as Brill's DdsParFormat does."""
    seats, _, rest = token.partition(' ')
    d = 0
    while d < len(rest) and rest[d].isdigit():
        d += 1
    if not seats or d == 0:
        return token
    level = int(rest[0])
    strain = rest[d:].rstrip('xX').upper()
    redoubled = rest[d:].lower().endswith('xx')
    doubled = not redoubled and rest[d:].lower().endswith('x')
    if strain not in _DENOM_INDEX or seats[0] not in _SEAT_INDEX:
        return token
    denom = 'N' if strain in ('N', 'NT') else strain
    tricks = dd_bytes[_SEAT_INDEX[seats[0]] * 5 + _DENOM_INDEX[denom]]
    offset = tricks - (6 + level)
    res = '=' if offset == 0 else (f'+{offset}' if offset > 0 else f'{offset}')
    return f"{seats} {level}{denom}{'XX' if redoubled else 'X' if doubled else ''}{res}"


def _format_contracts(dds_contracts, dd_bytes):
    """A DDS par contract string ('NS:NS 2H,NS 3C') as a list in tag notation, cheapest
    first: by level, then C/D/H/S/N."""
    text = dds_contracts.split(':', 1)[1] if ':' in dds_contracts else dds_contracts
    parts = [_format_contract(t.strip(), dd_bytes) for t in text.split(',') if t.strip()]

    def key(c):
        _, _, rest = c.partition(' ')
        return (int(rest[0]) if rest[:1].isdigit() else 9, _DENOM_RANK.get(rest[1:2], 9))
    return sorted(parts, key=key)


def _side_declaring(contract):
    seats = contract.split(' ')[0] if contract else ''
    if seats and all(c in 'NS' for c in seats):
        return 'NS'
    if seats and all(c in 'EW' for c in seats):
        return 'EW'
    return None


def par_tags(dd_bytes, vulnerable, dealer=None):
    """[OptimumScore] and [ParContract] VALUES for one board, from its deal's tricks.

    Par is DDS's Par() on the stored table at the board's own vulnerability -- the
    computation Brill's ParAnalyzer/ConsoleApp make (DdsNativeSolver.CalculatePar) -- and
    is written the way BBA's generator and Brill's DdsParFormat write it:
      - every contract that scores par, cheapest first, "; " between: "NS 2H=; NS 3C=";
      - the declaring SEAT when only one partner makes it ("W 1H+2"), else the side;
      - the lowest level that scores par, with its result; a doubled sacrifice "X";
      - the score from the DECLARING side's view, so a sacrifice is negative:
        "EW -100" with "EW 2SX-1".
    Where par depends on which side bids first (both sides make 1NT) DDS gives each
    side its own par, and both are written, as BBA does: "NS 90 EW 90" with "NE 1N=".
    A passed-out par is "NS 0" / "NSEW PASS". `dealer` is accepted for the callers'
    convenience and not used: Par() does not depend on it.
    """
    (s_ns, s_ew), (c_ns, c_ew) = _dds_par(dd_bytes, vulnerable)
    ns = int(s_ns.split()[1])
    ew = int(s_ew.split()[1])
    ns_list = _format_contracts(c_ns, dd_bytes)
    if not ns_list:
        return 'NS 0', 'NSEW PASS'
    if ew != -ns:
        # Side-dependent: each side has its own par.
        ew_list = _format_contracts(c_ew, dd_bytes)
        if len(ns_list) == 1 and len(ew_list) == 1                 and ns_list[0].split(' ', 1)[1] == ew_list[0].split(' ', 1)[1]:
            contract = ns_list[0].split(' ')[0] + ew_list[0].split(' ')[0] + ' ' + ns_list[0].split(' ', 1)[1]
        else:
            contract = '; '.join(ns_list + [c for c in ew_list if c not in ns_list])
        return f'NS {ns} EW {ew}', contract
    contract = '; '.join(ns_list)
    return (f'EW {-ns}' if _side_declaring(ns_list[0]) == 'EW' else f'NS {ns}'), contract
