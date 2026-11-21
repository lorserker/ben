"""bridge-solver as BEN's double-dummy solver, through its DDS-compatible C interface.

bridge-solver (https://github.com/BSalita/bridge-solver, a fork of macroxue/bridge-solver)
ships a single-file library with DDS 2.9's C API (SolveAllBoards, SolveBoardPBN,
CalcDDtablePBN, Par, ...). On BEN's recorded workload it gives the same results as dds3
about 2x faster per core (2026-11, see src/ddsolver/README.md).

BEN loads its own copy, bin/bsolver/<platform>-<arch>/<lib>, and not the dds library
beside BGADLL: the library's thread count is one global setting, and BGADLL sets it to 8.

BEN_BSOLVER_LIB=<path> loads a library from another place.
"""
import ctypes as C
import os
import platform
import sys
import threading

# The DDS 2.9 batch structs hold this many boards (MAXNOOFBOARDS in dll.h).
MAX_BOARDS = 5000


class FutureTricks(C.Structure):
    _fields_ = [("nodes", C.c_int), ("cards", C.c_int), ("suit", C.c_int * 13),
                ("rank", C.c_int * 13), ("equals", C.c_int * 13), ("score", C.c_int * 13)]


class DealPBN(C.Structure):
    _fields_ = [("trump", C.c_int), ("first", C.c_int), ("currentTrickSuit", C.c_int * 3),
                ("currentTrickRank", C.c_int * 3), ("remainCards", C.c_char * 80)]


class BoardsPBN(C.Structure):
    _fields_ = [("noOfBoards", C.c_int), ("deals", DealPBN * MAX_BOARDS),
                ("target", C.c_int * MAX_BOARDS), ("solutions", C.c_int * MAX_BOARDS),
                ("mode", C.c_int * MAX_BOARDS)]


class SolvedBoards(C.Structure):
    _fields_ = [("noOfBoards", C.c_int), ("solvedBoard", FutureTricks * MAX_BOARDS)]


class DdTableDealPBN(C.Structure):
    _fields_ = [("cards", C.c_char * 80)]


class DdTableResults(C.Structure):
    _fields_ = [("resTable", (C.c_int * 4) * 5)]


class ParResults(C.Structure):
    _fields_ = [("parScore", (C.c_char * 16) * 2), ("parContractsString", (C.c_char * 128) * 2)]


def _platform_dir():
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
    if sys.platform == "win32":
        return f"windows-{arch}", "bsolver-dds.dll"
    if sys.platform == "darwin":
        return f"macos-{arch}", "libbsolver-dds.dylib"
    return f"linux-{arch}", "libbsolver-dds.so"


def find_library():
    """The path of BEN's bridge-solver library, or None when there is none for this platform."""
    override = os.getenv("BEN_BSOLVER_LIB")
    if override:
        return override
    sub, name = _platform_dir()
    here = os.path.dirname(os.path.abspath(__file__))
    # The same layouts as the vendored dds3: the repository, Docker's flattened /app,
    # BEN_HOME and the current directory; PyInstaller unpacks bin/ beside the program.
    roots = [os.path.join(here, "..", "..", "bin"), os.path.join(here, "..", "bin")]
    if os.getenv("BEN_HOME"):
        roots.append(os.path.join(os.getenv("BEN_HOME"), "bin"))
    roots.append(os.path.join(os.getcwd(), "bin"))
    if getattr(sys, "frozen", False):
        roots.append(os.path.join(os.path.dirname(sys.executable), "bin"))
        if hasattr(sys, "_MEIPASS"):
            roots.append(os.path.join(sys._MEIPASS, "bin"))
    for root in roots:
        path = os.path.abspath(os.path.join(root, "bsolver", sub, name))
        if os.path.isfile(path):
            return path
    return None


class BridgeSolver:
    """The calls DDSolver needs, on one loaded copy of the library."""

    def __init__(self, path, threads):
        self.path = path
        self.lib = C.CDLL(path)
        self.lib.SetMaxThreads(int(threads))
        self.lib.SolveAllBoards.argtypes = [C.POINTER(BoardsPBN), C.POINTER(SolvedBoards)]
        self.lib.CalcDDtablePBN.argtypes = [DdTableDealPBN, C.POINTER(DdTableResults)]
        self.lib.Par.argtypes = [C.POINTER(DdTableResults), C.POINTER(ParResults), C.c_int]
        self.lib.ErrorMessage.argtypes = [C.c_int, C.c_char_p]
        # The batch structs are about 1.6 MB; each calling thread keeps its own pair.
        self._local = threading.local()

    def error_message(self, code):
        text = C.create_string_buffer(80)
        self.lib.ErrorMessage(code, text)
        return text.value.decode(errors="replace")

    def _buffers(self):
        buf = getattr(self._local, "buf", None)
        if buf is None:
            buf = self._local.buf = (BoardsPBN(), SolvedBoards())
        return buf

    def solve_boards(self, hands_pbn, trump, first, trick_suit, trick_rank, solutions, mode):
        """dds3.solve_board_pbn for every deal of hands_pbn, solved on the library's threads.
        Returns one dict per deal with dds3's keys: cards, suit, rank, equals, score."""
        boards, solved = self._buffers()
        out = []
        for start in range(0, len(hands_pbn), MAX_BOARDS):
            chunk = hands_pbn[start:start + MAX_BOARDS]
            boards.noOfBoards = len(chunk)
            for i, pbn in enumerate(chunk):
                d = boards.deals[i]
                d.trump = trump
                d.first = first
                d.currentTrickSuit[:] = trick_suit
                d.currentTrickRank[:] = trick_rank
                d.remainCards = pbn.encode("ascii")
                boards.target[i] = -1
                boards.solutions[i] = solutions
                boards.mode[i] = mode
            rc = self.lib.SolveAllBoards(C.byref(boards), C.byref(solved))
            if rc != 1:
                raise RuntimeError(f"bridge-solver SolveAllBoards returned {rc}: {self.error_message(rc)}")
            for i in range(len(chunk)):
                ft = solved.solvedBoard[i]
                n = ft.cards
                out.append({"cards": n, "suit": ft.suit[:n], "rank": ft.rank[:n],
                            "equals": ft.equals[:n], "score": ft.score[:n]})
        return out

    def par(self, deal_pbn, vulnerable):
        """The par scores (["NS <score>", "EW <score>"]) of a full deal, as dds3's
        calc_all_tables_pbn reports them. vulnerable: 0 none, 1 both, 2 NS, 3 EW."""
        table = DdTableResults()
        rc = self.lib.CalcDDtablePBN(DdTableDealPBN(deal_pbn.encode("ascii")), C.byref(table))
        if rc != 1:
            raise RuntimeError(f"bridge-solver CalcDDtablePBN returned {rc}: {self.error_message(rc)}")
        pres = ParResults()
        rc = self.lib.Par(C.byref(table), C.byref(pres), int(vulnerable))
        if rc != 1:
            raise RuntimeError(f"bridge-solver Par returned {rc}: {self.error_message(rc)}")
        return [pres.parScore[0].value.decode(), pres.parScore[1].value.decode()]
