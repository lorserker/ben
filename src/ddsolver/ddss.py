#! /usr/bin/python

"""Ctypes bindings for DDSS (Double Dummy Super Solver).

DDSS is a high-performance fork of DDS 2.9.0 by Robert Salita.
https://github.com/BSalita/ddss

Key differences from dds.py:
  - Dynamic structs: boards, solvedBoards, etc. use pointers, not fixed arrays.
    Use AllocBoardsPBN(n)/FreeBoardsPBN() etc. to allocate/free.
  - CalcAllTablesPBNx: dynamic batch DD table API using plain arrays
    of small, fixed-size per-deal structs (ddTableDealPBN, ddTableResults,
    parResults). No large MAXNOOFTABLES-sized wrapper structs needed.
  - GetDDSInfo: query library version and system info
  - Library file: ddss.dll / libddss.so / libddss.dylib

Build DDSS from source and rename the output library:
  Windows: ddss.dll   -> bin/ddss.dll
  Linux:   libdds.so  -> bin/libddss.so
  macOS:   libdds.dylib -> bin/darwin/libddss.dylib
"""

import sys
import os.path

from ctypes import (
    cdll, Structure, c_int, c_uint, c_char, c_char_p, c_bool,
    POINTER, pointer
)

# --- Path resolution (same logic as dds.py) ---

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(script_dir, "../..")
sys.path.append(parent_dir)

if "src" in script_dir and "dds" in script_dir:
    BIN_FOLDER = parent_dir + os.path.sep + 'bin'
else:
    BEN_HOME = os.getenv('BEN_HOME') or '.'
    if BEN_HOME == '.':
        BIN_FOLDER = 'bin'
    else:
        BIN_FOLDER = os.path.join(BEN_HOME, 'bin')

# --- Library loading ---

if sys.platform == 'win32':
    DDSS_LIB = 'ddss.dll'
elif sys.platform == 'darwin':
    DDSS_LIB = 'darwin/libddss.dylib'
else:
    DDSS_LIB = 'libddss.so'

DDSS_AVAILABLE = False
_ddss = None

try:
    DDSS_PATH = os.path.join(BIN_FOLDER, DDSS_LIB)
    _ddss = cdll.LoadLibrary(DDSS_PATH)
    DDSS_AVAILABLE = True
except Exception:
    if sys.platform == 'darwin':
        try:
            DDSS_PATH = 'libddss.dylib'
            _ddss = cdll.LoadLibrary(DDSS_PATH)
            DDSS_AVAILABLE = True
        except Exception:
            pass
    elif sys.platform != 'win32':
        try:
            DDSS_PATH = 'libddss.so'
            _ddss = cdll.LoadLibrary(DDSS_PATH)
            DDSS_AVAILABLE = True
        except Exception:
            pass

if DDSS_AVAILABLE:
    sys.stderr.write(f"Loaded DDSS library from {DDSS_PATH}\n")
else:
    sys.stderr.write(f"DDSS library not found ({DDSS_LIB}), DDSSolver will fall back to DDSolver\n")

# --- Constants ---

DDS_VERSION = 30000

DDS_HANDS = 4
DDS_SUITS = 4
DDS_STRAINS = 5

# Python-side chunking limit for very large batches.
# The DLL itself has no hard limit (structs are pointer-based / dynamic).
MAXNOOFBOARDS = 5000

# --- Error messages ---

error_messages = {
    1: "Success",
    -1: "General error",
    -2: "Zero cards",
    -3: "Target exceeds number of tricks",
    -4: "Cards duplicated",
    -5: "Target less than -1",
    -7: "Target is higher than 13",
    -8: "Solutions parameter is less than 1",
    -9: "Solutions parameter is higher than 3",
    -10: "Too many cards",
    -12: "currentTrickSuit or currentTrickRank has wrong data",
    -13: "Played card also remains in a hand",
    -14: "Wrong number of remaining cards in a hand",
    -15: "Thread index is not 0 .. maximum",
    -16: "Mode parameter is less than 0",
    -17: "Mode parameter is higher than 2",
    -18: "Trump is not in 1 .. 4",
    -19: "First is not in 0 .. 2",
    -98: "AnalysePlay input error",
    -99: "PBN string error",
    -101: "Too many boards requested",
    -102: "Could not create threads",
    -103: "Something failed waiting for thread to end",
    -201: "Denomination filter vector has no entries",
    -202: "Too many DD tables requested",
    -203: "Chunk size is less than 1",
    -301: "Chunk size error",
}


def get_error_message(error_code):
    return error_messages.get(error_code, "Unknown error code")


# =====================================================================
# Struct definitions — must match DDSS include/dll.h exactly
# =====================================================================

class futureTricks(Structure):
    _fields_ = [("nodes", c_int),
                ("cards", c_int),
                ("suit", c_int * 13),
                ("rank", c_int * 13),
                ("equals", c_int * 13),
                ("score", c_int * 13)]

class deal(Structure):
    _fields_ = [("trump", c_int),
                ("first", c_int),
                ("currentTrickSuit", c_int * 3),
                ("currentTrickRank", c_int * 3),
                ("remainCards", c_int * DDS_HANDS * DDS_SUITS)]

class dealPBN(Structure):
    _fields_ = [("trump", c_int),
                ("first", c_int),
                ("currentTrickSuit", c_int * 3),
                ("currentTrickRank", c_int * 3),
                ("remainCards", c_char * 80)]

class boards(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("deals", POINTER(deal)),
                ("target", POINTER(c_int)),
                ("solutions", POINTER(c_int)),
                ("mode", POINTER(c_int))]

class boardsPBN(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("deals", POINTER(dealPBN)),
                ("target", POINTER(c_int)),
                ("solutions", POINTER(c_int)),
                ("mode", POINTER(c_int))]

class solvedBoards(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("solvedBoard", POINTER(futureTricks))]

class ddTableDeal(Structure):
    _fields_ = [("cards", c_uint * DDS_HANDS * DDS_SUITS)]

class ddTableDealPBN(Structure):
    _fields_ = [("cards", c_char * 80)]

class ddTableResults(Structure):
    _fields_ = [("resTable", c_int * DDS_STRAINS * DDS_HANDS)]

class parResults(Structure):
    _fields_ = [("parScore", ((c_char * 16) * 2)),
                ("parContractsString", ((c_char * 128) * 2))]

# Note: The large wrapper structs (ddTableDeals, ddTableDealsPBN,
# ddTablesRes, allParResults) are NOT defined here. Use CalcAllTablesPBNx
# instead — it takes plain arrays of the small per-deal structs above.

class parResultsDealer(Structure):
    _fields_ = [("number", c_int),
                ("score", c_int),
                ("contracts", c_char * 10 * 10)]

class contractType(Structure):
    _fields_ = [("underTricks", c_int),
                ("overTricks", c_int),
                ("level", c_int),
                ("denom", c_int),
                ("seats", c_int)]

class parResultsMaster(Structure):
    _fields_ = [("score", c_int),
                ("number", c_int),
                ("contracts", contractType * 10)]

class parTextResults(Structure):
    _fields_ = [("parTextResults", c_char * 2 * 128),
                ("equal", c_bool)]

class playTraceBin(Structure):
    _fields_ = [("number", c_int),
                ("suit", c_int * 52),
                ("rank", c_int * 52)]

class playTracePBN(Structure):
    _fields_ = [("number", c_int),
                ("cards", c_char * 106)]

class solvedPlay(Structure):
    _fields_ = [("number", c_int),
                ("tricks", c_int * 53)]

class playTracesBin(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("plays", POINTER(playTraceBin))]

class playTracesPBN(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("plays", POINTER(playTracePBN))]

class solvedPlays(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("solved", POINTER(solvedPlay))]

class DDSInfo(Structure):
    _fields_ = [("major", c_int),
                ("minor", c_int),
                ("patch", c_int),
                ("versionString", c_char * 10),
                ("system", c_int),
                ("numBits", c_int),
                ("compiler", c_int),
                ("constructor", c_int),
                ("numCores", c_int),
                ("threading", c_int),
                ("noOfThreads", c_int),
                ("threadSizes", c_char * 128),
                ("systemString", c_char * 1024)]


# =====================================================================
# Function bindings — only bound when the library is available
# =====================================================================

if DDSS_AVAILABLE:

    # --- Thread management ---

    SetMaxThreads = _ddss.SetMaxThreads
    SetMaxThreads.argtypes = [c_int]
    SetMaxThreads.restype = None

    SetThreading = _ddss.SetThreading
    SetThreading.argtypes = [c_int]
    SetThreading.restype = c_int

    SetResources = _ddss.SetResources
    SetResources.argtypes = [c_int, c_int]
    SetResources.restype = None

    FreeMemory = _ddss.FreeMemory
    FreeMemory.argtypes = None
    FreeMemory.restype = None

    # --- Single board solving ---

    SolveBoard = _ddss.SolveBoard
    SolveBoard.argtypes = [deal, c_int, c_int, c_int, POINTER(futureTricks), c_int]
    SolveBoard.restype = c_int

    SolveBoardPBN = _ddss.SolveBoardPBN
    SolveBoardPBN.argtypes = [dealPBN, c_int, c_int, c_int, POINTER(futureTricks), c_int]
    SolveBoardPBN.restype = c_int

    # --- Batch board solving ---

    SolveAllBoards = _ddss.SolveAllBoards
    SolveAllBoards.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards)]
    SolveAllBoards.restype = c_int

    SolveAllBoardsBin = _ddss.SolveAllBoardsBin
    SolveAllBoardsBin.argtypes = [POINTER(boards), POINTER(solvedBoards)]
    SolveAllBoardsBin.restype = c_int

    SolveAllChunks = _ddss.SolveAllChunks
    SolveAllChunks.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards), c_int]
    SolveAllChunks.restype = c_int

    SolveAllChunksBin = _ddss.SolveAllChunksBin
    SolveAllChunksBin.argtypes = [POINTER(boards), POINTER(solvedBoards), c_int]
    SolveAllChunksBin.restype = c_int

    SolveAllChunksPBN = _ddss.SolveAllChunksPBN
    SolveAllChunksPBN.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards), c_int]
    SolveAllChunksPBN.restype = c_int

    # --- DD table calculation ---

    CalcDDtable = _ddss.CalcDDtable
    CalcDDtable.argtypes = [ddTableDeal, POINTER(ddTableResults)]
    CalcDDtable.restype = c_int

    CalcDDtablePBN = _ddss.CalcDDtablePBN
    CalcDDtablePBN.argtypes = [ddTableDealPBN, POINTER(ddTableResults)]
    CalcDDtablePBN.restype = c_int

    # --- DDSS dynamic batch DD table API ---
    # Uses plain arrays of small per-deal structs. No compile-time size limits.
    # int CalcAllTablesPBNx(int numDeals, ddTableDealPBN dealCards[],
    #                       int mode, int trumpFilter[5],
    #                       ddTableResults results[], parResults par[])

    CalcAllTablesPBNx = _ddss.CalcAllTablesPBNx
    CalcAllTablesPBNx.argtypes = [c_int, POINTER(ddTableDealPBN),
                                  c_int, c_int * DDS_STRAINS,
                                  POINTER(ddTableResults), POINTER(parResults)]
    CalcAllTablesPBNx.restype = c_int

    # --- Par calculation ---

    Par = _ddss.Par
    Par.argtypes = [POINTER(ddTableResults), POINTER(parResults), c_int]
    Par.restype = c_int

    CalcPar = _ddss.CalcPar
    CalcPar.argtypes = [ddTableDeal, c_int, POINTER(ddTableResults), POINTER(parResults)]
    CalcPar.restype = c_int

    CalcParPBN = _ddss.CalcParPBN
    CalcParPBN.argtypes = [ddTableDealPBN, POINTER(ddTableResults), c_int, POINTER(parResults)]
    CalcParPBN.restype = c_int

    SidesPar = _ddss.SidesPar
    SidesPar.argtypes = [POINTER(ddTableResults), parResultsDealer * 2, c_int]
    SidesPar.restype = c_int

    DealerPar = _ddss.DealerPar
    DealerPar.argtypes = [POINTER(ddTableResults), POINTER(parResultsDealer), c_int, c_int]
    DealerPar.restype = c_int

    DealerParBin = _ddss.DealerParBin
    DealerParBin.argtypes = [POINTER(ddTableResults), POINTER(parResultsMaster), c_int, c_int]
    DealerParBin.restype = c_int

    SidesParBin = _ddss.SidesParBin
    SidesParBin.argtypes = [POINTER(ddTableResults), parResultsMaster * 2, c_int]
    SidesParBin.restype = c_int

    ConvertToDealerTextFormat = _ddss.ConvertToDealerTextFormat
    ConvertToDealerTextFormat.argtypes = [POINTER(parResultsMaster), c_char_p]
    ConvertToDealerTextFormat.restype = c_int

    ConvertToSidesTextFormat = _ddss.ConvertToSidesTextFormat
    ConvertToSidesTextFormat.argtypes = [POINTER(parResultsMaster), POINTER(parTextResults)]
    ConvertToSidesTextFormat.restype = c_int

    # --- Play analysis ---

    AnalysePlayBin = _ddss.AnalysePlayBin
    AnalysePlayBin.argtypes = [deal, playTraceBin, POINTER(solvedPlay), c_int]
    AnalysePlayBin.restype = c_int

    AnalysePlayPBN = _ddss.AnalysePlayPBN
    AnalysePlayPBN.argtypes = [dealPBN, playTracePBN, POINTER(solvedPlay), c_int]
    AnalysePlayPBN.restype = c_int

    AnalyseAllPlaysBin = _ddss.AnalyseAllPlaysBin
    AnalyseAllPlaysBin.argtypes = [POINTER(boards), POINTER(playTracesBin),
                                   POINTER(solvedPlays), c_int]
    AnalyseAllPlaysBin.restype = c_int

    AnalyseAllPlaysPBN = _ddss.AnalyseAllPlaysPBN
    AnalyseAllPlaysPBN.argtypes = [POINTER(boardsPBN), POINTER(playTracesPBN),
                                   POINTER(solvedPlays), c_int]
    AnalyseAllPlaysPBN.restype = c_int

    # --- Info ---

    GetDDSInfo = _ddss.GetDDSInfo
    GetDDSInfo.argtypes = [POINTER(DDSInfo)]
    GetDDSInfo.restype = None

    ErrorMessage = _ddss.ErrorMessage
    ErrorMessage.argtypes = [c_int, c_char * 80]
    ErrorMessage.restype = None

    # --- Dynamic allocation helpers ---

    AllocBoards = _ddss.AllocBoards
    AllocBoards.argtypes = [c_int]
    AllocBoards.restype = POINTER(boards)

    FreeBoards = _ddss.FreeBoards
    FreeBoards.argtypes = [POINTER(boards)]
    FreeBoards.restype = None

    AllocBoardsPBN = _ddss.AllocBoardsPBN
    AllocBoardsPBN.argtypes = [c_int]
    AllocBoardsPBN.restype = POINTER(boardsPBN)

    FreeBoardsPBN = _ddss.FreeBoardsPBN
    FreeBoardsPBN.argtypes = [POINTER(boardsPBN)]
    FreeBoardsPBN.restype = None

    AllocSolvedBoards = _ddss.AllocSolvedBoards
    AllocSolvedBoards.argtypes = [c_int]
    AllocSolvedBoards.restype = POINTER(solvedBoards)

    FreeSolvedBoards = _ddss.FreeSolvedBoards
    FreeSolvedBoards.argtypes = [POINTER(solvedBoards)]
    FreeSolvedBoards.restype = None

    AllocPlayTracesBin = _ddss.AllocPlayTracesBin
    AllocPlayTracesBin.argtypes = [c_int]
    AllocPlayTracesBin.restype = POINTER(playTracesBin)

    FreePlayTracesBin = _ddss.FreePlayTracesBin
    FreePlayTracesBin.argtypes = [POINTER(playTracesBin)]
    FreePlayTracesBin.restype = None

    AllocPlayTracesPBN = _ddss.AllocPlayTracesPBN
    AllocPlayTracesPBN.argtypes = [c_int]
    AllocPlayTracesPBN.restype = POINTER(playTracesPBN)

    FreePlayTracesPBN = _ddss.FreePlayTracesPBN
    FreePlayTracesPBN.argtypes = [POINTER(playTracesPBN)]
    FreePlayTracesPBN.restype = None

    AllocSolvedPlays = _ddss.AllocSolvedPlays
    AllocSolvedPlays.argtypes = [c_int]
    AllocSolvedPlays.restype = POINTER(solvedPlays)

    FreeSolvedPlays = _ddss.FreeSolvedPlays
    FreeSolvedPlays.argtypes = [POINTER(solvedPlays)]
    FreeSolvedPlays.restype = None
