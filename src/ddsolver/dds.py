#! /usr/bin/python

"""Copyright 2014-2015 Foppe HEMMINGA

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import sys
import os.path

from ctypes import cdll, Structure, c_int, c_uint, c_char, c_char_p, POINTER

BEN_HOME = os.getenv('BEN_HOME') or '..'
BIN_FOLDER = os.path.join(BEN_HOME, 'bin')
if sys.platform == 'win32':
    DDS_LIB = 'dds.dll'
elif sys.platform == 'darwin':
    DDS_LIB = 'darwin/libdds.so'
else:
    DDS_LIB = 'libdds.so'

DDS_PATH = os.path.join(BIN_FOLDER, DDS_LIB)

dds = cdll.LoadLibrary(DDS_PATH)
sys.stderr.write(f"Loaded lib { os.path.basename(DDS_PATH)}\n")

DDS_VERSION = 20700    

DDS_HANDS = 4
DDS_SUITS = 4
DDS_STRAINS = 5

MAXNOOFBOARDS = 200

MAXNOOFHANDS = 32

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
    -203: "Chunk size is less than 1"
}

# Function to translate an error code to the corresponding message
def get_error_message(error_code):
    return error_messages.get(error_code, "Unknown error code")

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
                ("deals", deal * MAXNOOFBOARDS),
                ("target", c_int * MAXNOOFBOARDS),
                ("solutions", c_int * MAXNOOFBOARDS),
                ("mode", c_int * MAXNOOFBOARDS)]

class boardsPBN(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("deals", dealPBN * MAXNOOFBOARDS),
                ("target", c_int * MAXNOOFBOARDS),
                ("solutions", c_int * MAXNOOFBOARDS),
                ("mode", c_int * MAXNOOFBOARDS)]

class solvedBoards(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("solvedBoards", futureTricks * MAXNOOFBOARDS)]

class ddTableDeal(Structure):
    _fields_ = [("cards", c_uint * DDS_HANDS * DDS_SUITS)]

class ddTableDeals(Structure):
    _fields_ = [("noOfTables", c_int),
                ("deals", ddTableDeal * (MAXNOOFBOARDS >> 2))]

class ddTableDealPBN(Structure):
    _fields_ = [("cards", c_char * 80)]

class ddTableDealsPBN(Structure):
    _fields_ = [("noOfTables", c_int),
                ("deals", ddTableDealPBN * (MAXNOOFBOARDS >> 2))]

class ddTableResults(Structure):
    _fields_ = [("resTable", c_int * DDS_STRAINS * DDS_HANDS)]

class ddTablesRes(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("results", ddTableResults * (MAXNOOFBOARDS >> 2))]

class parResults(Structure):
    """     index = 0 is NS view and index = 1 
     is EW view. By 'view' is here meant 
     which side that starts the bidding."""
    _fields_ = [("parScore", ((c_char * 16) * 2)),
                ("parContractsString", ((c_char * 128) * 2))]

class allParResults(Structure):
    _fields_ = [("presults", parResults * (MAXNOOFBOARDS // 20))]

class parResultsDealer(Structure):
    _fields_ = [("number", c_int),
                ("score", c_int),
                ("contracts", c_char * 10 * 10)]

class contractType(Structure):
    """     undertricks: 0 = make; 1-13 = sacrifice
    overTricks: 0-3; e.g. 1 for 4S + 1
    level: 1-7
    denom: 0 = No Trumps, 1 = trump Spades, 2 = trump Hearts
        3 = trump Diamonds, 4 = trump Clubs
    seats: One of the cases N, E, S, W, NS, EW;
        0 = N, 1 = E, 2 = S, 3 = W, 4 = NS, 5 = EW"""
    _fields_ = [("underTricks", c_int),
                ("overTricks", c_int),
                ("level", c_int),
                ("denom", c_int),
                ("seats", c_int)]

class parResultsMaster(Structure):
    """     score: Sign acccording to NS iew
    number: Number of contracts giving the par score"""
    _fields_ = [("score", c_int),
                ("number", c_int),
                ("contracts", contractType * 10)]

class parTextResults(Structure):
    """     parText: Short text for par information, e.g.
        Par -110: EW 2S  EW 2D+1
    equal: TRUE in the normal case when it does not matter who
        starts the bidding. Otherwise, FALSE."""
    _fields_ = [("parTextResults", c_char * 2 * 128),
                ("equal", c_int)]

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
                ("plays", playTraceBin * (MAXNOOFBOARDS // 10))]

class playTracesPBN(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("plays", playTracePBN * (MAXNOOFBOARDS // 10))]

class solvedPlays(Structure):
    _fields_ = [("noOfBoards", c_int),
                ("solved", solvedPlay * (MAXNOOFBOARDS // 10))]

SetMaxThreads = dds.SetMaxThreads
"""int userThreads"""
SetMaxThreads.argtypes = [c_int]
SetMaxThreads.restype = None

FreeMemory = dds.FreeMemory
FreeMemory.argtypes = None
FreeMemory.restype = None

SolveBoard = dds.SolveBoard
"""deal dl
int target
int solutions
int mode,
pointer to struct futureTricks * futp
int threadIndex"""
SolveBoard.argtypes = [deal, c_int, c_int, c_int, POINTER(futureTricks), c_int]
SolveBoard.restype = c_int

SolveBoardPBN = dds.SolveBoardPBN
"""dealPBN dlpbn
int target
int solutions
int mode
pointer to struct futureTricks * futp
int thrId"""
SolveBoardPBN.argtypes = [dealPBN, c_int, c_int, c_int, \
    POINTER(futureTricks), c_int]
SolveBoardPBN.restype = c_int

CalcDDtable = dds.CalcDDtable
"""struct ddTableDeal tableDeal
pointer to struct ddTableResults * tablep"""
CalcDDtable.argtypes = [ddTableDeal, POINTER(ddTableResults)]
CalcDDtable.restype = c_int

CalcDDtablePBN = dds.CalcDDtablePBN
"""srtuct ddTableDealPBN tableDealPBN
pointer to struct ddTableResults * tablep"""
CalcDDtablePBN.argtypes = [ddTableDealPBN, POINTER(ddTableResults)]
CalcDDtablePBN.restype = c_int

CalcAllTables = dds.CalcAllTables
"""pointer to struct dd TableDeals * dealsp
int mode
int trumpFilter[DDS_STRAINS]
poiter to struct ddTablesRes * resp
pointer to struct allParResults'* presp"""
CalcAllTables.argtypes = [POINTER(ddTableDeals), c_int * DDS_STRAINS, \
    c_int, POINTER(ddTablesRes), POINTER(allParResults)]
CalcAllTables.restype = c_int

CalcAllTablesPBN = dds.CalcAllTablesPBN
"""pointer to struct ddTableDealsPBN * dealsp
int mode
int trumpFilter[DDS_STRINS]
pointer to struct ddTablesRes *resp
pointer to struct allParResults * presp"""
CalcAllTablesPBN.argtypes = [POINTER(ddTableDealsPBN), c_int * DDS_STRAINS, \
    c_int, POINTER(ddTablesRes), POINTER(allParResults)]
CalcAllTablesPBN.restype = c_int

SolveAllBoards = dds.SolveAllBoards
"""pointer to struct boardsPBN * bop
pointer to struct solvedBoards * solvedp"""
SolveAllBoards.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards)]
SolveAllBoards.restype = c_int

SolveAllChunks = dds.SolveAllChunks
"""pointer to struct boardsPBN * bop
pointer to struct solvedBoards * solvedP
int chunkSize"""
SolveAllChunks.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards), c_int]
SolveAllChunks.restype = c_int

solveAllChunksBin = dds.SolveAllChunksBin
"""pointer to struct boards * bop
pointer to struct solvedBoards * solvedp
int chunkSize"""
solveAllChunksBin.argtypes = [POINTER(boards), POINTER(solvedBoards), c_int]
solveAllChunksBin.restype = c_int

solveAllChunksPBN = dds.SolveAllChunksPBN
"""pointer to struct boardsPBN * bop
pointer to struct solvedBoards * solvedp
int chunkSize"""
solveAllChunksPBN.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards), c_int]
solveAllChunksPBN.restype = c_int

SolveAllChunksPBN = dds.SolveAllChunksPBN
"""pointer to struct boardsPBN * bop
pointer to struct solvedBoards * solvedp
int chunkSize"""
SolveAllChunksPBN.argtypes = [POINTER(boardsPBN), POINTER(solvedBoards), c_int]
SolveAllChunksPBN.restype = c_int

Par = dds.Par
"""pointer to struct ddTableResults * tablep
pointer to struct parResults * presp
int vulnerable"""
Par.argtypes = [POINTER(ddTableResults), POINTER(parResults), c_int]
Par.restype = c_int

CalcPar = dds.CalcPar
"""struct ddTableDeal
int ulnerable
pointer to struct ddTablesRes * tablep
pointer to parResults * presp"""
CalcPar.argtypes = [ddTableDeal, c_int, POINTER(ddTableResults), POINTER(parResults)]
CalcPar.restype = c_int

CalcPar = dds.CalcPar
"""struct ddTableDeal tableDeal
int vulnerable
pointer to struct ddTableResults * tablep
pointer to parResults * presp"""
CalcPar.argtypes = [ddTableDeal, c_int, POINTER(ddTableResults), POINTER(parResults)]
CalcPar.restype = c_int

CalcParPBN = dds.CalcParPBN
"""struct ddTableDealPBN tableDealPBN
pointer tostruct ddTableResults * tablep
int vulnerable
pointer to struct parResults * presp"""
CalcParPBN.argtypes = [ddTableDealPBN, POINTER(ddTableResults), c_int, POINTER(parResults)]
CalcParPBN.restype = c_int

SidesPar = dds.SidesPar
"""pointer to struct ddTableResults * tablep,
array struct parResultsDealer sidesRes[2],
int vulnerable"""
SidesPar.argtypes = [POINTER(ddTableResults), parResultsDealer * 2, c_int]
SidesPar.restypes = c_int

DealerPar = dds.DealerPar
"""pointer to struct ddTableResults * tablep
pointer to struct parResultsDealer * presp
int dealer
int vulnerable"""
DealerPar.argtypes = [POINTER(ddTableResults), POINTER(parResultsDealer), c_int, c_int]
DealerPar.restype = c_int

DealerParBin = dds.DealerParBin
"""pointer to struct ddTableResults * tablep
pointer to struct parResultsMaster * presp
int dealer
int vulnerable"""
DealerParBin.argtypes = [POINTER(ddTableResults), POINTER(parResultsMaster), c_int, c_int]
DealerParBin.restype = c_int

SidesParBin = dds.SidesParBin
"""pointer to struct ddTableResults * tablep
array struct parResultsMaster sidesRes[2]
int vulnerable"""
SidesParBin.argtypes = [POINTER(ddTableResults), parResultsMaster * 2, c_int]
SidesParBin.restype = c_int

ConvertToDealerTextFormat = dds.ConvertToDealerTextFormat
"""pointer to struct parResultsMaster *pres
pointer to char *resp"""
ConvertToDealerTextFormat.argtypes = [POINTER(parResultsMaster), c_char_p]
ConvertToDealerTextFormat.restype = c_int

ConvertToSidesTextFormat = dds.ConvertToSidesTextFormat
"""pointer to struct parResultsMaster * pres, 
pointer to struct parTextResults * resp"""
ConvertToSidesTextFormat.argtypes = [POINTER(parResultsMaster), POINTER(parTextResults)]
ConvertToSidesTextFormat.restype = c_int

AnalysePlayBin = dds.AnalysePlayBin
"""struct deal dl
struct playTraceBin play
pointer to struct solvedPlay * solved
int thrId"""
AnalysePlayBin.argtypes = [deal, playTraceBin, POINTER(solvedPlay), c_int]
AnalysePlayBin.restype = c_int

AnalysePlayPBN = dds.AnalysePlayPBN
"""struct dealPBN dlPBN
struct playTracePBN playPBN                                 
pointer to struct solvedPlay * solvedp
int thrId"""
AnalysePlayPBN.argtypes = [dealPBN, playTracePBN, POINTER(solvedPlay), c_int]
AnalysePlayPBN.restype = c_int

AnalyseAllPlaysBin = dds.AnalyseAllPlaysBin
"""pointer to struct boards * bop
pointer to struct playTracesBin * plp
pointer to struct solvedPlays * solvedp
int chunkSize"""
AnalyseAllPlaysBin.argtypes = [POINTER(boards), POINTER(playTracesBin), POINTER(solvedPlays), c_int]
AnalyseAllPlaysBin.restype = c_int

AnalyseAllPlaysPBN = dds.AnalyseAllPlaysPBN
"""pointer to struct boardsPBN * bopPBN
pointer to struct playTracesPBN * plpPBN
pointer to struct solvedPlays * solvedp
int chunkSize"""
AnalyseAllPlaysPBN.argtypes = [POINTER(boardsPBN), POINTER(playTracesPBN), POINTER(solvedPlays), c_int]
AnalyseAllPlaysPBN.restype = c_int
