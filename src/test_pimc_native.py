"""
Diagnostic script to test native BGADLL directly.
Checks which DDS backend is active and runs the problematic 3NT position.
"""
import ctypes
import os
import sys

# Add the pimc directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pimc.BGADLL_Native import (
    _get_lib, _read_string, is_available,
    NativePIMC, NativePIMCDef, NativeHand, NativePlay, NativeCard,
    NativeConstraints, NativeExtensions, NativeMacros
)

print(f"Native BGADLL available: {is_available()}")

lib = _get_lib()

# Check DDS backend
if hasattr(lib, 'bga_dds_backend'):
    lib.bga_dds_backend.argtypes = []
    lib.bga_dds_backend.restype = ctypes.c_void_p
    ptr = lib.bga_dds_backend()
    backend = _read_string(ptr)
    print(f"DDS backend: {backend}")
else:
    print("bga_dds_backend not available (old DLL)")

# Create PIMC
pimc = NativePIMC(maxThreads=12, verbose=False)
print(f"PIMC version: {pimc.version()}")

# Set up the problematic position
# 3NT, trick 3: DT(W) DA(N) D3(E) - South to play from DQ84
# Hands at trick 3 (after tricks 1-2 played):
# North: SA9873.-.D96.CK52 (H4,HJ played; DA in current trick)
# South: SKT.HT9.DQ84.CAQ98 (HA,H3 played)

fullDeck = NativeExtensions.Parse("AKQJT98765432.AKQJT98765432.AKQJT98765432.AKQJT98765432")
north = NativeExtensions.Parse("A9873..96.K52")
south = NativeExtensions.Parse("KT.T9.Q84.AQ98")

# Current trick: DT DA D3
current_trick = NativePlay()
current_trick.Add(NativeCard("TD"))
current_trick.Add(NativeCard("AD"))
current_trick.Add(NativeCard("3D"))

# Previous tricks: H6 H4 HQ HA | H3 HK HJ H2
previous_tricks = NativePlay()
for card_str in ["6H", "4H", "QH", "AH", "3H", "KH", "JH", "2H"]:
    previous_tricks.Add(NativeCard(card_str))

# Opposing cards
played_cards = NativePlay()
for card_str in ["TD", "AD", "3D", "6H", "4H", "QH", "AH", "3H", "KH", "JH", "2H"]:
    played_cards.Add(NativeCard(card_str))

oppos_hand = fullDeck.Except(north).Except(south)
for card_str in ["TD", "AD", "3D", "6H", "4H", "QH", "AH", "3H", "KH", "JH", "2H"]:
    oppos_hand.Remove(NativeCard(card_str))

print(f"North: {north}")
print(f"South: {south}")
print(f"Opposing: {oppos_hand} ({oppos_hand.Count} cards)")
print(f"Current trick: {current_trick.ListAsString()}")
print(f"Previous tricks: {previous_tricks.ListAsString()}")

east = NativeConstraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)
west = NativeConstraints(0, 13, 0, 13, 0, 13, 0, 13, 0, 37)

pimc.SetupEvaluation(
    [north, south], oppos_hand, current_trick, previous_tricks,
    [east, west], NativeMacros.Player.South, 200, False, False
)

pimc.Evaluate(NativeMacros.Trump.No)
pimc.AwaitEvaluation(10000)

import time
time.sleep(0.1)

print(f"Playouts: {pimc.Playouts}")
print(f"Combinations: {pimc.Combinations}")
print(f"Examined: {pimc.Examined}")

legal_moves = pimc.LegalMoves
print(f"Legal moves: {legal_moves}")

pimc.Output.SortResults()
for card in legal_moves:
    results = pimc.Output.GetTricksWithWeights(card)
    count = len(results)
    if count > 0:
        total_weight = sum(r.weight for r in results)
        tricks_avg = sum(r.tricks * r.weight for r in results) / total_weight if total_weight > 0 else 0
        print(f"  {card}: tricks={tricks_avg:.4f}, count={count}, total_weight={total_weight:.3f}")
    else:
        print(f"  {card}: no results")

# Check D4 vs DQ
tricks_d4 = sum(r.tricks * r.weight for r in pimc.Output.GetTricksWithWeights("4D")) / max(sum(r.weight for r in pimc.Output.GetTricksWithWeights("4D")), 0.001)
tricks_dq = sum(r.tricks * r.weight for r in pimc.Output.GetTricksWithWeights("QD")) / max(sum(r.weight for r in pimc.Output.GetTricksWithWeights("QD")), 0.001)
print(f"\nD4 tricks: {tricks_d4:.4f}")
print(f"DQ tricks: {tricks_dq:.4f}")
print(f"D4 > DQ: {tricks_d4 > tricks_dq} {'CORRECT' if tricks_d4 > tricks_dq else 'WRONG - DQ should not be preferred'}")
