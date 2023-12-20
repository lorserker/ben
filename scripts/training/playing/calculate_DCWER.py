import itertools
import sys
sys.path.append('../../../src')
from ddsolver import ddsolver

import shelve
import os
import argparse
import random

os.getcwd()

parser = argparse.ArgumentParser(description="DDOLAR calculator")
parser.add_argument("--db", default="gamedb", help="Database with boards played by BEN")
parser.add_argument("--ns", default="True", help="Calculate for NS")

args = parser.parse_args()

# Convert the string to a boolean
NS = args.ns.lower() == "true"

DB_NAME = os.getcwd() + "/" + args.db

def encode_card(card_str):
    suit_i = 'SHDC'.index(card_str[0])
    card_i = 'AKQJT98765432'.index(card_str[1])
    return suit_i*13 + card_i

def decode_card(card52):
    suit_i = card52 // 13
    card_i = card52 % 13
    return 'SHDC'[suit_i] + 'AKQJT98765432'[card_i]

def get_strain_i(contract):
    return 'NSHDC'.index(contract[1])

def get_suit_i(card):
    return 'NSHDC'.index(card[0])

def get_decl_i(decl):
    return 'WNES'.index(decl)

def validate_lead_gamedb(deal_str, outcome_str, play_str, trick_winners):
    decl_i = get_decl_i(outcome_str[-1])
    
    dd = ddsolver.DDSolver(dds_mode=2)
    strain_i = get_strain_i(outcome_str)
    hands_pbn = ["N:" + deal_str]
    dcer = 0
    dfer = 0
    current_trick52 = []
    cards_to_remove = []
    trick = 1
    #print(hands_pbn)
    #print(decl_i)
    leader_i = (decl_i + 1) % 4

    for i in range(0,47):
        card = play_str[i]['card']
        #print("Now playing: ", card)
        #print(strain_i, leader_i, current_trick52, hands_pbn)
        dd_solved = dd.solve(strain_i, leader_i, current_trick52, hands_pbn)
        #print("DD_Solved:",dd_solved, card)
        max_key = max(dd_solved, key=lambda k: dd_solved[k])
        declarer =  ((i % 4)+leader_i-1) % 2 == (decl_i+1) % 2
        if i > 0:
            if (declarer):
                if (dd_solved[encode_card(card)] != dd_solved[max_key]):
                    #print(card, " was wrong ", encode_card(card), dd_solved[encode_card(card)], dd_solved[max_key])
                    #print("DD_Solved:",dd_solved, card)
                    dcer += 1  
            else:
                if (dd_solved[encode_card(card)] != dd_solved[max_key]):
                    #print(card, " was wrong in defence", encode_card(card), dd_solved[encode_card(card)], dd_solved[max_key])
                    #print("DD_Solved:",dd_solved, card)
                    dfer += 1  
        deal_str = remove_played_card(deal_str, card, i, leader_i)
        hands_pbn = ["N:" + deal_str]           
        cards_to_remove.append(card)
        current_trick52.append(encode_card(card))
        if (len(current_trick52) > 3):
            #print("Trick: ", trick, " done")
            # deal_str = remove_played_cards(deal_str, cards_to_remove, leader_i)
            # Trickwinners are related to declarer, but leading is 0 for west
            if (trick_winners[trick-1] == 3):
                leader_i = decl_i    
            if (trick_winners[trick-1] == 2):
                leader_i = (decl_i + 3) % 4
            if (trick_winners[trick-1] == 1):
                leader_i = (decl_i + 2) % 4   
            if (trick_winners[trick-1] == 0):
                leader_i = (decl_i + 1) % 4   
            cards_to_remove = []
            current_trick52 = []
            trick = trick + 1
            #hands_pbn = ["N:" + deal_str]           
    #print("---------------------------------------")        
    return dcer, dfer

def remove_played_card(deal_str, card, i, leader_i):
    idx = ((i % 4)+leader_i-1) % 4
    suit = get_suit_i(card)
    hands = deal_str.split(' ')
    suits = hands[idx].split('.')
    suits[suit-1] = suits[suit-1].replace(card[1],"")
    hands[idx] = '.'.join(suits)
    new_deal_str = ' '.join(hands)
    if len(new_deal_str) == len(deal_str):
        print("Card not deleted")
    return new_deal_str

def calculate_DCWER_BEN():
    deals_n_def = 0
    deals_s_def = 0
    deals_n_dec = 0
    deals_s_dec = 0
    dcer = 0
    dfer = 0
    dcrn = 0
    dcrs = 0
    dfrn = 0
    dfrs = 0
    j = 0
    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            j += 1
            deal_str = deal["hands"]
            outcome_str = deal['contract']
            if outcome_str is None:
                continue
            play_str = deal["play"]
            trick_winners = deal["trick_winners"]
            #print(trick_winners)
            dcer, dfer = validate_lead_gamedb(deal_str, outcome_str, play_str,trick_winners)
            #print("errors: ",dcer, dfer)
            decl_i = get_decl_i(outcome_str[-1])
            if NS:
                declaring = (decl_i == 1) or (decl_i == 3)
            else:
                declaring = (decl_i == 0) or (decl_i == 2)

            if (get_strain_i(outcome_str) == 0):
                if declaring: 
                    dcrn += dcer
                    deals_n_dec += 1
                if not declaring: 
                    dfrn += dfer
                    deals_n_def += 1
            else:                   
                if declaring: 
                    dcrs += dcer
                    deals_s_dec += 1
                if not declaring: 
                    dfrs += dfer
                    deals_s_def += 1

            #print(NS, declaring, decl_i, deals_n_dec, deals_n_def, deals_s_dec, deals_s_def)
            #if deals_s > 2 and deals_n > 5:
            #    break
        
        print(f"Calculated on {j} deals.")
        print(f"DCWER  (suit) {deals_s_dec:>4} = {((dcrs)*100/ (deals_s_dec*24)):.2f}%")
        print(f"DCWER  (NT)   {deals_n_dec:>4} = {((dcrn)*100/ (deals_n_dec*24)):.2f}%")
        print(f"DFWER  (suit) {deals_s_def:>4} = {((dfrs)*100/ (deals_s_def*24)):.2f}%")
        print(f"DFWER  (NT)   {deals_n_def:>4} = {((dfrn)*100/ (deals_n_def*24)):.2f}%")
        print()
        print(f"DCWER  (suit) {deals_s_dec+deals_n_dec:>4} = {((dcrs+dcrn)*100/ ((deals_s_dec+deals_n_dec)*24)):.2f}%")
        print(f"DFWER  (suit) {deals_s_def+deals_n_def:>4} = {((dfrs+dfrn)*100/ ((deals_s_def+deals_n_def)*24)):.2f}%")

if __name__ == '__main__':

    calculate_DCWER_BEN()

