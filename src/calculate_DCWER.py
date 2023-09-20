import itertools
from ddsolver import ddsolver

import shelve
import os
import argparse
import random

os.getcwd()

parser = argparse.ArgumentParser(description="DDOLAR calculator")
parser.add_argument("--db", default="gamedb", help="Port for appserver")

args = parser.parse_args()

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
        min_key = min(dd_solved, key=lambda k: dd_solved[k])
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
        deal_str = remove_played_cards(deal_str, card, i, leader_i)
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
    print("---------------------------------------")        
    return dcer, dfer

def remove_played_cards(deal_str, card, i, leader_i):
    x = len(deal_str)
    #print("Removing: "+card)
    idx = ((i % 4)+leader_i-1) % 4
    suit = get_suit_i(card)
    hands = deal_str.split(' ')
    #print(hands, i, decl)
    suits = hands[idx].split('.')
    #print(suits)
    suits[suit-1] = suits[suit-1].replace(card[1],"")
    hands[idx] = '.'.join(suits)
    deal_str = ' '.join(hands)
    if x ==len(deal_str):
        print("Card not deleted")
        print(card)
        print(deal_str)
        print(suits)
        print(i)
        print(idx)
        print(leader_i)
    return deal_str

def calculate_DCWER_BEN():
    deals_n = 0
    deals_s = 0
    dcer = 0
    dfer = 0
    dcrn = 0
    dcrs = 0
    dfrn = 0
    dfrs = 0
    with shelve.open(DB_NAME) as db:
        deal_items = sorted(list(db.items()), key=lambda x: x[1]['timestamp'], reverse=True)
        for deal_id, deal in deal_items:
            deal_str = deal["hands"]
            outcome_str = deal['contract']
            if outcome_str is None:
                continue
            play_str = deal["play"]
            trick_winners = deal["trick_winners"]
            #print(trick_winners)
            dcer, dfer = validate_lead_gamedb(deal_str, outcome_str, play_str,trick_winners)
            print("errors: ",dcer, dfer)
            if (get_strain_i(outcome_str) == 0):
                dcrn += dcer
                dfrn += dfer
                deals_n += 1
            else:                   
                dcrs += dcer
                dfrs += dfer
                deals_s += 1
            #if deals_s > 2 and deals_n > 5:
            #    break
        
        print(f"DCWER  (suit) {deals_s:>4} = {((dcrn)*100/ (deals_s*24)):.2f}%")
        print(f"DCWER  (NT)   {deals_n:>4} = {((dcrn)*100/ (deals_n*24)):.2f}%")
        print(f"DFWER  (suit) {deals_s:>4} = {((dfrn)*100/ (deals_s*24)):.2f}%")
        print(f"DFWER  (NT)   {deals_n:>4} = {((dfrn)*100/ (deals_n*24)):.2f}%")

if __name__ == '__main__':

    calculate_DCWER_BEN()

