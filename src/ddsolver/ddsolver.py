import ctypes
from typing import Dict, List

from ddsolver import dds

dds.SetMaxThreads(0)

class DDSolver:

    # Default for dds_mode changes to 1
    # Transport table will be reused if same trump suit and the same or nearly the same cards distribution, deal.first can be different. 
    # Always search to find the score. Even when the hand to play has only one card, with possible equivalents, to play.  
    # If zero, we not always find the score
    # If 2 transport tables ignore trump
 
    def __init__(self, dds_mode=1):
        self.dds_mode = dds_mode
        self.bo = dds.boardsPBN()
        self.solved = dds.solvedBoards()

    # Solutions
    #1	Find the maximum number of tricks for the side to play.  Return only one of the optimum cards and its score.
    #2	Find the maximum number of tricks for the side to play.  Return all optimum cards and their scores.
    #3	Return all cards that can be legally played, with their scores in descending order.

    def solve(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn[:dds.MAXNOOFBOARDS], solutions)

        if len(hands_pbn) > dds.MAXNOOFBOARDS:
            i = dds.MAXNOOFBOARDS
            while i < len(hands_pbn):
                more_results = self.solve_helper(strain_i, leader_i, current_trick, hands_pbn[i:i+dds.MAXNOOFBOARDS], solutions)

                for card, values in more_results.items():
                    results[card] = results[card] + values

                i += dds.MAXNOOFBOARDS

        return results 

    def solve_helper(self, strain_i, leader_i, current_trick, hands_pbn, solutions):
        card_rank = [0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080, 0x0040, 0x0020, 0x0010, 0x0008, 0x0004]

        self.bo.noOfBoards = min(dds.MAXNOOFBOARDS, len(hands_pbn))

        for handno in range(self.bo.noOfBoards):
            self.bo.deals[handno].trump = (strain_i - 1) % 5
            self.bo.deals[handno].first = leader_i

            for i in range(3):
                self.bo.deals[handno].currentTrickSuit[i] = 0
                self.bo.deals[handno].currentTrickRank[i] = 0
                if i < len(current_trick):
                    self.bo.deals[handno].currentTrickSuit[i] = current_trick[i] // 13
                    self.bo.deals[handno].currentTrickRank[i] = 14 - current_trick[i] % 13

            self.bo.deals[handno].remainCards = hands_pbn[handno].encode('utf-8')

            self.bo.target[handno] = -1
            # Return all cards that can be legally played, with their scores in descending order.
            self.bo.solutions[handno] = solutions
            self.bo.mode[handno] = self.dds_mode

        res = dds.SolveAllBoards(ctypes.pointer(self.bo), ctypes.pointer(self.solved))
        if res != 1:
            error_message = dds.get_error_message(res)
            print(f"Error Code: {res}, Error Message: {error_message}")
            print(hands_pbn[0].encode('utf-8'))
            return None

        card_results = {}

        for handno in range(self.bo.noOfBoards):
            fut = ctypes.pointer(self.solved.solvedBoards[handno])
            for i in range(fut.contents.cards):
                suit_i = fut.contents.suit[i]
                card = suit_i * 13 + 14 - fut.contents.rank[i]
                if card not in card_results:
                    card_results[card] = []
                card_results[card].append(fut.contents.score[i])
                eq_cards_encoded = fut.contents.equals[i]
                for k, rank_code in enumerate(card_rank):
                    if rank_code & eq_cards_encoded > 0:
                        eq_card = suit_i * 13 + k
                        if eq_card not in card_results:
                            card_results[eq_card] = []
                        card_results[eq_card].append(fut.contents.score[i])
        return card_results


def expected_tricks_dds(card_results):
    return {card:(sum(values)/len(values)) for card, values in card_results.items()}

def expected_tricks_dds_probabiliy(card_results, probabilities_list : List[float]):
    return {card: sum([p*res for p, res in zip(probabilities_list, result_list)]) for card, result_list in card_results.items()}

def p_made_target(tricks_needed):

    def fun(card_results):
        return {card:round(sum(1 for x in values if x >= tricks_needed)/len(values),3) for card, values in card_results.items()}
    return fun
