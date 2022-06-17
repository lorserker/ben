import sys
sys.path.append('../../../src')

from collections import Counter

from nn.bidder import Bidder
from nn.models import Models
from bots import BotBid
from bidding import bidding


def get_models(bidder_model_path):
    models = Models(None, None, None, None, None)
    models.bidder_model = Bidder('bidder', bidder_model_path)
    return models


def main():
    bidder_model_path = sys.argv[1]
    
    models = get_models(bidder_model_path)

    for line in sys.stdin:
        parts = line.strip().split()
        hands = parts[2:]
        dealer_i = 'NESW'.index(parts[0])
        
        bidder_bots = [BotBid([False, False], hand, models) for hand in hands]

        auction = ['PAD_START'] * dealer_i

        turn_i = dealer_i

        while not bidding.auction_over(auction):
            candidates = bidder_bots[turn_i].get_bid_candidates(auction)
            bid = candidates[0].bid
            auction.append(bid)
            turn_i = (turn_i + 1) % 4  # next player's turn

        auction_str = "-".join(auction).replace("PASS", "P").replace('PAD_START-', '')
        print(f'{line.strip()} {auction_str}')


if __name__ == '__main__':
    main()
