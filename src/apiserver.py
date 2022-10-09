import json
import os

import numpy as np
from bottle import Bottle, request, run
from gevent import monkey

import bots
import deck52
import sample
from bidding import bidding
from bots import BotBid, BotLead
from nn.models import Models
from objects import BidResp, Card, CardResp

#import os



MODELS = Models.load('../models')

monkey.patch_all()

app = Bottle()


@app.route('/api/bid', method='POST')
def bid():
    req = request.json
    print("bid request", req)

    bot_bid = BotBid(req['vul'], req['hand'], MODELS)
    bid = bot_bid.bid(req['auction'])

    resp = {"bid": bid.bid}
    print("bid response", resp)
    return json.dumps(resp)


@app.route('/api/lead', method='POST')
def lead():
    req = request.json
    print("lead request", req)

    bot_lead = BotLead(req['vul'], req['hand'], MODELS)
    lead = bot_lead.lead(req['auction'])

    resp = {"card": lead.card.symbol()}
    print("lead response", resp)
    return json.dumps(resp)


@app.route('/api/play', method='POST')
def play():
    req = request.json
    print("play request", req)

    vuln = req['vul']
    hands = req['hands']
    auction = req['auction']
    play = req['play']

    contract = bidding.get_contract(auction)
    level = int(contract[0])
    strain_i = bidding.get_strain_i(contract)
    decl_i = bidding.get_decl_i(contract)
    is_decl_vuln = [vuln[0], vuln[1], vuln[0], vuln[1]][decl_i]

    lefty_hand = hands[(decl_i + 1) % 4]
    dummy_hand = hands[(decl_i + 2) % 4]
    righty_hand = hands[(decl_i + 3) % 4]
    decl_hand = hands[decl_i]

    card_players = [
        bots.CardPlayer(MODELS.player_models, 0, lefty_hand,
                        dummy_hand, contract, is_decl_vuln),
        bots.CardPlayer(MODELS.player_models, 1, dummy_hand,
                        decl_hand, contract, is_decl_vuln),
        bots.CardPlayer(MODELS.player_models, 2, righty_hand,
                        dummy_hand, contract, is_decl_vuln),
        bots.CardPlayer(MODELS.player_models, 3, decl_hand,
                        dummy_hand, contract, is_decl_vuln)
    ]

    player_cards_played = [[] for _ in range(4)]
    shown_out_suits = [set() for _ in range(4)]

    leader_i = 0

    tricks = []
    tricks52 = []
    trick_won_by = []

    opening_lead52 = Card.from_symbol(play[0]).code()
    opening_lead = deck52.card52to32(opening_lead52)

    current_trick = [opening_lead]
    current_trick52 = [opening_lead52]

    card_players[0].hand52[opening_lead52] -= 1
    card_i = 1

    for trick_i in range(12):
        print("trick", trick_i)
        for player_i in map(lambda x: x % 4, range(leader_i, leader_i + 4)):
            if trick_i == 0 and player_i == 0:
                for i, card_player in enumerate(card_players):
                    card_player.set_card_played(
                        trick_i=trick_i, leader_i=leader_i, i=0, card=opening_lead)
                continue

            if card_i >= len(play):
                rollout_states = sample.init_rollout_states(trick_i, player_i, card_players, player_cards_played, shown_out_suits,
                                                            current_trick, 200, auction, card_players[player_i].hand.reshape((-1, 32)), vuln, MODELS)

                card_resp = card_players[player_i].play_card(
                    trick_i, leader_i, current_trick52, rollout_states)
                print("play response", card_resp.card)
                return json.dumps({"card": card_resp.card.symbol()})

            cardObj = Card.from_symbol(play[card_i])

            card_i += 1
            card52 = cardObj.code()
            card = deck52.card52to32(card52)

            for card_player in card_players:
                card_player.set_card_played(
                    trick_i=trick_i, leader_i=leader_i, i=player_i, card=card)

            current_trick.append(card)

            current_trick52.append(card52)

            card_players[player_i].set_own_card_played52(card52)
            if player_i == 1:
                for i in [0, 2, 3]:
                    card_players[i].set_public_card_played52(card52)
            if player_i == 3:
                card_players[1].set_public_card_played52(card52)

            # update shown out state
            # card is different suit than lead card
            if card // 8 != current_trick[0] // 8:
                shown_out_suits[player_i].add(current_trick[0] // 8)

        # sanity checks after trick completed
        assert len(current_trick) == 4

        for i, card_player in enumerate(card_players):
            assert np.min(card_player.hand52) == 0
            assert np.min(card_player.public52) == 0
            assert np.sum(card_player.hand52) == 13 - trick_i - 1
            assert np.sum(card_player.public52) == 13 - trick_i - 1

        tricks.append(current_trick)
        tricks52.append(current_trick52)

        # initializing for the next trick
        # initialize hands
        for i, card in enumerate(current_trick):
            card_players[(leader_i + i) % 4].x_play[:, trick_i + 1,
                                                    0:32] = card_players[(leader_i + i) % 4].x_play[:, trick_i, 0:32]
            card_players[(leader_i + i) % 4].x_play[:,
                                                    trick_i + 1, 0 + card] -= 1

        # initialize public hands
        for i in (0, 2, 3):
            card_players[i].x_play[:, trick_i + 1,
                                   32:64] = card_players[1].x_play[:, trick_i + 1, 0:32]
        card_players[1].x_play[:, trick_i + 1,
                               32:64] = card_players[3].x_play[:, trick_i + 1, 0:32]

        for card_player in card_players:
            # initialize last trick
            for i, card in enumerate(current_trick):
                card_player.x_play[:, trick_i + 1, 64 + i * 32 + card] = 1

            # initialize last trick leader
            card_player.x_play[:, trick_i + 1, 288 + leader_i] = 1

            # initialize level
            card_player.x_play[:, trick_i + 1, 292] = level

            # initialize strain
            card_player.x_play[:, trick_i + 1, 293 + strain_i] = 1

        # sanity checks for next trick
        for i, card_player in enumerate(card_players):
            assert np.min(card_player.x_play[:, trick_i + 1, 0:32]) == 0
            assert np.min(card_player.x_play[:, trick_i + 1, 32:64]) == 0
            assert np.sum(
                card_player.x_play[:, trick_i + 1, 0:32], axis=1) == 13 - trick_i - 1
            assert np.sum(
                card_player.x_play[:, trick_i + 1, 32:64], axis=1) == 13 - trick_i - 1

        trick_winner = (
            leader_i + deck52.get_trick_winner_i(current_trick52, (strain_i - 1) % 5)) % 4
        trick_won_by.append(trick_winner)

        if trick_winner % 2 == 0:
            card_players[0].n_tricks_taken += 1
            card_players[2].n_tricks_taken += 1
        else:
            card_players[1].n_tricks_taken += 1
            card_players[3].n_tricks_taken += 1

        # update cards shown
        for i, card in enumerate(current_trick):
            player_cards_played[(leader_i + i) % 4].append(card)

        leader_i = trick_winner
        current_trick = []
        current_trick52 = []

    print("no response")
    resp = {"card": "nothing"}
    return json.dumps(resp)


run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), server='gevent')
