"""
Blackjack
van-john (England)
pontoon (Australia)

Beat the Dealer
- Rules vary from casino to casino


Rules
- n players = dealer plus 1-7 players
- fewer the players the better
- n_decks = number of decks shuffled together

"""

import numpy as np
import pandas as pd
import random
import copy
import itertools as it
from pprint import pprint

random.seed(42)
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 100)


# add if name is dealer automatically set bank=inf, bet=0, dealer=True, etc.
class Hand:
    def __init__(self):
        self.cards = []
        self.win_totals = []
        self.bust_totals = []
        self.soft = False
        self.natural = False
        self.bust = False
        self.hit = False
        self.final_total = 0

    def __repr__(self):
        return f'Hand({self.cards!r}, {self.win_totals!r}, {self.bust_totals!r}, {self.soft!r}, {self.natural!r}, ' \
            f'{self.bust}, {self.hit}, {self.final_total})'

    def receive_card(self, card):
        self.cards.append(card)
        if len(self.cards) >= 2:
            self._calc_hand()

    def _calc_hand(self):

        card_values = [card.value for card in self.cards]
        totals = {sum(i) for i in it.product(*card_values)}

        self.win_totals = []
        self.bust_totals = []
        for total in totals:
            if total <= 21:
                self.win_totals.append(total)
            else:
                self.bust_totals.append(total)

        if len(self.win_totals) == 0 and len(self.bust_totals) > 0:
            self.bust = True
            self.hit = False
            self.final_total = min(self.bust_totals)
        else:
            max_win_tot = max(self.win_totals)
            if max_win_tot < 17:
                self.hit = True
            else:
                self.hit = False
                self.final_total = max_win_tot

        if 2 in (len(self.cards[0].value), len(self.cards[0].value)):
            self.soft = True

        if 'A' in (self.cards[0].rank, self.cards[1].rank) and \
            10 in (self.cards[0].value[0], self.cards[1].value[0]):
            self.natural = True

    def clear_hand(self):
        self.cards = []
        self.win_totals = []
        self.bust_totals = []
        self.soft = False
        self.natural = False
        self.bust = False
        self.hit = False
        self.final_total = 0


class Player:
    def __init__(self, name, bank, strategy=None, dealer=False):
        self.name = name
        self.bank = bank
        self.strategy = strategy
        self.dealer = dealer
        self.bet = 0
        self.hand = Hand()

    def __repr__(self):
        return f'Player({self.name!r}, {self.bank!r}, {self.strategy!r}, {self.dealer}, {self.bet}, {self.hand})'


class Round:
    def __init__(self, players, dealer, hand_size=2, minimum_bet=5, maximum_bet=500, n_decks=2, n_cut=20):
        self.players = players
        self.dealer = dealer
        self.hand_size = hand_size
        self.minimum_bet = minimum_bet
        self.maximum_bet = maximum_bet
        self.n_decks = n_decks
        self.n_cut = n_cut
        self.shoe = self._create_shoe()

    def _create_shoe(self):
        shoe = []
        ranks = 'A 2 3 4 5 6 7 8 9 10 J Q K'.split()
        suits = 'C D H S'.split()
        values = [(1, 11)] + [(i,) for i in range(2, 10)] + [(10,) for i in range(4)]
        for n_deck in range(self.n_decks):
            for idx, rank in enumerate(ranks):
                for suit in suits:
                    shoe.append(Card(rank, suit, values[idx], True))

        # shuffle
        random.shuffle(shoe)
        shoe = shoe[-self.n_cut:] + shoe[:-self.n_cut]

        # burn card
        shoe[-1].view = False
        shoe = shoe[-1:] + shoe[:-1]

        print(f'n_decks: {self.n_decks} n_cards: {len(shoe)}')
        return shoe

    def _deal_card(self):
        if len(self.shoe) == 0:
            self.shoe = self._create_shoe
        return self.shoe.pop()

    def get_bets(self, get_in=False):

        for player in self.players:
            player.hand.clear_hand()
        self.dealer.hand.clear_hand()

        for player in self.players:
            if get_in:
                player.bet = int(input(f'{player.name} bet:'))
            else:
                player.bet = self.minimum_bet

    def init_deal_cards(self):

        for i in range(self.hand_size):
            for player in self.players:
                dealt_card = self._deal_card
                player.hand.receive_card(dealt_card)
            dealt_card = self._deal_card
            if i == 0:
                dealt_card.view = False
            self.dealer.hand.receive_card(dealt_card)

    def draw_cards(self):
        for player in self.players:
            while player.hand.hit == True:
                dealt_card = self._deal_card
                player.hand.receive_card(dealt_card)

        while self.dealer.hand.hit == True:
            dealt_card = self._deal_card
            self.dealer.hand.receive_card(dealt_card)

    def calc_payouts(self):
        for player in self.players:
            if player.hand.bust == True:
                player.bank -= player.bet
                self.dealer.bank += player.bet
            elif self.dealer.hand.bust == True:
                player.bank += player.bet
                self.dealer.bank -= player.bet
            elif player.hand.final_total > self.dealer.hand.final_total:
                player.bank += player.bet
                self.dealer.bank -= player.bet
            elif player.hand.final_total == self.dealer.hand.final_total:
                # push
                pass


class Card:
    def __init__(self, rank, suit, value, view):
        self.rank = rank
        self.suit = suit
        self.value = value
        self.view = view

    def __repr__(self):
        if self.view:
            return f'Card({self.rank!r}, {self.suit!r}, {self.value!r})'
        else:
            return f'Hidden Card'


class Game:
    def __init__(self, n_rounds, n_players, n_decks=2, n_cut=15):

        if n_players > 7 or n_players < 1:
            raise ValueError('n_players must be between 1 and 7')

        if n_decks not in (2, 4, 6, 8):
            raise ValueError('n_decks must be 2, 4, 6, or 8')

        if n_cut < 0 or n_cut > n_decks * 52:
            raise ValueError('n_cut must be an integer between 0 and n_decks * 52')

        self.n_rounds = n_rounds
        self.n_players = n_players
        self.n_decks = n_decks
        self.n_cut = n_cut

    def play_game(self):
        players = [Player('player_' + str(i), 1_000) for i in range(self.n_players)]
        dealer = Player('dealer', 100_000, None, True)

        rounds = Round(players, dealer)

        for n_round in range(n_rounds):
            print(n_round)
            rounds.get_bets(get_in=False)
            pprint(rounds.players)
            pprint(rounds.dealer)

            rounds.init_deal_cards()
            pprint(rounds.players)
            pprint(rounds.dealer)

            rounds.draw_cards()
            pprint(rounds.players)
            pprint(rounds.dealer)

            rounds.calc_payouts()
            pprint(rounds.players)
            pprint(rounds.dealer)
















