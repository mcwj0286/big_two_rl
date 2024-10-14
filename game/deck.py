import random
from .card import Card
class Deck:
    suits = ['Diamonds', 'Clubs', 'Hearts', 'Spades']
    ranks = [str(n) for n in range(3, 11)] + ['J', 'Q', 'K', 'A', '2']

    def __init__(self):
        self.cards = [Card(suit, rank) for suit in Deck.suits for rank in Deck.ranks]

    def shuffle(self):
        random.shuffle(self.cards)

    def deal(self, num_players):
        # Ensure the cards are shuffled before dealing
        self.shuffle()
        # Deal cards evenly to players
        return [self.cards[i::num_players] for i in range(num_players)]