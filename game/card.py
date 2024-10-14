class Card:
    suit_order = {'Diamonds': 0, 'Clubs': 1, 'Hearts': 2, 'Spades': 3}
    rank_order = {str(n): n for n in range(3, 11)}
    rank_order.update({'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15})

    def __init__(self, suit, rank):
        self.suit = suit  # 'Diamonds', 'Clubs', 'Hearts', 'Spades'
        self.rank = rank  # '3' to '10', 'J', 'Q', 'K', 'A', '2'

    def __lt__(self, other):
        if Card.rank_order[self.rank] == Card.rank_order[other.rank]:
            return Card.suit_order[self.suit] < Card.suit_order[other.suit]
        return Card.rank_order[self.rank] < Card.rank_order[other.rank]

    def __repr__(self):
        suit_symbols = {'Diamonds': '♦', 'Clubs': '♣', 'Hearts': '♥', 'Spades': '♠'}
        return f"{self.rank}{suit_symbols[self.suit]}"

    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit