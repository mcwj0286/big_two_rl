# game/player.py

from .hand_evaluator import (
    generate_singles,
    generate_pairs,
    generate_triples,
    generate_five_card_hands,
    is_valid_hand,
    compare_hands
)
from .hand_type import HandType

class Player:
    def __init__(self, name):
        self.name = name
        self.hand = []
        self.has_passed = False
        self.player_idx = None  # Will be set by the game engine

    def receive_cards(self, cards):
        self.hand.extend(cards)
        self.hand.sort()

    def play_cards(self, cards):
        for card in cards:
            self.hand.remove(card)

    def decide_move(self, game_state):
        raise NotImplementedError

    def get_valid_moves(self, game_state):
        valid_moves = []

        # Generate all possible hands
        singles = generate_singles(self.hand)
        pairs = generate_pairs(self.hand)
        triples = generate_triples(self.hand)
        five_card_hands_data = generate_five_card_hands(self.hand)

        five_card_hands = [hand for hand, _, _ in five_card_hands_data]

        # Combine all moves
        all_moves = singles + pairs + triples + five_card_hands

        # Include 'pass' as a valid move
        all_moves.append('pass')

        # Get the last played hand
        last_played_hand = game_state['last_played_hand']
        last_player_idx = game_state['last_player_idx']

        if last_played_hand is None or self.has_control(game_state):
            # If the player can lead, return all valid moves
            return all_moves
        else:
            # Filter moves to those that can beat the last played hand
            last_hand_type, last_key = last_played_hand
            valid_moves = []
            for move in all_moves:
                if move == 'pass':
                    valid_moves.append('pass')
                    continue
                hand_type, key = is_valid_hand(move)
                if hand_type != last_hand_type:
                    continue  # Must play the same type of hand
                if compare_hands((hand_type, key), last_played_hand) > 0:
                    valid_moves.append(move)
            # If no valid moves, must pass
            if not valid_moves:
                valid_moves.append('pass')
            return valid_moves

    def has_control(self, game_state):
        # Determine if the player can lead a new round
        return game_state['last_player_idx'] == self.player_idx