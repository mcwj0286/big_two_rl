from game.player import Player
from game.hand_evaluator import compare_cards
import random
class RandomPlayer(Player):
    # def __init__(self):
    #     super().__init__("Random Player")
    #     self.player_idx = None

    def decide_move(self, game_state):
        valid_moves = self.get_valid_moves(game_state)
        if valid_moves:
            move = random.choice(valid_moves)
            return move
        else:
            return 'pass'

    def get_valid_moves(self, game_state):
        # Generate all valid moves based on the current hand and game state
        # For simplicity, we'll assume all singles are valid (this is not accurate)
        last_hand_type, _ = game_state['last_played_hand'] if game_state['last_played_hand'] else (None, None)
        if last_hand_type is None:
            # Can play any valid hand
            return [[card] for card in self.hand]
        else:
            # Need to play a higher hand of the same type
            # We'll simplify and only consider singles
            return [ [card] for card in self.hand if compare_cards(card, game_state['last_played_hand'][1]) > 0 ]