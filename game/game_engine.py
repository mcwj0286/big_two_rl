
from game.hand_evaluator import (
    is_valid_hand,
    compare_hands
)
from game.deck import Deck
from game.hand_type import HandType
from game.player import Player

class BigTwoGame:
    def __init__(self, players):
        self.players = players
        for idx, player in enumerate(self.players):
            player.player_idx = idx
        self.current_player_idx = None
        self.last_played_hand = None
        self.last_player_idx = None  # The index of the player who last played a hand
        self.round_passes = 0  # Number of consecutive passes
        self.game_over = False

    def start_game(self):
        self.deal_cards()
        self.find_starting_player()
        while not self.game_over:
            self.play_turn()
        self.end_game()

    def deal_cards(self):
        deck = Deck()
        hands = deck.deal(len(self.players))
        for player, hand in zip(self.players, hands):
            player.receive_cards(hand)

    def find_starting_player(self):
        # Player who holds the 3♦ starts
        for idx, player in enumerate(self.players):
            if any(card.rank == '3' and card.suit == 'Diamonds' for card in player.hand):
                self.current_player_idx = idx
                print(f"{player.name} has the 3♦ and will start.")
                break

    def play_turn(self):
        player = self.players[self.current_player_idx]
        game_state = self.get_game_state()
        move = player.decide_move(game_state)

        if move == 'pass':
            player.has_passed = True
            self.round_passes += 1
            print(f"{player.name} passes.")
        else:
            # Validate the move
            hand_type, key = is_valid_hand(move)
            if hand_type == HandType.INVALID:
                print(f"{player.name} made an invalid move.")
            elif self.is_move_valid(hand_type, key):
                player.play_cards(move)
                player.has_passed = False
                self.last_played_hand = (hand_type, key)
                self.last_player_idx = self.current_player_idx
                self.round_passes = 0
                print(f"{player.name} plays {move}.")
            else:
                print(f"{player.name}'s move does not beat the last played hand.")
                player.has_passed = True
                self.round_passes += 1

        if self.check_win(player):
            self.game_over = True
            return

        if self.round_passes >= len(self.players) - 1:
            # All other players have passed, start a new round
            print("Starting a new round.")
            self.last_played_hand = None
            self.reset_passes()

        # Move to the next player
        self.current_player_idx = (self.current_player_idx + 1) % len(self.players)

    def is_move_valid(self, hand_type, key):
        if self.last_played_hand is None:
            return True
        else:
            comparison = compare_hands((hand_type, key), self.last_played_hand)
            return comparison > 0

    def reset_passes(self):
        for player in self.players:
            player.has_passed = False
        self.round_passes = 0

    def check_win(self, player):
        if not player.hand:
            print(f"{player.name} has won the game!")
            return True
        return False

    def get_game_state(self):
        return {
            'last_played_hand': self.last_played_hand,
            'last_player_idx': self.last_player_idx,
            'current_player_idx': self.current_player_idx,
            'player_hand': self.players[self.current_player_idx].hand,
            'players_passed': [player.has_passed for player in self.players],
        }

    def end_game(self):
        print("Game Over.")