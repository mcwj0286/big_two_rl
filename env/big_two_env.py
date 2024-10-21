# env/big_two_env.py

import gym
from gym import spaces
import numpy as np

from agents.rl_agent import RLAgentPlayer
from agents.random_agent import RandomPlayer
from game.game_engine import BigTwoGame
from game.player import Player
from game.card import Card

class BigTwoEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(BigTwoEnv, self).__init__()
        self.max_moves = 100  # Maximum number of possible moves
        self.action_space = spaces.Discrete(self.max_moves)
        self.observation_space = spaces.Dict({
            'player_hand': spaces.MultiBinary(52),
            'last_hand': spaces.MultiBinary(52),
            'action_mask': spaces.MultiBinary(self.max_moves),
        })
        # Initialize players
        self.players = [
            RLAgentPlayer("Agent"),
            RandomPlayer("Opponent1"),
            RandomPlayer("Opponent2"),
            RandomPlayer("Opponent3")
        ]
        self.game = BigTwoGame(self.players)
        self.current_state = None
        self.done = False

    def reset(self):
        self.game = BigTwoGame(self.players)
        self.game.deal_cards()
        self.game.find_starting_player()
        self.current_state = self.get_state()
        self.done = False
        return self.current_state

    def step(self, action):
        # Get the list of valid moves
        player = self.players[self.game.current_player_idx]
        valid_moves = player.get_valid_moves(self.game.get_game_state())
        num_valid_moves = len(valid_moves)
        action_mask = np.zeros(self.max_moves, dtype=int)
        action_mask[:num_valid_moves] = 1

        if action >= num_valid_moves or action_mask[action] == 0:
            # Invalid action
            reward = -1
            self.done = True  # Optionally end the episode
            info = {'invalid_action': True}
            return self.current_state, reward, self.done, info
        else:
            move = valid_moves[action]
            agent = self.players[self.game.current_player_idx]
            if isinstance(agent, RLAgentPlayer):
                agent.set_next_move(move)
            self.game.play_turn()
            self.current_state = self.get_state()
            reward = self.get_reward()
            self.done = self.game.game_over
            info = {'invalid_action': False}
            return self.current_state, reward, self.done, info

    def render(self, mode='human'):
        # Optional: Display the current game state
        pass

    def close(self):
        pass

    def get_state(self):
        player = self.players[self.game.current_player_idx]
        player_hand = self.cards_to_binary_vector(player.hand)
        last_hand_key = self.game.last_played_hand[1] if self.game.last_played_hand else []

        # Ensure last_hand_key is always a list
        if isinstance(last_hand_key, Card):
            last_hand_cards = [last_hand_key]
        elif isinstance(last_hand_key, list):
            last_hand_cards = last_hand_key
        else:
            last_hand_cards = []

        last_hand = self.cards_to_binary_vector(last_hand_cards)

        valid_moves = player.get_valid_moves(self.game.get_game_state())
        action_mask = np.zeros(self.max_moves, dtype=int)
        num_valid_moves = len(valid_moves)
        action_mask[:num_valid_moves] = 1

        state = {
            'player_hand': player_hand,
            'last_hand': last_hand,
            'action_mask': action_mask,
        }
        return state

    def cards_to_binary_vector(self, cards):
        vector = np.zeros(52, dtype=int)
        # Check if 'cards' is a single Card object
        if isinstance(cards, Card):
            cards = [cards]
        for card in cards:
            index = self.card_to_index(card)
            vector[index] = 1
        return vector

    def card_to_index(self, card):
        suit_idx = Card.suit_order[card.suit]
        rank_idx = list(Card.rank_order.values()).index(Card.rank_order[card.rank])
        return suit_idx * 13 + rank_idx

    def get_reward(self):
        # Define the reward function
        if self.done:
            winner = self.players[self.game.current_player_idx]
            if winner.name == "Agent":
                return 1  # Agent wins
            else:
                return -1  # Agent loses
        else:
            return 0  # No reward during the game