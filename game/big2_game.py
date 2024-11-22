# big2_game.py

import numpy as np
import random
from itertools import combinations

class Big2Game:
    '''Big Two Game Logic'''

    def __init__(self):
        self.num_players = 4
        self.players = []
        self.current_player_id = 0
        self.last_player_id = None
        self.last_play = []
        self.pass_count = 0
        self.control = True  # Indicates if the player has control
        self.game_over = False
        self.history = []

        self.card_order = self.init_card_order()
        self.card_id_to_str = self.init_card_id_to_str()
        self.card_str_to_id = {v: k for k, v in self.card_id_to_str.items()}
        self.action_id_to_action = self.init_action_id_to_action()
        self.init_game()
        

    def init_card_order(self):
        '''Initialize the card order for comparison'''
        # Suits in order: Diamonds < Clubs < Hearts < Spades
        suits = ['D', 'C', 'H', 'S']
        # Ranks in order: 3 < 4 < ... < A < 2
        ranks = ['3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A', '2']
        card_order = []
        for rank in ranks:
            for suit in suits:
                card_order.append(rank + suit)
        return card_order

    def init_card_id_to_str(self):
        '''Initialize mapping from card IDs to card strings'''
        card_id_to_str = {i: card for i, card in enumerate(self.card_order)}
        return card_id_to_str

    def init_action_id_to_action(self):
        '''Initialize mapping from action IDs to actions (tuples of card IDs)'''
        action_id_to_action = {}
        action_id = 0
        # Pass action
        action_id_to_action[action_id] = ()
        action_id += 1
        # Single cards
        for card_id in range(52):
            action_id_to_action[action_id] = (card_id,)
            action_id += 1
        # Pairs
        for combo in combinations(range(52), 2):
            if self.is_valid_pair(combo):
                action_id_to_action[action_id] = combo
                action_id += 1
        # Three of a kind
        for combo in combinations(range(52), 3):
            if self.is_valid_three_of_a_kind(combo):
                action_id_to_action[action_id] = combo
                action_id += 1
        # Four-card hands
        for combo in combinations(range(52), 4):
            if self.is_valid_four_card_hand(combo):
                action_id_to_action[action_id] = combo
                action_id += 1
        # Five-card hands
        for combo in combinations(range(52), 5):
            if self.is_valid_five_card_hand(combo):
                action_id_to_action[action_id] = combo
                action_id += 1
        return action_id_to_action

    def is_valid_pair(self, combo):
        '''Check if a combination is a valid pair'''
        card1 = self.card_id_to_str[combo[0]]
        card2 = self.card_id_to_str[combo[1]]
        return card1[0] == card2[0]  # Same rank

    def is_valid_three_of_a_kind(self, combo):
        '''Check if a combination is a valid three of a kind'''
        ranks = [self.card_id_to_str[card_id][0] for card_id in combo]
        return len(set(ranks)) == 1

    def is_valid_four_card_hand(self, combo):
        '''Check if a combination is a valid four-card hand (two pair or four of a kind)'''
        ranks = [self.card_id_to_str[card_id][0] for card_id in combo]
        rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
        return (sorted(rank_counts.values()) == [2, 2]) or (4 in rank_counts.values())

    def is_valid_five_card_hand(self, combo):
        '''Check if a combination is a valid five-card hand (straight, flush, full house, straight flush)'''
        # Implement logic to validate five-card hands
        # For simplicity, we'll consider only straight for now
        card_ids = list(combo)
        card_ids.sort()
        ranks = [self.card_id_to_str[card_id][0] for card_id in card_ids]
        rank_indices = [self.card_order.index(rank + 'D') // 4 for rank in ranks]  # Use 'D' as placeholder
        # Check for sequential ranks
        return max(rank_indices) - min(rank_indices) == 4 and len(set(rank_indices)) == 5

    def init_game(self):
        deck = list(range(52))
        random.shuffle(deck)
        self.players = [Big2Player(player_id=i, hand=[]) for i in range(self.num_players)]
        for i, card_id in enumerate(deck):
            self.players[i % self.num_players].hand.append(card_id)

        # Find who has '3D' using card_str_to_id
        try:
            three_d_id = self.card_str_to_id['3D']
            for player in self.players:
                if three_d_id in player.hand:
                    self.current_player_id = player.player_id
                    self.control = True
                    break

            # Remove '3D' from player's hand and set as last_play
            self.players[self.current_player_id].hand.remove(three_d_id)
            self.last_play = [three_d_id]
            self.last_player_id = self.current_player_id
            self.pass_count = 0
            
            # Record the action
            self.history.append({
                'player_id': self.current_player_id,
                'action': [three_d_id]
            })
            
            # Advance to next player
            self.current_player_id = (self.current_player_id + 1) % self.num_players
            
        except KeyError as e:
            raise ValueError(f"Could not find card '3D' in mappings") from e
    def get_card_str_to_id(self):
        '''Get the mapping from card string to card ID'''
        return self.card_str_to_id

    def get_action_id_to_action(self):
        '''Get the mapping from action ID to action'''
        return self.action_id_to_action

    def get_player_id(self):
        '''Get the current player ID'''
        return self.current_player_id

    def is_my_turn(self, player_id):
        '''Check if it's player's turn'''
        return self.current_player_id == player_id

    def get_state(self, player_id):
        '''Get the state for a player'''
        player = self.players[player_id]
        state = {
            'current_hand': [self.card_id_to_str[card_id] for card_id in player.hand],
            'last_play': [self.card_id_to_str[card_id] for card_id in self.last_play],
            'other_info': self.get_other_info(player_id),
            'legal_actions': self.get_legal_actions()
        }
        return state

    def get_other_info(self, player_id):
        '''Get additional info for state representation'''
        # For example: number of cards left for each player
        num_cards_left = [len(player.hand) for player in self.players]
        # Exclude the current player's hand size as it is represented in hand features
        return num_cards_left[:player_id] + num_cards_left[player_id + 1:]

    def get_legal_actions(self):
        '''Get legal actions for the current player'''
        player = self.players[self.current_player_id]
        legal_actions = []

        # Passing is always legal
        legal_actions.append(())

        # Generate all possible plays from player's hand
        hand_card_ids = player.hand
        possible_actions = []

        # Generate valid combinations based on hand size and game state
        for r in range(1, 6):
            for combo in combinations(hand_card_ids, r):
                if self.is_valid_combination(combo):
                    possible_actions.append(combo)

        # Filter valid actions based on last play
        for action in possible_actions:
            if self.is_higher(action, self.last_play) or self.control:
                legal_actions.append(action)

        return legal_actions

    def is_valid_combination(self, combo):
        '''Check if a combination is a valid hand'''
        # Implement validation logic based on hand type
        if len(combo) == 1:
            return True  # Any single card is valid
        elif len(combo) == 2:
            return self.is_valid_pair(combo)
        elif len(combo) == 3:
            return self.is_valid_three_of_a_kind(combo)
        elif len(combo) == 4:
            return self.is_valid_four_card_hand(combo)
        elif len(combo) == 5:
            return self.is_valid_five_card_hand(combo)
        else:
            return False

    def is_higher(self, new_play, last_play):
        '''Check if new play is higher than last play'''
        if not last_play:
            return True
        if len(new_play) != len(last_play):
            return False
        # Implement comparison logic based on hand types
        new_play_type = self.get_hand_type(new_play)
        last_play_type = self.get_hand_type(last_play)
        if new_play_type != last_play_type:
            return False  # Must be same hand type unless control
        # Compare the hands based on Big Two rules
        return self.compare_hands(new_play, last_play, new_play_type)

    def get_hand_type(self, hand):
        '''Determine the type of the hand'''
        if len(hand) == 1:
            return 'single'
        elif len(hand) == 2:
            return 'pair'
        elif len(hand) == 3:
            return 'three_of_a_kind'
        elif len(hand) == 4:
            if self.is_valid_four_card_hand(hand):
                return 'four_card_hand'
            else:
                return 'invalid'
        elif len(hand) == 5:
            if self.is_valid_five_card_hand(hand):
                return 'five_card_hand'
            else:
                return 'invalid'
        else:
            return 'invalid'

    def compare_hands(self, new_hand, old_hand, hand_type):
        '''Compare two hands of the same type'''
        # For simplicity, compare the highest card's index in card_order
        new_hand_max = max(new_hand, key=lambda card_id: self.card_order.index(self.card_id_to_str[card_id]))
        old_hand_max = max(old_hand, key=lambda card_id: self.card_order.index(self.card_id_to_str[card_id]))
        return new_hand_max > old_hand_max

    def step(self, action):
        '''Process the action and advance the game'''
        player = self.players[self.current_player_id]
        if action:
            # Player plays cards
            for card_id in action:
                player.hand.remove(card_id)
            self.last_play = action
            self.last_player_id = self.current_player_id
            self.pass_count = 0
            self.control = False
            # Record the action
            self.history.append({
                'player_id': self.current_player_id,
                'action': action,
            })
            if len(player.hand) == 0:
                self.game_over = True
        else:
            # Player passes
            self.pass_count += 1
            # Check if all other players have passed
            if self.pass_count >= self.num_players - 1:
                self.control = True
                self.last_play = []
                self.pass_count = 0
            # Record the action
            self.history.append({
                'player_id': self.current_player_id,
                'action': (),
            })

        # Advance to next player
        self.current_player_id = (self.current_player_id + 1) % self.num_players
        # Skip players who have no cards left
        while len(self.players[self.current_player_id].hand) == 0 and not self.game_over:
            self.current_player_id = (self.current_player_id + 1) % self.num_players

    def get_payoffs(self):
        '''Calculate and return payoffs when the game is over'''
        payoffs = []
        winner_id = None
        for player in self.players:
            if len(player.hand) == 0:
                winner_id = player.player_id
        total_cards_left = sum(len(player.hand) for player in self.players if player.player_id != winner_id)
        for player in self.players:
            if player.player_id == winner_id:
                payoffs.append(total_cards_left)
            else:
                payoffs.append(-len(player.hand))
        return payoffs

    def get_state_all_players(self):
        '''Get the perfect information of the game'''
        state = {
            'players': [{ 'player_id': p.player_id, 'hand': [self.card_id_to_str[card_id] for card_id in p.hand] } for p in self.players],
            'history': self.history,
            'last_play': [self.card_id_to_str[card_id] for card_id in self.last_play],
            'current_player': self.current_player_id,
        }
        return state

class Big2Player:
    def __init__(self, player_id, hand):
        self.player_id = player_id
        self.hand = hand  # List of card IDs
