# agents/base_agent.py

from abc import ABC, abstractmethod
from game.player import Player

class BaseAgent(Player, ABC):
    @abstractmethod
    def decide_move(self, game_state):
        pass