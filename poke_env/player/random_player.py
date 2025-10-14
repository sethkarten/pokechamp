"""This module defines a random players baseline
"""

from poke_env.environment import AbstractBattle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.player.battle_order import BattleOrder, DoubleBattleOrder
from poke_env.player.player import Player
import random


class RandomPlayer(Player):
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if isinstance(battle, DoubleBattle):
            return self._choose_simple_doubles_move(battle)
        return self.choose_random_move(battle)
    
    def _choose_simple_doubles_move(self, battle: DoubleBattle) -> DoubleBattleOrder:
        """Simple double battle implementation that chooses random moves for each slot."""
        orders = [None, None]
        
        # Handle force switch cases properly
        if any(battle.force_switch):
            for i in range(2):
                if battle.force_switch[i]:
                    if battle.available_switches[i]:
                        orders[i] = self.create_order(random.choice(battle.available_switches[i]))
                else:
                    # Pokemon not forced to switch - set to None for partial switches
                    orders[i] = None
            return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
        
        # Normal battle logic
        for i in range(2):
            # If Pokemon is None or fainted, must switch
            if battle.active_pokemon[i] is None or battle.active_pokemon[i].fainted:
                if battle.available_switches[i]:
                    orders[i] = self.create_order(random.choice(battle.available_switches[i]))
                continue
            
            # Choose randomly between moves and switches
            all_actions = []
            
            # Add moves with random targets
            for move in battle.available_moves[i]:
                # For doubles, target either opponent 1 or 2
                target = random.choice([1, 2])
                all_actions.append(self.create_order(move, move_target=target))
            
            # Add switches (only if not forced to switch)
            for switch in battle.available_switches[i]:
                all_actions.append(self.create_order(switch))
            
            if all_actions:
                orders[i] = random.choice(all_actions)
        
        return DoubleBattleOrder(first_order=orders[0], second_order=orders[1])
