"""
Timeout-enabled LLM Player with AbyssalPlayer fallback.

This module provides a wrapper around LLMPlayer that adds a watchdog timeout.
If the LLM takes too long to respond (default 90 seconds), it automatically
falls back to using AbyssalPlayer's decision-making logic.
"""

import time
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import BattleOrder
from pokechamp.llm_player import LLMPlayer


# Pre-load and cache JSON data for fast access
_CACHED_DATA = {}

def _load_cached_data():
    """Load and cache JSON data files once."""
    global _CACHED_DATA
    if not _CACHED_DATA:
        try:
            with open("./poke_env/data/static/moves/moves_effect.json", "r") as f:
                _CACHED_DATA['move_effect'] = json.load(f)
            with open("./poke_env/data/static/abilities/ability_effect.json", "r") as f:
                _CACHED_DATA['ability_effect'] = json.load(f)
            with open("./poke_env/data/static/items/item_effect.json", "r") as f:
                _CACHED_DATA['item_effect'] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load cached data: {e}")
            _CACHED_DATA['move_effect'] = {}
            _CACHED_DATA['ability_effect'] = {}
            _CACHED_DATA['item_effect'] = {}
    return _CACHED_DATA

# Load data once at module import
_load_cached_data()


class TimeoutLLMPlayer(LLMPlayer):
    """
    LLM Player with timeout protection and integrated fast fallback.
    
    This class wraps the normal LLMPlayer and adds a timeout mechanism.
    If the LLM doesn't respond within the timeout period, it automatically
    falls back to using fast heuristic-based decision making.
    """
    
    # Constants for Abyssal-style decision making
    ENTRY_HAZARDS = {
        "spikes": SideCondition.SPIKES,
        "stealhrock": SideCondition.STEALTH_ROCK,
        "stickyweb": SideCondition.STICKY_WEB,
        "toxicspikes": SideCondition.TOXIC_SPIKES,
    }
    ANTI_HAZARDS_MOVES = {"rapidspin", "defog"}
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4
    SWITCH_OUT_MATCHUP_THRESHOLD = -2
    
    def __init__(self, timeout_seconds=90, **kwargs):
        """
        Initialize TimeoutLLMPlayer.
        
        Args:
            timeout_seconds (int): Maximum time to wait for LLM response before fallback
            **kwargs: All other arguments passed to LLMPlayer
        """
        super().__init__(**kwargs)
        self.timeout_seconds = timeout_seconds
        self.timeout_count = 0
        self.total_moves = 0
        
        # Detailed timing statistics
        self.move_times = []  # List of all move times
        self.timeout_losses = 0  # Battles lost due to timeout
        self.last_move_time = 0.0
        self.battle_startup_messages = set()  # Track which battles have had startup messages
        self.battle_timeout_messages = set()  # Track which battles have had timeout messages to prevent duplicates
        
        # Thread pool for executing LLM calls and parallel fallback with cleaner timeout handling
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="llm_timeout")
        
        # Use cached data for fast fallback
        self.cached_data = _CACHED_DATA
        
        print(f"TimeoutLLMPlayer initialized with {timeout_seconds}s timeout")
    
    def _send_chat_message(self, battle: AbstractBattle, message: str):
        """
        Send a chat message during battle.
        
        Args:
            battle: The current battle
            message: Message to send
        """
        try:
            # Use the existing async infrastructure from the player
            import asyncio
            from poke_env.concurrency import POKE_LOOP
            
            # Create an async function to send the message
            async def send_message_async():
                try:
                    # Send chat message to the battle room using the correct format
                    # According to Pokemon Showdown protocol: ROOMID|MESSAGE
                    # But we need to make sure this gets interpreted as a chat message
                    print(f"   🔍 Attempting to send chat message...")
                    print(f"   🔍 Battle room: {battle.battle_tag}")
                    print(f"   🔍 Message: '{message}'")
                    print(f"   🔍 WebSocket connected: {hasattr(self.ps_client, 'websocket')}")
                    
                    # Send chat message to battle room
                    await self.ps_client.send_message(message, room=battle.battle_tag)
                    print(f"   💬 ✅ Chat message sent to {battle.battle_tag}: '{message}'")
                    
                except Exception as e:
                    print(f"   ⚠️  Failed to send chat message: {e}")
                    import traceback
                    print(f"   🔍 Full error: {traceback.format_exc()}")
            
            # Schedule the coroutine in the poke_env event loop without waiting
            asyncio.run_coroutine_threadsafe(send_message_async(), POKE_LOOP)
            
        except Exception as e:
            print(f"   💬 Would send message: '{message}' (chat unavailable: {e})")
    
    def _execute_llm_move(self, battle: AbstractBattle) -> BattleOrder:
        """Execute the LLM move selection in a separate thread context."""
        return super().choose_move(battle)
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        Choose a move with timeout protection.
        
        This method wraps the original choose_move with a timeout mechanism.
        The timeout is only applied when the battle timer has less than 2x the timeout remaining.
        """
        self.total_moves += 1
        
        # Send startup message on first move of each battle (non-blocking)
        if battle.battle_tag not in self.battle_startup_messages:
            self.battle_startup_messages.add(battle.battle_tag)
            try:
                self._send_chat_message(battle, "Good luck! I'm ready to battle!")
                print(f"   💬 Sent startup chat message for battle {battle.battle_tag}")
            except Exception as e:
                print(f"   ⚠️  Could not send startup chat message: {e}")
        
        # Check if we should apply timeout based on battle timer
        # Only apply timeout if battle timer has less than 2x the timeout remaining
        should_apply_timeout = True
        if battle.time_left is not None:
            if battle.time_left > 2 * self.timeout_seconds:
                should_apply_timeout = False
                print(f"   NO PRESSURE: Battle timer has {battle.time_left}s left (> {2 * self.timeout_seconds}s threshold)")
        
        # Log LLM attempt and send chat message when returning to normal mode after timeout
        had_previous_timeout = self.timeout_count > 0
        if had_previous_timeout:
            print(f"   🧠 LLM MODE: Attempting LLM decision (Turn {battle.turn}) - previous timeouts: {self.timeout_count}")
            
            # Send chat message about returning to normal mode (only once per battle after first timeout)
            return_key = f"{battle.battle_tag}_return_normal"
            if return_key not in self.battle_timeout_messages:
                self.battle_timeout_messages.add(return_key)
                try:
                    self._send_chat_message(battle, "Returning to normal mode!")
                    print(f"   💬 RETURN TO NORMAL: Sent chat message to {battle.battle_tag} (Turn {battle.turn})")
                except Exception as e:
                    print(f"   ⚠️  RETURN TO NORMAL: Could not send chat message: {e}")
        
        # Submit BOTH LLM and fallback tasks in parallel for instant timeout recovery
        start_time = time.time()
        llm_future = self.executor.submit(self._execute_llm_move, battle)
        fallback_future = self.executor.submit(self._fast_fallback_move, battle)
        
        try:
            # Wait for LLM result with timeout if applicable
            if should_apply_timeout:
                action = llm_future.result(timeout=self.timeout_seconds)
            else:
                # No timeout - wait indefinitely for LLM
                action = llm_future.result()
            elapsed_time = time.time() - start_time
            
            # LLM succeeded - cancel the fallback future to free resources
            fallback_future.cancel()
            
            # Success - track timing
            self.last_move_time = elapsed_time
            self.move_times.append(elapsed_time)
            print(f"✅ LLM response in {elapsed_time:.1f}s (Turn {battle.turn}) [fallback cancelled]")
            if len(self.move_times) >= 5:  # Show stats after a few moves
                print(f"   Move timing stats: min={min(self.move_times):.1f}s, max={max(self.move_times):.1f}s, avg={sum(self.move_times)/len(self.move_times):.1f}s")
            return action
        
        except FutureTimeoutError:
            # Timeout occurred - use fallback
            elapsed_time = time.time() - start_time
            self.timeout_count += 1
            self.last_move_time = elapsed_time
            self.move_times.append(elapsed_time)
            timeout_rate = (self.timeout_count / self.total_moves) * 100
            
            print(f"\n⚠️  LLM TIMEOUT after {elapsed_time:.1f}s (Turn {battle.turn})")
            print(f"   Timeout rate: {self.timeout_count}/{self.total_moves} ({timeout_rate:.1f}%)")
            if self.move_times:
                print(f"   Move timing stats: min={min(self.move_times):.1f}s, max={max(self.move_times):.1f}s, avg={sum(self.move_times)/len(self.move_times):.1f}s")
            print(f"   Falling back to AbyssalPlayer...")
            
            # Cancel the LLM future to free resources
            llm_future.cancel()
            
            # Send chat message about fallback (only once per timeout situation)
            timeout_key = f"{battle.battle_tag}_turn_{battle.turn}"
            if timeout_key not in self.battle_timeout_messages:
                self.battle_timeout_messages.add(timeout_key)
                try:
                    self._send_chat_message(battle, "Switching to fast mode!")
                    print(f"   💬 TIMEOUT: Sent fallback chat message to {battle.battle_tag} (Turn {battle.turn})")
                except Exception as e:
                    print(f"   ⚠️  TIMEOUT: Could not send chat message: {e}")
                    import traceback
                    print(f"   🔍 TIMEOUT: Full error: {traceback.format_exc()}")
            else:
                print(f"   🔇 TIMEOUT: Skipping duplicate chat message for {battle.battle_tag} (Turn {battle.turn})")
            
            # Use the pre-computed fallback result (should be ready instantly)
            try:
                fallback_action = fallback_future.result(timeout=1.0)  # Should complete almost instantly
                print(f"   ✅ INSTANT FALLBACK: Using pre-computed action: {fallback_action}")
                print(f"   🔄 FAST MODE: Will return to LLM on next turn")
                return fallback_action
            except Exception as e:
                print(f"   ⚠️  Pre-computed fallback not ready, calculating on-demand: {e}")
                # Fallback to on-demand calculation if parallel execution failed
                try:
                    fallback_action = self._fast_fallback_move(battle)
                    print(f"   ✅ ON-DEMAND FALLBACK: {fallback_action}")
                    return fallback_action
                except Exception as fallback_error:
                    print(f"   ❌ All fallbacks failed: {fallback_error}")
                    return self.choose_random_move(battle)
                
        except Exception as e:
            # LLM error - use fallback
            elapsed_time = time.time() - start_time
            print(f"\n❌ LLM error after {elapsed_time:.1f}s: {e}")
            print("   Falling back to AbyssalPlayer...")
            
            # Send chat message about fallback due to error (only once per error situation)
            error_key = f"{battle.battle_tag}_error_turn_{battle.turn}"
            if error_key not in self.battle_timeout_messages:
                self.battle_timeout_messages.add(error_key)
                try:
                    self._send_chat_message(battle, "Switching to fast mode!")
                    print(f"   💬 ERROR FALLBACK: Sent chat message to {battle.battle_tag} (Turn {battle.turn})")
                except Exception as e:
                    print(f"   ⚠️  ERROR FALLBACK: Could not send chat message: {e}")
                    import traceback
                    print(f"   🔍 ERROR FALLBACK: Full error: {traceback.format_exc()}")
            else:
                print(f"   🔇 ERROR FALLBACK: Skipping duplicate chat message for {battle.battle_tag} (Turn {battle.turn})")
            
            # Try to use the pre-computed fallback first, then on-demand
            try:
                fallback_action = fallback_future.result(timeout=1.0)  # Should be ready
                print(f"   ✅ INSTANT FALLBACK: Using pre-computed action: {fallback_action}")
                return fallback_action
            except Exception:
                # Pre-computed not ready, calculate on-demand
                try:
                    fallback_action = self._fast_fallback_move(battle)
                    print(f"   ✅ ON-DEMAND FALLBACK: {fallback_action}")
                    return fallback_action
                except Exception as fallback_error:
                    print(f"   ❌ All fallbacks failed: {fallback_error}")
                    return self.choose_random_move(battle)
    
    def get_timeout_stats(self):
        """Get comprehensive timeout and timing statistics."""
        if self.total_moves == 0:
            return {
                "total_moves": 0,
                "timeouts": 0,
                "timeout_rate": 0.0,
                "min_move_time": 0.0,
                "max_move_time": 0.0,
                "avg_move_time": 0.0,
                "timeout_losses": 0
            }
        
        stats = {
            "total_moves": self.total_moves,
            "timeouts": self.timeout_count,
            "timeout_rate": (self.timeout_count / self.total_moves) * 100,
            "timeout_losses": self.timeout_losses
        }
        
        if self.move_times:
            stats.update({
                "min_move_time": min(self.move_times),
                "max_move_time": max(self.move_times),
                "avg_move_time": sum(self.move_times) / len(self.move_times)
            })
        else:
            stats.update({
                "min_move_time": 0.0,
                "max_move_time": 0.0,
                "avg_move_time": 0.0
            })
        
        return stats
    
    def reset_timeout_stats(self):
        """Reset timeout statistics."""
        self.timeout_count = 0
        self.total_moves = 0
        self.move_times = []
        self.timeout_losses = 0
        self.last_move_time = 0.0
        self.battle_startup_messages.clear()  # Reset startup message tracking
        self.battle_timeout_messages.clear()  # Reset timeout message tracking
        print("Timeout statistics reset")
    
    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon) -> float:
        """Fast matchup estimation."""
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max([mon.damage_multiplier(t) for t in opponent.types if t is not None])
        
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT
        
        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        
        return score
    
    def _should_dynamax(self, battle: AbstractBattle, n_remaining_mons: int) -> bool:
        """Decide if should dynamax."""
        if battle.can_dynamax and self._dynamax_disable is False:
            # Last full HP mon
            if (len([m for m in battle.team.values() if m.current_hp_fraction == 1]) == 1
                and battle.active_pokemon.current_hp_fraction == 1):
                return True
            # Matchup advantage and full hp on full hp
            if (self._estimate_matchup(battle.active_pokemon, battle.opponent_active_pokemon) > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1):
                return True
            if n_remaining_mons == 1:
                return True
        return False
    
    def _should_switch_out(self, battle: AbstractBattle) -> bool:
        """Decide if should switch."""
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        # If there is a decent switch in...
        if [m for m in battle.available_switches if self._estimate_matchup(m, opponent) > 0]:
            # ...and a 'good' reason to switch out
            if active.boosts["def"] <= -3 or active.boosts["spd"] <= -3:
                return True
            if active.boosts["atk"] <= -3 and active.stats["atk"] >= active.stats["spa"]:
                return True
            if active.boosts["spa"] <= -3 and active.stats["atk"] <= active.stats["spa"]:
                return True
            if self._estimate_matchup(active, opponent) < self.SWITCH_OUT_MATCHUP_THRESHOLD:
                return True
        return False
    
    def _stat_estimation(self, mon: Pokemon, stat: str) -> float:
        """Estimate effective stat value."""
        # Stats boosts value
        if mon.boosts[stat] > 1:
            boost = (2 + mon.boosts[stat]) / 2
        else:
            boost = 2 / (2 - mon.boosts[stat])
        return ((2 * mon.base_stats[stat] + 31) + 5) * boost
    
    def _fast_fallback_move(self, battle: AbstractBattle) -> BattleOrder:
        """Fast heuristic-based move selection (Abyssal-style)."""
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        
        # Main mons shortcuts
        active = battle.active_pokemon
        opponent = battle.opponent_active_pokemon
        
        # Rough estimation of damage ratio
        physical_ratio = self._stat_estimation(active, "atk") / self._stat_estimation(opponent, "def")
        special_ratio = self._stat_estimation(active, "spa") / self._stat_estimation(opponent, "spd")
        
        next_action = None
        
        if battle.available_moves and (not self._should_switch_out(battle) or not battle.available_switches):
            n_remaining_mons = len([m for m in battle.team.values() if m.fainted is False])
            n_opp_remaining_mons = 6 - len([m for m in battle.opponent_team.values() if m.fainted is True])
            
            # Entry hazard setup
            for move in battle.available_moves:
                if (n_opp_remaining_mons >= 3 and move.id in self.ENTRY_HAZARDS
                    and self.ENTRY_HAZARDS[move.id] not in battle.opponent_side_conditions):
                    next_action = self.create_order(move)
                    break
                # Hazard removal
                elif (battle.side_conditions and move.id in self.ANTI_HAZARDS_MOVES and n_remaining_mons >= 2):
                    next_action = self.create_order(move)
                    break
            
            # Setup moves
            if (next_action is None and active.current_hp_fraction == 1
                and self._estimate_matchup(active, opponent) > 0):
                for move in battle.available_moves:
                    if (self._boost_disable is False and move.boosts
                        and sum(move.boosts.values()) >= 2 and move.target == "self"
                        and min([active.boosts[s] for s, v in move.boosts.items() if v > 0]) < 6):
                        next_action = self.create_order(move)
                        break
            
            # Best damage move
            if next_action is None:
                move = max(
                    battle.available_moves,
                    key=lambda m: m.base_power
                    * (1.5 if m.type in active.types else 1)
                    * (physical_ratio if m.category == MoveCategory.PHYSICAL else special_ratio)
                    * m.accuracy * m.expected_hits * opponent.damage_multiplier(m)
                )
                next_action = self.create_order(move, dynamax=self._should_dynamax(battle, n_remaining_mons))
        
        # Switch if needed
        if next_action is None and battle.available_switches:
            switches: List[Pokemon] = battle.available_switches
            next_action = self.create_order(
                max(switches, key=lambda s: self._estimate_matchup(s, opponent))
            )
        
        # Ultimate fallback
        if next_action is None:
            next_action = self.choose_random_move(battle)
        
        return next_action
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class PokellmonTimeoutLLMPlayer(TimeoutLLMPlayer):
    """Pokellmon-specific timeout player that uses max base power as fallback."""
    
    def __init__(self, **kwargs):
        # Initialize the base class first
        super().__init__(**kwargs)
        print(f"PokellmonTimeoutLLMPlayer initialized with {self.timeout_seconds}s timeout (max base power fallback)")
    
    def _fast_fallback_move(self, battle: AbstractBattle) -> BattleOrder:
        """Simple max base power fallback for Pokellmon."""
        if isinstance(battle, DoubleBattle):
            return self.choose_random_doubles_move(battle)
        
        # Simple strategy: choose the move with highest base power
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        
        # Switch if no moves available
        if battle.available_switches:
            # Pick the Pokemon with highest HP
            best_switch = max(battle.available_switches, 
                            key=lambda p: p.current_hp_fraction)
            return self.create_order(best_switch)
        
        return self.choose_random_move(battle)