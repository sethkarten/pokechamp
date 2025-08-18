"""
Timeout-enabled LLM Player with AbyssalPlayer fallback.

This module provides a wrapper around LLMPlayer that adds a watchdog timeout.
If the LLM takes too long to respond (default 90 seconds), it automatically
falls back to using AbyssalPlayer's decision-making logic.
"""

import time
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.player import BattleOrder
from pokechamp.llm_player import LLMPlayer
from poke_env.player.baselines import AbyssalPlayer, MaxBasePowerPlayer


class TimeoutLLMPlayer(LLMPlayer):
    """
    LLM Player with timeout protection and AbyssalPlayer fallback.
    
    This class wraps the normal LLMPlayer and adds a timeout mechanism.
    If the LLM doesn't respond within the timeout period, it automatically
    falls back to using AbyssalPlayer's heuristic-based decision making.
    """
    
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
        
        # Thread pool for executing LLM calls with cleaner timeout handling
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="llm_timeout")
        
        # Initialize fallback AbyssalPlayer with same configuration
        self.fallback_player = AbyssalPlayer(
            battle_format=self.format,
            team=kwargs.get('team'),
            account_configuration=kwargs.get('account_configuration'),
            server_configuration=kwargs.get('server_configuration')
        )
        
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
        If the LLM doesn't respond within timeout_seconds, it falls back to
        AbyssalPlayer's decision making.
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
        
        # Submit LLM task to executor for cleaner timeout handling
        start_time = time.time()
        future = self.executor.submit(self._execute_llm_move, battle)
        
        try:
            # Wait for result with timeout - this is more accurate than thread.join()
            action = future.result(timeout=self.timeout_seconds)
            elapsed_time = time.time() - start_time
            
            # Success - track timing
            self.last_move_time = elapsed_time
            self.move_times.append(elapsed_time)
            print(f"✅ LLM response in {elapsed_time:.1f}s (Turn {battle.turn})")
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
            
            # Cancel the future to free resources
            future.cancel()
            
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
            
            # Use AbyssalPlayer fallback
            try:
                fallback_action = self.fallback_player.choose_move(battle)
                print(f"   ✅ FAST MODE: Using AbyssalPlayer action: {fallback_action}")
                print(f"   🔄 FAST MODE: Will return to LLM on next turn")
                return fallback_action
            except Exception as e:
                print(f"   ❌ Fallback failed: {e}")
                # Ultimate fallback - choose random move
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
            
            try:
                fallback_action = self.fallback_player.choose_move(battle)
                print(f"   ✅ Fallback action: {fallback_action}")
                return fallback_action
            except Exception as fallback_error:
                print(f"   ❌ Fallback failed: {fallback_error}")
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
    
    def __del__(self):
        """Cleanup thread pool on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)

class PokellmonTimeoutLLMPlayer(TimeoutLLMPlayer):
    """Pokellmon-specific timeout player that uses MaxBasePowerPlayer as fallback."""
    
    def __init__(self, **kwargs):
        # Initialize the base class first
        super().__init__(**kwargs)
        
        # Replace the AbyssalPlayer fallback with MaxBasePowerPlayer for pokellmon
        self.fallback_player = MaxBasePowerPlayer(
            battle_format=self.format,
            team=kwargs.get('team'),
            account_configuration=kwargs.get('account_configuration'),
            server_configuration=kwargs.get('server_configuration')
        )
        
        print(f"PokellmonTimeoutLLMPlayer initialized with {self.timeout_seconds}s timeout (MaxBasePowerPlayer fallback)")