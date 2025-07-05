#!/usr/bin/env python3
"""
Starter Kit Bot: Basic LLM-based Pokémon Bot

This is a simple example showing how to create your own LLM-based Pokémon bot
using the PokéChamp framework. This example uses the "io" (input-output) prompt format,
which is the most basic and easiest to understand.

To create your own bot:
1. Copy this file and rename it
2. Modify the system prompt and constraint prompts to match your strategy
3. Add your own custom logic in the choose_move method if needed
4. Test your bot using the provided scripts

Author: PokéChamp Team
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
from poke_env.player.llm_player import LLMPlayer
from poke_env.player.gpt_player import GPTPlayer
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.local_simulation import LocalSim
from poke_env.player.prompts import state_translate


class StarterKitBot(LLMPlayer):
    """
    A basic example bot that demonstrates how to create your own LLM-based Pokémon bot.
    
    This bot uses the "io" (input-output) prompt format, which is the simplest approach:
    - It takes the current battle state as input
    - Outputs a JSON response with the chosen action
    - Uses a simple system prompt to define the bot's personality and strategy
    
    Key features to customize:
    1. System prompt: Defines your bot's personality and overall strategy
    2. Constraint prompts: Define the output format and specific instructions
    3. Custom logic: Add your own decision-making logic if needed
    """
    
    def __init__(self, 
                 battle_format: str = "gen9ou",
                 api_key: str = "",
                 backend: str = "gpt-4o",
                 temperature: float = 0.3,
                 log_dir: str = "./battle_log/starter_kit",
                 **kwargs):
        """
        Initialize your custom bot.
        
        Args:
            battle_format: The Pokémon format to play (e.g., "gen9ou", "gen9randombattle")
            api_key: Your OpenAI API key (or other LLM provider key)
            backend: The LLM model to use (e.g., "gpt-4o", "gpt-4o-mini", "llama")
            temperature: Controls randomness in LLM responses (0.0 = deterministic, 1.0 = very random)
            log_dir: Directory to save battle logs and LLM outputs
            **kwargs: Additional arguments passed to the parent LLMPlayer class
        """
        super().__init__(
            battle_format=battle_format,
            api_key=api_key,
            backend=backend,
            temperature=temperature,
            prompt_algo="io",  # Use the simple input-output format
            log_dir=log_dir,
            **kwargs
        )
        
        # You can add custom attributes here
        self.bot_name = "StarterKitBot"
        self.bot_personality = "aggressive"  # Example: could be "defensive", "balanced", etc.
        
    def get_custom_system_prompt(self) -> str:
        """
        Define your bot's system prompt - this is where you set the personality and strategy.
        
        The system prompt is sent to the LLM at the beginning of each conversation
        and defines how the bot should behave. You can customize this to create
        different types of bots (aggressive, defensive, strategic, etc.).
        
        Returns:
            str: The system prompt that defines your bot's behavior
        """
        # This is a simple example system prompt - customize it for your bot!
        system_prompt = f"""You are {self.bot_name}, a {self.bot_personality} Pokémon trainer.

Your goal is to win Pokémon battles by making smart decisions. You should:

1. **Analyze the current situation**: Consider your Pokémon's HP, status, moves, and the opponent's Pokémon
2. **Choose the best action**: Either use a move or switch to a different Pokémon
3. **Think strategically**: Consider type advantages, move effectiveness, and battle momentum
4. **Be decisive**: Make clear, confident decisions based on the available information
5. **Use only available moves**: Only choose from the moves that are actually available to your Pokémon

You are playing in the {self.format} format. Always respond with valid JSON containing your chosen action.

Remember: Your response must be a valid JSON object with either a "move" or "switch" key."""
        
        return system_prompt
    
    def get_custom_constraint_prompt(self, battle: Battle) -> str:
        """
        Define the constraint prompt that tells the LLM how to format its response.
        
        This prompt is added to the battle state information and tells the LLM
        exactly what format to use for its response. You can customize this to
        add specific instructions or change the output format.
        
        Args:
            battle: The current battle state
            
        Returns:
            str: The constraint prompt that defines the output format
        """
        # Check if we need to switch (Pokémon fainted or no moves available)
        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:
            constraint_prompt = '''Choose the most suitable Pokémon to switch to. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}

Consider:
- Type advantages against the opponent's active Pokémon
- Your Pokémon's HP and status conditions
- The overall team composition'''
            
        # Check if we can only use moves (no switches available)
        elif len(battle.available_switches) == 0:
            constraint_prompt = '''Choose the best move to use. Your output MUST be a JSON like: {"move":"<move_name>"}

Consider:
- Move effectiveness against the opponent's type
- Your Pokémon's current HP and status
- Whether the move will KO the opponent
- Any status effects or stat changes the move might cause'''
            
        # We can both move and switch
        else:
            constraint_prompt = '''Choose the best action. Your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}

Consider:
- Whether to attack or switch based on the current matchup
- Type advantages and disadvantages
- Your Pokémon's HP and status
- The opponent's potential moves and strategy'''
        
        return constraint_prompt
    
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        The main method that decides what action to take in the battle.
        
        This method is called by the game engine whenever it's your turn.
        You can override this method to add custom logic before or after
        the LLM makes its decision.
        
        Args:
            battle: The current battle state
            
        Returns:
            BattleOrder: The action to take (move or switch)
        """
        # Handle special cases first (these are already handled by the parent class)
        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                # Only one switch available, use it
                return BattleOrder(battle.available_switches[0])
            elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
                # Only one move available, use it
                return self.choose_max_damage_move(battle)
        
        # Get the standard battle state information
        sim = LocalSim(battle, 
                      self.move_effect,
                      self.pokemon_move_dict,
                      self.ability_effect,
                      self.pokemon_ability_dict,
                      self.item_effect,
                      self.pokemon_item_dict,
                      self.gen,
                      self._dynamax_disable,
                      format=self.format,
                      prompt_translate=self.prompt_translate
        )
        
        # Get the system prompt and battle state
        system_prompt = self.get_custom_system_prompt()
        _, state_prompt, state_action_prompt = sim.state_translate(battle)
        
        # Get available actions for validation
        moves = [move.id for move in battle.available_moves]
        switches = [pokemon.species for pokemon in battle.available_switches]
        actions = [moves, switches]
        
        # Get the constraint prompt based on the current situation
        constraint_prompt = self.get_custom_constraint_prompt(battle)
        
        # Combine all prompts
        full_prompt = state_prompt + state_action_prompt + constraint_prompt
        
        # Add a small thinking prompt (optional)
        thinking_prompt = '\nIn fewer than 3 sentences, think step by step about your decision:'
        full_prompt += thinking_prompt
        
        # Call the LLM to get the decision
        retries = 2
        for attempt in range(retries):
            try:
                # Get the LLM's response
                llm_output = self.get_LLM_action(
                    system_prompt=system_prompt,
                    user_prompt=full_prompt,
                    model=self.backend,
                    temperature=self.temperature,
                    max_tokens=300,
                    json_format=True,
                    actions=actions
                )
                
                # Parse the JSON response
                llm_action_json = json.loads(llm_output)
                
                # Handle the response based on the action type
                if "move" in llm_action_json:
                    # The LLM chose to use a move
                    move_name = llm_action_json["move"].strip()
                    
                    # Find the move in the available moves
                    for move in battle.available_moves:
                        if move.id.lower().replace(' ', '') == move_name.lower().replace(' ', ''):
                            print(f"Using move: {move.id}")
                            return self.create_order(move)
                            
                elif "switch" in llm_action_json:
                    # The LLM chose to switch Pokémon
                    switch_name = llm_action_json["switch"].strip()
                    
                    # Find the Pokémon in the available switches
                    for pokemon in battle.available_switches:
                        if pokemon.species.lower().replace(' ', '') == switch_name.lower().replace(' ', ''):
                            print(f"Switching to: {pokemon.species}")
                            return self.create_order(pokemon)
                
                # If we get here, the action wasn't valid
                raise ValueError(f"Invalid action in LLM response: {llm_action_json}")
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt == retries - 1:
                    # If all retries failed, use a fallback strategy
                    print("All retries failed, using fallback strategy")
                    return self.choose_max_damage_move(battle)
                continue
        
        # This should never be reached, but just in case
        return self.choose_max_damage_move(battle)


# Example usage function
async def test_starter_kit_bot():
    """
    Example function showing how to use your custom bot.
    
    This function demonstrates how to create and test your bot against
    other bots or human players.
    """
    # Create your custom bot
    bot = StarterKitBot(
        battle_format="gen9ou",
        api_key=os.getenv('OPENAI_API_KEY'),  # Replace with your actual API key
        backend="gpt-4o",
        temperature=0.3,
        log_dir="./battle_log/starter_kit"
    )
    
    # You can now use this bot in battles
    # For example, you could import and use it in local_1v1.py or other scripts
    
    print("Starter Kit Bot created successfully!")
    print("To use this bot:")
    print("1. Replace 'your-api-key-here' with your actual API key")
    print("2. Import this bot in your battle scripts")
    print("3. Customize the system prompt and constraint prompts")
    print("4. Test it against other bots!")


if __name__ == "__main__":
    # Run the example
    asyncio.run(test_starter_kit_bot()) 