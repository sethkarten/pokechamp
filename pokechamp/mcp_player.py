"""
MCP Player - MCP-controlled Pokemon player

Pokemon player that has built-in LLM loop using MCP endpoints.
Gets state, calls LLM via MCP, gets decision, executes action.
"""

import json
import logging
import requests
import time
from typing import Dict, Any
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player.player import BattleOrder
from pokechamp.llm_player import LLMPlayer

logger = logging.getLogger(__name__)

class MCPPlayer(LLMPlayer):
    """
    Pokemon player controlled via MCP protocol.
    Exposes full battle state via endpoints and gets decisions from external LLM.
    """
    
    def __init__(self, mcp_host="localhost", mcp_port=8000, **kwargs):
        # Set prompt_algo to mcp but keep backend as normal LLM
        kwargs['prompt_algo'] = 'mcp'
        if 'backend' not in kwargs:
            kwargs['backend'] = 'gemini-2.5-flash'  # Default LLM backend
        super().__init__(**kwargs)
        self.mcp_host = mcp_host
        self.mcp_port = mcp_port
        self.mcp_base_url = f"http://{mcp_host}:{mcp_port}"
        
    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        """
        MCP/LLM Loop: Get state -> Call LLM with state -> Get action -> Execute
        This is the core loop that replaces normal LLM decision making.
        """
        print("MCPPlayer choose_move")
        # Handle forced actions first
        if self._is_forced_action(battle):
            return self._handle_forced_action(battle)
        
        # MCP/LLM LOOP START
        try:
            # Step 1: Get full battle state (automatic)
            battle_state = self._get_full_battle_state(battle)
            print(f"[MCP] Turn {battle.turn}: Got battle state")
            
            # Step 2: Call LLM with battle state to get action
            decision = self._call_llm_with_state(battle_state, battle)
            print(f"[MCP] LLM decision: {decision.get('action')} -> {decision.get('target')}")
            
            # Step 3: Execute the LLM's decision
            return self._execute_decision(battle, decision)
            
        except Exception as e:
            logger.warning(f"MCP/LLM loop failed: {e}, using fallback")
            return self.choose_max_damage_move(battle)
    
    def _is_forced_action(self, battle: AbstractBattle) -> bool:
        """Check if there's only one possible action"""
        if battle.active_pokemon and battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            return True
        elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
            return True
        return False
    
    def _handle_forced_action(self, battle: AbstractBattle) -> BattleOrder:
        """Handle forced actions"""
        if battle.active_pokemon and battle.active_pokemon.fainted and len(battle.available_switches) == 1:
            return self.create_order(battle.available_switches[0])
        elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1:
            return self.create_order(battle.available_moves[0])
        else:
            return self.choose_max_damage_move(battle)
    
    def _get_full_battle_state(self, battle: AbstractBattle) -> Dict[str, Any]:
        """Get comprehensive battle state for MCP exposure"""
        
        # Available moves with full details
        available_moves = []
        if battle.available_moves:
            for move in battle.available_moves:
                available_moves.append({
                    "id": move.id,
                    "name": move.id,
                    "type": str(move.type) if move.type else None,
                    "category": str(move.category) if move.category else None,
                    "power": move.base_power,
                    "accuracy": move.accuracy,
                    "pp": move.current_pp,
                    "max_pp": move.max_pp,
                    "priority": move.priority,
                    "crit_ratio": move.crit_ratio
                })
        
        # Available switches with full details
        available_switches = []
        if battle.available_switches:
            for pokemon in battle.available_switches:
                available_switches.append({
                    "species": pokemon.species,
                    "hp_fraction": pokemon.current_hp_fraction,
                    "status": str(pokemon.status) if pokemon.status else None,
                    "level": pokemon.level,
                    "types": [str(t) for t in pokemon.types if t],
                    "ability": str(pokemon.ability) if pokemon.ability else None,
                    "item": str(pokemon.item) if pokemon.item else None,
                    "stats": dict(pokemon.stats) if pokemon.stats else {}
                })
        
        # Full team information
        team_info = []
        for pokemon in battle.team.values():
            team_info.append({
                "species": pokemon.species,
                "hp_fraction": pokemon.current_hp_fraction,
                "fainted": pokemon.fainted,
                "active": pokemon.active,
                "status": str(pokemon.status) if pokemon.status else None,
                "level": pokemon.level,
                "types": [str(t) for t in pokemon.types if t],
                "ability": str(pokemon.ability) if pokemon.ability else None,
                "item": str(pokemon.item) if pokemon.item else None,
                "stats": dict(pokemon.stats) if pokemon.stats else {},
                "boosts": dict(pokemon.boosts) if pokemon.boosts else {},
                "moves": [str(move) for move in pokemon.moves] if pokemon.moves else []
            })
        
        # Visible opponent team information
        opponent_team_info = []
        for pokemon in battle.opponent_team.values():
            if pokemon.species:
                opponent_team_info.append({
                    "species": pokemon.species,
                    "hp_fraction": pokemon.current_hp_fraction if pokemon.active else None,
                    "fainted": pokemon.fainted,
                    "active": pokemon.active,
                    "level": pokemon.level,
                    "types": [str(t) for t in pokemon.types if t],
                    "ability": str(pokemon.ability) if pokemon.ability else None,
                    "item": str(pokemon.item) if pokemon.item else None,
                    "stats": dict(pokemon.stats) if pokemon.stats else {},
                    "boosts": dict(pokemon.boosts) if pokemon.boosts else {},
                    "moves": [str(move) for move in pokemon.moves] if pokemon.moves else []
                })
        
        # Active Pokemon details
        active_pokemon = None
        if battle.active_pokemon:
            active_pokemon = {
                "species": battle.active_pokemon.species,
                "hp_fraction": battle.active_pokemon.current_hp_fraction,
                "fainted": battle.active_pokemon.fainted,
                "status": str(battle.active_pokemon.status) if battle.active_pokemon.status else None,
                "level": battle.active_pokemon.level,
                "types": [str(t) for t in battle.active_pokemon.types if t],
                "ability": str(battle.active_pokemon.ability) if battle.active_pokemon.ability else None,
                "item": str(battle.active_pokemon.item) if battle.active_pokemon.item else None,
                "stats": dict(battle.active_pokemon.stats) if battle.active_pokemon.stats else {},
                "boosts": dict(battle.active_pokemon.boosts) if battle.active_pokemon.boosts else {},
                "moves": [str(move) for move in battle.active_pokemon.moves] if battle.active_pokemon.moves else []
            }
        
        # Opponent active Pokemon details
        opponent_active_pokemon = None
        if battle.opponent_active_pokemon:
            opponent_active_pokemon = {
                "species": battle.opponent_active_pokemon.species,
                "hp_fraction": battle.opponent_active_pokemon.current_hp_fraction,
                "fainted": battle.opponent_active_pokemon.fainted,
                "status": str(battle.opponent_active_pokemon.status) if battle.opponent_active_pokemon.status else None,
                "level": battle.opponent_active_pokemon.level,
                "types": [str(t) for t in battle.opponent_active_pokemon.types if t],
                "ability": str(battle.opponent_active_pokemon.ability) if battle.opponent_active_pokemon.ability else None,
                "item": str(battle.opponent_active_pokemon.item) if battle.opponent_active_pokemon.item else None,
                "stats": dict(battle.opponent_active_pokemon.stats) if battle.opponent_active_pokemon.stats else {},
                "boosts": dict(battle.opponent_active_pokemon.boosts) if battle.opponent_active_pokemon.boosts else {},
                "moves": [str(move) for move in battle.opponent_active_pokemon.moves] if battle.opponent_active_pokemon.moves else []
            }
        
        return {
            "battle_tag": battle.battle_tag,
            "turn": battle.turn,
            "battle_active": not battle.finished,
            "won": battle.won if battle.finished else None,
            "lost": battle.lost if battle.finished else None,
            "format": self.format,
            "active_pokemon": active_pokemon,
            "opponent_active_pokemon": opponent_active_pokemon,
            "available_moves": available_moves,
            "available_switches": available_switches,
            "can_dynamax": battle.can_dynamax,
            "can_tera": battle.can_tera,
            "team": team_info,
            "opponent_team": opponent_team_info,
            "weather": str(battle.weather) if battle.weather else None,
            "terrain": str(battle.fields.get('terrain', '')) if battle.fields else None,
            "side_conditions": {
                "player": [str(effect) for effect in battle.side_conditions] if battle.side_conditions else [],
                "opponent": [str(effect) for effect in battle.opponent_side_conditions] if battle.opponent_side_conditions else []
            },
            "force_switch": battle.force_switch,
            "maybe_trapped": battle.maybe_trapped,
            "trapped": battle.trapped
        }
    
    def _call_llm_with_state(self, battle_state: Dict[str, Any], battle: AbstractBattle) -> Dict[str, Any]:
        """Call LLM directly with battle state to get action decision"""
        
        # Create prompt for LLM with full battle state
        system_prompt = "You are a Pokemon battle AI. Analyze the battle state and choose the best action."
        
        # Format battle state into readable prompt
        user_prompt = self._format_state_for_llm(battle_state)
        
        # Call LLM using existing infrastructure
        try:
            llm_output = self.get_LLM_action(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.backend,
                temperature=self.temperature,
                json_format=True,
                max_tokens=200,
                battle=battle
            )
            # Parse LLM response
            decision = json.loads(llm_output)
            return decision
            
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            # Simple fallback decision
            moves = battle_state.get("available_moves", [])
            if moves:
                return {"action": "move", "target": moves[0]["name"]}
            else:
                return {"action": "move", "target": "struggle"}
    
    def _format_state_for_llm(self, state: Dict[str, Any]) -> str:
        """Format battle state into readable prompt for LLM"""
        
        active = state.get("active_pokemon", {})
        opponent = state.get("opponent_active_pokemon", {}) 
        moves = state.get("available_moves", [])
        switches = state.get("available_switches", [])
        
        prompt = f"""Battle State Analysis:

Turn: {state.get('turn', 0)}
Format: {state.get('format', 'unknown')}

Your Active Pokemon:
- Species: {active.get('species', 'Unknown')}
- HP: {active.get('hp_fraction', 0):.1%}
- Types: {', '.join(active.get('types', []))}
- Status: {active.get('status') or 'None'}
- Ability: {active.get('ability', 'Unknown')}
- Item: {active.get('item', 'Unknown')}
- Stats: {active.get('stats', {})}
- Stat Boosts: {active.get('boosts', {})}

Opponent's Active Pokemon:
- Species: {opponent.get('species', 'Unknown')}
- HP: {opponent.get('hp_fraction', 0):.1%}
- Types: {', '.join(opponent.get('types', []))}
- Status: {opponent.get('status') or 'None'}
- Item: {opponent.get('item', 'Unknown')}
- Stat Boosts: {opponent.get('boosts', {})}

Available Moves:
"""
        
        for i, move in enumerate(moves):
            prompt += f"- {move['name']}: {move.get('type', 'Unknown')} type, {move.get('power', 0)} power, {move.get('accuracy', 100)} accuracy\n"
        
        if switches:
            prompt += "\nAvailable Switches:\n"
            for switch in switches:
                prompt += f"- {switch['species']}: {switch.get('hp_fraction', 0):.1%} HP, {', '.join(switch.get('types', []))} type"
                if switch.get('ability'):
                    prompt += f", {switch['ability']} ability"
                if switch.get('status'):
                    prompt += f", {switch['status']} status"
                if switch.get('item'):
                    prompt += f", {switch['item']} item"
                if switch.get('stats'):
                    stats = switch['stats']
                    prompt += f", Stats: {stats.get('atk', 0)}/{stats.get('def', 0)}/{stats.get('spa', 0)}/{stats.get('spd', 0)}/{stats.get('spe', 0)}"
                if switch.get('moves'):
                    prompt += f", Known moves: {', '.join(switch['moves'][:4])}"  # Show first 4 moves
                prompt += "\n"
        
        prompt += f"""
Battle Conditions:
- Weather: {state.get('weather') or 'None'}
- Terrain: {state.get('terrain') or 'None'}
- Can Dynamax: {state.get('can_dynamax', False)}
- Can Terastallize: {state.get('can_tera', False)}

Choose the best action. Output format:
{{"action": "move", "target": "move_name"}} or {{"action": "switch", "target": "pokemon_species"}}
"""
        print(prompt)
        print("--------------------------------")
        return prompt
    
    def _execute_decision(self, battle: AbstractBattle, decision: Dict[str, str]) -> BattleOrder:
        """Execute the decision"""
        action = decision.get("action")
        target = decision.get("target")
        
        if action == "move" and target:
            for move in battle.available_moves:
                if move.id.lower().replace(' ', '') == target.lower().replace(' ', ''):
                    return self.create_order(move)
            # Fallback if move not found
            return self.choose_max_damage_move(battle)
            
        elif action == "switch" and target:
            for pokemon in battle.available_switches:
                if pokemon.species.lower().replace(' ', '') == target.lower().replace(' ', ''):
                    return self.create_order(pokemon)
            # Fallback if pokemon not found
            return self.choose_max_damage_move(battle)
        
        return self.choose_max_damage_move(battle)