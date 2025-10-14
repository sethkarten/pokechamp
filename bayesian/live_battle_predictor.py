#!/usr/bin/env python3
"""
Live Battle Predictor - Single integrated solution

Hooks into the existing battle system to show real-time Bayesian predictions
using already-parsed battle data. Shows predictions after each turn.
"""

import asyncio
import os
import sys
from typing import Dict, List, Set
from collections import defaultdict

from common import PNUMBER1
from poke_env.player.team_util import get_llm_player, load_random_team
from poke_env.environment.battle import Battle
from poke_env.player.baselines import AbyssalPlayer
from bayesian.predictor_singleton import get_pokemon_predictor

# Visual effects import (optional)
try:
    from pokechamp.visual_effects import visual, print_banner, print_status
    VISUAL_EFFECTS = True
except ImportError:
    VISUAL_EFFECTS = False


class LiveBattlePredictor(AbyssalPlayer):
    """Abyssal player that shows live Bayesian predictions every turn."""
    
    def __init__(self, *args, **kwargs):
        # Initialize attributes BEFORE calling super().__init__
        self.last_prediction_turn = 0
        self.predictor = None
        
        # Create name mapping for common Pokemon
        self.name_mapping = {
            'slowkinggalar': 'Slowking-Galar',
            'slowbrogalar': 'Slowbro-Galar',
            'tinglu': 'Ting-Lu',
            'chiyu': 'Chi-Yu',
            'wochien': 'Wo-Chien',
            'chienpao': 'Chien-Pao',
            'ironmoth': 'Iron Moth',
            'ironvaliant': 'Iron Valiant',
            'irontreads': 'Iron Treads',
            'ironbundle': 'Iron Bundle',
            'ironhands': 'Iron Hands',
            'ironjugulis': 'Iron Jugulis',
            'ironthorns': 'Iron Thorns',
            'ironboulder': 'Iron Boulder',
            'ironcrown': 'Iron Crown',
            'greattusk': 'Great Tusk',
            'screamtail': 'Scream Tail',
            'brutebonnet': 'Brute Bonnet',
            'fluttermane': 'Flutter Mane',
            'slitherwing': 'Slither Wing',
            'sandyshocks': 'Sandy Shocks',
            'roaringmoon': 'Roaring Moon',
            'walkingwake': 'Walking Wake',
            'ragingbolt': 'Raging Bolt',
            'gougingfire': 'Gouging Fire',
            'ogerponwellspring': 'Ogerpon-Wellspring',
            'ogerponhearthflame': 'Ogerpon-Hearthflame',
            'ogerponcornerstone': 'Ogerpon-Cornerstone',
            'ogerpontealtera': 'Ogerpon-Teal',
            'ursalunabloodmoon': 'Ursaluna-Bloodmoon',
            'ninetalesalola': 'Ninetales-Alola',
            'sandslashalola': 'Sandslash-Alola',
            'tapukoko': 'Tapu Koko',
            'tapulele': 'Tapu Lele',
            'tapubulu': 'Tapu Bulu',
            'tapufini': 'Tapu Fini',
            'hydrapple': 'Hydrapple',
            'zapdos': 'Zapdos',
            'zamazenta': 'Zamazenta',
            'tinkaton': 'Tinkaton',
        }
        
        # Call parent init
        super().__init__(*args, **kwargs)
        
        # Initialize predictor after parent init
        if VISUAL_EFFECTS:
            print_banner("PREDICTOR", "purplepink")
        else:
            print("[INIT] Initializing Live Battle Predictor...")
        try:
            self.predictor = get_pokemon_predictor()
            print("[OK] Bayesian predictor ready!")
        except Exception as e:
            print(f"[ERROR] Failed to load predictor: {e}")
            self.predictor = None
    
    def normalize_pokemon_name(self, name: str) -> str:
        """Normalize Pokemon name from battle format to training data format."""
        # Ensure name_mapping exists
        if not hasattr(self, 'name_mapping'):
            return name.capitalize()
            
        # First check if it's in our mapping
        lower_name = name.lower()
        if lower_name in self.name_mapping:
            return self.name_mapping[lower_name]
        
        # Otherwise, capitalize first letter of each word part
        # Handle special cases like "mr-mime" -> "Mr. Mime"
        if lower_name == 'mrmime':
            return 'Mr. Mime'
        elif lower_name == 'mimejr':
            return 'Mime Jr.'
        elif lower_name == 'typenull':
            return 'Type: Null'
        elif lower_name == 'hooh':
            return 'Ho-Oh'
        elif lower_name == 'porygonz':
            return 'Porygon-Z'
        elif lower_name == 'porygon2':
            return 'Porygon2'
        
        # Default: capitalize first letter
        return name.capitalize()
    
    def normalize_move_name(self, move_id: str) -> str:
        """Normalize move name from battle format to training data format."""
        # Common move name transformations
        move_mapping = {
            'chillyreception': 'Chilly Reception',
            'thunderwave': 'Thunder Wave', 
            'stealthrock': 'Stealth Rock',
            'earthquake': 'Earthquake',
            'ruination': 'Ruination',
            'whirlwind': 'Whirlwind',
            'spikes': 'Spikes',
            'rest': 'Rest',
            'closecombat': 'Close Combat',
            'crunch': 'Crunch',
            'gigadrain': 'Giga Drain',
            'earthpower': 'Earth Power',
            'nastyplot': 'Nasty Plot',
            'ficklebeam': 'Fickle Beam',
            'leafstorm': 'Leaf Storm',
            'dracometeor': 'Draco Meteor',
            'futuresight': 'Future Sight',
            'sludgebomb': 'Sludge Bomb',
            'psychicnoise': 'Psychic Noise',
            'flamethrower': 'Flamethrower',
            'gigatonhammer': 'Gigaton Hammer',
            'encore': 'Encore',
            'knockoff': 'Knock Off',
            'playrough': 'Play Rough'
        }
        
        # Check if we have a direct mapping
        lower_move = move_id.lower()
        if lower_move in move_mapping:
            return move_mapping[lower_move]
        
        # Default: capitalize first letter and add spaces before capital letters
        # Convert "iceBeam" to "Ice Beam"
        import re
        # Add space before capital letters that follow lowercase letters
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', move_id)
        return spaced.title()
    
    def _battle_started_callback(self, battle: Battle):
        """Called when battle starts."""
        super()._battle_started_callback(battle)
        print(f"\n{'='*70}")
        if VISUAL_EFFECTS:
            print(visual.battle_header("YOU", "OPPONENT"))
            print(f"Battle ID: {battle.battle_tag}")
        else:
            print(f"[BATTLE] BATTLE STARTED: {battle.battle_tag}")
        print(f"{'='*70}")
        print(f"[TEAM-BLUE] Your team: {', '.join([p.species for p in battle.team.values() if p])}")
        print(f"[TEAM-RED] Opponent: Predictions will appear as Pokemon are revealed...")
        print(f"{'='*70}\n")
    
    def _battle_finished_callback(self, battle: Battle):
        """Called when battle ends."""
        super()._battle_finished_callback(battle)
        
        print(f"\n{'='*70}")
        if VISUAL_EFFECTS:
            winner = "YOU" if self.n_won_battles > 0 else "OPPONENT"
            print(visual.victory_banner(winner, battle.turn))
        else:
            print(f"[FINISH] BATTLE FINISHED")
        print(f"{'='*70}")
        
        # Show final team analysis
        opponent_pokemon = []
        revealed_moves = defaultdict(set)
        
        for pokemon in battle.opponent_team.values():
            if pokemon and pokemon.species:
                opponent_pokemon.append(pokemon.species)
                for move in pokemon.moves.values():
                    if move:
                        revealed_moves[pokemon.species].add(move.id)
        
        print(f"[TEAM-RED] Final opponent team ({len(opponent_pokemon)}/6):")
        for i, species in enumerate(opponent_pokemon, 1):
            moves = list(revealed_moves[species])
            if moves:
                print(f"   {i}. {species}: {', '.join(moves[:4])}")
            else:
                print(f"   {i}. {species}: No moves revealed")
        
        result = "Victory! [WIN]" if self.n_won_battles > 0 else "Defeat"
        print(f"\n[RESULT] {result}")
        print(f"⏱️  Total turns: {battle.turn}")
        print(f"[STATS] Win rate: {self.win_rate*100:.1f}%")
        print(f"{'='*70}")
    
    def show_live_predictions(self, battle: Battle):
        """Show live predictions for moves, items, and EVs since all Pokemon are revealed in team preview."""
        # Initialize if needed
        if not hasattr(self, 'last_prediction_turn'):
            self.last_prediction_turn = 0
            
        if battle.turn <= self.last_prediction_turn or battle.turn < 1:
            return
        
        # Extract all opponent Pokemon (all 6 revealed in team preview)
        opponent_pokemon_raw = []  # Raw names from battle
        opponent_pokemon_normalized = []  # Normalized for predictor
        revealed_moves = defaultdict(set)
        revealed_items = {}
        revealed_abilities = {}
        
        # Get data from opponent_team (all 6 Pokemon revealed from team preview)
        for pokemon in battle.opponent_team.values():
            if pokemon and pokemon.species:
                opponent_pokemon_raw.append(pokemon.species)
                normalized_name = self.normalize_pokemon_name(pokemon.species)
                opponent_pokemon_normalized.append(normalized_name)
                
                # Track revealed moves (normalize move names to match training data)
                for move in pokemon.moves.values():
                    if move:
                        # Convert move.id to proper case with spaces for matching training data
                        normalized_move = self.normalize_move_name(move.id)
                        revealed_moves[pokemon.species].add(normalized_move)
                
                # Track revealed item and ability (only if actually revealed, not just placeholder)
                if hasattr(pokemon, 'item') and pokemon.item and str(pokemon.item) != "None":
                    revealed_items[pokemon.species] = str(pokemon.item)
                if hasattr(pokemon, 'ability') and pokemon.ability and str(pokemon.ability) != "None":
                    revealed_abilities[pokemon.species] = str(pokemon.ability)
        
        if not opponent_pokemon_raw:
            return
        
        if VISUAL_EFFECTS:
            print(visual.battle_turn(battle.turn, "Analyzing opponent..."))
        else:
            print(f"\n[TURN {battle.turn}] MOVE/ITEM/EV PREDICTIONS")
        print("-" * 60)
        
        # Current battle state
        if battle.active_pokemon and battle.opponent_active_pokemon:
            your_hp = battle.active_pokemon.current_hp_fraction * 100
            opp_hp = battle.opponent_active_pokemon.current_hp_fraction * 100
            your_status = f" ({battle.active_pokemon.status.name})" if battle.active_pokemon.status else ""
            opp_status = f" ({battle.opponent_active_pokemon.status.name})" if battle.opponent_active_pokemon.status else ""
            
            print(f"[MATCHUP] CURRENT BATTLE:")
            print(f"[BLUE] YOU: {battle.active_pokemon.species} - {your_hp:.0f}% HP{your_status}")
            print(f"[RED] OPP: {battle.opponent_active_pokemon.species} - {opp_hp:.0f}% HP{opp_status}")
        
        print(f"\n[ROSTER] OPPONENT TEAM ({len(opponent_pokemon_raw)}/6 revealed from team preview):")
        for i, species in enumerate(opponent_pokemon_raw, 1):
            moves = list(revealed_moves[species])
            item = revealed_items.get(species)
            ability = revealed_abilities.get(species)
            
            if moves:
                move_display = f"Moves: {', '.join(moves[:4])}"
            else:
                move_display = "No moves revealed yet"
            
            item_display = f"Item: {item}" if item else "Item: ?"
            print(f"   {i}. {species} - {move_display} | {item_display}")
        
        # Predict moves, items, and EVs for each Pokemon
        if VISUAL_EFFECTS:
            print(visual.create_banner("PREDICT", font="small", style="greenblue"))
        else:
            print(f"\n[PREDICT] BAYESIAN MOVE/ITEM/EV PREDICTIONS:")
        if self.predictor is None:
            print("   [ERROR] Predictor not loaded")
            return
            
        try:
            for i, species_raw in enumerate(opponent_pokemon_raw, 1):
                species_norm = self.normalize_pokemon_name(species_raw)
                observed_moves = list(revealed_moves[species_raw])
                known_item = revealed_items.get(species_raw)
                
                # Skip if all info already revealed
                if len(observed_moves) >= 4 and known_item and known_item != "unknown":
                    continue
                
                # Enhanced prediction display
                move_predictions = [(move, prob) for move, prob in probabilities.get('moves', {}).items() if prob > 0.1]
                if move_predictions:
                    print(visual.prediction_display(species_raw, move_predictions[:5]))
                else:
                    print(f"\n   {i}. [TARGET] {species_raw}:")
                
                # Get detailed probability breakdown for all components
                probabilities = self.predictor.predict_component_probabilities(
                    species_norm, 
                    teammates=opponent_pokemon_normalized,
                    observed_moves=observed_moves
                )
                
                if 'error' in probabilities:
                    print(f"      [ERROR] No prediction data available")
                    continue
                
                # Show move probabilities
                if 'moves' in probabilities and probabilities['moves']:
                    print(f"      [MOVES] Move Probabilities:")
                    for move, prob in probabilities['moves'][:6]:  # Top 6 moves
                        if move in observed_moves:
                            print(f"        [OK] {move:<20} {prob:>6.1%} (confirmed)")
                        else:
                            print(f"        [?] {move:<20} {prob:>6.1%}")
                
                # Show item probabilities
                if 'items' in probabilities and probabilities['items']:
                    if known_item:
                        print(f"      [ITEM] {known_item} [OK] (confirmed)")
                    else:
                        print(f"      [ITEM] Probabilities:")
                        for item, prob in probabilities['items'][:3]:  # Top 3 items
                            print(f"        • {item:<20} {prob:>6.1%}")
                
                # Show nature probabilities
                if 'natures' in probabilities and probabilities['natures']:
                    print(f"      🧬 Nature Probabilities:")
                    for nature, prob in probabilities['natures'][:3]:  # Top 3 natures
                        print(f"        • {nature:<20} {prob:>6.1%}")
                
                # Show EV spread probabilities
                if 'ev_spreads' in probabilities and probabilities['ev_spreads']:
                    print(f"      [EVS] EV Spread Probabilities:")
                    for ev_spread, prob in probabilities['ev_spreads'][:3]:  # Top 3 spreads
                        print(f"        • {ev_spread:<20} {prob:>6.1%}")
                
                # Show ability probabilities
                if 'abilities' in probabilities and probabilities['abilities']:
                    known_ability = revealed_abilities.get(species_raw)
                    if known_ability:
                        print(f"      [ABILITY] {known_ability} [OK] (confirmed)")
                    else:
                        print(f"      [ABILITY] Probabilities:")
                        for ability, prob in probabilities['abilities'][:3]:  # Top 3 abilities
                            print(f"        • {ability:<20} {prob:>6.1%}")
                
                # Show how many moves are revealed vs predicted
                total_revealed = len(observed_moves)
                print(f"      [INFO] Status: {total_revealed}/4 moves revealed")
                
        except Exception as e:
            print(f"   [ERROR] Prediction error: {e}")
        
        print("-" * 60)
        self.last_prediction_turn = battle.turn
    
    def get_opponent_current_moves(self, mon=None, return_switch=False, is_player=False, return_separate=False, battle=None):
        """Override to use Bayesian predictions for opponent moves."""
        # Get battle context
        current_battle = battle or getattr(self, 'battle', None)
        
        if is_player and current_battle:
            return list(current_battle.active_pokemon.moves.keys())
        
        if mon is None and current_battle:
            mon = current_battle.opponent_active_pokemon
        
        # Get confirmed moves from battle
        confirmed_moves = []
        if mon.moves:
            for move_id, opponent_move in mon.moves.items():
                confirmed_moves.append(opponent_move.id)
        
        # Get Bayesian predictions if we have a predictor
        predicted_moves = []
        if self.predictor:
            try:
                # Get battle context - use the battle from the method parameter
                current_battle = getattr(self, 'battle', None)
                if not current_battle:
                    # Try to get battle from the mon's context or fallback
                    return confirmed_moves if len(confirmed_moves) > 0 else []
                
                # Get all opponent Pokemon names for teammate context
                opponent_pokemon = []
                for pokemon in current_battle.opponent_team.values():
                    if pokemon and pokemon.species:
                        normalized_name = self.normalize_pokemon_name(pokemon.species)
                        opponent_pokemon.append(normalized_name)
                
                # Get observed moves for this Pokemon (normalized)
                observed_moves = []
                for move in mon.moves.values():
                    if move:
                        normalized_move = self.normalize_move_name(move.id)
                        observed_moves.append(normalized_move)
                
                # Get Bayesian predictions
                species_norm = self.normalize_pokemon_name(mon.species)
                probabilities = self.predictor.predict_component_probabilities(
                    species_norm, 
                    teammates=opponent_pokemon,
                    observed_moves=observed_moves
                )
                
                # Extract top predicted moves (not already confirmed)
                if 'moves' in probabilities:
                    for move_name, prob in probabilities['moves']:
                        # Convert back to battle format (lowercase, no spaces)
                        battle_format_move = move_name.lower().replace(' ', '').replace('-', '')
                        if battle_format_move not in confirmed_moves and len(predicted_moves) < 4:
                            predicted_moves.append(battle_format_move)
                            
            except Exception as e:
                print(f"Error getting Bayesian move predictions: {e}")
        
        # Combine confirmed and predicted moves
        all_moves = confirmed_moves + predicted_moves
        
        # Fill to 4 moves if needed (fallback to original method)
        if len(all_moves) < 4:
            # Fall back to original method for remaining slots
            try:
                original_moves = super().get_opponent_current_moves(mon, return_switch, is_player, return_separate)
                if isinstance(original_moves, tuple):  # return_separate=True
                    original_moves = original_moves[0] + original_moves[1]
                
                for move in original_moves:
                    if move not in all_moves and len(all_moves) < 4:
                        all_moves.append(move)
            except:
                pass
        
        # Truncate to 4 moves
        all_moves = all_moves[:4]
        
        if return_separate:
            return confirmed_moves, predicted_moves
        
        return all_moves
    
    def guess_opponent_stats(self, mon=None, battle=None):
        """Use Bayesian predictions to guess opponent Pokemon stats (nature, EVs)."""
        # Get battle context
        current_battle = battle or getattr(self, 'battle', None)
        if mon is None and current_battle:
            mon = current_battle.opponent_active_pokemon
        
        # Try to get Bayesian predictions first
        if self.predictor:
            try:
                # Get battle context
                current_battle = getattr(self, 'battle', None)
                if not current_battle:
                    # Fallback to basic stats
                    return {'hp': 252, 'atk': 0, 'def': 0, 'spa': 252, 'spd': 4, 'spe': 0}, 'Modest'
                
                # Get all opponent Pokemon names for teammate context
                opponent_pokemon = []
                for pokemon in current_battle.opponent_team.values():
                    if pokemon and pokemon.species:
                        normalized_name = self.normalize_pokemon_name(pokemon.species)
                        opponent_pokemon.append(normalized_name)
                
                # Get observed moves for this Pokemon (normalized)
                observed_moves = []
                for move in mon.moves.values():
                    if move:
                        normalized_move = self.normalize_move_name(move.id)
                        observed_moves.append(normalized_move)
                
                # Get Bayesian predictions
                species_norm = self.normalize_pokemon_name(mon.species)
                probabilities = self.predictor.predict_component_probabilities(
                    species_norm, 
                    teammates=opponent_pokemon,
                    observed_moves=observed_moves
                )
                
                # Extract most likely nature and EV spread
                predicted_nature = None
                predicted_evs = None
                
                if 'natures' in probabilities and probabilities['natures']:
                    predicted_nature = probabilities['natures'][0][0]  # Top nature
                
                if 'ev_spreads' in probabilities and probabilities['ev_spreads']:
                    ev_spread_str = probabilities['ev_spreads'][0][0]  # Top EV spread
                    # Parse EV spread string like "252 HP / 252 SpA" to EV values
                    predicted_evs = self._parse_ev_spread_string(ev_spread_str)
                
                if predicted_nature and predicted_evs:
                    return predicted_evs, predicted_nature
                    
            except Exception as e:
                print(f"Error getting Bayesian stat predictions: {e}")
        
        # Fallback to original guess_stats if available
        try:
            return mon.guess_stats('most_likely')
        except:
            # Ultimate fallback - common competitive spreads
            return {'hp': 252, 'atk': 0, 'def': 0, 'spa': 252, 'spd': 4, 'spe': 0}, 'Modest'
    
    def _parse_ev_spread_string(self, ev_spread_str: str) -> dict:
        """Parse EV spread string like '252 HP / 252 SpA' into EV dictionary."""
        ev_dict = {'hp': 0, 'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
        
        # Map spread names to stat keys
        stat_mapping = {
            'HP': 'hp',
            'Atk': 'atk', 
            'Def': 'def',
            'SpA': 'spa',
            'SpD': 'spd',
            'Spe': 'spe'
        }
        
        if not ev_spread_str or ev_spread_str == "No major investments":
            return ev_dict
        
        # Parse "252 HP / 252 SpA / 4 SpD" format
        parts = ev_spread_str.split(' / ')
        for part in parts:
            try:
                value, stat = part.strip().split(' ', 1)
                if stat in stat_mapping:
                    ev_dict[stat_mapping[stat]] = int(value)
            except:
                continue
                
        return ev_dict
    
    def choose_move(self, battle: Battle):
        """Choose move and show live predictions for moves/items/EVs."""
        
        # Show live predictions starting from turn 1 (all Pokemon revealed in team preview)
        if battle.turn >= 1:
            self.show_live_predictions(battle)
            
            # Demonstrate the new Bayesian-powered methods
            if battle.opponent_active_pokemon:
                print(f"\n🧠 BAYESIAN-POWERED BATTLE ANALYSIS:")
                print("-" * 50)
                
                # Test get_opponent_current_moves
                try:
                    predicted_moves = self.get_opponent_current_moves(battle.opponent_active_pokemon, battle=battle)
                    print(f"[PREDICT] Predicted moves for {battle.opponent_active_pokemon.species}:")
                    print(f"   {', '.join(predicted_moves)}")
                except Exception as e:
                    print(f"   [ERROR] Move prediction error: {e}")
                
                # Test guess_opponent_stats  
                try:
                    predicted_evs, predicted_nature = self.guess_opponent_stats(battle.opponent_active_pokemon, battle)
                    print(f"[STATS] Predicted stats for {battle.opponent_active_pokemon.species}:")
                    print(f"   Nature: {predicted_nature}")
                    print(f"   EVs: {predicted_evs}")
                except Exception as e:
                    print(f"   [ERROR] Stat prediction error: {e}")
                
                print("-" * 50)
            
        # Make the move decision
        return super().choose_move(battle)


async def run_live_battle():
    """Run a battle with live turn-by-turn predictions."""
    print("[BATTLE] Live Battle Predictor")
    print("=" * 50)
    print("Shows real-time Bayesian predictions every turn!")
    print("Uses parsed battle data for accurate predictions.")
    print("=" * 50)
    
    # Create players
    player = LiveBattlePredictor(
        battle_format="gen9ou",
        server_configuration=None
    )
    
    opponent = get_llm_player(
        args=type('Args', (), {
            'temperature': 0.3,
            'log_dir': './battle_log/live_predictor'
        })(),
        backend='random',
        prompt_algo='random',
        name='random',
        PNUMBER1=PNUMBER1 + "_opp",
        battle_format='gen9ou'
    )
    
    # Load teams
    print("[LOAD] Loading teams...")
    player.update_team(load_random_team(1))
    opponent.update_team(load_random_team(2))
    
    print("[START] Starting live battle with predictions!")
    print("[INFO] Watch for predictions after each turn!")
    
    try:
        await player.battle_against(opponent, n_battles=1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Battle interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Battle error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    # Setup environment
    if not os.getenv('METAMON_CACHE_DIR'):
        os.environ['METAMON_CACHE_DIR'] = '/tmp/metamon_cache'
    
    try:
        asyncio.run(run_live_battle())
    except KeyboardInterrupt:
        print("\n[END] Live battle predictor ended!")


if __name__ == "__main__":
    main()