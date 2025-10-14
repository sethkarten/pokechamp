#!/usr/bin/env python3
"""
Bayesian Team Predictor for Pokemon Gen9OU

Predicts unrevealed team members, moves, EVs/IVs, items, and abilities
based on observed information using Naive Bayes approach trained on ~1M teams.
"""

import os
import re
import pickle
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from tqdm import tqdm

from poke_env.player.team_util import get_metamon_teams


@dataclass
class PokemonConfig:
    """Complete configuration for a single Pokemon."""
    species: str
    item: str
    ability: str
    moves: List[str]  # Exactly 4 moves
    nature: str
    evs: Dict[str, int]  # HP, Atk, Def, SpA, SpD, Spe
    ivs: Dict[str, int]  # HP, Atk, Def, SpA, SpD, Spe  
    tera_type: str


@dataclass 
class TeamData:
    """Complete team of 6 Pokemon configurations."""
    pokemon: List[PokemonConfig]
    
    def get_species_list(self) -> List[str]:
        return [p.species for p in self.pokemon]


class TeamParser:
    """Parse Showdown team format into structured data."""
    
    def __init__(self):
        self.stat_names = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']
    
    def parse_team_file(self, file_path: str) -> TeamData:
        """Parse a single team file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        return self.parse_team_string(content)
    
    def parse_team_string(self, team_str: str) -> TeamData:
        """Parse team string into TeamData."""
        # Split into individual Pokemon sections
        pokemon_sections = re.split(r'\n\s*\n', team_str.strip())
        pokemon_configs = []
        
        for section in pokemon_sections:
            if not section.strip():
                continue
            config = self._parse_pokemon_section(section)
            if config:
                pokemon_configs.append(config)
        
        return TeamData(pokemon=pokemon_configs)
    
    def _parse_pokemon_section(self, section: str) -> Optional[PokemonConfig]:
        """Parse a single Pokemon section."""
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if not lines:
            return None
        
        # Parse first line: species @ item
        first_line = lines[0]
        species_match = re.match(r'^(.+?)(?:\s+@\s+(.+))?$', first_line)
        if not species_match:
            return None
            
        species = species_match.group(1).strip()
        item = species_match.group(2).strip() if species_match.group(2) else ""
        
        # Handle gender/nickname in species
        species = re.sub(r'\s+\([MF]\)$', '', species)  # Remove (M)/(F)
        
        # Initialize defaults
        ability = ""
        moves = []
        nature = "Hardy"
        evs = {stat: 0 for stat in self.stat_names}
        ivs = {stat: 31 for stat in self.stat_names}
        tera_type = ""
        
        for line in lines[1:]:
            line = line.strip()
            
            # Ability
            if line.startswith('Ability:'):
                ability = line[8:].strip()
            
            # Tera Type
            elif line.startswith('Tera Type:'):
                tera_type = line[10:].strip()
            
            # EVs
            elif line.startswith('EVs:'):
                ev_str = line[4:].strip()
                evs.update(self._parse_stat_line(ev_str))
            
            # IVs
            elif line.startswith('IVs:'):
                iv_str = line[4:].strip()
                ivs.update(self._parse_stat_line(iv_str))
            
            # Nature
            elif line.endswith('Nature'):
                nature = line.replace(' Nature', '').strip()
            
            # Moves
            elif line.startswith('-'):
                move = line[1:].strip()
                if move:
                    moves.append(move)
        
        # Ensure exactly 4 moves (pad with empty if needed)
        while len(moves) < 4:
            moves.append("")
        moves = moves[:4]
        
        return PokemonConfig(
            species=species,
            item=item,
            ability=ability,
            moves=moves,
            nature=nature,
            evs=evs,
            ivs=ivs,
            tera_type=tera_type
        )
    
    def _parse_stat_line(self, stat_line: str) -> Dict[str, int]:
        """Parse EV/IV line like '252 HP / 252 Atk / 4 Def'."""
        stats = {}
        parts = [part.strip() for part in stat_line.split('/')]
        
        for part in parts:
            match = re.match(r'(\d+)\s+(\w+)', part)
            if match:
                value, stat = match.groups()
                if stat in self.stat_names:
                    stats[stat] = int(value)
        
        return stats


class BayesianTeamPredictor:
    """Naive Bayes predictor for Pokemon team configurations."""
    
    def __init__(self, cache_file: str = "gen9ou_team_predictor_full.pkl", battle_format: str = "gen9ou"):
        self.cache_file = cache_file
        self.battle_format = battle_format
        self.parser = TeamParser()
        
        # Set up cache directory
        self.cache_dir = os.getenv('METAMON_CACHE_DIR', '/tmp/metamon_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Full path for cache file
        self.cache_path = os.path.join(self.cache_dir, cache_file)
        
        # Probability tables
        self.species_counts = Counter()  # P(species)
        self.teammate_counts = defaultdict(Counter)  # P(species_B | species_A on team)
        self.config_given_species = defaultdict(Counter)  # P(config | species) - fixed structure
        self.config_given_teammates = defaultdict(lambda: defaultdict(Counter))  # P(config | teammates)
        self.move_given_species = defaultdict(Counter)  # P(move | species)
        self.move_pairs = defaultdict(Counter)  # P(move_B | move_A, species)
        
        self.total_teams = 0
        self.is_trained = False
    
    def load_and_train(self, force_retrain: bool = False):
        """Load cached model or train from scratch."""
        if not force_retrain and os.path.exists(self.cache_path):
            print(f"Loading cached model from {self.cache_path}...")
            self._load_cache()
            self.is_trained = True
            return
        
        print("Training new model from full team dataset...")
        print("This may take several minutes for ~1M teams...")
        self._train_from_data()
        self._save_cache()
        self.is_trained = True
    
    def _train_from_data(self):
        """Train the model on team data."""
        # Get team data
        team_set = get_metamon_teams(self.battle_format, "modern_replays")
        team_files = team_set.team_files
        
        print(f"Training on {len(team_files)} teams...")
        
        for file_path in tqdm(team_files, desc="Processing teams"):
            try:
                team_data = self.parser.parse_team_file(file_path)
                self._update_counts(team_data)
                self.total_teams += 1
                
                # Progress reporting for large datasets
                if self.total_teams % 10000 == 0:
                    print(f"Processed {self.total_teams} teams...")
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        print(f"Trained on {self.total_teams} teams")
        print(f"Found {len(self.species_counts)} unique species")
    
    def _update_counts(self, team_data: TeamData):
        """Update probability counts from a single team."""
        species_list = team_data.get_species_list()
        
        for i, pokemon in enumerate(team_data.pokemon):
            species = pokemon.species
            
            # Update species frequency
            self.species_counts[species] += 1
            
            # Update teammate associations
            teammates = [s for j, s in enumerate(species_list) if j != i]
            for teammate in teammates:
                self.teammate_counts[species][teammate] += 1
            
            # Update config given species (with error handling)
            try:
                config_key = self._pokemon_to_config_key(pokemon)
                self.config_given_species[species][config_key] += 1
                
                # Update config given teammates
                teammate_key = tuple(sorted(teammates))
                self.config_given_teammates[teammate_key][species][config_key] += 1
            except Exception as e:
                print(f"Error creating config key for {species}: {e}")
                continue
            
            # Update move associations
            for move in pokemon.moves:
                if move:  # Skip empty moves
                    self.move_given_species[species][move] += 1
                    
                    # Move pairs within this Pokemon
                    for other_move in pokemon.moves:
                        if other_move and other_move != move:
                            self.move_pairs[(species, move)][other_move] += 1
    
    def _pokemon_to_config_key(self, pokemon: PokemonConfig) -> str:
        """Convert Pokemon config to a key for counting."""
        # Debug and ensure EVs is a dictionary
        if not isinstance(pokemon.evs, dict):
            raise ValueError(f"EVs should be dict, got {type(pokemon.evs)}: {pokemon.evs}")
            
        # Simplified config key - you might want to make this more granular
        ev_spread = tuple(pokemon.evs[stat] for stat in ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'])
        moves_tuple = tuple(sorted([m for m in pokemon.moves if m]))
        
        return f"{pokemon.item}|{pokemon.ability}|{pokemon.nature}|{ev_spread}|{moves_tuple}|{pokemon.tera_type}"
    
    def predict_unrevealed_pokemon(self, revealed_species: List[str], max_predictions: int = 5) -> List[Tuple[str, float]]:
        """Predict most likely unrevealed team members."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")
        
        if self.total_teams == 0:
            return []  # No training data
        
        # Calculate P(species | revealed_teammates)
        species_probs = {}
        
        for species in self.species_counts:
            if species in revealed_species:
                continue  # Skip already revealed
            
            # Base probability P(species)
            base_prob = self.species_counts[species] / self.total_teams
            
            # Conditional probability given teammates
            teammate_prob = 1.0
            for revealed in revealed_species:
                if revealed in self.species_counts and self.species_counts[revealed] > 0:
                    if species in self.teammate_counts[revealed]:
                        # P(species | revealed_teammate)
                        conditional = self.teammate_counts[revealed][species] / self.species_counts[revealed]
                        teammate_prob *= conditional
                    else:
                        teammate_prob *= 0.001  # Small probability for unseen combinations
                else:
                    teammate_prob *= 0.001  # Unknown revealed species
            
            species_probs[species] = base_prob * teammate_prob
        
        # Sort by probability
        ranked_species = sorted(species_probs.items(), key=lambda x: x[1], reverse=True)
        return ranked_species[:max_predictions]
    
    def predict_pokemon_config(self, species: str, teammates: List[str] = None, revealed_moves: List[str] = None) -> Dict:
        """Predict full configuration for a specific Pokemon."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")
        
        revealed_moves = revealed_moves or []
        
        # Get most common configs for this species
        if species not in self.config_given_species:
            return {"error": f"No data for species {species}"}
        
        configs = self.config_given_species[species]
        total_configs = sum(configs.values())
        
        # Calculate probabilities for each config
        config_probs = {}
        for config_key, count in configs.items():
            base_prob = count / total_configs
            
            # Boost probability if revealed moves match
            move_match_bonus = 1.0
            if revealed_moves:
                config_moves = self._extract_moves_from_config_key(config_key)
                matches = len(set(revealed_moves) & set(config_moves))
                move_match_bonus = (1 + matches) ** 2  # Exponential bonus for matches
            
            config_probs[config_key] = base_prob * move_match_bonus
        
        # Get top prediction
        best_config = max(config_probs, key=config_probs.get)
        probability = config_probs[best_config]
        
        # Parse config back to readable format
        parsed_config = self._parse_config_key(best_config)
        parsed_config['probability'] = probability
        parsed_config['species'] = species
        
        return parsed_config
    
    def predict_component_probabilities(self, species: str, teammates: List[str] = None, revealed_moves: List[str] = None) -> Dict:
        """Predict probabilities for individual components (moves, items, natures, abilities, EVs)."""
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call load_and_train() first.")
        
        revealed_moves = revealed_moves or []
        
        if species not in self.config_given_species:
            return {"error": f"No data for species {species}"}
        
        configs = self.config_given_species[species]
        total_configs = sum(configs.values())
        
        # Collect probabilities for each component
        move_probs = {}
        item_probs = {}
        nature_probs = {}
        ability_probs = {}
        ev_spread_probs = {}
        
        for config_key, count in configs.items():
            base_prob = count / total_configs
            
            # Apply move match bonus if revealed moves match
            move_match_bonus = 1.0
            if revealed_moves:
                config_moves = self._extract_moves_from_config_key(config_key)
                matches = len(set(revealed_moves) & set(config_moves))
                move_match_bonus = (1 + matches) ** 2
            
            adjusted_prob = base_prob * move_match_bonus
            
            # Parse config and accumulate probabilities
            parsed = self._parse_config_key(config_key)
            if parsed:
                # Moves
                if 'moves' in parsed:
                    for move in parsed['moves']:
                        if move:  # Skip empty move slots
                            move_probs[move] = move_probs.get(move, 0) + adjusted_prob
                
                # Items
                if 'item' in parsed and parsed['item']:
                    item_probs[parsed['item']] = item_probs.get(parsed['item'], 0) + adjusted_prob
                
                # Natures
                if 'nature' in parsed and parsed['nature']:
                    nature_probs[parsed['nature']] = nature_probs.get(parsed['nature'], 0) + adjusted_prob
                
                # Abilities
                if 'ability' in parsed and parsed['ability']:
                    ability_probs[parsed['ability']] = ability_probs.get(parsed['ability'], 0) + adjusted_prob
                
                # EV spreads (simplified to main stats)
                if 'ev_spread' in parsed and parsed['ev_spread']:
                    ev_key = self._summarize_ev_spread(parsed['ev_spread'])
                    if ev_key:
                        ev_spread_probs[ev_key] = ev_spread_probs.get(ev_key, 0) + adjusted_prob
        
        # Handle confirmed moves specially - they should have 100% probability
        def normalize_and_sort_with_confirmed(prob_dict, confirmed_items=None):
            confirmed_items = confirmed_items or []
            total = sum(prob_dict.values()) if prob_dict else 1
            
            # Set confirmed items to 100% and normalize others
            normalized = {}
            for item, prob in prob_dict.items():
                if item in confirmed_items:
                    normalized[item] = 1.0  # 100% for confirmed items
                else:
                    normalized[item] = prob / total
            
            return sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        
        # Regular normalize and sort for non-move components
        def normalize_and_sort(prob_dict):
            total = sum(prob_dict.values()) if prob_dict else 1
            normalized = {k: v/total for k, v in prob_dict.items()}
            return sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'species': species,
            'moves': normalize_and_sort_with_confirmed(move_probs, revealed_moves),
            'items': normalize_and_sort(item_probs),
            'natures': normalize_and_sort(nature_probs),
            'abilities': normalize_and_sort(ability_probs),
            'ev_spreads': normalize_and_sort(ev_spread_probs),
            'revealed_moves': revealed_moves
        }
    
    def _summarize_ev_spread(self, ev_spread: Dict[str, int]) -> str:
        """Create a summary string for EV spread showing main investments."""
        main_investments = []
        for stat, value in ev_spread.items():
            if value >= 200:  # Significant investment
                main_investments.append(f"{value} {stat}")
        return " / ".join(main_investments) if main_investments else "No major investments"
    
    def _extract_moves_from_config_key(self, config_key: str) -> List[str]:
        """Extract moves from config key."""
        parts = config_key.split('|')
        if len(parts) >= 5:
            moves_str = parts[4]  # moves tuple string
            # Parse the tuple string back to list
            try:
                moves_tuple = eval(moves_str)  # Safe here since we created it
                return list(moves_tuple)
            except:
                return []
        return []
    
    def _parse_config_key(self, config_key: str) -> Dict:
        """Parse config key back to readable format."""
        parts = config_key.split('|')
        if len(parts) < 6:
            print(f"Warning: Config key has insufficient parts ({len(parts)}): {config_key[:100]}")
            return {}
        
        try:
            ev_spread = eval(parts[3])  # EV tuple (fixed index)
            moves = list(eval(parts[4]))  # Moves tuple
            
            return {
                'item': parts[0],
                'ability': parts[1], 
                'nature': parts[2],
                'ev_spread': dict(zip(['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe'], ev_spread)),
                'moves': moves,
                'tera_type': parts[5]
            }
        except Exception as e:
            print(f"Warning: Failed to parse config key: {str(e)}")
            print(f"  Config key: {config_key[:200]}")
            print(f"  Parts[3]: {parts[3] if len(parts) > 3 else 'N/A'}")
            print(f"  Parts[4]: {parts[4] if len(parts) > 4 else 'N/A'}")
            
            # Try to return a partial parse with at least the basic components
            try:
                return {
                    'item': parts[0] if parts[0] else None,
                    'ability': parts[1] if parts[1] else None,
                    'nature': parts[2] if parts[2] else None,
                    'ev_spread': {},  # Empty EV spread as fallback
                    'moves': [],     # Empty moves as fallback
                    'tera_type': parts[5] if len(parts) > 5 and parts[5] else None,
                    'parse_error': True
                }
            except:
                return {'raw_config': config_key, 'parse_error': True}
    
    def _save_cache(self):
        """Save trained model to cache."""
        cache_data = {
            'species_counts': dict(self.species_counts),
            'teammate_counts': {k: dict(v) for k, v in self.teammate_counts.items()},
            'config_given_species': {k: dict(v) for k, v in self.config_given_species.items()},
            'config_given_teammates': {str(k): {k2: dict(v2) for k2, v2 in v.items()} 
                                     for k, v in self.config_given_teammates.items()},
            'move_given_species': {k: dict(v) for k, v in self.move_given_species.items()},
            'move_pairs': {str(k): dict(v) for k, v in self.move_pairs.items()},
            'total_teams': self.total_teams
        }
        
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Model cached to {self.cache_path}")
    
    def _load_cache(self):
        """Load trained model from cache."""
        with open(self.cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        self.species_counts = Counter(cache_data['species_counts'])
        self.teammate_counts = defaultdict(Counter)
        for k, v in cache_data['teammate_counts'].items():
            self.teammate_counts[k] = Counter(v)
        
        self.config_given_species = defaultdict(Counter)
        for k, v in cache_data['config_given_species'].items():
            self.config_given_species[k] = Counter(v)
        
        # Restore other attributes...
        self.total_teams = cache_data['total_teams']
        print(f"Loaded model trained on {self.total_teams} teams")


def main():
    """Test the team predictor."""
    predictor = BayesianTeamPredictor()
    
    # Train or load model
    predictor.load_and_train(force_retrain=False)
    
    # Test prediction
    revealed_species = ["Gliscor", "Latios", "Zamazenta"]
    print(f"\nGiven revealed Pokemon: {revealed_species}")
    
    # Predict unrevealed teammates
    predictions = predictor.predict_unrevealed_pokemon(revealed_species)
    print("\nMost likely unrevealed teammates:")
    for species, prob in predictions:
        print(f"  {species}: {prob:.4f}")
    
    # Predict config for a specific Pokemon
    if predictions:
        test_species = predictions[0][0]
        config = predictor.predict_pokemon_config(test_species, revealed_species)
        print(f"\nPredicted config for {test_species}:")
        for key, value in config.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()