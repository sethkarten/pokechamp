#!/usr/bin/env python3
"""
Production interface for the Bayesian Pokemon Team Predictor.

This module provides a simple interface to load the trained model and make predictions
during battles or team analysis.
"""

import os
from typing import List, Dict, Tuple, Optional
from team_predictor import BayesianTeamPredictor


class PokemonPredictor:
    """Production interface for Pokemon team predictions."""
    
    def __init__(self, cache_dir: str = None):
        """
        Initialize the predictor and load the trained model.
        
        Args:
            cache_dir: Directory containing the trained model cache.
                      Defaults to METAMON_CACHE_DIR environment variable.
        """
        if cache_dir:
            os.environ['METAMON_CACHE_DIR'] = cache_dir
        
        self.predictor = BayesianTeamPredictor()
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained model."""
        try:
            self.predictor.load_and_train(force_retrain=False)
            print(f"✅ Model loaded successfully!")
            print(f"   Trained on {self.predictor.total_teams:,} teams")
            print(f"   Knows {len(self.predictor.species_counts):,} Pokemon species")
        except Exception as e:
            raise RuntimeError(f"Failed to load trained model: {e}")
    
    def predict_teammates(self, revealed_pokemon: List[str], max_predictions: int = 5) -> List[Tuple[str, float]]:
        """
        Predict the most likely unrevealed teammates.
        
        Args:
            revealed_pokemon: List of Pokemon species already revealed on the team
            max_predictions: Maximum number of predictions to return
            
        Returns:
            List of (species, probability) tuples, sorted by probability descending
        """
        return self.predictor.predict_unrevealed_pokemon(revealed_pokemon, max_predictions)
    
    def predict_moveset(self, species: str, teammates: List[str] = None, 
                       observed_moves: List[str] = None) -> Dict:
        """
        Predict the full configuration (moves, items, EVs, etc.) for a Pokemon.
        
        Args:
            species: The Pokemon species to predict for
            teammates: Known teammates (helps with prediction accuracy)
            observed_moves: Any moves already observed for this Pokemon
            
        Returns:
            Dictionary containing predicted configuration with confidence score
        """
        return self.predictor.predict_pokemon_config(species, teammates, observed_moves)
    
    def predict_component_probabilities(self, species: str, teammates: List[str] = None, 
                                      observed_moves: List[str] = None) -> Dict:
        """
        Predict probabilities for individual components (moves, items, natures, abilities, EVs).
        
        Args:
            species: The Pokemon species to predict for
            teammates: Known teammates (helps with prediction accuracy)
            observed_moves: Any moves already observed for this Pokemon
            
        Returns:
            Dictionary containing probability distributions for each component
        """
        return self.predictor.predict_component_probabilities(species, teammates, observed_moves)
    
    def get_usage_stats(self, top_n: int = 20) -> List[Tuple[str, int, float]]:
        """
        Get usage statistics for Pokemon in the dataset.
        
        Args:
            top_n: Number of top Pokemon to return
            
        Returns:
            List of (species, count, percentage) tuples
        """
        total_teams = self.predictor.total_teams
        stats = []
        
        for species, count in self.predictor.species_counts.most_common(top_n):
            percentage = (count / total_teams) * 100
            stats.append((species, count, percentage))
        
        return stats
    
    def analyze_team_core(self, core_pokemon: List[str], max_suggestions: int = 10) -> Dict:
        """
        Analyze a team core and suggest complementary Pokemon.
        
        Args:
            core_pokemon: List of Pokemon species in the core
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            Dictionary with analysis results and suggestions
        """
        if len(core_pokemon) >= 6:
            return {"error": "Team core cannot have more than 6 Pokemon"}
        
        # Get teammate predictions
        suggestions = self.predict_teammates(core_pokemon, max_suggestions)
        
        # Get detailed configs for top suggestions
        detailed_suggestions = []
        for species, prob in suggestions[:5]:  # Get details for top 5
            config = self.predict_moveset(species, core_pokemon)
            detailed_suggestions.append({
                'species': species,
                'teammate_probability': prob,
                'predicted_config': config
            })
        
        return {
            'core_pokemon': core_pokemon,
            'core_size': len(core_pokemon),
            'remaining_slots': 6 - len(core_pokemon),
            'suggested_pokemon': suggestions,
            'detailed_suggestions': detailed_suggestions,
            'usage_in_dataset': [
                (pokemon, self.predictor.species_counts.get(pokemon, 0))
                for pokemon in core_pokemon
            ]
        }


def main():
    """Example usage of the Pokemon Predictor."""
    print("🔮 Pokemon Team Predictor - Production Interface")
    print("=" * 50)
    
    try:
        # Initialize predictor (loads trained model)
        predictor = PokemonPredictor()
        
        # Example 1: Predict teammates for a popular core
        print("\n📊 Example 1: Predicting teammates for Kingambit + Gholdengo")
        core = ["Kingambit", "Gholdengo"]
        teammates = predictor.predict_teammates(core)
        
        print(f"Given core: {core}")
        print("Most likely teammates:")
        for species, prob in teammates:
            print(f"  • {species}: {prob:.1%}")
        
        # Example 2: Predict moveset for a specific Pokemon
        print(f"\n🎯 Example 2: Predicting moveset for {teammates[0][0]}")
        moveset = predictor.predict_moveset(teammates[0][0], core)
        
        print(f"Predicted configuration for {teammates[0][0]}:")
        for key, value in moveset.items():
            if key not in ['probability', 'species']:
                print(f"  • {key}: {value}")
        print(f"  • Confidence: {moveset.get('probability', 0):.1%}")
        
        # Example 3: Usage statistics
        print("\n📈 Example 3: Top 10 most used Pokemon")
        usage = predictor.get_usage_stats(10)
        
        for i, (species, count, pct) in enumerate(usage, 1):
            print(f"  {i:2}. {species:<20} {count:,} uses ({pct:.1f}%)")
        
        # Example 4: Team core analysis
        print(f"\n🔍 Example 4: Analyzing team core")
        analysis = predictor.analyze_team_core(["Dragonite", "Zamazenta"])
        
        print(f"Core: {analysis['core_pokemon']}")
        print(f"Remaining slots: {analysis['remaining_slots']}")
        print("Top suggestions with details:")
        for suggestion in analysis['detailed_suggestions'][:3]:
            species = suggestion['species']
            prob = suggestion['teammate_probability']
            config = suggestion['predicted_config']
            print(f"\n  🎯 {species} ({prob:.1%} probability)")
            if 'moves' in config:
                print(f"     Moves: {', '.join(config['moves'])}")
            if 'item' in config:
                print(f"     Item: {config['item']}")
            if 'nature' in config:
                print(f"     Nature: {config['nature']}")
        
        print(f"\n✨ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()