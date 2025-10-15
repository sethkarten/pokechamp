# Format-aware singleton module for PokemonPredictor
# Maintains separate instances for different battle formats

_predictor_instances = {}

def get_pokemon_predictor(battle_format: str = "gen9ou"):
    """Get the PokemonPredictor instance for the specified format."""
    global _predictor_instances
    if battle_format not in _predictor_instances:
        from bayesian.pokemon_predictor import PokemonPredictor
        
        # Create format-specific predictor with appropriate cache file
        if "vgc" in battle_format.lower():
            cache_file = f"{battle_format}_team_predictor_full.pkl"
        else:
            cache_file = f"{battle_format}_team_predictor_full.pkl"
            
        _predictor_instances[battle_format] = PokemonPredictor(battle_format=battle_format)
    return _predictor_instances[battle_format]