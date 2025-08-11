# Singleton module for PokemonPredictor to avoid circular imports
# and ensure only one instance is created

_predictor_instance = None

def get_pokemon_predictor():
    """Get the shared PokemonPredictor instance, creating it if necessary."""
    global _predictor_instance
    if _predictor_instance is None:
        from pokemon_predictor import PokemonPredictor
        _predictor_instance = PokemonPredictor()
    return _predictor_instance