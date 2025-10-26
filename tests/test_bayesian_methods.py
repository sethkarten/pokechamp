"""
Tests for integrated Bayesian-powered methods.

This test suite verifies that the Bayesian predictor and live battle predictor
work correctly together.
"""

import pytest
import os
from bayesian.live_battle_predictor import LiveBattlePredictor
from bayesian.pokemon_predictor import PokemonPredictor


class TestBayesianMethods:
    """Test class for integrated Bayesian methods."""

    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Set up environment for Bayesian tests."""
        if not os.getenv('METAMON_CACHE_DIR'):
            os.environ['METAMON_CACHE_DIR'] = '/tmp/metamon_cache'

    @pytest.mark.bayesian
    @pytest.mark.integration
    def test_pokemon_predictor_direct(self):
        """Test PokemonPredictor functionality directly."""
        predictor = PokemonPredictor()
        
        # Test component predictions for a common Pokemon
        probabilities = predictor.predict_component_probabilities(
            "Slowking-Galar",
            teammates=["Ting-Lu", "Tinkaton", "Zapdos", "Hydrapple", "Zamazenta"],
            observed_moves=["Chilly Reception"]
        )
        
        assert isinstance(probabilities, dict), "Probabilities should be a dictionary"
        assert 'moves' in probabilities, "Should have move predictions"
        assert 'natures' in probabilities, "Should have nature predictions"
        
        # Verify structure of predictions
        if probabilities['moves']:
            for move, prob in probabilities['moves'][:3]:
                assert isinstance(move, str), "Move name should be a string"
                assert isinstance(prob, (int, float)), "Probability should be numeric"
                assert 0 <= prob <= 1, "Probability should be between 0 and 1"

    @pytest.mark.bayesian
    @pytest.mark.integration
    def test_live_battle_predictor(self):
        """Test LiveBattlePredictor functionality."""
        predictor = LiveBattlePredictor()
        
        # Test that predictor has expected methods
        assert hasattr(predictor, 'get_opponent_current_moves'), "Should have move prediction method"
        assert hasattr(predictor, 'guess_opponent_stats'), "Should have stats prediction method"
        assert hasattr(predictor, 'normalize_pokemon_name'), "Should have name normalization method"
        assert hasattr(predictor, 'normalize_move_name'), "Should have move normalization method"
        
        # Test name normalization
        normalized = predictor.normalize_pokemon_name('slowkinggalar')
        assert normalized == 'Slowking-Galar', "Should normalize Pokemon names correctly"
        
        # Test move normalization
        normalized_move = predictor.normalize_move_name('terablast')
        assert normalized_move == 'Terablast', "Should normalize move names correctly"

    @pytest.mark.bayesian
    def test_team_prediction_integration(self):
        """Test team prediction with PokemonPredictor directly."""
        predictor = PokemonPredictor()
        
        # Test with a popular competitive core
        core = ["Kingambit", "Gholdengo", "Great Tusk"]
        predictions = predictor.predict_teammates(
            revealed_pokemon=core,
            max_predictions=10
        )
        
        assert isinstance(predictions, list), "Predictions should be a list"
        assert len(predictions) > 0, "Should have some predictions for popular core"
        
        # Verify prediction format
        for species, prob in predictions:
            assert isinstance(species, str), "Species should be a string"
            assert len(species) > 0, "Species name should not be empty"
            assert isinstance(prob, (int, float)), "Probability should be numeric"
            assert prob > 0, "Probability should be positive"

    @pytest.mark.bayesian
    def test_usage_stats(self):
        """Test usage statistics functionality."""
        predictor = PokemonPredictor()
        
        usage_stats = predictor.get_usage_stats(top_n=10)
        
        assert isinstance(usage_stats, list), "Usage stats should be a list"
        assert len(usage_stats) <= 10, "Should respect top_n limit"
        
        if usage_stats:
            for species, count, percentage in usage_stats:
                assert isinstance(species, str), "Species should be a string"
                assert isinstance(count, int), "Count should be an integer"
                assert isinstance(percentage, (int, float)), "Percentage should be numeric"
                assert count > 0, "Count should be positive"
                assert percentage > 0, "Percentage should be positive"

    @pytest.mark.bayesian
    def test_team_core_analysis(self):
        """Test team core analysis functionality."""
        predictor = PokemonPredictor()
        
        analysis = predictor.analyze_team_core(
            core_pokemon=["Dragonite", "Zamazenta"],
            max_suggestions=5
        )
        
        assert isinstance(analysis, dict), "Analysis should be a dictionary"
        assert 'core_pokemon' in analysis, "Should include core Pokemon"
        assert 'suggested_pokemon' in analysis, "Should have suggestions"
        assert 'detailed_suggestions' in analysis, "Should have detailed suggestions"
        
        # Verify core analysis structure
        assert analysis['core_pokemon'] == ["Dragonite", "Zamazenta"], "Should preserve input core"
        assert isinstance(analysis['suggested_pokemon'], list), "Suggestions should be a list"
        assert len(analysis['suggested_pokemon']) <= 5, "Should respect max_suggestions"


# Standalone test function for backward compatibility
def test_bayesian_methods():
    """Standalone version for backward compatibility."""
    import os
    # Set up environment
    if not os.getenv('METAMON_CACHE_DIR'):
        os.environ['METAMON_CACHE_DIR'] = '/tmp/metamon_cache'
    
    test_instance = TestBayesianMethods()
    test_instance.test_pokemon_predictor_direct()
    test_instance.test_live_battle_predictor()
    print("[OK] Bayesian methods tests completed!")


if __name__ == "__main__":
    test_bayesian_methods()