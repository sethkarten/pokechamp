"""
Shared test fixtures and configuration for pytest.

This file contains common fixtures used across multiple test files.
"""

import pytest
import os
import sys
from collections import Counter

# Add the project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class MockBattle:
    """Mock battle object for testing Pokemon functionality."""
    
    def __init__(self, battle_format="gen9ou"):
        self._battle_format = battle_format
        self._gen = 9
        self.opponent_team = {}


class MockMove:
    """Mock move object for testing move functionality."""
    
    def __init__(self, move_id: str):
        self.id = move_id.lower().replace(' ', '').replace('-', '')


@pytest.fixture
def mock_battle():
    """Provide a mock battle object for testing."""
    return MockBattle()


@pytest.fixture
def sample_pokemon_species():
    """Provide a list of common Pokemon species for testing."""
    return [
        'Dragonite', 'Kingambit', 'Gholdengo', 'Great Tusk', 'Zamazenta',
        'Dragapult', 'Iron Valiant', 'Slowking-Galar', 'Raging Bolt', 
        'Ogerpon-Wellspring', 'Kyurem', 'Iron Moth', 'Corviknight',
        'Landorus-Therian', 'Ting-Lu', 'Cinderace', 'Hatterene'
    ]


@pytest.fixture
def sample_battle_forms():
    """Provide a list of battle forms that should map to base forms."""
    return [
        ("mimikyubusted", "Mimikyu-Busted"),
        ("miniormeteor", "Minior-Meteor"), 
        ("morpekohangry", "Morpeko-Hangry"),
        ("eiscuenoice", "Eiscue-Noice"),
        ("cramorantgulping", "Cramorant-Gulping")
    ]


@pytest.fixture
def sample_moves():
    """Provide a list of common moves for testing."""
    return [
        'Knock Off', 'U-turn', 'Earthquake', 'Stealth Rock', 'Swords Dance',
        'Shadow Ball', 'Close Combat', 'Calm Mind', 'Ice Spinner', 'Tera Blast',
        'Earth Power', 'Sucker Punch', 'Roost', 'Will-O-Wisp', 'Thunder Wave',
        'Rapid Spin', 'Protect', 'Psychic Noise', 'Body Press', 'Volt Switch'
    ]


@pytest.fixture(scope="session")
def predictor():
    """Provide a shared Pokemon predictor instance for the test session."""
    from bayesian.pokemon_predictor import PokemonPredictor
    return PokemonPredictor()


# Test markers for organizing tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "bayesian: marks tests related to Bayesian functionality"
    )
    config.addinivalue_line(
        "markers", "moves: marks tests related to move functionality"
    )
    config.addinivalue_line(
        "markers", "teamloader: marks tests related to team loading"
    )


# Set up environment variables for testing
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables."""
    os.environ.setdefault('METAMON_CACHE_DIR', '/tmp/metamon_cache')
    yield
    # Cleanup could go here if needed