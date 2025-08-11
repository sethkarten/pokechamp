"""
Tests for the bot system functionality.

This test suite verifies that the bot discovery and loading system works correctly
and that all available bots can be instantiated and used.
"""

import pytest
import asyncio
from common import bot_choices, get_available_bots
from poke_env.player.team_util import get_llm_player
from unittest.mock import MagicMock


class TestBotSystem:
    """Test class for bot system functionality."""

    @pytest.fixture
    def mock_args(self):
        """Provide mock arguments for bot creation."""
        args = MagicMock()
        args.temperature = 0.3
        args.log_dir = "./test_logs"
        return args

    @pytest.mark.integration
    def test_get_available_bots(self):
        """Test that get_available_bots returns expected bots."""
        available_bots = get_available_bots()
        
        assert isinstance(available_bots, list), "Should return a list"
        # Available bots may be empty if no custom bots in bots/ folder
        assert isinstance(available_bots, list), "Should be a list even if empty"
        
        # Check bot_choices includes built-in bots
        expected_builtin = ['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random']
        
        for bot_name in expected_builtin:
            assert bot_name in bot_choices, f"Built-in bot {bot_name} should be in bot_choices"

    @pytest.mark.integration
    def test_bot_choices_consistency(self):
        """Test that bot_choices is consistent with available bots."""
        available_bots = get_available_bots()
        
        assert isinstance(bot_choices, list), "bot_choices should be a list"
        assert len(bot_choices) > 0, "Should have bot choices available"
        
        # Check that built-in bots are in bot_choices
        built_in_bots = ['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random']
        for bot in built_in_bots:
            assert bot in bot_choices, f"Built-in bot {bot} should be in bot_choices"

    @pytest.mark.integration
    def test_built_in_bot_instantiation(self, mock_args):
        """Test that built-in bots can be instantiated."""
        built_in_bots = ['one_step', 'abyssal', 'max_power', 'random']
        
        for bot_name in built_in_bots:
            if bot_name not in bot_choices:
                continue  # Skip if not in available choices
                
            try:
                # Test bot instantiation without API key (for non-LLM bots)
                player = get_llm_player(
                    args=mock_args,
                    backend='None',
                    prompt_algo='random',
                    name=bot_name,
                    KEY='',
                    battle_format='gen9ou',
                    device=0,
                    PNUMBER1='test',
                    online=False
                )
                
                assert player is not None, f"Should create {bot_name} player"
                assert hasattr(player, 'choose_move'), f"{bot_name} should have choose_move method"
                # Some baseline players might not have battle_format attribute
                if hasattr(player, 'battle_format'):
                    assert player.battle_format == 'gen9ou', f"{bot_name} should have correct battle format"
                # But they should have _format or similar
                elif hasattr(player, '_format'):
                    assert player._format == 'gen9ou', f"{bot_name} should have correct _format"
                
            except Exception as e:
                # Some bots might require API keys or have other requirements
                if "API key" in str(e) or "backend" in str(e):
                    pytest.skip(f"Bot {bot_name} requires API key or special configuration")
                else:
                    pytest.fail(f"Failed to instantiate {bot_name}: {e}")

    @pytest.mark.integration 
    def test_custom_bot_discovery(self):
        """Test that custom bots are properly discovered."""
        available_bots = get_available_bots()
        
        # get_available_bots returns a list of custom bot names
        assert isinstance(available_bots, list), "Should return a list"
        
        # Check if any custom bots are in bot_choices
        built_in_bots = ['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random']
        custom_bots = [bot for bot in bot_choices if bot not in built_in_bots]
        
        # If custom bots exist, they should be from the available_bots list
        for bot_name in custom_bots:
            assert bot_name in available_bots, f"Custom bot {bot_name} should be in available_bots"

    @pytest.mark.integration
    def test_bot_list_formatting(self):
        """Test that bot information is properly formatted."""
        available_bots = get_available_bots()
        
        # available_bots is a list of strings
        for bot_name in available_bots:
            assert isinstance(bot_name, str), "Bot name should be a string"
            assert len(bot_name) > 0, "Bot name should not be empty"
        
        # Check bot_choices formatting
        for bot_name in bot_choices:
            assert isinstance(bot_name, str), "Bot name should be a string"
            assert len(bot_name) > 0, "Bot name should not be empty"

    @pytest.mark.integration
    @pytest.mark.slow
    def test_bot_battle_compatibility(self, mock_args):
        """Test that bots are compatible with battle system."""
        # Test a simple bot that doesn't require API keys
        bot_name = 'random'
        
        if bot_name not in bot_choices:
            pytest.skip(f"Bot {bot_name} not available")
        
        try:
            player1 = get_llm_player(
                args=mock_args,
                backend='None', 
                prompt_algo='random',
                name=bot_name,
                KEY='',
                battle_format='gen9ou',
                device=0,
                PNUMBER1='1',
                online=False
            )
            
            player2 = get_llm_player(
                args=mock_args,
                backend='None',
                prompt_algo='random', 
                name=bot_name,
                KEY='',
                battle_format='gen9ou',
                device=0,
                PNUMBER1='2',
                online=False
            )
            
            assert player1 is not None, "Should create first player"
            assert player2 is not None, "Should create second player"
            
            # Verify they have different identifiers
            assert player1 != player2, "Players should be different instances"
            
        except Exception as e:
            if "API key" in str(e):
                pytest.skip(f"Bot {bot_name} requires API key")
            else:
                pytest.fail(f"Failed bot compatibility test: {e}")


# Standalone test functions for backward compatibility
def list_available_bots():
    """Standalone function for listing bots."""
    available_bots = get_available_bots()
    print("Available custom bots from bots/ folder:")
    print("=" * 50)
    
    for bot_name in available_bots:
        print(f"  - {bot_name}")
    
    print("\nAll available bot choices:")
    print("=" * 50)
    for bot_name in bot_choices:
        print(f"  - {bot_name}")
    
    return available_bots


def test_bot_system():
    """Standalone function for basic bot system test."""
    test_instance = TestBotSystem()
    test_instance.test_get_available_bots()
    test_instance.test_bot_choices_consistency()
    print("✅ Bot system tests completed!")


if __name__ == "__main__":
    test_bot_system()
    list_available_bots()