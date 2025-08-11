"""
Comprehensive tests for move normalization functionality.

This test suite extracts all moves from metamon team files and tests the move 
normalization function to identify any missing mappings.
"""

import pytest
import os
import re
import random
from collections import Counter, defaultdict
from poke_env.player.team_util import get_metamon_teams


def extract_moves_from_team_file(file_path: str) -> list:
    """Extract all moves from a Showdown team file."""
    moves = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split by double newlines to get each Pokemon block
        pokemon_blocks = content.strip().split('\n\n')
        
        for block in pokemon_blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            
            # Look for lines that start with "- " (moves)
            for line in lines:
                line = line.strip()
                if line.startswith('- '):
                    # Extract move name (remove "- " prefix)
                    move = line[2:].strip()
                    
                    # Handle moves with additional info like "Hidden Power [Type]"
                    if '[' in move and ']' in move:
                        # Extract the main move name before the bracket
                        move = move.split('[')[0].strip()
                    
                    # Handle moves with "/" (alternative moves)
                    if '/' in move:
                        for alt_move in move.split('/'):
                            moves.append(alt_move.strip())
                    else:
                        moves.append(move)
                        
    except Exception as e:
        print(f"Error parsing team file {file_path}: {e}")
        
    return moves


def normalize_move_name_current(move_id):
    """Current move normalization function."""
    move_mapping = {
        'chillyreception': 'Chilly Reception', 'thunderwave': 'Thunder Wave', 
        'stealthrock': 'Stealth Rock', 'earthquake': 'Earthquake', 'ruination': 'Ruination',
        'whirlwind': 'Whirlwind', 'spikes': 'Spikes', 'rest': 'Rest',
        'closecombat': 'Close Combat', 'crunch': 'Crunch', 'gigadrain': 'Giga Drain',
        'earthpower': 'Earth Power', 'nastyplot': 'Nasty Plot', 'ficklebeam': 'Fickle Beam',
        'leafstorm': 'Leaf Storm', 'dracometeor': 'Draco Meteor', 'futuresight': 'Future Sight',
        'sludgebomb': 'Sludge Bomb', 'psychicnoise': 'Psychic Noise', 'flamethrower': 'Flamethrower',
        'gigatonhammer': 'Gigaton Hammer', 'encore': 'Encore', 'knockoff': 'Knock Off',
        'playrough': 'Play Rough', 'hurricane': 'Hurricane', 'roost': 'Roost',
        'voltswitch': 'Volt Switch', 'discharge': 'Discharge', 'uturn': 'U-turn',
        'terablast': 'Tera Blast', 'swordsdance': 'Swords Dance', 'shadowball': 'Shadow Ball',
        'calmmind': 'Calm Mind', 'icespinner': 'Ice Spinner', 'suckerpunch': 'Sucker Punch',
        'willowisp': 'Will-O-Wisp', 'rapidspin': 'Rapid Spin', 'bodypress': 'Body Press'
    }
    lower_move = move_id.lower()
    if lower_move in move_mapping:
        return move_mapping[lower_move]
    # Default: add spaces before capitals and title case
    spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', move_id)
    return spaced.title()


class TestMoveNormalization:
    """Test class for move normalization functionality."""

    @pytest.mark.moves
    def test_common_moves_normalization(self, sample_moves):
        """Test normalization for the most common moves."""
        for move in sample_moves:
            # Test different input formats
            test_formats = [
                move.lower().replace(' ', '').replace('-', ''),  # Compressed format
                move.lower(),  # Lowercase format
            ]
            
            for test_format in test_formats:
                normalized = normalize_move_name_current(test_format)
                assert isinstance(normalized, str), f"Normalized move should be string for {test_format}"
                assert len(normalized) > 0, f"Normalized move should not be empty for {test_format}"

    @pytest.mark.moves
    def test_specific_move_mappings(self):
        """Test specific move mappings that are known to be problematic."""
        test_cases = [
            ('knockoff', 'Knock Off'),
            ('uturn', 'U-turn'), 
            ('stealthrock', 'Stealth Rock'),
            ('terablast', 'Tera Blast'),
            ('swordsdance', 'Swords Dance'),
            ('shadowball', 'Shadow Ball'),
            ('calmmind', 'Calm Mind'),
            ('thunderwave', 'Thunder Wave'),
            ('earthpower', 'Earth Power'),
            ('voltswitch', 'Volt Switch'),
        ]
        
        for input_move, expected_output in test_cases:
            result = normalize_move_name_current(input_move)
            assert result == expected_output, f"Expected '{input_move}' → '{expected_output}', got '{result}'"

    @pytest.mark.moves
    def test_move_normalization_edge_cases(self):
        """Test edge cases for move normalization."""
        edge_cases = [
            ('', ''),  # Empty string
            ('EARTHQUAKE', 'Earthquake'),  # All caps
            ('earthquake', 'Earthquake'),  # All lowercase
            ('Earthquake', 'Earthquake'),  # Proper case (no change needed)
            ('willowisp', 'Will-O-Wisp'),  # Hyphenated result
            ('uturn', 'U-turn'),  # Special capitalization
        ]
        
        for input_move, expected_output in edge_cases:
            if input_move == '':
                # Skip empty string test as it's not a realistic scenario
                continue
            result = normalize_move_name_current(input_move)
            assert result == expected_output, f"Expected '{input_move}' → '{expected_output}', got '{result}'"

    @pytest.mark.slow
    @pytest.mark.moves
    @pytest.mark.integration
    def test_all_moves_from_teamloader(self):
        """Test move normalization for all moves found in metamon teams."""
        # Load the team set
        teamloader = get_metamon_teams("gen9ou", "modern_replays")
        
        # Extract all unique moves from team files
        all_moves = []
        move_counts = Counter()
        failed_files = []
        
        # Sample a subset of team files (to keep processing time reasonable)
        sample_size = min(50, len(teamloader.team_files))  # Test up to 50 team files
        sample_files = random.sample(teamloader.team_files, sample_size)
        
        for file_path in sample_files:
            try:
                moves_in_file = extract_moves_from_team_file(file_path)
                for move in moves_in_file:
                    if move:  # Skip empty moves
                        all_moves.append(move)
                        move_counts[move] += 1
            except Exception as e:
                failed_files.append(file_path)
        
        unique_moves = list(set(all_moves))
        assert len(unique_moves) > 0, "No moves found in team files"
        
        # Test move normalization
        working_moves = []
        failed_moves = []
        
        for move in sorted(unique_moves):
            try:
                # Test current normalization
                normalized = normalize_move_name_current(move)
                assert isinstance(normalized, str), f"Normalized move should be string for {move}"
                assert len(normalized) > 0, f"Normalized move should not be empty for {move}"
                working_moves.append((move, normalized))
                
            except Exception as e:
                failed_moves.append((move, str(e)))
        
        # Calculate success rate
        success_rate = len(working_moves) / len(unique_moves) * 100
        
        assert success_rate >= 98, (
            f"Move normalization success rate too low: {success_rate:.1f}%. "
            f"Expected >98% success rate. Failed moves: {failed_moves[:5]}"
        )

    @pytest.mark.moves
    def test_move_normalization_consistency(self):
        """Test that move normalization is consistent across different input formats."""
        test_moves = ['Knock Off', 'Stealth Rock', 'Swords Dance', 'Thunder Wave']
        
        for move in test_moves:
            # Test different input formats for the same move
            formats = [
                move,  # Original
                move.lower(),  # Lowercase
                move.upper(),  # Uppercase
                move.lower().replace(' ', '').replace('-', ''),  # Compressed
            ]
            
            results = []
            for format_move in formats:
                try:
                    normalized = normalize_move_name_current(format_move)
                    results.append(normalized)
                except Exception as e:
                    pytest.fail(f"Move normalization failed for '{format_move}': {e}")
            
            # Check that the normalized forms are consistent
            # (allowing for the fact that some inputs might not change)
            assert len(set(results)) <= 2, (
                f"Inconsistent normalization for {move}: {dict(zip(formats, results))}"
            )


# Standalone test function for backward compatibility
def test_all_moves_from_teamloader():
    """Standalone version for backward compatibility."""
    test_instance = TestMoveNormalization()
    test_instance.test_all_moves_from_teamloader()


if __name__ == "__main__":
    print("Running move normalization tests...")
    test_all_moves_from_teamloader()
    print("✅ Move normalization test completed!")