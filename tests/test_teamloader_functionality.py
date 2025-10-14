"""
Tests for teamloader functionality and team rejection handling.

This test suite verifies that the teamloader correctly handles team rejections
and provides fallback teams when battles fail.
"""

import pytest
import asyncio
import time
from poke_env.player.random_player import RandomPlayer
from poke_env.player.team_util import get_metamon_teams


class TestTeamloaderFunctionality:
    """Test class for teamloader functionality."""

    @pytest.mark.teamloader
    @pytest.mark.asyncio
    async def test_basic_teamloader_functionality(self):
        """Test basic teamloader functionality."""
        teamloader = get_metamon_teams("gen9ou", "modern_replays")
        
        assert teamloader is not None, "Teamloader should be created successfully"
        assert len(teamloader.team_files) > 0, "Should have team files available"
        
        # Test team generation
        team1 = teamloader.yield_team()
        team2 = teamloader.yield_team()
        
        assert team1 is not None, "Should generate a valid team"
        assert team2 is not None, "Should generate another valid team"
        assert isinstance(team1, str), "Team should be a string"
        assert isinstance(team2, str), "Team should be a string"

    @pytest.mark.teamloader
    @pytest.mark.asyncio
    async def test_teamloader_rejection_recovery(self):
        """Test that rejected teams get replaced with teamloader.yield_team()."""
        # Create players
        player1 = RandomPlayer(battle_format="gen9ou")
        player2 = RandomPlayer(battle_format="gen9ou")
        
        # Set up teamloader
        teamloader = get_metamon_teams("gen9ou", "modern_replays")
        
        # Set teamloader on both players for rejection recovery
        player1.set_teamloader(teamloader)
        player2.set_teamloader(teamloader)
        
        # Load initial teams (might get rejected)
        team1 = teamloader.yield_team()
        team2 = teamloader.yield_team()
        
        player1.update_team(team1)
        player2.update_team(team2)
        
        assert player1._team is not None, "Player 1 should have a team"
        assert player2._team is not None, "Player 2 should have a team"
        
        # Test a quick battle with timeout
        start_time = time.time()
        
        try:
            # Use asyncio.wait_for to add a timeout
            await asyncio.wait_for(
                player1.battle_against(player2, n_battles=1),
                timeout=30.0
            )
            
            end_time = time.time()
            battle_duration = end_time - start_time
            
            # Battle should complete within reasonable time
            assert battle_duration < 25.0, f"Battle took too long: {battle_duration:.1f}s"
            
            # Verify that both players participated
            total_battles = player1.n_finished_battles + player2.n_finished_battles
            assert total_battles >= 1, "At least one battle should be recorded"
            
        except asyncio.TimeoutError:
            pytest.fail("Battle timed out - possible team rejection issue")

    @pytest.mark.teamloader
    def test_teamloader_team_variety(self):
        """Test that teamloader provides variety in teams."""
        teamloader = get_metamon_teams("gen9ou", "modern_replays")
        
        # Generate multiple teams and check for variety
        teams = []
        for _ in range(5):
            team = teamloader.yield_team()
            teams.append(team)
        
        # All teams should be different (very high probability)
        unique_teams = set(teams)
        assert len(unique_teams) > 1, "Should generate different teams"
        
        # Each team should have Pokemon
        for i, team in enumerate(teams):
            assert len(team) > 0, f"Team {i} should not be empty"
            # Basic format check - should contain Pokemon names
            assert any(char.isalpha() for char in team), f"Team {i} should contain Pokemon names"

    @pytest.mark.teamloader
    @pytest.mark.slow
    def test_teamloader_file_integrity(self):
        """Test that teamloader files are valid and readable."""
        teamloader = get_metamon_teams("gen9ou", "modern_replays")
        
        # Check that we can read a sample of files
        sample_size = min(10, len(teamloader.team_files))
        sample_files = teamloader.team_files[:sample_size]
        
        for file_path in sample_files:
            assert os.path.exists(file_path), f"Team file should exist: {file_path}"
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    assert len(content) > 0, f"Team file should not be empty: {file_path}"
                    # Basic format check
                    assert '@' in content or 'Ability:' in content, f"Should be valid team format: {file_path}"
            except Exception as e:
                pytest.fail(f"Failed to read team file {file_path}: {e}")

    @pytest.mark.teamloader
    def test_teamloader_different_formats(self):
        """Test teamloader with different battle formats - focus on gen9ou."""
        # Test gen9ou primarily, but also check if gen8ou available
        formats = ["gen9ou", "gen8ou"]
        
        for battle_format in formats:
            try:
                teamloader = get_metamon_teams(battle_format, "modern_replays")
                assert teamloader is not None, f"Should create teamloader for {battle_format}"
                assert len(teamloader.team_files) > 0, f"Should have teams for {battle_format}"
                
                # Test team generation
                team = teamloader.yield_team()
                assert team is not None, f"Should generate team for {battle_format}"
                assert len(team) > 0, f"Team should not be empty for {battle_format}"
                
                # gen9ou must work, gen8ou can be skipped if unavailable
                if battle_format == "gen9ou":
                    # gen9ou should always work
                    assert team is not None, "gen9ou teams must be available"
                
            except Exception as e:
                # gen8ou might not be available, but gen9ou must work
                if battle_format == "gen8ou" and ("Cannot locate valid team directory" in str(e) or "404" in str(e)):
                    pytest.skip(f"Teams not available for {battle_format} - this is acceptable")
                elif battle_format == "gen9ou":
                    # gen9ou should always work
                    pytest.fail(f"gen9ou teams must be available: {e}")
                else:
                    raise


# Import required for async tests
import os


# Standalone test function for backward compatibility
@pytest.mark.asyncio
async def test_teamloader_rejection():
    """Standalone version for backward compatibility."""
    test_instance = TestTeamloaderFunctionality()
    await test_instance.test_teamloader_rejection_recovery()


if __name__ == "__main__":
    # Run the async test
    asyncio.run(test_teamloader_rejection())
    print("✅ Teamloader functionality test completed!")