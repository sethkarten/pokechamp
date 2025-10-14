"""
Minimax Performance Optimization System

This module provides optimizations specifically for the minimax tree search
to reduce LocalSim node creation overhead and improve performance.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from copy import copy, deepcopy
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from poke_env.environment.battle import Battle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.local_simulation import LocalSim


@dataclass
class BattleStateHash:
    """Lightweight battle state representation for hashing and caching."""
    active_pokemon_hp: Tuple[int, int]  # (player, opponent)
    active_pokemon_species: Tuple[str, str]
    team_remaining: Tuple[int, int]  # remaining Pokemon count
    turn_number: int
    weather: str
    terrain: str
    
    def __hash__(self):
        return hash((
            self.active_pokemon_hp,
            self.active_pokemon_species, 
            self.team_remaining,
            self.turn_number,
            self.weather,
            self.terrain
        ))


def create_battle_state_hash(battle: Battle) -> BattleStateHash:
    """Create a lightweight hash representation of battle state."""
    try:
        player_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0
        opp_hp = battle.opponent_active_pokemon.current_hp_fraction if battle.opponent_active_pokemon else 0
        
        player_species = battle.active_pokemon.species if battle.active_pokemon else ""
        opp_species = battle.opponent_active_pokemon.species if battle.opponent_active_pokemon else ""
        
        player_remaining = len([p for p in battle.team.values() if not p.fainted])
        opp_remaining = len([p for p in battle.opponent_team.values() if not p.fainted])
        
        weather = str(battle.weather) if hasattr(battle, 'weather') else ""
        terrain = str(battle.fields) if hasattr(battle, 'fields') else ""
        
        return BattleStateHash(
            active_pokemon_hp=(int(player_hp * 100), int(opp_hp * 100)),
            active_pokemon_species=(player_species, opp_species),
            team_remaining=(player_remaining, opp_remaining),
            turn_number=battle.turn,
            weather=weather,
            terrain=terrain
        )
    except Exception as e:
        # Fallback hash based on turn number only
        return BattleStateHash((0, 0), ("", ""), (0, 0), battle.turn, "", "")


class LocalSimPool:
    """Object pool for LocalSim instances to avoid repeated creation."""
    
    def __init__(self, initial_size: int = 1):
        self._available_sims: List[LocalSim] = []
        self._in_use_sims: List[LocalSim] = []
        self._initial_size = initial_size
        self._creation_template: Optional[Dict[str, Any]] = None
        
    def initialize_pool(self, template_battle: Battle, **localsim_kwargs):
        """Initialize the pool with template LocalSim instances."""
        self._creation_template = localsim_kwargs
        
        print(f"🔄 Initializing LocalSim pool with {self._initial_size} instance (memory optimized)...")
        for i in range(self._initial_size):
            sim = LocalSim(
                battle=deepcopy(template_battle),
                **localsim_kwargs
            )
            self._available_sims.append(sim)
        print(f"✅ LocalSim pool initialized with {len(self._available_sims)} instance")
    
    def acquire_sim(self, battle: Battle) -> LocalSim:
        """Get a LocalSim from the pool, creating new one if needed."""
        if self._available_sims:
            sim = self._available_sims.pop()
            # Reset the simulation with new battle state
            sim.battle = deepcopy(battle)
            self._in_use_sims.append(sim)
            return sim
        else:
            # Pool exhausted, create new instance
            if self._creation_template is None:
                raise RuntimeError("Pool not initialized - call initialize_pool() first")
                
            sim = LocalSim(
                battle=deepcopy(battle),
                **self._creation_template
            )
            self._in_use_sims.append(sim)
            return sim
    
    def release_sim(self, sim: LocalSim):
        """Return a LocalSim to the pool for reuse."""
        if sim in self._in_use_sims:
            self._in_use_sims.remove(sim)
            # Clean up the simulation state
            sim.battle = None  # Clear battle reference
            self._available_sims.append(sim)
    
    def release_all(self):
        """Release all in-use sims back to the pool."""
        for sim in self._in_use_sims[:]:  # Copy list to avoid modification during iteration
            self.release_sim(sim)
    
    def get_stats(self) -> Tuple[int, int, int]:
        """Get pool statistics: (available, in_use, total)."""
        return len(self._available_sims), len(self._in_use_sims), len(self._available_sims) + len(self._in_use_sims)


class MinimaxCache:
    """Cache for minimax evaluation results to avoid recomputation."""
    
    def __init__(self, max_size: int = 1000):
        self._cache: Dict[Tuple[BattleStateHash, str, str], float] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def get_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str) -> Optional[float]:
        """Get cached evaluation for a battle state + action combination."""
        key = (battle_state, player_action, opp_action)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        else:
            self._misses += 1
            return None
    
    def set_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str, value: float):
        """Cache an evaluation result."""
        key = (battle_state, player_action, opp_action)
        
        # Simple LRU: if cache is full, remove oldest 25% of entries
        if len(self._cache) >= self._max_size:
            items_to_remove = list(self._cache.keys())[:self._max_size // 4]
            for item in items_to_remove:
                del self._cache[item]
        
        self._cache[key] = value
    
    def clear(self):
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
    
    def get_stats(self) -> Tuple[int, int, float]:
        """Get cache statistics: (hits, misses, hit_rate)."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return self._hits, self._misses, hit_rate


class OptimizedSimNode:
    """Optimized version of SimNode that uses object pooling and caching."""
    
    def __init__(self, battle: Battle, sim_pool: LocalSimPool, depth: int = 0):
        self.simulation = sim_pool.acquire_sim(battle)
        self.sim_pool = sim_pool
        self.depth = depth
        self.action: Optional[BattleOrder] = None
        self.action_opp: Optional[BattleOrder] = None
        self.parent_node = None
        self.parent_action = None
        self.hp_diff = 0
        self.children: List['OptimizedSimNode'] = []
        self.battle_state_hash = create_battle_state_hash(battle)
    
    def __del__(self):
        """Return simulation to pool when node is destroyed."""
        if hasattr(self, 'simulation') and self.simulation is not None:
            self.sim_pool.release_sim(self.simulation)
    
    def create_child_node(self, player_action: BattleOrder, opp_action: BattleOrder) -> 'OptimizedSimNode':
        """Create a child node efficiently."""
        # Create new battle state by stepping forward
        child_battle = deepcopy(self.simulation.battle)
        
        # Create child node
        child_node = OptimizedSimNode(child_battle, self.sim_pool, self.depth + 1)
        child_node.action = player_action
        child_node.action_opp = opp_action
        child_node.parent_node = self
        child_node.parent_action = self.action
        
        # Step the simulation forward
        child_node.simulation.step(player_action, opp_action)
        
        # Update relationships
        self.children.append(child_node)
        
        return child_node
    
    def cleanup(self):
        """Recursively cleanup all child nodes and return sims to pool."""
        for child in self.children:
            child.cleanup()
        
        if self.simulation is not None:
            self.sim_pool.release_sim(self.simulation)
            self.simulation = None


class MinimaxOptimizer:
    """Main optimizer for minimax tree search."""
    
    def __init__(self):
        self.sim_pool = LocalSimPool(initial_size=1)  # Single instance to reduce memory usage
        self.cache = MinimaxCache(max_size=2000)
        self.stats = {
            'nodes_created': 0,
            'cache_hits': 0,
            'pool_reuses': 0,
            'total_time': 0.0
        }
    
    def initialize(self, battle: Battle, **localsim_kwargs):
        """Initialize the optimizer with battle template."""
        self.sim_pool.initialize_pool(battle, **localsim_kwargs)
        print(f"🚀 MinimaxOptimizer initialized")
    
    def create_optimized_root(self, battle: Battle) -> OptimizedSimNode:
        """Create an optimized root node for minimax search."""
        root = OptimizedSimNode(battle, self.sim_pool, depth=1)
        self.stats['nodes_created'] += 1
        return root
    
    def cleanup_tree(self, root: OptimizedSimNode):
        """Cleanup entire tree and return all sims to pool."""
        root.cleanup()
        self.sim_pool.release_all()
    
    def get_cached_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str) -> Optional[float]:
        """Try to get cached evaluation for this state."""
        result = self.cache.get_evaluation(battle_state, player_action, opp_action)
        if result is not None:
            self.stats['cache_hits'] += 1
        return result
    
    def cache_evaluation(self, battle_state: BattleStateHash, player_action: str, opp_action: str, value: float):
        """Cache an evaluation result."""
        self.cache.set_evaluation(battle_state, player_action, opp_action, value)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        pool_available, pool_in_use, pool_total = self.sim_pool.get_stats()
        cache_hits, cache_misses, cache_hit_rate = self.cache.get_stats()
        
        return {
            'nodes_created': self.stats['nodes_created'],
            'pool_stats': {
                'available': pool_available,
                'in_use': pool_in_use,
                'total': pool_total,
                'reuse_rate': self.stats['pool_reuses'] / max(1, self.stats['nodes_created'])
            },
            'cache_stats': {
                'hits': cache_hits,
                'misses': cache_misses,
                'hit_rate': cache_hit_rate
            },
            'total_time': self.stats['total_time']
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.stats = {
            'nodes_created': 0,
            'cache_hits': 0,
            'pool_reuses': 0,
            'total_time': 0.0
        }
        self.cache.clear()


# Global optimizer instance
_minimax_optimizer = MinimaxOptimizer()


def get_minimax_optimizer() -> MinimaxOptimizer:
    """Get the global minimax optimizer instance."""
    return _minimax_optimizer


def initialize_minimax_optimization(battle: Battle, **localsim_kwargs):
    """Initialize minimax optimizations for a battle."""
    _minimax_optimizer.initialize(battle, **localsim_kwargs)


@lru_cache(maxsize=500)
def fast_battle_evaluation(
    active_hp_player: int,
    active_hp_opp: int, 
    team_count_player: int,
    team_count_opp: int,
    turn: int
) -> float:
    """
    Fast heuristic evaluation function that avoids LLM calls.
    
    This provides a quick approximation of battle state value based on:
    - HP advantage
    - Team size advantage  
    - Turn progression penalty
    """
    # HP advantage (0-100 scale)
    hp_advantage = (active_hp_player - active_hp_opp) * 20
    
    # Team advantage (each Pokemon worth ~15 points)
    team_advantage = (team_count_player - team_count_opp) * 15
    
    # Turn penalty (encourages quicker wins)
    turn_penalty = min(turn * 0.5, 10)
    
    # Base score starts at 50 (neutral)
    score = 50 + hp_advantage + team_advantage - turn_penalty
    
    # Clamp to valid range
    return max(0, min(100, score))