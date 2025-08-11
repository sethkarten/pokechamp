"""
Static constants and reusable data for Pokemon simulation optimization.

This module contains constants, lookups, and cached computations that can be
shared across simulation instances to avoid repeated calculations.
"""

from functools import lru_cache
from typing import Dict, List, Tuple, Any

# Pokemon type constants (avoid recreating this list repeatedly)
TYPE_LIST = [
    'BUG', 'DARK', 'DRAGON', 'ELECTRIC', 'FAIRY', 'FIGHTING', 'FIRE', 
    'FLYING', 'GHOST', 'GRASS', 'GROUND', 'ICE', 'NORMAL', 'POISON', 
    'PSYCHIC', 'ROCK', 'STEEL', 'WATER'
]

# Type effectiveness categories
TYPE_EFFECTIVENESS = {
    'SUPER_EFFECTIVE': 2.0,
    'SUPER_SUPER_EFFECTIVE': 4.0,
    'RESISTED': 0.5,
    'SUPER_RESISTED': 0.25,
    'IMMUNE': 0.0,
    'NORMAL': 1.0
}


@lru_cache(maxsize=1000)  # Cache up to 1000 type combinations
def calculate_move_type_damage_multiplier_cached(
    type_1: str, 
    type_2: str, 
    type_chart_tuple: tuple,  # Convert type_chart to tuple for hashing
    constraint_types: tuple = None
) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
    """
    Cached version of move type damage multiplier calculation.
    
    Args:
        type_1: Primary type of the Pokemon
        type_2: Secondary type of the Pokemon (or empty string if None)
        type_chart_tuple: Type effectiveness chart as a tuple of tuples
        constraint_types: Optional constraint type list as tuple
        
    Returns:
        Tuple of (extreme_effective, effective, resistant, extreme_resistant, immune) type lists
    """
    # Convert tuple back to dict for processing
    type_chart = {}
    for i, type_name in enumerate(TYPE_LIST):
        type_chart[type_name] = {}
        for j, target_type in enumerate(TYPE_LIST):
            if i < len(type_chart_tuple) and j < len(type_chart_tuple[i]):
                type_chart[type_name][target_type] = type_chart_tuple[i][j]
            else:
                type_chart[type_name][target_type] = 1.0

    move_type_damage_multiplier_list = []

    if type_2:
        for target_type in TYPE_LIST:
            if 'STELLAR' not in [target_type, type_1, type_2]:
                multiplier = type_chart[type_1][target_type] * type_chart[type_2][target_type]
                move_type_damage_multiplier_list.append(multiplier)
            else:
                move_type_damage_multiplier_list.append(1.0)
        move_type_damage_multiplier_dict = dict(zip(TYPE_LIST, move_type_damage_multiplier_list))
    else:
        if type_1 == 'STELLAR':
            move_type_damage_multiplier_dict = {type_name: 1.0 for type_name in TYPE_LIST}
        else:
            move_type_damage_multiplier_dict = type_chart[type_1]

    # Categorize types by effectiveness
    effective_type_list = []
    extreme_effective_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    
    for type_name, value in move_type_damage_multiplier_dict.items():
        if value == TYPE_EFFECTIVENESS['SUPER_EFFECTIVE']:
            effective_type_list.append(type_name)
        elif value == TYPE_EFFECTIVENESS['SUPER_SUPER_EFFECTIVE']:
            extreme_effective_type_list.append(type_name)
        elif value == TYPE_EFFECTIVENESS['RESISTED']:
            resistant_type_list.append(type_name)
        elif value == TYPE_EFFECTIVENESS['SUPER_RESISTED']:
            extreme_resistant_type_list.append(type_name)
        elif value == TYPE_EFFECTIVENESS['IMMUNE']:
            immune_type_list.append(type_name)

    return (extreme_effective_type_list, effective_type_list, 
            resistant_type_list, extreme_resistant_type_list, immune_type_list)


def convert_type_chart_to_tuple(type_chart: Dict[str, Dict[str, float]]) -> tuple:
    """
    Convert a type chart dictionary to a tuple for caching purposes.
    
    Args:
        type_chart: Dictionary mapping type -> type -> effectiveness multiplier
        
    Returns:
        Tuple representation of the type chart
    """
    return tuple(
        tuple(type_chart.get(type1, {}).get(type2, 1.0) for type2 in TYPE_LIST)
        for type1 in TYPE_LIST
    )


class SimulationOptimizer:
    """
    Class to hold optimized simulation utilities and cached computations.
    """
    
    def __init__(self):
        self._type_chart_cache = {}
    
    def get_cached_type_effectiveness(
        self, 
        pokemon_type1: str, 
        pokemon_type2: str, 
        type_chart: Dict[str, Dict[str, float]],
        constraint_types: List[str] = None
    ) -> Tuple[List[str], List[str], List[str], List[str], List[str]]:
        """
        Get cached type effectiveness calculations for a Pokemon.
        
        Args:
            pokemon_type1: Primary type
            pokemon_type2: Secondary type (can be None)
            type_chart: Type effectiveness chart
            constraint_types: Optional list of types to constrain to
            
        Returns:
            Tuple of type effectiveness lists
        """
        # Convert type chart to cacheable format
        type_chart_tuple = convert_type_chart_to_tuple(type_chart)
        type_2_str = pokemon_type2 if pokemon_type2 else ""
        constraint_tuple = tuple(constraint_types) if constraint_types else None
        
        return calculate_move_type_damage_multiplier_cached(
            pokemon_type1, type_2_str, type_chart_tuple, constraint_tuple
        )


# Global optimizer instance
_optimizer = SimulationOptimizer()


def get_simulation_optimizer() -> SimulationOptimizer:
    """Get the global simulation optimizer instance."""
    return _optimizer


def clear_simulation_cache():
    """Clear all simulation caches."""
    calculate_move_type_damage_multiplier_cached.cache_clear()
    _optimizer._type_chart_cache.clear()


# Common Pokemon stat calculation constants
NATURE_MODIFIERS = {
    'boost': 1.1,
    'nerf': 0.9,
    'neutral': 1.0
}

# EV/IV constants
MAX_IV = 31
MAX_EV = 252
MAX_TOTAL_EVS = 510

# Level constants for stat calculations
LEVEL_50 = 50
LEVEL_100 = 100