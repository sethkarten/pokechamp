from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from poke_env.data import GenData, to_id_str
from poke_env.environment.effect import Effect
from poke_env.environment.move import SPECIAL_MOVES, Move
from poke_env.environment.pokemon_gender import PokemonGender
from poke_env.environment.pokemon_type import PokemonType
from poke_env.environment.status import Status
from poke_env.environment.z_crystal import Z_CRYSTAL
import math


class Pokemon:
    
    __slots__ = (
        "_ability",
        "_active",
        "_active",
        "_base_stats",
        "_boosts",
        "_current_hp",
        "_data",
        "_effects",
        "_first_turn",
        "_gender",
        "_heightm",
        "_item",
        "_last_details",
        "_last_request",
        "_level",
        "_max_hp",
        "_moves",
        "_must_recharge",
        "_possible_abilities",
        "_preparing_move",
        "_preparing_target",
        "_protect_counter",
        "_sets",
        "_shiny",
        "_revealed",
        "_species",
        "_status",
        "_status_counter",
        "_terastallized",
        "_terastallized_type",
        #"_tera_type"
        "_type_1",
        "_type_2",
        "_weightkg",
    )

    def __init__(
        self,
        gen: int,
        *,
        species: Optional[str] = None,
        request_pokemon: Optional[Dict[str, Any]] = None,
        details: Optional[str] = None,
    ):
        # Base data
        self._data = GenData.from_gen(gen)

        # Species related attributes
        self._base_stats: Dict[str, int]
        self._heightm: int
        self._possible_abilities: List[str]
        self._species: str = ""
        self._type_1: PokemonType
        self._type_2: Optional[PokemonType] = None
        self._weightkg: int

        # Individual related attributes
        self._ability: Optional[str] = None
        self._active: bool
        self._gender: Optional[PokemonGender] = None
        self._level: int = 100
        self._max_hp: Optional[int] = 0
        self._moves: Dict[str, Move] = {}
        self._shiny: Optional[bool] = False

        # Battle related attributes
        self._active: bool = False
        self._boosts: Dict[str, int] = {
            "accuracy": 0,
            "atk": 0,
            "def": 0,
            "evasion": 0,
            "spa": 0,
            "spd": 0,
            "spe": 0,
            "hp": 0,                             # TODO: for dynamax
        }
        self._current_hp: Optional[int] = 0
        self._effects: Dict[Effect, int] = {}
        self._first_turn: bool = False
        self._terastallized: bool = False
        self._terastallized_type: Optional[PokemonType] = None
        #self._tera_type : Optional[PokemonType] = None
        self._item: Optional[str] = self._data.UNKNOWN_ITEM
        self._last_request: Optional[Dict[str, Any]] = {}
        self._last_details: str = ""
        self._must_recharge: bool = False
        self._preparing_move: Optional[Move] = None
        self._preparing_target = None
        self._protect_counter: int = 0
        self._revealed: bool = False
        self._status: Optional[Status] = None
        self._status_counter: int = 0

        with open('poke_env/data/static/gen9/ou/sets_1500.json', 'r') as f:
            sets = json.load(f)
        self._sets = sets
        
        
        if request_pokemon:
            self.update_from_request(request_pokemon)
        elif details:
            self._update_from_details(details)
        elif species:
            self._update_from_pokedex(species)
            
        
        
    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        if self._status is None:
            status_repr = None
        else:
            status_repr = self._status.name

        return (
            f"{self._species} (pokemon object) "
            f"[Active: {self._active}, Status: {status_repr}]"
        )

    def _add_move(self, move_id: str, use: bool = False) -> Optional[Move]:
        """Store the move if applicable."""
        id_ = Move.retrieve_id(move_id)

        if not Move.should_be_stored(id_, self._data.gen):
            return

        if id_ not in self._moves:
            move = Move(move_id=id_, raw_id=move_id, gen=self._data.gen)
            self._moves[id_] = move
        if use:
            self._moves[id_].use()

        return self._moves[id_]

    def boost(self, stat: str, amount: int):
        self._boosts[stat] += amount
        if self._boosts[stat] > 6:
            self._boosts[stat] = 6
        elif self._boosts[stat] < -6:
            self._boosts[stat] = -6

    def cant_move(self):
        self._first_turn = False
        self._protect_counter = 0

        if self._status == Status.SLP:
            self._status_counter += 1

    def clear_active(self):
        self._active = False

    def clear_boosts(self):
        for stat in self._boosts:
            self._boosts[stat] = 0

    def _clear_effects(self):
        self._effects = {}

    def clear_negative_boosts(self):
        for stat, value in self._boosts.items():
            if value < 0:
                self._boosts[stat] = 0

    def clear_positive_boosts(self):
        for stat, value in self._boosts.items():
            if value > 0:
                self._boosts[stat] = 0

    def copy_boosts(self, mon: Pokemon):
        self._boosts = dict(mon._boosts.items())

    def cure_status(self, status: Optional[str] = None):
        if status and Status[status.upper()] == self._status:
            self._status = None
            self._status_counter = 0
        elif status is None and not self.fainted:
            self._status = None

    def damage(self, hp_status: str):
        self.set_hp_status(hp_status)

    def end_effect(self, effect_str: str):
        effect = Effect.from_showdown_message(effect_str)
        if effect in self._effects:
            self._effects.pop(effect)

    def end_item(self, item: str):
        self._item = None

        if item == "powerherb":
            self._preparing_move = None
            self._preparing_target = False

    def end_turn(self):
        if self._status == Status.TOX:
            self._status_counter += 1
        for effect in self.effects:
            if effect.is_turn_countable:
                self.effects[effect] += 1

    def faint(self):
        self._current_hp = 0
        self._status = Status.FNT

    def forme_change(self, species: str):
        species = species.split(",")[0]
        self._update_from_pokedex(species, store_species=False)

    def heal(self, hp_status: str):
        self.set_hp_status(hp_status)

    def invert_boosts(self):
        self._boosts = {k: -v for k, v in self._boosts.items()}

    def mega_evolve(self, stone: str):
        species_id_str = to_id_str(self.species)
        mega_species = (
            species_id_str + "mega"
            if not species_id_str.endswith("mega")
            else species_id_str
        )
        if mega_species in self._data.pokedex:
            self._update_from_pokedex(mega_species, store_species=False)
        elif stone[-1] in "XYxy":
            mega_species = mega_species + stone[-1].lower()
            self._update_from_pokedex(mega_species, store_species=False)

    def moved(self, move_id: str, failed: bool = False, use: bool = True):
        self._must_recharge = False
        self._preparing_move = None
        self._preparing_target = None
        move = self._add_move(move_id, use=use)

        if move and move.is_protect_counter and not failed:
            self._protect_counter += 1
        else:
            self._protect_counter = 0

        if self._status == Status.SLP:
            self._status_counter += 1

        if len(self._moves) > 4:
            new_moves = {}

            # Keep the current move
            if move and move in self._moves.values():
                new_moves = {
                    move_id: m for move_id, m in self._moves.items() if m is move
                }

            for move in self._moves:
                if len(new_moves) == 4:
                    break
                elif move not in new_moves:
                    new_moves[move] = self._moves[move]

            self._moves = new_moves

    def prepare(self, move_id: str, target: Optional[Pokemon]):
        self.moved(move_id, use=False)

        move_id = Move.retrieve_id(move_id)
        if move_id in self.moves:
            move = self.moves[move_id]
        else:
            # Handle moves that aren't stored (like special moves)
            move = Move(move_id, gen=self._data.gen)
        
        self._preparing_move = move
        self._preparing_target = target

    def primal(self):
        species_id_str = to_id_str(self._species)
        primal_species = (
            species_id_str + "primal"
            if not species_id_str.endswith("primal")
            else species_id_str
        )
        self._update_from_pokedex(primal_species, store_species=False)

    def set_boost(self, stat: str, amount: int):
        assert (
            abs(amount) <= 6
        ), f"{stat} of mon {self._species} is not <= 6. Got {amount}"
        self._boosts[stat] = int(amount)

    def set_hp(self, hp_status: str):
        self.set_hp_status(hp_status)

    def set_hp_status(self, hp_status: str):
        if hp_status == "0 fnt":
            self.faint()
            return

        if " " in hp_status:
            hp, status = hp_status.split(" ")
            self._status = Status[status.upper()]
        else:
            hp = hp_status

        hp = "".join([c for c in hp if c in "0123456789/"]).split("/")
        self._current_hp = int(hp[0])
        self._max_hp = int(hp[1])

    def start_effect(self, effect_str: str):
        effect = Effect.from_showdown_message(effect_str)
        if effect not in self._effects:
            self._effects[effect] = 0
        elif effect.is_action_countable:
            self._effects[effect] += 1

        if effect.breaks_protect:
            self._protect_counter = 0

    def _swap_boosts(self):
        self._boosts["atk"], self._boosts["spa"] = (
            self._boosts["spa"],
            self._boosts["atk"],
        )

    def switch_in(self, details: Optional[str] = None):
        self._active = True

        if details:
            self._update_from_details(details)

        self._first_turn = True
        self._revealed = True

    def switch_out(self):
        self._active = False
        self.clear_boosts()
        self._clear_effects()
        self._first_turn = False
        self._must_recharge = False
        self._preparing_move = None
        self._preparing_target = None
        self._protect_counter = 0

        if self._status == Status.TOX:
            self._status_counter = 0

    def terastallize(self, type_: str = ''):
        if type_ == '' and self._terastallized_type == None:
            type_ = self.guess_tera()
            if type_ == '':
                return
            
        if type_ != '':
            self._terastallized_type = PokemonType.from_name(type_)        

        self._terastallized = True
        
    def unterastallize(self):
        # self._terastallized_type = None
        self._terastallized = False

    def transform(self, into: Pokemon):
        current_hp = self.current_hp
        self._update_from_pokedex(into.species, store_species=False)
        self._current_hp = int(current_hp)
        self._boosts = into.boosts.copy()

    def _update_from_pokedex(self, species: str, store_species: bool = True):
        species = to_id_str(species)
        dex_entry = self._data.pokedex[species]
        if store_species:
            self._species = species
        self._base_stats = dex_entry["baseStats"]
        self._type_1 = PokemonType.from_name(dex_entry["types"][0])
        if len(dex_entry["types"]) == 1:
            self._type_2 = None
        else:
            self._type_2 = PokemonType.from_name(dex_entry["types"][1])

        self._possible_abilities = [
            to_id_str(ability) for ability in dex_entry["abilities"].values()
        ]

        if len(self._possible_abilities) == 1:
            self.ability = self._possible_abilities[0]

        self._heightm = dex_entry["heightm"]
        self._weightkg = dex_entry["weightkg"]

    def _update_from_details(self, details: str):
        if details == self._last_details:
            return
        else:
            self._last_details = details

        if ", shiny" in details:
            self._shiny = True
            details = details.replace(", shiny", "")
        else:
            self._shiny = False

        split_details = details.split(", ")

        gender = None
        level = None

        for split_detail in split_details:
            if split_detail.startswith("tera:"):
                self._terastallized_type = PokemonType.from_name(split_detail[5:])

                split_details.remove(split_detail)
                break

        if len(split_details) == 3:
            species, level, gender = split_details
        elif len(split_details) == 2:
            if split_details[1].startswith("L") or split_details[1].startswith("l"):
                species, level = split_details
            else:
                species, gender = split_details
        else:
            species = to_id_str(split_details[0])

        if gender:
            self._gender = PokemonGender.from_request_details(gender.upper())
        else:
            self._gender = PokemonGender.NEUTRAL

        if level:
            self._level = int(level[1:])
        else:
            self._level = 100

        if species != self._species:
            self._update_from_pokedex(species)

    def update_from_request(self, request_pokemon: Dict[str, Any]):
        self._active = request_pokemon["active"]

        if request_pokemon == self._last_request:
            return

        if "ability" in request_pokemon:
            self.ability = request_pokemon["ability"]
        elif "baseAbility" in request_pokemon:
            self.ability = request_pokemon["baseAbility"]

        self._last_request = request_pokemon

        condition = request_pokemon["condition"]
        self.set_hp_status(condition)

        self._item = request_pokemon["item"]

        details = request_pokemon["details"]
        self._update_from_details(details)

        for move in request_pokemon["moves"]:
            self._add_move(move)

        if len(self._moves) > 4:
            moves_to_keep = {
                Move.retrieve_id(move_id) for move_id in request_pokemon["moves"]
            }
            self._moves = {
                move_id: move
                for move_id, move in self._moves.items()
                if move_id in moves_to_keep
            }

    def used_z_move(self):
        self._item = None

    def was_illusioned(self):
        self._current_hp = None
        self._max_hp = None
        self._status = None

        last_request = self._last_request
        self._last_request = None

        if last_request:
            self.update_from_request(last_request)

        self.switch_out()

    def available_moves_from_request(self, request: Dict[str, Any]) -> List[Move]:
        moves: List[Move] = []

        request_moves: List[str] = [
            move["id"] for move in request["moves"] if not move.get("disabled", False)
        ]
        for move in request_moves:
            if move in self.moves:
                if self.is_dynamaxed:
                    moves.append(self.moves[move].dynamaxed)
                else:
                    moves.append(self.moves[move])
            elif move in SPECIAL_MOVES:
                moves.append(Move(move, gen=self._data.gen))
            elif (
                move == "hiddenpower"
                and len([m for m in self.moves if m.startswith("hiddenpower")]) == 1
            ):
                moves.append(
                    [v for m, v in self.moves.items() if m.startswith("hiddenpower")][0]
                )
            elif self.is_dynamaxed:
                moves.append(Move(move, gen=self._data.gen).dynamaxed)
            else:
                moves.append(Move(move, gen=self._data.gen))
        return moves

    def damage_multiplier(self, type_or_move: Union[PokemonType, Move]) -> float:
        """
        Returns the damage multiplier associated with a given type or move on this
        pokemon.

        This method is a shortcut for PokemonType.damage_multiplier with relevant types.

        :param type_or_move: The type or move of interest.
        :type type_or_move: PokemonType or Move
        :return: The damage multiplier associated with given type on the pokemon.
        :rtype: float
        """
        if isinstance(type_or_move, Move):
            type_or_move = type_or_move.type
        return type_or_move.damage_multiplier(
            self._type_1, self._type_2, type_chart=self._data.type_chart
        )

    @property
    def ability(self) -> Optional[str]:
        """
        :return: The pokemon's ability. None if unknown.
        :rtype: str, optional
        """
        return self._ability

    @ability.setter
    def ability(self, ability: Optional[str]):
        if ability is None:
            self._ability = None
        else:
            self._ability = to_id_str(ability)

    @property
    def active(self) -> Optional[bool]:
        """
        :return: Boolean indicating whether the pokemon is active.
        :rtype: bool
        """
        return self._active

    @active.setter
    def active(self, value: Optional[bool]):
        self.active = value

    @property
    def available_z_moves(self) -> List[Move]:
        """
        Caution: this property is not properly tested yet.

        :return: The set of moves that pokemon can use as z-moves.
        :rtype: List[Move]
        """
        if isinstance(self.item, str) and self.item.endswith("iumz"):
            type_, move = Z_CRYSTAL[self.item]
            if type_:
                return [
                    move
                    for move in self._moves.values()
                    if move.type == type_ and move.can_z_move
                ]
            elif move in self._moves:
                return [self._moves[move]]
        return []

    @property
    def base_species(self) -> str:
        """
        :return: The pokemon's base species.
        :rtype: str
        """
        dex_entry = self._data.pokedex[self._species]
        if "baseSpecies" in dex_entry:
            return to_id_str(dex_entry["baseSpecies"])
        return self._species

    @property
    def base_stats(self) -> Dict[str, int]:
        """
        :return: The pokemon's base stats.
        :rtype: Dict[str, int]
        """
        return self._base_stats

    @property
    def boosts(self) -> Dict[str, int]:
        """
        :return: The pokemon's boosts.
        :rtype: Dict[str, int]
        """
        return self._boosts

    @boosts.setter
    def boosts(self, value: Dict[str, int]):
        self._boosts = value

    @property
    def current_hp(self) -> int:
        """
        :return: The pokemon's current hp. For your pokemons, this is the actual value.
            For opponent's pokemon, this value depends on showdown information: it can
            be on a scale from 0 to 100 or on a pixel scale.
        :rtype: int
        """
        return self._current_hp or 0

    @property
    def current_hp_fraction(self) -> float:
        """
        :return: The pokemon's current remaining hp fraction.
        :rtype: float
        """
        if self.current_hp:
            return self.current_hp / self.max_hp
        return 0

    @property
    def effects(self) -> Dict[Effect, int]:
        """
        :return: A Dict mapping the effects currently affecting the pokemon and the
            associated counter.
        :rtype: Dict[Effect, int]
        """
        return self._effects

    @property
    def fainted(self) -> bool:
        """
        :return: Wheter the pokemon has fainted.
        :rtype: bool
        """
        return Status.FNT == self._status

    @property
    def first_turn(self) -> bool:
        """
        :return: Wheter this is this pokemon's first action since its last switch in.
        :rtype: bool
        """
        return self._first_turn

    @property
    def gender(self) -> Optional[PokemonGender]:
        """
        :return: The pokemon's gender.
        :rtype: PokemonGender, optional
        """
        return self._gender

    @property
    def height(self) -> float:
        """
        :return: The pokemon's height, in meters.
        :rtype: float
        """
        return self._heightm

    @property
    def is_dynamaxed(self) -> bool:
        """
        :return: Whether the pokemon is currently dynamaxed
        :rtype: bool
        """
        return Effect.DYNAMAX in self.effects

    @property
    def item(self) -> Optional[str]:
        """
        :return: The pokemon's item.
        :rtype: str | None
        """
        return self._item

    @item.setter
    def item(self, item: Optional[str]):
        self._item = item

    @property
    def level(self) -> int:
        """
        :return: The pokemon's level.
        :rtype: int
        """
        return self._level

    @property
    def max_hp(self) -> int:
        """
        :return: The pokemon's max hp. For your pokemons, this is the actual value.
            For opponent's pokemon, this value depends on showdown information: it can
            be on a scale from 0 to 100 or on a pixel scale.
        :rtype: int
        """
        return self._max_hp or 0

    @property
    def moves(self) -> Dict[str, Move]:
        """
        :return: A dictionary of the pokemon's known moves.
        :rtype: Dict[str, Move]
        """
        return self._moves

    @property
    def must_recharge(self) -> bool:
        """
        :return: A boolean indicating whether the pokemon must recharge.
        :rtype: bool
        """
        return self._must_recharge

    @must_recharge.setter
    def must_recharge(self, value: bool):
        self._must_recharge = value

    @property
    def pokeball(self) -> Optional[str]:
        """
        :return: The pokeball in which is the pokemon.
        :rtype: str | None
        """
        if self._last_request is not None:
            return self._last_request.get("pokeball", None)

    @property
    def possible_abilities(self) -> List[str]:
        """
        :return: The list of possible abilities for this pokemon.
        :rtype: List[str]
        """
        return self._possible_abilities

    @property
    def preparing(self) -> bool:
        """
        :return: Whether this pokemon is preparing a multi-turn move.
        :rtype: bool
        """
        return bool(self._preparing_target) or bool(self._preparing_move)

    @property
    def preparing_target(self) -> Optional[Union[bool, Pokemon]]:
        """
        :return: The moves target - optional.
        :rtype: Any
        """
        return self._preparing_target

    @property
    def preparing_move(self) -> Optional[Move]:
        """
        :return: The move being prepared - optional.
        :rtype: Move, optional
        """
        return self._preparing_move

    @property
    def protect_counter(self) -> int:
        """
        :return: How many protect-like moves where used in a row by this pokemon.
        :rtype: int
        """
        return self._protect_counter

    @property
    def revealed(self) -> bool:
        """
        :return: Whether this pokemon has appeared in the current battle.
        :rtype: bool
        """
        return self._revealed
    
    @property
    def sets(self) -> Dict:
        return self._sets

    @property
    def shiny(self) -> bool:
        """
        :return: Whether this pokemon is shiny.
        :rtype: bool
        """
        return bool(self._shiny)

    @property
    def species(self) -> str:
        """
        :return: The pokemon's species.
        :rtype: str | None
        """
        if self._species.lower() == 'keldeoresolute':
            return 'keldeo'
        elif self._species.lower() == 'sinistchamasterpiece':
            return 'sinistcha'
        return self._species

    @property
    def stats(self) -> Optional[Dict[str, Optional[int]]]:
        """
        :return: The pokemon's stats, as a dictionary.
        :rtype: Dict[str, int | None]
        """
        if self._last_request is not None:
            return self._last_request.get(
                "stats",
                {"atk": None, "def": None, "spa": None, "spd": None, "spe": None},
            )

    @property
    def status(self) -> Optional[Status]:
        """
        :return: The pokemon's status.
        :rtype: Optional[Status]
        """
        return self._status
    
    def calc_indiv_stat(self):
        
        return
    
    def guess_tera(self, guess_type='most_likely'):
        sets = self._sets
        if self.species.lower() not in sets:
            return ''

        if guess_type == 'most_likely':
            # most likely based on stats
            set = sets[self.species.lower()]['tera'][0]
            tera = set['name']
        
        else:
            # statistically weighted choice (copy paste lol)
            def get_weighted_choice(category, id, size=1):
                category_dict = sets[self.species.lower()][category]
                p = np.array([float(category_dict[i]['percentage'])/100. for i in range(len(category_dict))])
                p = p / p.sum()
                if size > len(category_dict):
                    size = len(category_dict)
                item = np.random.choice(category_dict, p=p, size=size, replace=False)
                if category == 'moves':
                    out = [item[i][id] if item[i][id] != 'Nothing' else '' for i in range(len(item))]
                else:
                    item = item[0]
                    out = item[id]
                if id == 'stats':
                    return out, item['nature']
                return out
            
            tera = get_weighted_choice('tera', 'name')
            
        return tera
        
    def guess_stats(self, guess_type='bayesian', observed_moves=None, battle=None):
        """
        Guess Pokemon stats using Bayesian predictions when possible.
        
        Args:
            guess_type: 'most_likely', 'bayesian', or original statistical methods
            observed_moves: List of observed moves to improve Bayesian predictions
            battle: Battle context for team information
        """
        # Try Bayesian predictions first if requested or if we have context
        if guess_type == 'bayesian' or (observed_moves and battle):
            bayesian_result = self._get_bayesian_stats(observed_moves, battle)
            if bayesian_result:
                return bayesian_result
            else:
                print('Bayesian stats failed')
        
        # Original implementation
        stat_types = ['hp', 'atk', 'def', 'spa', 'spd', 'spe']
        sets = self._sets
        if guess_type == 'most_likely':
            # most likely based on statistics
            set = sets[self.species.lower()]['spreads'][0]
            spread = set['stats']
            nature = set['nature']
        else:
            # statistically weighted choice
            def get_weighted_choice(category, id, size=1):
                category_dict = sets[self.species.lower()][category]
                p = np.array([float(category_dict[i]['percentage'])/100. for i in range(len(category_dict))])
                p = p / p.sum()
                if size > len(category_dict):
                    size = len(category_dict)
                item = np.random.choice(category_dict, p=p, size=size, replace=False)
                if category == 'moves':
                    out = [item[i][id] if item[i][id] != 'Nothing' else '' for i in range(len(item))]
                else:
                    item = item[0]
                    out = item[id]
                if id == 'stats':
                    return out, item['nature']
                return out
            
            spread, nature = get_weighted_choice('spreads', 'stats')
        
        return spread, nature
    
    def _get_bayesian_stats(self, observed_moves=None, battle=None):
        """Get Bayesian stat predictions for this Pokemon."""
        # Use singleton predictor to avoid loading multiple models
        from bayesian.predictor_singleton import get_pokemon_predictor
        predictor = get_pokemon_predictor()
        
        # Normalize Pokemon names
        def normalize_pokemon_name(name):
            name_mapping = {
                'slowkinggalar': 'Slowking-Galar', 'slowbrogalar': 'Slowbro-Galar',
                'tinglu': 'Ting-Lu', 'chiyu': 'Chi-Yu', 'wochien': 'Wo-Chien',
                'chienpao': 'Chien-Pao', 'ironmoth': 'Iron Moth', 'ironvaliant': 'Iron Valiant',
                'irontreads': 'Iron Treads', 'ironbundle': 'Iron Bundle', 'ironhands': 'Iron Hands',
                'ironjugulis': 'Iron Jugulis', 'ironthorns': 'Iron Thorns', 'ironboulder': 'Iron Boulder',
                'ironcrown': 'Iron Crown', 'greattusk': 'Great Tusk', 'screamtail': 'Scream Tail',
                'brutebonnet': 'Brute Bonnet', 'fluttermane': 'Flutter Mane', 'slitherwing': 'Slither Wing',
                'sandyshocks': 'Sandy Shocks', 'roaringmoon': 'Roaring Moon', 'walkingwake': 'Walking Wake',
                'ragingbolt': 'Raging Bolt', 'gougingfire': 'Gouging Fire', 'ogerponwellspring': 'Ogerpon-Wellspring',
                'ogerponhearthflame': 'Ogerpon-Hearthflame', 'ogerponcornerstone': 'Ogerpon-Cornerstone',
                'ogerponteal': 'Ogerpon', 'ogerpontealtera': 'Ogerpon', 'ursalunabloodmoon': 'Ursaluna',
                'ninetalesalola': 'Ninetales-Alola', 'sandslashalola': 'Sandslash-Alola',
                'tapukoko': 'Zapdos', 'tapulele': 'Clefable', 'tapubulu': 'Zapdos',
                'tapufini': 'Primarina', 'hydrapple': 'Hydrapple', 'zapdos': 'Zapdos',
                'zamazenta': 'Zamazenta', 'tinkaton': 'Tinkaton', 'hoopaunbound': 'Hoopa-Unbound',
                'mausholdfour': 'Maushold-Four', 'polteageistantique': 'Polteageist-Antique',
                'deoxysspeed': 'Deoxys-Speed', 'deoxysdefense': 'Deoxys-Defense', 'deoxysattack': 'Deoxys-Attack',
                'goodrahisui': 'Goodra-Hisui', 'kommoo': 'Kommo-o',
                'landorustherian': 'Landorus-Therian', 'moltresgalar': 'Moltres-Galar',
                'porygonz': 'Porygon-Z', 'rotomwash': 'Rotom-Wash', 'rotomheat': 'Rotom-Heat',
                'rotomfrost': 'Rotom-Frost', 'rotomfan': 'Rotom-Fan', 'rotommow': 'Rotom-Mow',
                'samurotthisui': 'Samurott-Hisui',
                'thundurustherian': 'Thundurus-Therian', 'tornadustherian': 'Tornadus-Therian',
                'weezinggalar': 'Weezing-Galar', 'zapdosgalar': 'Zapdos-Galar',
                'arcaninehisui': 'Arcanine-Hisui', 'braviaryhisui': 'Braviary-Hisui',
                'enamorustherian': 'Enamorus-Therian', 'lilliganthisui': 'Lilligant-Hisui',
                'sneaselhisui': 'Sneasel-Hisui', 'taurospaldeablaze': 'Tauros-Paldea-Blaze',
                'zarudedada': 'Zarude-Dada', 'zoroarkhisui': 'Zoroark-Hisui',
                'decidueyehisui': 'Decidueye-Hisui', 'mimikyubusted': 'Mimikyu',
                'miniormeteor': 'Minior', 'morpekohangry': 'Morpeko', 'eiscuenoice': 'Eiscue',
                'cramorantgulping': 'Cramorant', 'cramorantgorging': 'Cramorant',
                'sawsbucksummer': 'Sawsbuck', 'sawsbuckautumn': 'Sawsbuck', 'sawsbuckwinter': 'Sawsbuck',
                # Additional gen9ou Pokemon normalizations
                'basculegionf': 'Basculegion-F', 'basculegionm': 'Basculegion',
                'mukalola': 'Muk-Alola', 'raichualola': 'Raichu-Alola',
                'golemalaola': 'Golem-Alola', 'magnetonalola': 'Magnezone', 'dugtrioalola': 'Dugtrio-Alola',
                'grimerala': 'Grimer-Alola', 'marrowaka': 'Marowak-Alola', 'exeggutoralola': 'Exeggutor-Alola',
                'vulpixalola': 'Vulpix-Alola', 'persianalola': 'Persian-Alola', 'meowthala': 'Meowth-Alola',
                'rattataalola': 'Rattata-Alola', 'raticatealola': 'Raticate-Alola',
                'geodudealola': 'Geodude-Alola', 'graveleralola': 'Graveler-Alola',
                'magnemitealola': 'Magnemite', 'magnetonalola': 'Magneton',
                # More gender/form variants (map to common gen9ou Pokemon)
                'indeedf': 'Clefable', 'indeedeem': 'Indeedee',  # Female form maps to common psychic type
                'indeedeef': 'Clefable',  # Internal Pokemon constructor name for Indeedee-F
                'meowsticf': 'Meowstic', 'meowsticm': 'Meowstic',  # Female form maps to base Meowstic
                'unfezantf': 'Staraptor', 'unfezantm': 'Staraptor',  # Map to common Normal/Flying type
                # Paldean forms (map to common gen9ou Pokemon)
                'taurospaldeaaqua': 'Tauros-Paldea-Aqua', 'taurospaldeacombat': 'Tauros-Paldea-Aqua',
                'wooperpalda': 'Wooper-Paldea', 'clodsirepalda': 'Clodsire',
                # Hisuian forms that might be missed (map to known Hisuian forms or common Pokemon)
                'voltorbhisui': 'Electrode-Hisui', 'electrodehisui': 'Electrode-Hisui',  # Map to known Hisui form
                'typhloshionhisui': 'Typhlosion-Hisui', 'typhlosionhisui': 'Typhlosion-Hisui',  # Both variants
                'qwilfishhisui': 'Overqwil',  # Map to known evolution
                'growlithehisui': 'Arcanine-Hisui', 'sneaslerhisui': 'Sneasler',
                'overqwilhisui': 'Overqwil', 'kleavohisui': 'Kleavor',
                'basculinhisui': 'Basculegion', 'basculinwhitestriped': 'Basculegion',  # Map to known evolution
                # Additional common variants (map to common rock types)
                'lycanrocmidday': 'Lycanroc', 'lycanrocmidnight': 'Lycanroc-Dusk', 'lycanrocdusk': 'Lycanroc-Dusk',
                'oricoriomeadow': 'Kilowattrel', 'oricoriopompom': 'Kilowattrel',  # Map to common Electric/Flying
                'oricoriopau': 'Kilowattrel', 'oricoriosensu': 'Kilowattrel',  # All Oricorio forms to common similar type
                'toxapexgmax': 'Toxapex', 'corviknightgmax': 'Corviknight', 'grimmsnarrgmax': 'Grimmsnarl'
            }
            lower_name = name.lower()
            return name_mapping.get(lower_name, name.capitalize())
        
        # Normalize move names
        def normalize_move_name(move_id):
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
            import re
            spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', move_id)
            return spaced.title()
        
        # Get opponent team for context
        opponent_pokemon = []
        if battle and hasattr(battle, 'opponent_team'):
            for pokemon in battle.opponent_team.values():
                if pokemon and pokemon.species:
                    normalized_name = normalize_pokemon_name(pokemon.species)
                    opponent_pokemon.append(normalized_name)
        
        # Normalize observed moves
        normalized_moves = []
        if observed_moves:
            for move in observed_moves:
                if hasattr(move, 'id'):
                    normalized_moves.append(normalize_move_name(move.id))
                else:
                    normalized_moves.append(normalize_move_name(str(move)))
        
        # Get Bayesian predictions
        species_norm = normalize_pokemon_name(self.species)
        try:
            probabilities = predictor.predict_component_probabilities(
                species_norm, 
                teammates=opponent_pokemon,
                observed_moves=normalized_moves
            )
        except Exception as e:
            print(f'Bayesian prediction failed for {species_norm}: {e}')
            return None
        
        # Check if we got an error response
        if 'error' in probabilities:
            print(f'Bayesian prediction error for {species_norm}: {probabilities["error"]}')
            return None
        
        # Extract most likely nature and EV spread
        predicted_nature = None
        predicted_evs = None
        
        if 'natures' in probabilities and probabilities['natures']:
            predicted_nature = probabilities['natures'][0][0]  # Top nature
        else:
            print(f'No nature predictions available for {species_norm}')
        
        if 'ev_spreads' in probabilities and probabilities['ev_spreads']:
            ev_spread_str = probabilities['ev_spreads'][0][0]  # Top EV spread
            predicted_evs = self._parse_ev_spread_string(ev_spread_str)
            if not predicted_evs:
                print(f'Failed to parse EV spread for {species_norm}: {ev_spread_str}')
        else:
            print(f'No EV spread predictions available for {species_norm}')
        
        if predicted_nature and predicted_evs:
            return predicted_evs, predicted_nature
        
        print(f'Incomplete Bayesian prediction for {species_norm}: nature={predicted_nature}, evs={bool(predicted_evs)}')
        return None
    
    def _parse_ev_spread_string(self, ev_spread_str):
        """Parse EV spread string like '252 HP / 252 SpA' into EV list format."""
        # Default EV array [HP, Atk, Def, SpA, SpD, Spe]
        ev_array = [0, 0, 0, 0, 0, 0]
        
        # Map spread names to array indices
        stat_mapping = {
            'HP': 0, 'Atk': 1, 'Def': 2, 'SpA': 3, 'SpD': 4, 'Spe': 5
        }
        
        if not ev_spread_str or ev_spread_str == "No major investments":
            return ev_array
        
        # Parse "252 HP / 252 SpA / 4 SpD" format
        parts = ev_spread_str.split(' / ')
        for part in parts:
            try:
                value, stat = part.strip().split(' ', 1)
                if stat in stat_mapping:
                    ev_array[stat_mapping[stat]] = int(value)
            except:
                continue
                
        return ev_array
        # # more common natures ordered first
        # nature_common = [
        #     'Adamant',
        #     'Modest',
        #     'Jolly',
        #     'Timid',
        #     'Bold',
        #     'Brave',
        #     'Calm',
        #     'Careful',
        #     'Gentle',
        #     'Hasty',
        #     'Impish',
        #     'Lax',
        #     'Lonely',
        #     'Mild',
        #     'Naive',
        #     'Naughty',
        #     'Quiet',
        #     'Rash',
        #     'Relaxed',
        #     'Sassy',
        #     'Hardy',
        # ]
        
        # # brute force
        # for nature in nature_common:
        #     calculated_stats = {'hp': 0, 'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0}
        #     for stat in ['hp', 'atk', 'def', 'spa', 'spd', 'spe']:
        #         if ignore_hp:
        #             guessed_ivs['hp'] = 31
        #             guessed_evs['hp'] = 0
        #             continue
        #         for iv in range(31, -1, step=-1):
        #             for ev in range(0, 253, step=4):
        #                 calculated_stats[stat] = self.calc_indiv_stat(stat, iv, ev, nature)
                            
        return

    def calculate_stats(self, ivs=(31,) * 6, evs=(85,) * 6, battle_format='random'):
        nature = None
        if not 'random' in battle_format:
            # brute force the iv/ev
            evs, nature = self.guess_stats()
        def common_pkmn_stat_calc(stat: int, iv: int, ev: int, level: int):
            return math.floor(((2 * stat + iv + math.floor(ev / 4)) * level) / 100)

        new_stats = dict()
        new_stats['hp'] = common_pkmn_stat_calc(
                            self._base_stats['hp'],
                            ivs[0],
                            evs[0],
                            self._level
                        ) + self._level + 10

        new_stats['atk'] = common_pkmn_stat_calc(
                            self._base_stats['atk'],
                            ivs[1],
                            evs[1],
                            self._level
                        ) + 5

        new_stats['def'] = common_pkmn_stat_calc(
                            self._base_stats['def'],
                            ivs[2],
                            evs[2],
                            self._level
                        ) + 5

        new_stats['spa'] = common_pkmn_stat_calc(
                            self._base_stats['spa'],
                            ivs[3],
                            evs[3],
                            self._level
                        ) + 5

        new_stats['spd'] = common_pkmn_stat_calc(
                            self._base_stats['spd'],
                            ivs[4],
                            evs[4],
                            self._level
                        ) + 5

        new_stats['spe'] = common_pkmn_stat_calc(
                            self._base_stats['spe'],
                            ivs[5],
                            evs[5],
                            self._level
                        ) + 5
        if nature is not None:
            # first +, second -
            nature_boosts = {
                'Adamant': ['atk', 'spa'],
                'Bashful': [],
                'Bold': ['def', 'atk'],
                'Brave': ['atk', 'spe'],
                'Calm': ['spd', 'atk'],
                'Careful': ['spd', 'spa'],
                'Docile': [],
                'Gentle': ['spd', 'def'],
                'Hardy': [],
                'Hasty': ['spe', 'def'],
                'Impish': ['def', 'spa'],
                'Jolly': ['spe', 'spa'],
                'Lax': ['def', 'spd'],
                'Lonely': ['atk', 'def'],
                'Mild': ['spa', 'def'],
                'Modest': ['spa', 'atk'],
                'Naive': ['spe', 'spd'],
                'Naughty': ['atk', 'spd'],
                'Quiet': ['spa', 'spe'],
                'Quirky': [],
                'Rash': ['spa', 'spd'],
                'Relaxed': ['def', 'spe'],
                'Sassy': ['spd', 'spe'],
                'Serious': [],
                'Timid': ['spe', 'atk'],
            }
            nature_effect = nature_boosts[nature]
            # Only apply nature bonuses if the nature has an effect (not neutral)
            if len(nature_effect) == 2:
                buff, nerf = nature_effect
                new_stats[buff] = math.floor(1.1*new_stats[buff])
                new_stats[nerf] = math.floor(0.9*new_stats[nerf])
        
        new_stats = {k: int(v) for k, v in new_stats.items()}
        return new_stats

    @property
    def status_counter(self) -> int:
        """
        :return: The pokemon's status turn count. Only counts TOXIC and SLEEP statuses.
        :rtype: int
        """
        return self._status_counter

    @status.setter
    def status(self, status: Optional[Union[Status, str]]):
        self._status = Status[status.upper()] if isinstance(status, str) else status

    @property
    def stab_multiplier(self) -> float:
        """
        :return: The pokemon's STAB multiplier.
        :rtype: float
        """
        if self._terastallized and self._terastallized_type in (
            self._type_1,
            self._type_2,
        ):
            return 2
        return 1.5

    @property
    def terastallized(self) -> bool:
        """
        :return: Whether the pokemon is currently terastallized
        :rtype: bool
        """
        return self._terastallized

    @property
    def type_1(self) -> PokemonType:
        """
        :return: The pokemon's first type.
        :rtype: PokemonType
        """
        if self._terastallized and self._terastallized_type is not None:
            return self._terastallized_type
        return self._type_1

    @property
    def type_2(self) -> Optional[PokemonType]:
        """
        :return: The pokemon's second type.
        :rtype: Optional[PokemonType]
        """
        if self._terastallized:
            return None
        return self._type_2

    @property
    def types(self) -> Tuple[PokemonType, Optional[PokemonType]]:
        """
        :return: The pokemon's types, as a tuple.
        :rtype: Tuple[PokemonType, Optional[PokemonType]]
        """
        return self.type_1, self._type_2

    @property
    def weight(self) -> float:
        """
        :return: The pokemon's weight, in kilograms.
        :rtype: float
        """
        return self._weightkg
