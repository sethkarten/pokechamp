import json
import sys
from time import sleep
from typing import Callable, Dict, List
import numpy as np
from copy import deepcopy

import orjson

from poke_env.data.gen_data import GenData
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.environment.status import Status
from poke_env.player.battle_order import BattleOrder
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer

# Avoid circular import by importing here
try:
    from pokechamp.data_cache import get_cached_moves_set
    from pokechamp.sim_constants import get_simulation_optimizer, TYPE_LIST
except ImportError:
    # Fallback if optimization modules are not available
    get_cached_moves_set = None
    get_simulation_optimizer = None
    TYPE_LIST = 'BUG,DARK,DRAGON,ELECTRIC,FAIRY,FIGHTING,FIRE,FLYING,GHOST,GRASS,GROUND,ICE,NORMAL,POISON,PSYCHIC,ROCK,STEEL,WATER'.split(",")

DEBUG = False

def calculate_move_type_damage_multipier(type_1, type_2, type_chart, constraint_type_list):
    TYPE_list = TYPE_LIST  # Use cached constant instead of recreating

    move_type_damage_multiplier_list = []

    if type_2:
        for type in TYPE_list:
            if 'STELLAR' not in [type, type_1, type_2]:
                move_type_damage_multiplier_list.append(type_chart[type_1][type] * type_chart[type_2][type])
            else: 
                move_type_damage_multiplier_list.append(1)
        move_type_damage_multiplier_dict = dict(zip(TYPE_list, move_type_damage_multiplier_list))
    else:
        if type_1 == 'STELLAR':
            move_type_damage_multiplier_dict = {'BUG': 1, 'DARK': 1, 'DRAGON': 1, 'ELECTRIC': 1, 'FAIRY': 1, 'FIGHTING': 1, 'FIRE': 1, 'FLYING': 1, 'GHOST': 1, 'GRASS': 1, 'GROUND': 1, 'ICE': 1, 'NORMAL': 1, 'POISON': 1, 'PSYCHIC': 1, 'ROCK': 1, 'STEEL': 1, 'WATER': 1}
        else:
            move_type_damage_multiplier_dict = type_chart[type_1]

    effective_type_list = []
    extreme_type_list = []
    resistant_type_list = []
    extreme_resistant_type_list = []
    immune_type_list = []
    for type, value in move_type_damage_multiplier_dict.items():
        if value == 2:
            effective_type_list.append(type)
        elif value == 4:
            extreme_type_list.append(type)
        elif value == 1 / 2:
            resistant_type_list.append(type)
        elif value == 1 / 4:
            extreme_resistant_type_list.append(type)
        elif value == 0:
            immune_type_list.append(type)
        else:  # value == 1
            continue

    if constraint_type_list:
        extreme_type_list = list(set(extreme_type_list).intersection(set(constraint_type_list)))
        effective_type_list = list(set(effective_type_list).intersection(set(constraint_type_list)))
        resistant_type_list = list(set(resistant_type_list).intersection(set(constraint_type_list)))
        extreme_resistant_type_list = list(set(extreme_resistant_type_list).intersection(set(constraint_type_list)))
        immune_type_list = list(set(immune_type_list).intersection(set(constraint_type_list)))

    return (list(map(lambda x: x.capitalize(), extreme_type_list)),
           list(map(lambda x: x.capitalize(), effective_type_list)),
           list(map(lambda x: x.capitalize(), resistant_type_list)),
           list(map(lambda x: x.capitalize(), extreme_resistant_type_list)),
           list(map(lambda x: x.capitalize(), immune_type_list)))

def move_type_damage_wrapper(pokemon, type_chart, constraint_type_list=None):
    if pokemon is None:
        return ""
    type_1 = None
    type_2 = None
    if pokemon.type_1:
        type_1 = pokemon.type_1.name
        if pokemon.type_2:
            type_2 = pokemon.type_2.name

    move_type_damage_prompt = ""
    extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
        type_1, type_2, type_chart, constraint_type_list)

    move_type_damage_prompt = ""
    if extreme_effective_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(extreme_effective_type_list) +
                                   f"-type attack is extremely-effective (4x damage) to {pokemon.species}.")

    if effective_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(effective_type_list) +
                                   f"-type attack is super-effective (2x damage) to {pokemon.species}.")

    if resistant_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(resistant_type_list) +
                                   f"-type attack is ineffective (0.5x damage) to {pokemon.species}.")

    if extreme_resistant_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(extreme_resistant_type_list) +
                                   f"-type attack is highly ineffective (0.25x damage) to {pokemon.species}.")

    if immune_type_list:
        move_type_damage_prompt = (move_type_damage_prompt + " " + ", ".join(immune_type_list) +
                                   f"-type attack is zero effect (0x damage) to {pokemon.species}.")

    return move_type_damage_prompt

class LocalSim():
    def __init__(self, 
                 battle: Battle,
                 move_effect: Dict,
                 pokemon_move_dict: Dict,
                 ability_effect: Dict,
                 pokemon_ability_dict: Dict,
                 item_effect: Dict,
                 pokemon_item_dict: Dict,
                 gen: GenData,
                 _dynamax_disable: bool,
                _strategy: str='',
                format: str='gen9randombattle',
                prompt_translate: Callable=None
        ):
        self.battle = deepcopy(battle)
        self.move_effect = move_effect
        self.pokemon_move_dict = pokemon_move_dict
        self.ability_effect = ability_effect
        self.pokemon_ability_dict = pokemon_ability_dict
        self.item_effect = item_effect
        # unused
        # self.pokemon_item_dict = pokemon_item_dict
        self.gen = gen
        self._dynamax_disable = _dynamax_disable
        self._tera_disable = False # TODO: working on tera
        self.strategy = _strategy
        self.format = format
        self.prompt_translate = prompt_translate

        self.switch_set = set()

        self.SPEED_TIER_COEFICIENT = 0.1
        self.HP_FRACTION_COEFICIENT = 0.4
        
        # Use cached moves set data instead of loading file
        if get_cached_moves_set is not None:
            self.moves_set = get_cached_moves_set(self.format)
        else:
            # Fallback to file loading if cache is not available
            if self.format == 'gen9ou':
                file = f'poke_env/data/static/gen9/ou/sets_1000.json'
                with open(file, 'r') as f:
                    self.moves_set = orjson.loads(f.read())
            else:
                self.moves_set = {}


    def get_llm_system_prompt(self, _format: str, llm: GPTPlayer | LLAMAPlayer = None, team_str: str=None, model: str='gpt-4o'):
        # sleep to make sure server has sent pokemon team information first
        # llm = GPTPlayer(api_key=KEY)
        if 'random' in _format:
            if llm is not None:
                sleep(1)
                strategy_prompt = f""
                for poke_str in self.battle.team.keys():
                    mon = self.battle.team[poke_str]
                    strategy_prompt += f"{mon.species}"
                    try:
                        if mon.item: strategy_prompt += f" @ {self.item_effect[mon.item]['name']}"
                    except:
                        pass
                    strategy_prompt += '\n'
                    if mon.ability: strategy_prompt += f"Ability: {mon.ability}\n"
                    # if self.gen.gen == 9: 
                    #     if mon._terastallized_type.name: f"Tera Type: {mon._terastallized_type.name}\n"
                    strategy_prompt += "EVs: 85 HP/ 85 Atk / 85 Def / 85 SpA / 85 SpD / 85 Spe\n"
                    strategy_prompt += f"Bashful Nature\n"
                    for move in mon.moves.values():
                        strategy_prompt += f'- {move.id}\n'
                    strategy_prompt += '\n'

                # print(strategy_prompt)
                strategy_prompt += "\nI play competitive Pokémon battles. How do I play this teams effectively?"
                self.strategy, _ = llm.get_LLM_query("", strategy_prompt, max_tokens=1000, model=model)
                # print()
                # print(self.strategy)
        elif team_str != None and llm is not None:
            # Read OU team
            # with open(f"poke_env/data/static/teams/gen{self.gen.gen}ou{team_str}.txt","r") as f:
            strategy_prompt = team_str
            strategy_prompt += "\n\nI play competitive Pokémon battles. How do I play this team effectively?"
            self.strategy, _ = llm.get_LLM_query("", strategy_prompt, max_tokens=1000, model=model)
            # print(self.strategy)
        return self.strategy
    
    def step_llm(self, action1, action2, llm, model, m1, m2, temperature=0.7, max_tokens=200):
        # get active pokemon stats
        # opponent pokemon
        opponent_type = ""
        opponent_type_list = []
        if self.battle.opponent_active_pokemon.type_1:
            type_1 = self.battle.opponent_active_pokemon.type_1.name
            opponent_type += type_1.capitalize()
            opponent_type_list.append(type_1)

            if self.battle.opponent_active_pokemon.type_2:
                type_2 = self.battle.opponent_active_pokemon.type_2.name
                opponent_type = opponent_type + " and " + type_2.capitalize()
                opponent_type_list.append(type_2)
        species = self.battle.opponent_active_pokemon.species
        if species == 'polteageistantique':
            species = 'polteageist'
        if self.battle.opponent_active_pokemon.ability:
            opponent_ability = self.battle.opponent_active_pokemon.ability
        elif species in self.pokemon_ability_dict:
            opponent_ability = self.pokemon_ability_dict[species][0]
        else:
            opponent_ability = ""

        if opponent_ability:
            try:
                ability_name = self.ability_effect[opponent_ability]["name"]
                ability_effect = self.ability_effect[opponent_ability]["effect"]
                opponent_ability = f"{ability_name}({ability_effect})"
            except:
                pass
        opponent_hp_fraction = round(self.battle.opponent_active_pokemon.current_hp / self.battle.opponent_active_pokemon.max_hp * 100)
        opponent_stats = self.battle.opponent_active_pokemon.calculate_stats(battle_format=self.format)
        opponent_boosts = self.battle.opponent_active_pokemon._boosts
        active_stats = self.battle.active_pokemon.stats
        active_boosts = self.battle.active_pokemon._boosts
        opponent_status = self.battle.opponent_active_pokemon.status
        opponent_is_dynamax = self.battle.opponent_active_pokemon.is_dynamaxed
        opponent_prompt = (
                f"Opposing pokemon:{self.battle.opponent_active_pokemon.species},Type:{opponent_type},HP:{opponent_hp_fraction}%,Is dynamax:{opponent_is_dynamax}," +
                (f"Status:{self.check_status(opponent_status)}," if self.check_status(opponent_status) else "") +
                (f"Attack:{opponent_stats['atk']}," if opponent_boosts['atk']==0 else f"Attack:{round(opponent_stats['atk'] * self.boost_multiplier('atk', opponent_boosts['atk']))}({opponent_boosts['atk']} stage boosted),") +
                (f"Defense:{opponent_stats['def']}," if opponent_boosts['def']==0 else f"Defense:{round(opponent_stats['def'] * self.boost_multiplier('def', opponent_boosts['def']))}({opponent_boosts['def']} stage boosted),") +
                (f"Special attack:{opponent_stats['spa']}," if opponent_boosts['spa']==0 else f"Special attack:{round(opponent_stats['spa'] * self.boost_multiplier('spa', opponent_boosts['spa']))}({opponent_boosts['spa']} stage boosted),") +
                (f"Special defense:{opponent_stats['spd']}," if opponent_boosts['spd']==0 else f"Special defense:{round(opponent_stats['spd'] * self.boost_multiplier('spd', opponent_boosts['spd']))}({opponent_boosts['spd']} stage boosted),") +
                (f"Speed:{opponent_stats['spe']}," if opponent_boosts['spe'] == 0 else f"Speed:{round(opponent_stats['spe'] * self.boost_multiplier('spe', opponent_boosts['spe']))}({opponent_boosts['spe']} stage boosted),") +
                (f"Ability:{opponent_ability}" if opponent_ability else "")
        )
        opponent_speed = round(opponent_stats['spe'] * self.boost_multiplier('spe', opponent_boosts['spe']))
        # player pokemon
        active_hp_fraction = round(self.battle.active_pokemon.current_hp / self.battle.active_pokemon.max_hp * 100)
        active_status = self.battle.active_pokemon.status

        active_type = ""
        if self.battle.active_pokemon.type_1:
            active_type += self.battle.active_pokemon.type_1.name.capitalize()
            if self.battle.active_pokemon.type_2:
                active_type = active_type + " and " + self.battle.active_pokemon.type_2.name.capitalize()

        active_move_type_damage_prompt = move_type_damage_wrapper(self.battle.active_pokemon, self.gen.type_chart, opponent_type_list)
        speed_active_stats = active_stats['spe']
        if speed_active_stats == None: speed_active_stats = 0
        active_speed = round(speed_active_stats*self.boost_multiplier('spe', active_boosts['spe']))

        try:
            active_ability = self.ability_effect[self.battle.active_pokemon.ability]["name"]
            ability_effect = self.ability_effect[self.battle.active_pokemon.ability]["effect"]
        except:
            active_ability = self.battle.active_pokemon.ability
            ability_effect = ""

        # item
        if self.battle.active_pokemon.item:
            try:
                active_item = self.item_effect[self.battle.active_pokemon.item]["name"]
                item_effect = self.item_effect[self.battle.active_pokemon.item]["effect"]
                active_item = f"{active_item}({item_effect})"
            except:
                active_item = self.battle.active_pokemon.item
        else:
            active_item = ""
        active_pokemon_prompt = (
            f"Your current pokemon:{self.battle.active_pokemon.species},Type:{active_type},HP:{active_hp_fraction}%," +
            (f"Status:{self.check_status(active_status)}," if self.check_status(active_status) else "" ) +
            (f"Attack:{active_stats['atk']}," if active_boosts['atk']==0 else f"Attack:{round(active_stats['atk']*self.boost_multiplier('atk', active_boosts['atk']))}({active_boosts['atk']} stage boosted),") +
            (f"Defense:{active_stats['def']}," if active_boosts['def']==0 else f"Defense:{round(active_stats['def']*self.boost_multiplier('def', active_boosts['def']))}({active_boosts['def']} stage boosted),") +
            (f"Special attack:{active_stats['spa']}," if active_boosts['spa']==0 else f"Special attack:{round(active_stats['spa']*self.boost_multiplier('spa', active_boosts['spa']))}({active_boosts['spa']} stage boosted),") +
            (f"Special defense:{active_stats['spd']}," if active_boosts['spd']==0 else f"Special defense:{round(active_stats['spd']*self.boost_multiplier('spd', active_boosts['spd']))}({active_boosts['spd']} stage boosted),") +
            (f"Speed:{active_stats['spe']}" if active_boosts['spe']==0 else f"Speed:{round(active_stats['spe']*self.boost_multiplier('spe', active_boosts['spe']))}({active_boosts['spe']} stage boosted),") +
            (f"(slower than {self.battle.opponent_active_pokemon.species})." if active_speed < opponent_speed else f"(faster than {self.battle.opponent_active_pokemon.species}).") +
            (f"Ability:{active_ability}({ability_effect})," if ability_effect else f"Ability:{active_ability},") +
            (f"Item:{active_item}" if active_item else "")
        )
        # moves + history
        n_turn = 5
        if "p1" in list(self.battle.team.keys())[0]:
            context_prompt = (f"Historical turns:\n" + "\n".join(
                self.battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).
                                          replace("p1a: ", "player:").
                                          replace("p2a:","opponent:").
                                          replace("Player1", "Player").
                                          replace("Player2", "Opponent"))
        else:
            context_prompt = (f"Historical turns:\n" + "\n".join(
                self.battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).
                                          replace("p1a: ", "player:").
                                          replace("p2a:","opponent:").
                                          replace("Player1", "Player").
                                          replace("Player2", "Opponent"))
        move_prompt = f'Player used {action1}.\nOpponent used {action2}.\n'
        # json response 
        json_action = 'Output the remaining player and opponent pokemon health remaining after their actions\'. \
            Each pokemon\'s health should be a percentage between 0 and 100. \
            Your output MUST be a JSON like where <pokemon_health> is an integer: {"player":"<pokemon_health>", "opponent":"<pokemon_health>"}\n'
        # calculate damage based on actions
        system_prompt = 'You are an expert pokemon battle simulator. \
            Your job is to accurately calculate the remaining health of pokemon based on their stats and the current moves/switches performed. \
            Consider the type advantages, stats, abilities, and items in addition to the moves/switches performed in order to accurately calculate the damage.'
        user_prompt = context_prompt + opponent_prompt + '\n' + active_pokemon_prompt + '\n' + move_prompt + json_action

        # update state based on actions, damage via string updates
        # return relevant state parameters as request to battle object
        msg_all = []
        msg = ['', ]
        # p1a is opponent, p2a is player
        player_tag = 'p2a'
        opponent_tag = 'p1a'
        if "p1" in list(self.battle.team.keys())[0]:
            # p1a is player, p2a is opponent
            player_tag = 'p1a'
            opponent_tag = 'p2a'
        if DEBUG: print(f'PLAYER {player_tag}')
        if DEBUG: print(f'OPPONENT {opponent_tag}')
        action1_name = action1.split(' ')[-1].title()
        if DEBUG: print('action 1 name', action1_name)
        action2_name = action2.split(' ')[-1].title()
        if 'switch' in action1:
            # get switched pokemon health
            player_health = 100
            for avail_mon in self.battle.available_switches:
                if avail_mon.species == action1_name:
                    player_health = int(avail_mon.current_hp_fraction * 100)
                    break
            msg = ['', 'switch', f'{player_tag}: {action1_name}', '', f'{player_health}/100']
            msg_all.append(msg)
        if 'switch' in action2:
            # get switched pokemon health
            opponent_health = 100
            for avail_mon in self.battle.opponent_team.values():
                if avail_mon.species == action2_name:
                    opponent_health = int(avail_mon.current_hp_fraction * 100)
                    break
            msg = ['', 'switch', f'{opponent_tag}: {action2_name}', '', f'{opponent_health}/100']
            msg_all.append(msg)
        for request in msg_all:
            # print("[local sim msg]", request)
            self._handle_battle_message(request)


        '''ADVANCE WORLD'''
        move1 = m1 if isinstance(m1, Move) else None
        move2 = m2 if isinstance(m2, Move) else None
        if DEBUG: print(self.battle.active_pokemon, self.battle.opponent_active_pokemon)
        if DEBUG: 
            for avail_mon in self.battle.opponent_team.values():
                print('opp', avail_mon)
        assert self.battle.opponent_active_pokemon != None
        player_health, opponent_health, m1_success, m2_success = self.calculate_remaining_hp(
                                                                     self.battle.active_pokemon,
                                                                     self.battle.opponent_active_pokemon,
                                                                     move1,
                                                                     move2,
                                                                     team=self.battle.team,
                                                                     opp_team=self.battle.opponent_team,
                                                                    )

        msg_all = []
        msg = ['', ]
        if 'move' in action1:
            msg = ['', 'move', f'{player_tag}: {self.battle.active_pokemon.species.title()}', f'{action1_name}', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}']
            msg_all.append(msg)
            if int(opponent_health) == 0:
                msg = ['', '-damage', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', '0 fnt']
            else:
                msg = ['', '-damage', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', f'{opponent_health}/100']
            msg_all.append(msg)
        if 'move' in action2:
            msg = ['', 'move', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', f'{action2_name}', f'{player_tag}: {self.battle.active_pokemon.species.title()}']
            msg_all.append(msg)
            if int(player_health) == 0:
                msg = ['', '-damage', f'{player_tag}: {self.battle.active_pokemon.species.title()}', '0 fnt']
            else:
                msg = ['', '-damage', f'{player_tag}: {self.battle.active_pokemon.species.title()}', f'{player_health}/100']
            msg_all.append(msg)
        if m1_success:
            if move1.status != None:
                msg = ['', '-status', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', f'{move1.status}']
        if m2_success:
            if move2.status != None:
                msg = ['', '-status', f'{player_tag}: {self.battle.active_pokemon.species.title()}', f'{move1.status}']
        # check for stat changes
        
        for request in msg_all:
            if DEBUG: print("[local sim msg]", request)
            self._handle_battle_message(request)

        # process end of turn
        return
    
    def step(self, action1: BattleOrder, action2: BattleOrder):
        # print(action1, action2)
        m1 = action1.order
        action1 = action1.message
        m2 = None
        if action2 is not None:
            m2 = action2.order
            action2 = action2.message
        # update state based on actions, damage via string updates
        # return relevant state parameters as request to battle object
        msg_all = []
        msg = ['', ]
        # p1a is player, p2a is opponent
        player_tag = 'p1a'
        opponent_tag = 'p2a'
        if "p2" in list(self.battle.team.keys())[0]:
            # p1a is player, p2a is opponent
            player_tag = 'p2a'
            opponent_tag = 'p1a'
        if DEBUG: print(f'PLAYER {player_tag}')
        if DEBUG: print(f'OPPONENT {opponent_tag}')
        action1_name = action1.split(' ')[-1].title()
        if DEBUG: print('action 1 name', action1_name)
        if action2 is not None:
            action2_name = action2.split(' ')[-1].title()
        if 'switch' in action1 and not 'move' in action1:
            # get switched pokemon health
            player_health = 100
            for avail_mon in self.battle.available_switches:
                if avail_mon.species == action1_name.lower().replace(' ','').replace('-',''):
                    player_health = int(avail_mon.current_hp_fraction * 100)
                    # print('player', avail_mon.current_hp, avail_mon.max_hp)
                    break
            msg = ['', 'switch', f'{player_tag}: {action1_name}', '', f'{player_health}/100']
            msg_all.append(msg)
        if action2 is not None:
            if 'switch' in action2 and not 'move' in action2:
                # get switched pokemon health
                opponent_health = 100
                for avail_mon in self.battle.opponent_team.values():
                    if avail_mon.species == action2_name.lower().replace(' ','').replace('-',''):
                        opponent_health = int(avail_mon.current_hp_fraction * 100)
                        # print('opp', avail_mon.current_hp, avail_mon.max_hp)
                        break
                msg = ['', 'switch', f'{opponent_tag}: {action2_name}', '', f'{opponent_health}/100']
                msg_all.append(msg)
        for request in msg_all:
            # print("[local sim msg]", request)
            self._handle_battle_message(request)
        # sleep(0.05)


        '''ADVANCE WORLD'''
        move1 = m1 if isinstance(m1, Move) else None
        move2 = m2 if isinstance(m2, Move) else None
        if DEBUG: print(self.battle.active_pokemon, self.battle.opponent_active_pokemon)
        if DEBUG: 
            for avail_mon in self.battle.opponent_team.values():
                print('opp', avail_mon)
        assert self.battle.opponent_active_pokemon != None
        player_health, opponent_health, m1_success, m2_success = self.calculate_remaining_hp(
                                                                     self.battle.active_pokemon,
                                                                     self.battle.opponent_active_pokemon,
                                                                     move1,
                                                                     move2,
                                                                     team=self.battle.team,
                                                                     opp_team=self.battle.opponent_team,
                                                                    )
        # print(m1_success, m2_success, action1, action2)
        # print(player_health, opponent_health)
        msg_all = []
        msg = ['', ]
        if action1 is not None:
            if 'move' in action1:
                msg = ['', 'move', f'{player_tag}: {self.battle.active_pokemon.species.title()}', f'{action1_name}', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}']
                msg_all.append(msg)
                if int(opponent_health) == 0:
                    msg = ['', '-damage', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', '0 fnt']
                else:
                    msg = ['', '-damage', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', f'{opponent_health}/100']
                msg_all.append(msg)
                pokemon, hp_status = msg[2:4]
                self.battle.get_pokemon(pokemon).damage(hp_status)
        if action2 is not None:
            if 'move' in action2:
                msg = ['', 'move', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', f'{action2_name}', f'{player_tag}: {self.battle.active_pokemon.species.title()}']
                msg_all.append(msg)
                if int(player_health) == 0:
                    msg = ['', '-damage', f'{player_tag}: {self.battle.active_pokemon.species.title()}', '0 fnt']
                else:
                    msg = ['', '-damage', f'{player_tag}: {self.battle.active_pokemon.species.title()}', f'{player_health}/100']
                msg_all.append(msg)
                pokemon, hp_status = msg[2:4]
                self.battle.get_pokemon(pokemon).damage(hp_status)
        if m1_success:
            if move1.status != None:
                msg = ['', '-status', f'{opponent_tag}: {self.battle.opponent_active_pokemon.species.title()}', f'{move1.status}']
        if m2_success:
            if move2.status != None:
                msg = ['', '-status', f'{player_tag}: {self.battle.active_pokemon.species.title()}', f'{move2.status}']
        # check for stat changes
        
        for request in msg_all:
            # print("[local sim msg]", request)
            self._handle_battle_message(request)
        # sleep(0.05)

        # process end of turn
        return
    
    def get_hp_diff(self):
        # calculate expected hp difference between p1 and p2
        hp_diff = 0.

        for mon in self.battle.team.values():
            hp_diff += mon.current_hp_fraction

        for mon in self.battle.opponent_team.values():
            hp_diff -= mon.current_hp_fraction
        remaining_pokemon = 6 - len(self.battle.opponent_team.values())
        hp_diff -= remaining_pokemon

        return hp_diff 
    
    def get_all_hp(self):
        hps = []
        
        hps.append(self.battle.active_pokemon.current_hp_fraction)
        for mon in self.battle.team.values():
            if mon.species != self.battle.active_pokemon.species:
                hps.append(mon.current_hp_fraction)

        hps.append(self.battle.opponent_active_pokemon.current_hp_fraction)
        for mon in self.battle.opponent_team.values():
            if mon.species != self.battle.opponent_active_pokemon.species:
                hps.append(mon.current_hp_fraction)
            
        hps_str = str(hps)
        return hps_str
    
    def get_player_prompt(self, return_actions=False, return_choices=False):
        if return_actions:
            system_prompt, state_prompt, state_action_prompt, action_prompt_switch, action_prompt_move = self.prompt_translate(self, self.battle, return_actions=return_actions) # add lower case
        elif return_choices:
            system_prompt, state_prompt, state_action_prompt, action_choice_switch, action_choice_move = self.prompt_translate(self, self.battle, return_choices=return_choices) # add lower case
        else:
            system_prompt, state_prompt, state_action_prompt = self.prompt_translate(self, self.battle) # add lower case

        if self.battle.active_pokemon.fainted or len(self.battle.available_moves) == 0:

            constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
        elif len(self.battle.available_switches) == 0:
            constraint_prompt_io = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>"}\n'''
        else:
            constraint_prompt_io = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
        if return_actions:
            return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move
        elif return_choices:
            return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_choice_switch, action_choice_move
        return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt

    def get_opponent_possible_mons(self):
        observable_switches = []
        opponent_fainted_num = 0
        for _, opponent_pokemon in self.battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1
            elif not opponent_pokemon.active:
                # print(opponent_pokemon)
                observable_switches.append(opponent_pokemon.species)
        opponent_unfainted_num = 6 - opponent_fainted_num
        
        # get definite moves for the current pokemon
        opponent_moves = []
        if self.battle.opponent_active_pokemon.moves:
            for move_id, opponent_move in self.battle.opponent_active_pokemon.moves.items():
                opponent_moves.append(f'{opponent_move}')
                
        
    def get_opponent_current_moves(self, mon=None, return_switch=False, is_player=False, return_separate=False):
        if is_player:
            return list(self.battle.active_pokemon.moves.keys())
        if mon == None: #fainted pokemon 
            mon = self.battle.opponent_active_pokemon
        # get definite moves for the current pokemon
        opponent_moves = []
        if mon.moves:
            for move_id, opponent_move in mon.moves.items():
                opponent_moves.append(f'{opponent_move.id}')

        # Try Bayesian predictions first for gen9ou - includes both confirmed and predicted moves
        bayesian_result = []
        if self.format == 'gen9ou':
            try:
                bayesian_result = self._get_bayesian_move_predictions(mon)
                # If we got good Bayesian predictions, use them directly
                if bayesian_result and len(bayesian_result) >= len(opponent_moves):
                    if return_separate:
                        return opponent_moves, bayesian_result[len(opponent_moves):]  # Separate confirmed vs predicted
                    return bayesian_result[:4]  # Return top 4 Bayesian moves (includes 100% confirmed moves)
            except:
                pass  # Fall back to original method

        # Fallback: get possible moves for current pokemon (original method)
        species = mon.species
        possible_moves = []
        if self.format == 'gen9ou':
            try:
                possible_moves = [move_set['name'].lower().replace(' ', '').replace('-','') for move_set in self.moves_set[species]['moves']]
            except:
                if species in self.pokemon_move_dict:
                    possible_moves = [move[0] for move in self.pokemon_move_dict[species].values()]
        else:
            if species in self.pokemon_move_dict:
                possible_moves = [move[0] for move in self.pokemon_move_dict[species].values()]
                # possible_moves = self.pokemon_move_dict[species].keys()
        
        if return_separate:
            return opponent_moves, possible_moves
        
        # Combine confirmed + fallback moves (original logic)
        all_moves = opponent_moves[:]
        while len(all_moves) != 4 and len(possible_moves) > 0:
            move_unseen = possible_moves.pop(0)
            if move_unseen not in all_moves:
                all_moves.append(move_unseen)

        # need to create order after return
        return all_moves

    def _get_bayesian_move_predictions(self, mon):
        """Get Bayesian move predictions for opponent Pokemon."""
        try:
            # Get the singleton predictor from the battle class
            predictor = self.get_pokemon_predictor()
            
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
                    'ogerpontealtera': 'Ogerpon-Teal', 'ursalunabloodmoon': 'Ursaluna-Bloodmoon',
                    'ninetalesalola': 'Ninetales-Alola', 'sandslashalola': 'Sandslash-Alola',
                    'tapukoko': 'Tapu Koko', 'tapulele': 'Tapu Lele', 'tapubulu': 'Tapu Bulu',
                    'tapufini': 'Tapu Fini', 'hydrapple': 'Hydrapple', 'zapdos': 'Zapdos',
                    'zamazenta': 'Zamazenta', 'tinkaton': 'Tinkaton'
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
                    'voltswitch': 'Volt Switch', 'discharge': 'Discharge', 'uturn': 'U-turn'
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
            for pokemon in self.battle.opponent_team.values():
                if pokemon and pokemon.species:
                    normalized_name = normalize_pokemon_name(pokemon.species)
                    opponent_pokemon.append(normalized_name)
            
            # Get observed moves for this Pokemon
            observed_moves = []
            for move in mon.moves.values():
                if move:
                    normalized_move = normalize_move_name(move.id)
                    observed_moves.append(normalized_move)
            
            # Get Bayesian predictions
            species_norm = normalize_pokemon_name(mon.species)
            probabilities = predictor.predict_component_probabilities(
                species_norm, 
                teammates=opponent_pokemon,
                observed_moves=observed_moves
            )
            
            # Extract ALL moves with their probabilities (confirmed + predicted)
            all_moves_with_probs = []
            if 'moves' in probabilities:
                for move_name, prob in probabilities['moves']:
                    # Convert back to battle format (lowercase, no spaces)
                    battle_format_move = move_name.lower().replace(' ', '').replace('-', '')
                    all_moves_with_probs.append((battle_format_move, prob))
            
            # Sort by probability (confirmed 100% moves first, then by descending probability)
            all_moves_with_probs.sort(key=lambda x: x[1], reverse=True)
            
            # Return just the move names (top 4)
            return [move for move, prob in all_moves_with_probs[:4]]
            
        except Exception as e:
            # Silently fall back to original method
            return []
    
    def get_turn_summary(self,
                        battle: Battle,
                        n_turn: int=5
                        ) -> str:
        if "p1" in list(battle.team.keys())[0]:
            context_prompt = (f"Historical turns:\n" + "\n".join(
                battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).
                                            replace("p1a: ", "").
                                            replace("p2a:","opposing").
                                            replace("Player1", "You").
                                            replace("Player2", "Opponent"))
        else:
            context_prompt = (f"Historical turns:\n" + "\n".join(
                battle.battle_msg_history.split("[sep]")[-1 * (n_turn + 1):]).
                                replace("p2a: ", "").
                                replace("p1a:", "opposing").
                                replace("Player2", "You").
                                replace("Player1", "Opponent"))
        
        battle_prompt = context_prompt + " Current battle state:\n"
        return battle_prompt

    
    def get_opponent_prompt(self, state_prompt, return_actions=False):
        # _, state_prompt, _ = self.state_translate(self.battle) # add lower case

        
        system_prompt = (
                "You are a pokemon battler that targets to win the pokemon battle by predicting the action that the opposing battler will use. Your opponent can choose to take a move or switch in another pokemon. Here are some battle tips:"
                " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
                " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
                " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
                " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
                " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
                )
        system_prompt += ' Output the action the opposing battler will use. \
            You may not be able to observe all possible moves and pokemon that the opposing battler will use so you will need to both select the action based on their move history \
            and any possible unseen moves/pokemon. '
        state_switch_prompt = ''
        
        battle_prompt = self.get_turn_summary(self.battle, n_turn=16)

        # get viewable opponent pokemon
        # add prompt about unknowable pokemon
        observable_switches = []
        opponent_fainted_num = 0
        for _, opponent_pokemon in self.battle.opponent_team.items():
            if opponent_pokemon.fainted:
                opponent_fainted_num += 1
            elif not opponent_pokemon.active:
                # print(opponent_pokemon)
                observable_switches.append(opponent_pokemon.species)
        opponent_unfainted_num = 6 - opponent_fainted_num
        swich_prompt = ''
        if opponent_unfainted_num > 1:
            state_switch_prompt += f'The opponent may switch to any of their remaining {opponent_unfainted_num} pokemon.\n'
        if len(observable_switches) > 0:
            swich_prompt += f'[<opponent_switch_pokemon_name>] = {observable_switches}\n'
        
        
        # get definite moves for the current pokemon
        opponent_moves = self.get_opponent_current_moves()
        action_prompt = f' Opponent\'s current Pokemon: {self.battle.opponent_active_pokemon.species}.\nChoose only from the following opponent action choices:\n'
        move_prompt = ''
        if len(opponent_moves) > 0:
            move_prompt += f"[<opponent_move_name>] = {opponent_moves}\n"

        if self.battle.opponent_active_pokemon.fainted:
            constraint_prompt_io = '''Choose the most suitable opponent pokemon to switch. Your output MUST be a JSON like: {"switch":"<opponent_switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable opponent pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<opponent_step-by-step-thinking>", "switch":"<opponent_switch_pokemon_name>"}\n'''
        elif opponent_unfainted_num == 1:
            constraint_prompt_io = '''Choose the best opponent action and your output MUST be a JSON like: {"move":"<opponent_move_name>"}\n'''
            constraint_prompt_cot = '''Choose the best opponent action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<opponent_step-by-step-thinking>", "move":"<opponent_move_name>"} or {"thought":"<opponent_step-by-step-thinking>"}\n'''
        else:
            constraint_prompt_io = '''Choose the best opponent action and your output MUST be a JSON like: {"move":"<opponent_move_name>"} or {"switch":"<opponent_switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the best opponent action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<opponent_step-by-step-thinking>", "move":"<opponent_move_name>"} or {"thought":"<opponent_step-by-step-thinking>", "switch":"<opponent_switch_pokemon_name>"}\n'''


        state_action_prompt = battle_prompt + action_prompt + move_prompt + state_switch_prompt + swich_prompt
        if return_actions:
            return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, move_prompt, swich_prompt
        return system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt
    
    def is_terminal(self):
        return self.battle._finished
    
    def _estimate_matchup(self, mon: Pokemon, opponent: Pokemon):
        score = max([opponent.damage_multiplier(t) for t in mon.types if t is not None])
        score -= max(
            [mon.damage_multiplier(t) for t in opponent.types if t is not None]
        )
        if mon.base_stats["spe"] > opponent.base_stats["spe"]:
            score += self.SPEED_TIER_COEFICIENT
        elif opponent.base_stats["spe"] > mon.base_stats["spe"]:
            score -= self.SPEED_TIER_COEFICIENT

        score += mon.current_hp_fraction * self.HP_FRACTION_COEFICIENT
        score -= opponent.current_hp_fraction * self.HP_FRACTION_COEFICIENT

        return score

    def _should_dynamax(self, battle: Battle):
        n_remaining_mons = len(
            [m for m in battle.team.values() if m.fainted is False]
        )
        if battle.can_dynamax and self._dynamax_disable is False:
            # Last full HP mon
            if (
                len([m for m in battle.team.values() if m.current_hp_fraction == 1])
                == 1
                and battle.active_pokemon.current_hp_fraction == 1
            ):
                return True
            # Matchup advantage and full hp on full hp
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                return True
            if n_remaining_mons == 1:
                return True
        return False
    
    def _should_terastallize(self, battle: Battle):
        if battle.can_tera and self._tera_disable is False:
            # return True

            # matchup adv and full hp on full hp
            battle.active_pokemon.terastallize()
            if (
                self._estimate_matchup(
                    battle.active_pokemon, battle.opponent_active_pokemon
                )
                > 0
                and battle.active_pokemon.current_hp_fraction == 1
                and battle.opponent_active_pokemon.current_hp_fraction == 1
            ):
                battle.active_pokemon.unterastallize()
                return True

            battle.active_pokemon.unterastallize()

        return False
    
    

    def state_translate(self, battle: Battle, idx: int, return_actions: bool = False, return_choices: bool = False):
        try:
            if return_actions:
                return self.prompt_translate(self, battle, return_actions=return_actions, idx=idx)
            if return_choices:
                return self.prompt_translate(self, battle, return_choices=return_choices, idx=idx)
            return self.prompt_translate(self, battle, idx=idx)
        except TypeError:
            # Prompt function may not accept idx; fall back gracefully
            if return_actions:
                return self.prompt_translate(self, battle, return_actions=return_actions)
            if return_choices:
                return self.prompt_translate(self, battle, return_choices=return_choices)
            return self.prompt_translate(self, battle)

    def check_status(self, status):
        if status:
            if status.value == 1:
                return "burnt"
            elif status.value == 2:
                return "fainted"
            elif status.value == 3:
                return "frozen"
            elif status.value == 4:
                return "paralyzed"
            elif status.value == 5:
                return "poisoned"
            elif status.value == 7:
                return "toxic"
            elif status.value == 6:
                return "sleeping"
        else:
            return ""

    def boost_multiplier(self, state, level):
        if state == "accuracy":
            if level == 0:
                return 1.0
            if level == 1:
                return 1.33
            if level == 2:
                return 1.66
            if level == 3:
                return 2.0
            if level == 4:
                return 2.5
            if level == 5:
                return 2.66
            if level == 6:
                return 3.0
            if level == -1:
                return 0.75
            if level == -2:
                return 0.6
            if level == -3:
                return 0.5
            if level == -4:
                return 0.43
            if level == -5:
                return 0.36
            if level == -6:
                return 0.33
        else:
            if level == 0:
                return 1.0
            if level == 1:
                return 1.5
            if level == 2:
                return 2.0
            if level == 3:
                return 2.5
            if level == 4:
                return 3.0
            if level == 5:
                return 3.5
            if level == 6:
                return 4.0
            if level == -1:
                return 0.67
            if level == -2:
                return 0.5
            if level == -3:
                return 0.4
            if level == -4:
                return 0.33
            if level == -5:
                return 0.29
            if level == -6:
                return 0.25
        raise ValueError(f'Boost level not found {state} {level}')


    def apply_protosynthesis(self, mon: Pokemon, returned_stat):
        if (mon.ability == 'protosynthesis' or mon.ability == 'quarkdrive') and mon.item == 'boosterdrive':
            best_stat_val = mon.stats['atk']
            best_stat = 'atk'
            for stat in ['def', 'spa', 'spd', 'spe']:
                if best_stat_val < mon.stats[stat]:
                    best_stat_val = mon.stats[stat]
                    best_stat = stat
            if best_stat == returned_stat:
                if best_stat == 'spe':
                    return 1.5
                else:
                    return 1.3
        return 1.0

    def calculate_remaining_hp(self, 
                               p1: Pokemon, 
                               p2: Pokemon, 
                               m1: Move, 
                               m2: Move, 
                               boosts1: Dict[str, int]=None, 
                               boosts2: Dict[str, int]=None, 
                               return_turns: bool=False,
                               team=None,
                               opp_team=None,
                               ):
        # calculate move damage based on stats IF NOT status change
        # damage done by pokemon 1
        d1, d2 = 0, 0
        id1, id2 = None, None
        if boosts1 is None:
            boosts1 = p1._boosts
        if boosts2 is None:
            boosts2 = p2._boosts
        if m1 != None:
            id1 = m1.id
            if m1.category != MoveCategory.STATUS:
                d1 = self.calc_base_dmg(p1, p2, m1, boosts1=boosts1, boosts2=boosts2, team=team) 
                # print(f'base damage 1: {d1}')
                d1 = self.modify_damage(d1, p1, p2, m1, m2)
                # print(f'modified damage 1: {d1}')
        # damage done by pokemon 2
        if m2 != None:
            id2 = m2.id
            if m2.category != MoveCategory.STATUS:
                d2 = self.calc_base_dmg(p2, p1, m2, boosts1=boosts2, boosts2=boosts1, team=opp_team) 
                # print(f'base damage 2: {d2}')
                d2 = self.modify_damage(d2, p2, p1, m2, m1)
                # print(f'modified damage 2: {d2}')
        # get HP
        stats1 = p1.calculate_stats(battle_format=self.format)
        hp1_total = stats1['hp']
        hp1 = p1.current_hp_fraction * hp1_total
        stats2 = p2.calculate_stats(battle_format=self.format)
        hp2_total = stats2['hp']
        hp2 = p2.current_hp_fraction * hp2_total
        turns_to_faint = hp2 / max(d1, 0.001)
        
        # apply in order of speed
        p1_speed = round(stats1['spe'] * self.boost_multiplier('spe', boosts1['spe'])) * self.apply_protosynthesis(p1, 'spe')
        p2_speed = round(stats2['spe'] * self.boost_multiplier('spe', boosts2['spe'])) * self.apply_protosynthesis(p1, 'spe')
        p1_priority = False
        if m1 is not None:
            if m1.priority == 1:
                p1_priority = True
                if m2 is not None:
                    if m1.priority == 1 and m2.priority == 1:
                        p1_priority = False
        if p1_speed > p2_speed or p1_priority:
            # check healing
            if m1 != None:
                if m1.heal > 0:
                    hp1 += m1.heal
            hp2 = max(hp2 - d1, 0)
            # check if HP is 0 before second move
            if hp2 != 0:
                # check healing
                if m2 != None:
                    if m2.heal > 0:
                        hp2 += m2.heal
                hp1 = max(hp1 - d2, 0)
        else:
            # check healing
            if m2 != None:
                if m2.heal > 0:
                    hp2 += m2.heal
            hp1 = max(hp1 - d2, 0)
            # check if HP is 0 before second move
            if hp1 != 0:
                # check healing
                if m1 != None:
                    if m1.heal > 0:
                        hp1 += m1.heal
                hp2 = max(hp2 - d1, 0)
        m1_success = ((p1_speed > p2_speed) or hp1 > 0) and m1 != None
        m2_success = ((p1_speed <= p2_speed) or hp2 > 0) and m2 != None
        # print(f'[DAMAGE PRED A] {p1.species} {id1} {p2.species} {id2}')
        # print(f'[DAMAGE PRED B] {d1} {d2} {hp1} {hp2}')
        hp1 = int(hp1 / hp1_total * 100)
        hp2 = int(hp2 / hp2_total * 100)
        if return_turns:
            return hp1, hp2, m1_success, m2_success, turns_to_faint
        return hp1, hp2, m1_success, m2_success

    def modify_base_power(self, mon: Pokemon, target: Pokemon, move: Move, team=None) -> float:
        power = move.base_power
        # weight based modifiers based on difference in health
        weight_moves_diff = {'heavyslam', 'heatcrash'}
        if weight_moves_diff.intersection([move.id]):
            relative_weight = target.weight / mon.weight
            if relative_weight < 0.2:
                power = 120
            elif relative_weight < 0.25:
                power = 100
            elif relative_weight < 0.334:
                power = 80
            elif relative_weight < 0.5:
                power = 60
            else:
                power = 40
        # weight based modifiers based on opponent's weight
        weight_moves_opp = {'grassknot', 'lowkick'}
        if weight_moves_opp.intersection([move.id]):
            if target.weight > 200:
                power = 120
            elif target.weight > 100:
                power = 100
            elif target.weight > 50:
                power = 80
            elif target.weight > 25:
                power = 60
            elif target.weight > 10:
                power = 40
            else:
                power = 20
        # ability based modifiers
        if mon.ability == 'technician' and power <= 60:
            power *= 1.5
        if move.id == 'acrobatics':
            if mon.item == None or mon.item == 'flyinggem':
                power *= 2
        if mon.ability == 'supremeoverlord' and team is not None:
            boost_atk = 0
            for teammate in team.values():
                # print(teammate)
                if teammate.fainted:
                    boost_atk += 0.1
            power *= 1.0 + boost_atk
        return power
    
    def apply_item(self, mon: Pokemon, boosts: Dict[str, int]) -> Dict[str, int]:
        if boosts is None:
            print('found None') 
            return None
        boosts = boosts.copy()
        item = mon.item
        def boost(stat: str, amount: float):
                boosts[stat] += amount
                boosts[stat] = np.floor(boosts[stat])
                if boosts[stat] > 6:
                    boosts[stat] = 6
                elif boosts[stat] < -6:
                    boosts[stat] = -6
                return
        # @TODO: check for embargo in field that prevents item use
        if item != None:
            # status
            if mon.status:
                if mon.ability == 'guts':
                    boost('atk', 1.5)
                elif mon.ability == 'quickfeet':
                    boost('spe', 1.5)
                elif mon.status == Status.BRN:
                    boost('atk', 0.5)
                elif mon.status == Status.PAR:
                    boost('spe', 0.5)
                if mon.ability == 'marvelscale':
                    boost('def', 1.5)
                    
            if mon.species == 'pikachu' and item == 'lightball':
                boost('atk', 2.0)   # @TODO: gen > 4
                boost('spa', 2.0)
            if mon.species in ['marowak', 'cubone'] and item == 'thickclub':
                boost('atk', 2.0)
            if mon.species == 'ditto':
                if item == 'quickpowder':
                    boost('spe', 2.0)
                elif item == 'metalpowder':
                    boost('def', 2.0) # @TODO: gen > 2
            if mon.is_dynamaxed:
                boost('hp', 2.0)
            # items
            if item == 'choiceband':
                boost('atk', 1.5)
            elif item == 'choicespecs':
                boost('spa', 1.5)
            elif item == 'choicescarf':
                boost('spe', 1.5)
            elif item == 'assaultvest':
                boost('spd', 1.5)
            elif item == 'furcoat':
                boost('def', 2.0)
            elif mon.species == 'clamperl':
                if item == 'deepseatooth':
                    boost('spa', 2)
                elif item == 'deepseascale':
                    boost('spd', 2)
            # @TODO: ignoring latios soul dew since it is gen 7 only
            # elif item =='souldew' and gen == 7 and mon.species in ['latios', 'latias']
            #     boost('spa', 1.5)
            #     boost('sda', 1.5)
            elif item == 'ironball':
                boost('spe', 0.5)
            # @TODO: check eviolite
            # if gen{gen}pokedex.json [mon.species][evos] and item == 'eviolite':
            #   boost('def', 1.5)
            #   boost('spd', 1.5)
                
            # boost for abilities
        if mon.ability in ['purepower', 'hugepower']:
            boost('atk', 2.0)
        elif mon.ability == 'hustle' or (mon.ability == 'gorillatactics' and not mon.is_dynamaxed):
            boost('atk', 1.5)
        elif mon.ability == 'furcoat':
            boost('def', 2.0)
        elif mon.ability == 'defeatist' and mon.current_hp_fraction <= 0.5:
            boost('atk', 0.5)
            
        # @TODO: ruin ability requires all pokemon to be checked
        '''
        // apply "Ruin" ability effects that'll ruin me (gen 9)
        // update (2022/12/14): Showdown fixed the Ruin stacking bug, so apply only once now
        // update (2023/01/23): apparently Ruin abilities will CANCEL each other out if BOTH Pokemon have it
        if (allPlayers?.length && ruinAbilitiesActive(...allPlayers.map((p) => p?.side))) {
            const ruinCounts = countRuinAbilities(...allPlayers.map((p) => p?.side));

            // 25% SPD reduction if there's at least one Pokemon with the "Beads of Ruin" ability (excluding this `pokemon`)
            const ruinBeadsCount = Math.max(ruinCounts.beads - (ability === 'beadsofruin' ? ruinCounts.beads : 0), 0);

            if (ruinBeadsCount) {
            record.apply('spd', 0.75, 'abilities', 'Beads of Ruin');
            }

            // 25% DEF reduction if there's at least one Pokemon with the "Sword of Ruin" ability (excluding this `pokemon`)
            const ruinSwordCount = Math.max(ruinCounts.sword - (ability === 'swordofruin' ? ruinCounts.sword : 0), 0);

            if (ruinSwordCount) {
            record.apply('def', 0.75, 'abilities', 'Sword of Ruin');
            }

            // 25% ATK reduction if there's at least one Pokemon with the "Tablets of Ruin" ability (excluding this `pokemon`)
            const ruinTabletsCount = Math.max(ruinCounts.tablets - (ability === 'tabletsofruin' ? ruinCounts.tablets : 0), 0);

            if (ruinTabletsCount) {
            record.apply('atk', 0.75, 'abilities', 'Tablets of Ruin');
            }

            // 25% SPA reduction if there's at least one Pokemon with the "Vessel of Ruin" ability (excluding this `pokemon`)
            const ruinVesselCount = Math.max(ruinCounts.vessel - (ability === 'vesselofruin' ? ruinCounts.vessel : 0), 0);

            if (ruinVesselCount) {
            record.apply('spa', 0.75, 'abilities', 'Vessel of Ruin');
            }
        }
        '''
        # weather effects
        weather = self.battle.weather
        if weather:
            if weather == 'sand':
                # @TODO: only if gen > 3
                if 'rock' in mon.types:
                    boost('spd', 1.5)
                if mon.ability == 'sandrush':
                    boost('spe', 2.0)
            if weather in ['hail', 'snow'] and mon.ability == 'slushrush':
                boost('spe', 2.0)
            if weather == 'snow' and 'ice' in mon.types:    # @TODO check type case
                boost('def', 1.5)
            if item != 'utilityumbrella':
                if weather in ['sun', 'harshsunshine']:
                    if mon.ability == 'solarpower':
                        boost('spa', 1.5)
                    elif mon.ability == 'chlorophyll':
                        boost('spe', 1.5)
                    elif mon.ability == 'orichalcumpulse':
                        boost('atk', 1.3)
                    # elif mon.ability == 'flowergrift': # @TODO: check if baseform is cherrim
            if weather in ['rain', 'heavyrain'] and mon.ability == 'swiftswim':
                boost('spe', 2.0)
            boost('spa', 0.5)
                
            # terrain effects: @TODO implement trerrains
            '''
            // apply terrain effects
            const terrain = id(field.terrain);

            // 50% DEF boost if ability is "Grass Pelt" w/ terrain of the grassy nature
            if (ability === 'grasspelt' && terrain === 'grassy') {
                record.apply('def', 1.5, 'abilities', 'Grass Pelt');
            }

            if (terrain === 'electric') {
                // 2x SPE modifier if ability is "Surge Surfer" w/ electric terrain
                if (ability === 'surgesurfer') {
                record.apply('spe', 2, 'abilities', 'Surge Surfer');
                }

                // 30% SPA boost if ability is "Hadron Engine" w/ electric terrain
                if (ability === 'hadronengine') {
                record.apply('spa', 1.3, 'abilities', 'Hadron Engine');
                }
            }
            '''
            # @TODO: tailwind (field)
            # @TODO: grass pledge (field)
            '''
            // 2x SPE modifier if "Tailwind" is active on the field
            if (playerSide?.isTailwind) {
                record.apply('spe', 2, 'field', 'Tailwind');
            }

            // 0.25x SPE modifier if "Grass Pledge" is active on the field
            if (playerSide?.isGrassPledge) {
                record.apply('spe', 0.25, 'field', 'Grass Pledge');
            }
            '''
            # @TODO: apply toggleable abilities
            '''
            // apply toggleable abilities
            if (pokemon.abilityToggled) {
                // 50% ATK/SPE reduction if ability is "Slow Start"
                if (ability === 'slowstart') {
                record.apply('atk', 0.5, 'abilities', 'Slow Start');
                record.apply('spe', 0.5, 'abilities', 'Slow Start');
                }

                // 2x SPE modifier if ability is "Unburden" and item was removed
                if (ability === 'unburden') {
                record.apply('spe', 2, 'abilities', 'Unburden');
                }

                /**
                * @todo Implement ally Pokemon support for "Minus" and "Plus" toggleable abilities.
                * @see https://github.com/smogon/pokemon-showdown-client/blob/master/src/battle-tooltips.ts#L1159-L1172
                */

                // 30% highest stat boost (or 1.5x SPE modifier) if ability is "Protosynthesis" or "Quark Drive"
                // update (2023/05/15): highest boosted stat can now be overwritten by specifying pokemon.boostedStat
                // (which it is, wherever highestBoostedStat is declared above)
                if (['protosynthesis', 'quarkdrive'].includes(ability) && highestBoostedStat) {
                // if the Pokemon has a booster volatile, use its reported stat
                // e.g., 'protosynthesisatk' -> boosterVolatileStat = 'atk'
                // const boosterVolatile = Object.keys(pokemon.volatiles || {}).find((k) => /^(?:proto|quark)/i.test(k));
                // const boosterVolatileStat = <Showdown.StatNameNoHp> boosterVolatile?.replace(/(?:protosynthesis|quarkdrive)/i, '');
                // const stat = boosterVolatileStat || highestBoostedStat;

                record.apply(
                    highestBoostedStat,
                    highestBoostedStat === 'spe' ? 1.5 : 1.3,
                    'abilities',
                    dex.abilities.get(ability)?.name || ability,
                );
                }
            '''
            # @TODO: swaps
            '''
            // swap ATK & DEF if the move "Power Trick" was used
            if ('powertrick' in pokemon.volatiles) {
                record.swap('atk', 'def', 'moves', 'Power Trick');
            }

            // swap DEF & SPD if Wonder Room is active on the field
            if (field?.isWonderRoom) {
                record.swap('def', 'spd', 'moves', 'Wonder Room');
            }
            '''
            # @TODO: screens
            '''
            if (player.side.isReflect) {
            record.apply('def', 2, gen === 1 ? 'moves' : 'field', 'Reflect');
            }

            // 100% SPD boost if the "Light Screen" player side condition is active (gens 1-2)
            if (player.side.isLightScreen) {
            record.apply('spd', 2, gen === 1 ? 'moves' : 'field', 'Light Screen');
            '''

        return boosts
    
    def calc_base_dmg(self, 
                      pokemon: Pokemon, 
                      target: Pokemon, 
                      move: Move,
                      boosts1: Dict[str, int]=None, 
                      boosts2: Dict[str, int]=None, 
                      team=None,
                      ) -> float:
        baseDamage = 2
        level = pokemon.level
        power = self.modify_base_power(pokemon, target, move, team)
        stats = pokemon.calculate_stats(battle_format=self.format)
        if boosts1 is None:
            boosts1 = pokemon._boosts
        active_boosts = self.apply_item(pokemon, boosts1)
        stats_target = target.calculate_stats(battle_format=self.format)
        if boosts2 is None:
            boosts2 = target._boosts
        target_boosts = self.apply_item(target, boosts2)
        A = -1
        D = -1
        if move.category == MoveCategory.PHYSICAL:
            A = stats['atk'] if active_boosts['atk']==0 else round(stats['atk']*self.boost_multiplier('atk', active_boosts['atk']))
            A = A * self.apply_protosynthesis(pokemon, 'atk')
            D = stats_target['def'] if target_boosts['def']==0 else round(stats_target['def']*self.boost_multiplier('def', target_boosts['def']))
            D = D * self.apply_protosynthesis(pokemon, 'def')
        elif move.category == MoveCategory.SPECIAL:
            A = stats['spa'] if active_boosts['spa']==0 else round(stats['spa']*self.boost_multiplier('spa', active_boosts['spa']))
            A = A * self.apply_protosynthesis(pokemon, 'spa')
            D = stats_target['spd'] if target_boosts['spd']==0 else round(stats_target['spd']*self.boost_multiplier('spd', target_boosts['spd']))
            D = D * self.apply_protosynthesis(pokemon, 'spd')
        assert A != -1
        baseDamage += (((2*level) / 5. + 2) * power * A / D) / 50. + 2
        return baseDamage

    def modify_damage(self, baseDamage: float, pokemon: Pokemon, target: Pokemon, move: Move, target_move: Move, use_expected: bool=True) -> float:
        if (not move.type):
            move.type = '???'
        type = move.type

        # if (move.spreadHit) {
        #     # multi-target modifier (doubles only)
        #     const spreadModifier = move.spreadModifier or (self.battle.gameType === 'freeforall' ? 0.5 : 0.75);
        #     self.battle.debug('Spread modifier: ' + spreadModifier);
        #     baseDamage = self.battle.modify(baseDamage, spreadModifier);
        # if (move.multihitType == 'parentalbond' and move.expected_hits() > 1):
        #     # Parental Bond modifier
        #     const bondModifier = self.battle.gen > 6 ? 0.25 : 0.5;
        #     self.battle.debug(`Parental Bond modifier: ${bondModifier}`);
        #     baseDamage = self.battle.modify(baseDamage, bondModifier);
        

        # weather modifier
        # baseDamage = self.battle.runEvent('WeatherModifyDamage', pokemon, target, move, baseDamage);

        # crit - not a modifier
        # crit_stage = 1
        # match move.crit_ratio:
        #     case 0:
        #         crit_stage = 1. / 24.
        #     case 1:
        #         crit_stage = 1. / 8.
        #     case 2:
        #         crit_stage = 1. / 2.
        #     case _:
        #         crit_stage = 1
        # isCrit = np.random.random_sample() < crit_stage
        # if (isCrit):
        #     baseDamage *= 1.5

        # random factor - also not a modifier
        # baseDamage *= np.random.randint(85, 101) / 100.
        # print(f'random {baseDamage}')
        type_1 = pokemon.type_1.name
        type_2 = None
        if pokemon.type_2 != None:
            type_2 = pokemon.type_2.name
        # STAB
        # The "???" type never gets STAB
        # Not even if you Roost in Gen 4 and somehow manage to use
        # Struggle in the same turn.
        # (On second thought, it might be easier to get a MissingNo.)
        stab = 1
        if (type != '???'):
            if type.name in [type_1, type_2]:
                stab = pokemon.stab_multiplier
                
        
        ## TODO: change stab based on tera    
        #tera_stab = 1
        #if type.name in [type_1, type_2]:
            

        baseDamage *= stab
        # print(f'stab {move.id} {stab} {baseDamage}')
        

        # types
        opponent_type_list = []
        if target.terastallized:
            opponent_type_list.append(target._terastallized_type)
        else:
            if target.type_1:
                type_1 = target.type_1.name
                opponent_type_list.append(type_1)

                if target.type_2:
                    type_2 = target.type_2.name
                    opponent_type_list.append(type_2)
        # print('typing', type_1, type_2)
        extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list = calculate_move_type_damage_multipier(
                                    type_1, type_2, self.gen.type_chart, [type.name])
        # print(extreme_effective_type_list, effective_type_list, resistant_type_list, extreme_resistant_type_list, immune_type_list)

        if extreme_effective_type_list: # extremely-effective (4x damage)
            # print('double super effective')
            baseDamage *= 4

        elif effective_type_list:   # super-effective (2x damage)
            # print('super effective')
            baseDamage *= 2

        elif resistant_type_list:   # ineffective (0.5x damage)
            # print('ineffective')
            baseDamage *= 0.5

        elif extreme_resistant_type_list:   # highly ineffective (0.25x damage)
            # print('highly ineffective')
            baseDamage *= 0.25

        elif immune_type_list:  # zero effect (0x damage)
            # print('immune')
            baseDamage *= 0
        # print(pokemon.species, pokemon.item, target.item, type)
        # check for item immunity
        if target.item is not None:
            if target.item.lower() == 'airballoon':
                # print(target.species, target.item)
                if type == 'ground':
                    baseDamage *= 0
        # @TODO: check for ability immunity, such as sound moves, bullet moves etc. https://pokemondb.net/pokebase/188704/what-are-all-the-abilities-that-grant-immunities
        # @TODO: add healing for absorbs
        if target.ability == 'voltabsorb' and move.type == 'electric':
            baseDamage *= 0
        if (target.ability == 'waterabsorb' or target.ability == 'dryskin') and move.type == 'water':
            baseDamage *= 0
        if target.ability == 'levitate' and move.type == 'ground':
            baseDamage *= 0
        if target.ability == 'flashfire' and move.type == 'fire':
            baseDamage *= 0
        if target.ability == 'dryskin' and move.type == 'fire':
            baseDamage *= 1.25
        if target.ability == 'wonderguard' and not (extreme_effective_type_list or effective_type_list):
            baseDamage *= 0
        

        # print(f'type {baseDamage}')

        if (pokemon.status == Status.BRN and move.category == MoveCategory.PHYSICAL and not pokemon.ability == 'guts'):
            if (self.gen.gen < 6 or move.id != 'facade'):
                baseDamage *= 0.5

        # Generation 5, but nothing later, sets damage to 1 before the final damage modifiers
        if (self.gen.gen == 5 and baseDamage == 0): baseDamage = 1

        # Final modifier. Modifiers that modify damage after min damage check, such as Life Orb.
        if pokemon.item == 'LifeOrb':
            baseDamage *= 1.3

        if target_move != None:
            if ((move.is_z or pokemon.is_dynamaxed) and target_move.id == 'protect'):
                baseDamage *= 0.25
            if target_move.category == MoveCategory.STATUS and (move.id == 'suckerpunch' or move.id == 'thunderclap'):
                baseDamage *= 0

        # Generation 6-7 moves the check for minimum 1 damage after the final modifier...
        if (self.gen.gen != 5 and baseDamage == 0): baseDamage = 1

        # expected value with accuracy
        if use_expected:
            baseDamage *= move.accuracy
        
        # print(f'gen {baseDamage}')
        # ...but 16-bit truncation happens even later, and can truncate to 0
        return int(baseDamage) * move.expected_hits

    def _handle_battle_message(self, split_message: List[str]):
        """Handles a battle message.

        :param split_message: The received battle message.
        :type split_message: str
        """
        description = ''
        battle = self.battle
        if split_message[1] == "switch":
            # update hp information
            self.switch_set.add(split_message[2])
            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[4])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[4]]

            description = " " + split_message[2].split(" ")[0] + " sent out " + split_message[2].split(": ")[-1] + "."
            description = description.replace("p2a:", "Player2").replace("p1a:", "Player1")
        elif split_message[1] == "turn":
            if len(battle.speed_list) == 2:
                description = f" {battle.speed_list[0]} outspeeded {battle.speed_list[1]} in this turn."
            description += "[sep]Turn " + split_message[2] + ":"
        elif split_message[1] == "drag":
            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[4])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[4]]

            description = " " + split_message[2] + "was dragged out."

        elif split_message[1] == "faint":
            description = " " + split_message[2] + " faint."

        elif split_message[1] == "move":
            description = " " + split_message[2] + " used "+ split_message[3] + "."
            battle.speed_list.append(split_message[2])

        elif split_message[1] == "cant":
            if split_message[3] == "frz":
                reason = "frozen"
            elif split_message[3] == "par":
                reason = "paralyzed"
            elif split_message[3] == "slp":
                reason = "sleeping"
            else:
                reason = split_message[3]

            description = " " + split_message[2] + " cannot move because of " + reason + "."

        elif split_message[1] == "-start":
            description = " " + split_message[2] + " started " + split_message[3] + "."
            if len(split_message) > 4:
                if split_message[4]:
                    description = " " + split_message[2] + " started " + split_message[3] + " due to " + split_message[4] + "."

        elif split_message[1] == "-end":
            description = " " + split_message[2] + " stop " + split_message[3] + "."

        elif split_message[1] == "-fieldstart":
            description = " Field start: " + split_message[2] + " ran across the battlefield."

        elif split_message[1] == "-fieldend":
            description = " Field end: " + split_message[2] + " disappeared from the battlefield."

        elif split_message[1] == "-ability":
            description = " " + split_message[2] + "'s ability: " + split_message[3] + "."

        elif split_message[1] == "-supereffective":
            description = " The move was super effective to " + split_message[2] + "."

        elif split_message[1] == "-resisted":
            description = " The move was ineffective to " + split_message[2] + "."

        elif split_message[1] == "-heal":
            try:
                previous_hp = battle.pokemon_hp_log_dict[split_message[2]][-1].split(" ")[0]
            except:
                previous_hp = "100/100"

            if previous_hp == "0":
                previous_hp_fraction = 0
            else:
                previous_hp_fraction = round(float(previous_hp.split("/")[0]) / float(previous_hp.split("/")[1]) * 100)

            current_hp = split_message[3].split(" ")[0]
            if current_hp == "0":
                current_hp_fraction = 0
            else:
                current_hp_fraction = round(float(current_hp.split("/")[0]) / float(current_hp.split("/")[1]) * 100)

            delta_hp_fraction = current_hp_fraction - previous_hp_fraction

            if len(split_message) > 4:
                description = f" {split_message[2]} restored {delta_hp_fraction}% of HP ({current_hp_fraction}% left) {split_message[4]}."
            else:
                description = f" {split_message[2]} restored {delta_hp_fraction}% of HP ({current_hp_fraction}% left)."
            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[3])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[3]]

        elif split_message[1] == "-damage":
            try:
                previous_hp = battle.pokemon_hp_log_dict[split_message[2]][-1].split(" ")[0]
            except:
                previous_hp = "100/100"

            if previous_hp == "0":
                previous_hp_fraction = 0
            else:
                previous_hp_fraction = round(float(previous_hp.split("/")[0]) / float(previous_hp.split("/")[1]) * 100)

            try:
                battle.pokemon_hp_log_dict[split_message[2]].append(split_message[3])
            except:
                battle.pokemon_hp_log_dict[split_message[2]] = [split_message[3]]

            current_hp = split_message[3].split(" ")[0]
            if current_hp == "0":
                current_hp_fraction = 0
            else:
                current_hp_fraction = round(float(current_hp.split("/")[0]) / float(current_hp.split("/")[1]) * 100)

            delta_hp_fraction = previous_hp_fraction - current_hp_fraction

            if "oroark" in split_message[2]:  # Zoroark
                if len(split_message) > 4:
                    description = f" {split_message[2]}'s HP was damaged to {current_hp_fraction}% {split_message[4]}."
                else:
                    description = f" It damaged {split_message[2]}'s HP to {current_hp_fraction}%."
            else:
                if len(split_message) > 4:
                    description = f" {split_message[2]}'s HP was damaged by {delta_hp_fraction}% {split_message[4]} ({current_hp_fraction}% left)."
                else:
                    description = f" It damaged {split_message[2]}'s HP by {delta_hp_fraction}% ({current_hp_fraction}% left)."

        elif split_message[1] == "-unboost":
            description = " It decreased " + split_message[2] + "'s " + split_message[3] + " " + split_message[4] + " level."

        elif split_message[1] == "-boost":
            description = " It boosted " + split_message[2] + "'s " + split_message[3] + " " + split_message[4] + " level."

        elif split_message[1] == "-fail":
            description = " But it failed."

        elif split_message[1] == "-miss":
            description = " It missed."

        elif split_message[1] == "-activate":
            description = " " + split_message[2] + " activated " + split_message[3] + "."

        elif split_message[1] == "-immune":
            description = f" but had zero effect to {split_message[2]}."

        elif split_message[1] == "-crit":
            description = " A critical hit."

        elif split_message[1] == "-status":
            status_dict = {"brn": "burnt", "frz": "frozen", "par": "paralyzed", "slp": "sleeping", "tox": "toxic", "psn": "poisoned"}
            description = " It caused " + split_message[2] + " " + status_dict[split_message[3]] + "."
        # else:
        #     print('unhandled description msg', split_message)
        if description:
            battle.battle_msg_history = battle.battle_msg_history + description
            # print(description)

        self.battle.parse_message(split_message)

class SimNode():
    def __init__(self, 
                 battle: Battle, 
                 move_effect,
                 pokemon_move_dict,
                 ability_effect,
                 pokemon_ability_dict,
                 item_effect,
                 pokemon_item_dict,
                 gen,
                 _dynamax_disable,
                 depth: int=0,
                 format='gen9randombattle',
                 prompt_translate=None,
                 sim=None,
                ):
        if sim is None:
            self.simulation = LocalSim(battle, 
                                    move_effect,
                                    pokemon_move_dict,
                                    ability_effect,
                                    pokemon_ability_dict,
                                    item_effect,
                                    pokemon_item_dict,
                                    gen,
                                    _dynamax_disable,
                                    format=format,
                                    prompt_translate=prompt_translate,
                                    )
        else:
            self.simulation = sim
        self.depth = depth
        self.action = None
        self.action_opp: BattleOrder = None
        self.parent_node = None
        self.parent_action = None
        self.hp_diff = 0
        self.children: List[SimNode] = []