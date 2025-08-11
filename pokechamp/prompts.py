
from typing import Dict
import numpy as np
from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.local_simulation import LocalSim, move_type_damage_wrapper

def get_turn_summary(sim: LocalSim,
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

def get_current_status(sim, battle):
    opponent_prompt = ''
    # opponent side conditions
    opponent_side_condition_list = [] # I should add the description for the side condition. and the status.
    for side_condition in battle.opponent_side_conditions:
        opponent_side_condition_list.append(" ".join(side_condition.name.lower().split("_")))

    opponent_side_condition = ",".join(opponent_side_condition_list)
    if opponent_side_condition:
        opponent_prompt = opponent_prompt + "Opponent team's side condition: " + opponent_side_condition

    opponent_prompt += "\n"

    active_pokemon_prompt = ''

    side_condition_list = []
    for side_condition in battle.side_conditions:

        side_condition_name = " ".join(side_condition.name.lower().split("_"))
        if side_condition == SideCondition.SPIKES:
            effect = " (cause damage to your pokémon when switch in except flying type)"
        elif side_condition == SideCondition.STEALTH_ROCK:
            effect = " (cause rock-type damage to your pokémon when switch in)"
        elif side_condition == SideCondition.STICKY_WEB:
            effect = " (reduce the speed stat of your pokémon when switch in)"
        elif side_condition == SideCondition.TOXIC_SPIKES:
            effect = " (cause your pokémon toxic when switch in)"
        else:
            effect = ""

        side_condition_name = side_condition_name + effect
        side_condition_list.append(side_condition_name)

    side_condition_prompt = ",".join(side_condition_list)

    if side_condition_prompt:
        active_pokemon_prompt = active_pokemon_prompt + "Your team's side condition: " + side_condition_prompt + "\n"


    # The active pokemon
    active_hp_fraction = round(battle.active_pokemon.current_hp / battle.active_pokemon.max_hp * 100)
    active_type = ""
    if battle.active_pokemon.type_1:
        active_type += battle.active_pokemon.type_1.name.capitalize()
        if battle.active_pokemon.type_2:
            active_type = active_type + " and " + battle.active_pokemon.type_2.name.capitalize()

    try:
        active_ability = sim.ability_effect[battle.active_pokemon.ability]["name"]
        ability_effect = sim.ability_effect[battle.active_pokemon.ability]["effect"]
    except:
        active_ability = battle.active_pokemon.ability
        ability_effect = ""

    # item
    if battle.active_pokemon.item:
        try:
            active_item = sim.item_effect[battle.active_pokemon.item]["name"]
            item_effect = sim.item_effect[battle.active_pokemon.item]["effect"]
            active_item = f"{active_item}({item_effect})"
        except:
            active_item = battle.active_pokemon.item
    else:
        active_item = ""


    active_pokemon_prompt = (
            f"Your current pokemon:{battle.active_pokemon.species},Type:{active_type},HP:{active_hp_fraction}%" +
            (f"Status:{sim.check_status(battle.active_pokemon.status)}," if sim.check_status(battle.active_pokemon.status) else "" ) +
            (f"Ability:{active_ability}({ability_effect})," if ability_effect else f"Ability:{active_ability},") +
            (f"Item:{active_item}" if active_item else "")
        )
    
    return opponent_prompt + active_pokemon_prompt

def get_status_mon(mon: Pokemon, sim: LocalSim):
    active_hp_fraction = round(mon.current_hp / mon.max_hp * 100)
    active_type = ""
    if mon.type_1:
        active_type += mon.type_1.name.capitalize()
        if mon.type_2:
            active_type = active_type + " and " + mon.type_2.name.capitalize()

    try:
        active_ability = sim.ability_effect[mon.ability]["name"]
        ability_effect = sim.ability_effect[mon.ability]["effect"]
    except:
        active_ability = mon.ability
        ability_effect = ""

    # item
    if mon.item:
        try:
            active_item = sim.item_effect[mon.item]["name"]
            item_effect = sim.item_effect[mon.item]["effect"]
            active_item = f"{active_item}({item_effect})"
        except:
            active_item = mon.item
    else:
        active_item = ""

    pokemon_prompt = (
            f"Pokemon:{mon.species},Type:{active_type},HP:{active_hp_fraction}%" +
            (f"Status:{sim.check_status(mon.status)}," if sim.check_status(mon.status) else "" ) +
            (f"Ability:{active_ability}({ability_effect})," if ability_effect else f"Ability:{active_ability},") +
            (f"Item:{active_item}" if active_item else "") + 
            (f"Tera type:{mon._terastallized_type} Is terastallized? {mon.terastallized}")
        )
    return pokemon_prompt + '\n'

def get_macro_strat(sim: LocalSim,
                    battle: Battle
                    ) -> str:
    '''
    This should be the system prompt.
    '''
    # with open('poke_env/data/static/teams/gen9ou1_strat.txt', 'r') as f:
    #     text = f.read()
    return sim.strategy
    return ''

def get_number_turns_faint(mon: Pokemon,
                           move: Move,
                           mon_opp: Pokemon,
                           sim: LocalSim,
                           boosts1: Dict[str, int]=None, 
                           boosts2: Dict[str, int]=None, 
                           return_hp=False,
                           ) -> int:
    # @TODO: power up punch stat boost
    _, hp_remaining, _, _, turns = sim.calculate_remaining_hp(mon, mon_opp, move, None, boosts1=boosts1, boosts2=boosts2, return_turns = True, team=sim.battle.team, opp_team=sim.battle.opponent_team)
    turns = int(np.ceil(turns))
    # print(mon.species, mon_opp.species, move.id, hp_remaining, turns)
    # input()
    if return_hp:
        return turns, hp_remaining
    return turns

def get_status_num_turns_fnt(mon: Pokemon,
                             move: Move,
                             mon_opp: Pokemon,
                             sim: LocalSim,
                             boosts: Dict[str, int]=None, 
                             ) -> int:
    def boost(stat: str, amount: float):
        boosts[stat] += amount
        boosts[stat] = np.floor(boosts[stat])
        if boosts[stat] > 6:
            boosts[stat] = 6
        elif boosts[stat] < -6:
            boosts[stat] = -6
        return
    turns = []
    multiple_stat_raises = True
    if move.id == 'swordsdance':
        if boosts['atk'] == 6:
            return np.inf
        boost('atk', 2.0)
    elif move.id == 'curse' and 'ghost' not in mon.types:
        if boosts['atk'] == 6:
            return np.inf
        boost('atk', 1.5)
        boost('def', 1.5)
        boost('spe', 0.5)
    elif move.id == 'noretreat':
        boost('hp', 1.5)
        boost('atk', 1.5)
        boost('def', 1.5)
        boost('spa', 1.5)
        boost('spd', 1.5)
        boost('spe', 1.5)
        multiple_stat_raises = False
    elif move.id == 'bellydrum':
        # @TODO: cut own hp in half
        boost('atk', 6.0)
        multiple_stat_raises = False
    elif move.id == 'filletaway':
        # @TODO: cut own hp in half
        boost('atk', 2.0)
        boost('spa', 2.0)
        boost('spe', 2.0)
        multiple_stat_raises = False
    elif move.id == 'nastyplot':
        if boosts['spa'] == 6:
            return np.inf
        boost('spa', 2.0)
    elif move.id == 'tailflow':
        if boosts['spa'] == 6:
            return np.inf
        boost('spa', 2.5)
    elif move.id == 'calmmind':
        if boosts['spa'] == 6:
            return np.inf
        boost('spa', 1.5)
        boost('spd', 1.5)
    elif move.id == 'dragondance':
        if boosts['atk'] == 6:
            return np.inf
        boost('atk', 1.5)
        boost('spe', 1.5)
    elif move.id == 'shellsmash':
        if boosts['atk'] == 6 and boosts['spa'] == 6:
            return np.inf
        boost('atk', 2.0)
        boost('spa', 2.0)
        boost('def', 0.5)
        boost('spd', 0.5)
    elif move.id == 'quiverdance':
        if boosts['spa'] == 6:
            return np.inf
        boost('spa', 1.5)
        boost('spd', 1.5)
        boost('spe', 1.5)
    elif move.id == 'geomancy':
        if boosts['spa'] == 6:
            return np.inf
        boost('spa', 2.0)
        boost('spd', 2.0)
        boost('spe', 2.0)
    elif move.id == 'bulkup':
        if boosts['atk'] == 6:
            return np.inf
        boost('atk', 1.5)
    else:
        return np.inf
    if multiple_stat_raises:
        t_new = 1 + get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=boosts.copy())
        turns.append(t_new)
        
    for move_id in mon.moves:
        if move_id == move.id: continue
        if mon.moves[move_id].category == MoveCategory.STATUS: continue
        t_new = 1 + get_number_turns_faint(mon, mon.moves[move_id], mon_opp, sim, boosts1=boosts.copy())
        turns.append(t_new)
    if len(turns) > 0:
        return np.min(turns)
    return np.inf

def get_move_prompt(mon: Pokemon,
                    mon_opp: Pokemon,
                    sim: LocalSim,
                    is_player: bool=False,
                    ):
    move_prompt = ''
    moves = sim.get_opponent_current_moves(mon=mon, is_player=is_player)
    
    def call_dmg_calc(mon: Pokemon, mon_opp: Pokemon, move: Move):
        move_prompt = ''
        t = 0
        if move.category == MoveCategory.STATUS:
            # apply stat boosting effects to see if it will KO in fewer turns
            t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            move_prompt += f'{move_id}: {sim.move_effect[move.id]} {t} turns to KO opponent\'s pokemon\n'
            # move_prompt += f'{move_id}: {sim.move_effect[move.id]}\n'
        else:
            t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            move_prompt += f'{move_id}: {t} turns to KO opponent\'s pokemon\n'
            
        return move_prompt, t
    
    for move_id in moves:
        # @TODO: fix nothing coming up
        if 'nothing' == move_id:
            continue
        # move = mon.moves[move_id]
        move = Move(move_id, gen=sim.gen.gen)
        if mon.is_dynamaxed:
            move = move.dynamaxed
            # check if the move is status move -> change to max guard
            if move.category == MoveCategory.STATUS:
                move_prompt += f'{move_id}: inf turn to KO opponent\'s pokemon. This will fully protect your pokemon from all damage.'
                continue
            
        prompt_new, _ = call_dmg_calc(mon, mon_opp, move)
        move_prompt += prompt_new

        #if move.category == MoveCategory.STATUS:
        #    # apply stat boosting effects to see if it will KO in fewer turns
        #    t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
        #    move_prompt += f'{move_id}: {sim.move_effect[move.id]} {t} turns to KO opponent\'s pokemon\n'
        #    # move_prompt += f'{move_id}: {sim.move_effect[move.id]}\n'
        #else:
        #    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
        #    move_prompt += f'{move_id}: {t} turns to KO opponent\'s pokemon\n'
    
    if sim.battle._data.gen == 8 and sim.battle.can_dynamax:
        # give data about if bot were to dynamax
        move_prompt += f"If {mon.species} uses \'dynamax\':\n"
        for move_id in moves:
            if 'nothing' == move_id:
                continue
            move = Move(move_id, gen=sim.gen.gen).dynamaxed
            prompt_new, _ = call_dmg_calc(mon, mon_opp, move)
            move_prompt += prompt_new
                
    if sim.battle._data.gen == 9 and sim.battle.can_tera:

        if not mon_opp.terastallized:

            # untera'd mon vs tera'd opp
            move_prompt += f"{mon.species}\'s moves if opponent\'s {mon_opp.species} uses \'terastallize\':\n"
            mon_opp.terastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                prompt_new, _ = call_dmg_calc(mon, mon_opp, move)
                move_prompt += prompt_new

            # tera'd mon vs tera'd opp
            move_prompt += f"{mon.species}\'s moves if it uses \'terastallize\' and opponent\'s {mon_opp.species} uses \'terastallize\':\n"
            mon.terastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                prompt_new, _ = call_dmg_calc(mon, mon_opp, move)
                move_prompt += prompt_new

            # tera'd mon vs untera'd opp
            move_prompt += f"{mon.species}\'s moves if it uses \'terastallize\' and opponent\'s {mon_opp.species} does NOT use \'terastallize\':\n"
            mon_opp.unterastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                prompt_new, _ = call_dmg_calc(mon, mon_opp, move)
                move_prompt += prompt_new

            mon.unterastallize()

        else:
            # tera'd mon vs opp (tera'd or untera'd)
            move_prompt += f"{mon.species}\'s moves if it uses \'terastallize\'"
            mon.terastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                prompt_new, _ = call_dmg_calc(mon, mon_opp, move)
                move_prompt += prompt_new
            mon.unterastallize()
            
    return move_prompt

def get_move_opp_prompt(mon: Pokemon,
                    mon_opp: Pokemon,
                    sim: LocalSim,
                    ):
    move_prompt = ''
    moves = sim.get_opponent_current_moves()
    
    def call_dmg_calc(mon: Pokemon, mon_opp: Pokemon, move: Move):
        move_prompt = ''
        
        if move.category == MoveCategory.STATUS:
            # apply stat boosting effects to see if it will KO in fewer turns
            t = 1+get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            move_prompt += f'{move_id}: {sim.move_effect[move.id]} {t} turns to KO your pokemon\n'
            # move_prompt += f'{move_id}: {sim.move_effect[move.id]}\n'
        else:
            t = 1+get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            move_prompt += f'{move_id}: {t} turns to KO your pokemon\n'
            
        return move_prompt
    

    for move_id in moves:
        # @TODO: fix nothing coming up
        if 'nothing' == move_id:
            continue
        move = Move(move_id, gen=sim.gen.gen)
        if mon.is_dynamaxed:
            move = move.dynamaxed
            # check if the move is status move -> change to max guard
            if move.category == MoveCategory.STATUS:
                move_prompt += f'{move_id}: inf turn to KO your pokemon. This will fully protect opponent\'s pokemon from all damage.'
                continue    

        move_prompt += call_dmg_calc(mon_opp, mon, move)

        # if move.category == MoveCategory.STATUS:
        #     # apply stat boosting effects to see if it will KO in fewer turns
        #     t = 1+get_status_num_turns_fnt(mon_opp, move, mon, sim, boosts=mon_opp._boosts.copy())
        #     move_prompt += f'{move_id}: {sim.move_effect[move.id]} {t} turns to KO your pokemon\n'
        #     # move_prompt += f'{move_id}: {sim.move_effect[move.id]}\n'
        # else:
        #     t = 1+get_number_turns_faint(mon_opp, move, mon, sim, boosts1=mon_opp._boosts.copy(), boosts2=mon.boosts.copy())
        #     move_prompt += f'{move_id}: {t} turns to KO your pokemon\n'
            
    # give additional information of what happens if bot dynamaxes
    if sim.battle._data.gen == 8 and sim.battle.opponent_can_dynamax and mon_opp.active and mon.active: # only show relevant info
        move_prompt += f"If opponent's {mon_opp.species} uses \'dynamax\':\n"
        for move_id in moves:
            if 'nothing' == move_id:
                continue
            move = Move(move_id, gen=sim.gen.gen).dynamaxed

            move_prompt += call_dmg_calc(mon_opp, mon, move)

            # if move.category == MoveCategory.STATUS:
            #     # apply stat boosting effects to see if it will KO in fewer turns
            #     t = 1+get_status_num_turns_fnt(mon_opp, move, mon, sim, boosts=mon_opp._boosts.copy())
            #     move_prompt += f'{move_id}: {sim.move_effect[move.id]} {t} turns to KO your pokemon\n'
            #     # move_prompt += f'{move_id}: {sim.move_effect[move.id]}\n'
            # else:
            #     t = 1+get_number_turns_faint(mon_opp, move, mon, sim, boosts1=mon_opp._boosts.copy(), boosts2=mon.boosts.copy())
            #     move_prompt += f'{move_id}: {t} turns to KO your pokemon\n'
        
    if sim.battle._data.gen == 9 and sim.battle.opponent_can_tera and mon_opp.active and mon.active:
        if not mon.terastallized:
            # untera'd opp vs tera'd mon
            move_prompt += f"opponent\'s {mon_opp.species} moves if {mon.species} uses \'terastallize\':\n"
            mon.terastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                move_prompt += call_dmg_calc(mon_opp, mon, move)

            # tera'd opp vs tera'd mon
            move_prompt += f"opponent\'s{mon_opp.species} moves if it uses \'terastallize\' and {mon.species} uses \'terastallize\':\n"
            mon_opp.terastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                move_prompt += call_dmg_calc(mon_opp, mon, move)

            # tera'd opp vs untera'd mon
            move_prompt += f"opponent\'s {mon_opp.species} moves if it uses \'terastallize\' and {mon.species} does NOT use \'terastallize\':\n"
            mon.unterastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                move_prompt += call_dmg_calc(mon_opp, mon, move)
            mon_opp.unterastallize() 

        else:
            # tera'd opp vs mon (tera'd or untera'd)
            move_prompt += f"opponent\'s {mon_opp.species} moves if it uses \'terastallize\'"
            mon_opp.terastallize()
            for move_id in moves:
                if 'nothing' == move_id:
                    continue
                move = Move(move_id, gen=sim.gen.gen)
                move_prompt += call_dmg_calc(mon_opp, mon, move)
            mon_opp.unterastallize()

    return move_prompt

def get_speed_prompt(mon: Pokemon,
                    mon_opp: Pokemon,
                    sim: LocalSim
                    ):
    mon_stats = mon.calculate_stats(battle_format=sim.format)
    mon_opp_stats = mon_opp.calculate_stats(battle_format=sim.format)
    if mon_stats['spe'] > mon_opp_stats['spe']:
        return f'{mon.species} outspeeds {mon_opp.species}\n'
    else:
        return f'{mon_opp.species} outspeeds {mon.species}\n'


def estimate_matchup(sim: LocalSim, battle: Battle, mon: Pokemon, mon_opp: Pokemon, is_opp: bool=False):
        hp_remaining = []
        hps = []
        moves = list(mon.moves.keys())
        if len(moves) == 0:
            moves = sim.get_opponent_current_moves(mon=mon)
        if battle.active_pokemon.species == mon.species and not is_opp:
            moves = [move.id for move in battle.available_moves]
        for move_id in moves:
            move = Move(move_id, gen=sim.gen.gen)
            t = np.inf
            if move.category == MoveCategory.STATUS:
                # apply stat boosting effects to see if it will KO in fewer turns
                t, hp = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy(),)
            else:
                t, hp = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            hp_remaining.append(t)
            hps.append(hp)
            # _, hp2, _, _ = sim.calculate_remaining_hp(battle.active_pokemon, battle.opponent_active_pokemon, move, None)
            # hp_remaining.append(hp2)
        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        return Move(best_move, gen=sim.gen.gen), best_move_turns, hps
    
def get_micro_strat(sim: LocalSim,
                    battle: Battle
                    ) -> str:
    '''
    Create matchup information about moves, speed
    '''
    micro_prompt = ''
    for mon in battle.team.values():
        micro_prompt += get_status_mon(mon, sim)
    for mon_opp in battle.opponent_team.values():
        micro_prompt += get_status_mon(mon_opp, sim)
    for mon in battle.team.values():
        if mon.fainted: continue
        for mon_opp in battle.opponent_team.values():
            if mon_opp.fainted: continue
            if battle.active_pokemon.species == mon.species:
                micro_prompt += 'Current pokemon:\n'
                micro_prompt += f'{mon.species} vs. {mon_opp.species}:\n'
                micro_prompt += get_speed_prompt(mon, mon_opp, sim)
                micro_prompt += f'{mon.species}\'s moves:\n'
                micro_prompt += get_move_prompt(mon, mon_opp, sim, is_player=True)
                # opponent moves
                micro_prompt += f'Opponent moves: {mon_opp.species}\n'
                micro_prompt += get_move_opp_prompt(mon, mon_opp, sim)
                micro_prompt += '\n'
            else:
                micro_prompt += 'Requires switch:\n'
                micro_prompt += f'{mon.species} vs. {mon_opp.species}:\n'
                micro_prompt += get_speed_prompt(mon, mon_opp, sim)
                micro_prompt += f'{mon.species}\'s moves:\n'
                micro_prompt += get_move_prompt(mon, mon_opp, sim)
                # opponent moves
                micro_prompt += f'Opponent moves: {mon_opp.species}\n'
                micro_prompt += get_move_opp_prompt(mon, mon_opp, sim)
                micro_prompt += '\n'
            # else:
            #     mon_opp_stats = mon_opp.calculate_stats(battle_format=sim.format)
            #     mon_stats = mon.calculate_stats(battle_format=sim.format)
            #     # estimate player side matchup
            #     _, best_move_turns = estimate_matchup(sim, battle, mon, mon_opp)
            #     best_move_turns = best_move_turns + 1   # switch
            #     # estimate opponent side matchup
            #     _, opp_move_turns = estimate_matchup(sim, battle, mon_opp, mon, is_opp=True)
            #     # ignore scenario where opponent wins
            #     macro_prompt = f'{best_move_turns} turns to KO opponent. '
            #     if opp_move_turns > best_move_turns or (opp_move_turns == best_move_turns and mon_stats['spe'] <= mon_opp_stats['spe']):
            #         macro_prompt = 'Opponent wins matchup'
            #     micro_prompt += 'Requires switch:\n'
            #     micro_prompt += f'{mon.species} vs. {mon_opp.species}:\n'
            #     micro_prompt += get_speed_prompt(mon, mon_opp, sim)
            #     micro_prompt += f'{mon.species}\'s moves:\n'
            #     micro_prompt += macro_prompt
            #     # opponent moves
            #     micro_prompt += f'Opponent moves: {mon_opp.species}\n'
            #     micro_prompt += get_move_opp_prompt(mon, mon_opp, sim)
            #     micro_prompt += '\n'
            
    return micro_prompt

def get_avail_actions(sim: LocalSim,
                      battle: Battle
                      ) -> str:
    move_choices = [move.id for move in battle.available_moves]

    # add gimmick action to available actions
    # if battle._data.gen == 8:
    #     if battle.can_dynamax and not battle._dynamax_intent:
    #         move_choices.append('dynamax')

    # if battle._data.gen == 9:
    #     if battle.can_tera and not battle._tera_intent:
    #         move_choices.append('terastallize')

    action_prompt_move = f' Your current Pokemon: {battle.active_pokemon.species}.\nChoose only from the following action choices:\n'
    if len(move_choices) > 0:
        action_prompt_move += f"[<move_name>] = {move_choices}\n"

    # Switch
    action_prompt_switch = f"You have {len(battle.available_switches)} pokemons:\n"
    switch_choices = []
    for mon in battle.available_switches:
        if mon.species not in switch_choices:
            switch_choices.append(mon.species)
    # switch_choices = [pokemon.species for pokemon in battle.available_switches]
    if len(switch_choices) > 0:
        action_prompt_switch += f"[<switch_pokemon_name>] = {switch_choices}\n"

    return action_prompt_move, action_prompt_switch

def get_gimmick_prompt(sim: LocalSim, battle: Battle):
    gen = battle._data.gen

    gimmick_prompt = ''

    if gen == 8:
        if battle.can_dynamax or battle.opponent_can_dynamax:
            # gimmick_prompt = '\'dynamax\' temporarily increases a Pokemon\'s max hit points as well as the damage dealt by its moves for 3 turns. You can only \'dynamax\' one Pokemon per battle. You can choose to \'dynamax\' and use another move in this same turn.'
            # gimmick_prompt += 'If you choose to \'dyanamax\' this turn, please structure your response as [<move_name>, \'dynamax\'], where <move_name> is your intended move.\n'
            gimmick_prompt = '\'dynamax\' temporarily increases a Pokemon\'s max hit points as well as the damage dealt by its moves for 3 turns. You can only \'dynamax\' one Pokemon per battle. You can choose to \'dynamax\' and use another move in the same turn.\n'
            # gimmick_prompt += 'If you choose to \'dyanamax\' this turn, please structure your response as [<move_name>, \'dynamax\'], where <move_name> is your intended move.\n'

    elif gen == 9:
        if battle.can_tera or battle.opponent_can_tera:
            # gimmick_prompt = '\'terastallize\' changes a Pokemon\'s defensive typing to solely their tera type, meaning their resistances and weaknesses can change. It also gives them a boost to moves of their new typing. You can only \'terastallize\' one Pokemon per battle, and it will last on that Pokemon until they are KO\'d or the battle ends. You can choose to \'terastallize\' and use another move in this same turn.'
            # gimmick_prompt += 'If you choose to \'terastallize\' this turn, please structure your response as [<move_name>, \'terastallize\'], where <move_name> is your intended move.\n'
            gimmick_prompt = '\'terastallize\' changes a Pokemon\'s defensive typing to solely their tera type, meaning their resistances and weaknesses can change. It also gives them a boost to moves of their new typing. You can only \'terastallize\' one Pokemon per battle, and it will last on that Pokemon until they are KO\'d or the battle ends. You can choose to \'terastallize\' and use another move in the same turn.\n'
            # gimmick_prompt += 'If you choose to \'terastallize\' this turn, please structure your response as [<move_name>, \'terastallize\'], where <move_name> is your intended move.\n'

    return gimmick_prompt

def get_gimmick_motivation(sim: LocalSim, battle: Battle):
    gen = battle._data.gen

    gimmick_motiviation_prompt = ''


    if gen == 8:
        # if battle._dynamax_intent and not battle.active_pokemon.is_dynamaxed:
        #     gimmick_motiviation_prompt += 'You are about to \'dynamax\' this turn, you should choose a move with this in mind.\n'

        if battle.can_dynamax:
            gimmick_motiviation_prompt += "You are able to use [\'dynamax\'] this turn as well.\n"

        if sim._should_dynamax(battle):
            # gimmick_motiviation_prompt += "You are able to use [\'dynamax\'] this turn as well. It is recommended you choose \'dynamax\' this turn as your move over the other moves listed.\n"
            gimmick_motiviation_prompt += "It is recommended you choose to \'dynamax\' this turn paired with a move from your available moves.\n"
            
        elif battle.active_pokemon.is_dynamaxed:
            if battle.dynamax_turns_left == 0:
                gimmick_motiviation_prompt += f"You are currently \'dynamaxed\' for {battle.dynamax_turns_left} more turns.\n" 
            else:
                gimmick_motiviation_prompt += f"This is your last turn while \'dynamaxed\'.\n" 
                
            if battle.dynamax_turns_left > 0:
                gimmick_motiviation_prompt += "Switching now will end the \'dynamax\' condition early. It is generally not recommended to switch to another Pokemon while you are currently \'dynamaxed\'. You will not be able to \'dynamax\' later.\n"

        if battle.opponent_active_pokemon.is_dynamaxed:
            gimmick_motiviation_prompt += f"Opponent is \'dynamaxed\' for {sim.battle.opponent_dynamax_turns_left} more turns.\n"

    elif gen == 9:
        # if battle._tera_intent and not battle.active_pokemon.terastallized:
        #     gimmick_motiviation_prompt += 'You are about to \'terastallize\', you should choose your next move with the knowledge that you will be a different type.\n'

        if battle.can_tera:
            gimmick_motiviation_prompt += "You are able to use [\'terastallize\'] this turn as well.\n"

        if sim._should_terastallize(battle):
            # gimmick_motiviation_prompt += "You are able to use [\'terastallize\'] this turn as well. It is recommended you choose \'terastallize\' this turn as your move over the other moves listed.\n"
            gimmick_motiviation_prompt += "It is recommended you choose to \'terastallize\' this turn paired with a move from your available moves.\n"


    return gimmick_motiviation_prompt
    
def prompt_translate(sim: LocalSim, 
                    battle: Battle,
                    return_actions=False
                    ) -> str:
    battle_prompt = get_turn_summary(sim, battle, n_turn=16)
    macro_prompt = get_macro_strat(sim, battle)
    # print(f'macro prompt:\n{macro_prompt}')
    micro_prompt = get_micro_strat(sim, battle)
    # print(f'micro prompt:\n{micro_prompt}')
    action_prompt_move, action_prompt_switch = get_avail_actions(sim, battle)

    state_prompt = get_current_status(sim, battle)

    gimmick_prompt = get_gimmick_prompt(sim, battle)

    gimmick_motivation_prompt = get_gimmick_motivation(sim, battle)
    # action_prompt_list = get_avail_actions_list(sim, battle)
    # print(f'avail actions prompt:\n{action_prompt_move}\n{action_prompt_switch}')
    # action_prompt = "Choose the best action to KO the opponent's pokemon in the least turns.\n"
    action_prompt = f"Recall the information about each of {battle.active_pokemon.species}'s move actions and available switch actions. Which move or switch will KO the opponent's pokemon in the fewest turns in order to win against the opponent?\n"
     
    if battle.active_pokemon.fainted: # passive switching
        
        system_prompt = (
            f"You are a pokemon battler that targets to win the pokemon battle. Your {battle.active_pokemon.species} just fainted. Choose a suitable pokemon to continue the battle. Here are some tips:"
            " Compare the speeds of your pokemon to the opposing pokemon, which determines who take the move first."
            " Consider the defense state and type-resistance of your pokemon when its speed is lower than the opposing pokemon."
            " Consider the move-type advantage of your pokemon pokemon when its speed is higher than the opposing pokemon.")

        system_prompt = system_prompt + macro_prompt
        state_prompt = battle_prompt + micro_prompt + gimmick_prompt
        state_action_prompt = action_prompt + gimmick_motivation_prompt + action_prompt_switch

    else: # take a move or active switch
        
        system_prompt = (
            "You are a pokemon battler that targets to win the pokemon battle. You can choose to take a move or switch in another pokemon. Here are some battle tips:"
            " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
            " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
            " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
            " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
            " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
            )

        system_prompt = system_prompt + macro_prompt

        # state_prompt = battle_prompt + micro_prompt + gimmick_prompt
        # state_action_prompt = action_prompt + gimmick_motivation_prompt + action_prompt_switch + action_prompt_move

        state_prompt = battle_prompt + micro_prompt + gimmick_prompt + state_prompt
        # state_action_prompt = action_prompt + action_prompt_switch + action_prompt_move
        state_action_prompt = action_prompt + gimmick_motivation_prompt + action_prompt_switch + action_prompt_move

    # print(system_prompt)
    # print(state_prompt)
    # print(state_action_prompt)
    # input()
    

    # state_action_prompt = action_prompt + action_prompt_list
    if return_actions:
        return system_prompt, state_prompt, action_prompt, action_prompt_switch, action_prompt_move

        # print(system_prompt)
        # print(state_prompt)
        # print(state_action_prompt)
        # #input()

    return system_prompt, state_prompt, state_action_prompt




def state_translate(sim: LocalSim, 
                    battle: Battle,
                    return_actions=False,
                    ):

    n_turn = 5
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

    # number of fainted pokemon
    opponent_fainted_num = 0
    for _, opponent_pokemon in battle.opponent_team.items():
        if opponent_pokemon.fainted:
            opponent_fainted_num += 1

    opponent_unfainted_num = 6 - opponent_fainted_num
    opponent_hp_fraction = round(battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp * 100)
    opponent_stats = battle.opponent_active_pokemon.calculate_stats(battle_format=sim.format)
    opponent_boosts = battle.opponent_active_pokemon._boosts
    active_stats = battle.active_pokemon.stats
    if active_stats['atk'] is None:
        active_stats = battle.active_pokemon.base_stats
    active_boosts = battle.active_pokemon._boosts
    opponent_status = battle.opponent_active_pokemon.status
    opponent_is_dynamax = battle.opponent_active_pokemon.is_dynamaxed

    # Type information
    opponent_type = ""

    opponent_type_list = []
    if battle.opponent_active_pokemon.type_1:
        type_1 = battle.opponent_active_pokemon.type_1.name
        opponent_type += type_1.capitalize()
        opponent_type_list.append(type_1)

        if battle.opponent_active_pokemon.type_2:
            type_2 = battle.opponent_active_pokemon.type_2.name
            opponent_type = opponent_type + " and " + type_2.capitalize()
            opponent_type_list.append(type_2)
    species = battle.opponent_active_pokemon.species
    if species == 'polteageistantique':
        species = 'polteageist'
    if battle.opponent_active_pokemon.ability:
        opponent_ability = battle.opponent_active_pokemon.ability
    elif species in sim.pokemon_ability_dict:
        opponent_ability = sim.pokemon_ability_dict[species][0]
    else:
        opponent_ability = ""

    if opponent_ability:
        try:
            ability_name = sim.ability_effect[opponent_ability]["name"]
            ability_effect = sim.ability_effect[opponent_ability]["effect"]
            opponent_ability = f"{ability_name}({ability_effect})"
        except:
            pass

    opponent_prompt = (
            f"Opponent has {opponent_unfainted_num} pokemons left.\n" +
            f"Opposing pokemon:{battle.opponent_active_pokemon.species},Type:{opponent_type},HP:{opponent_hp_fraction}%" +
            (f"Status:{sim.check_status(opponent_status)}," if sim.check_status(opponent_status) else "") +
            (f"Attack:{opponent_stats['atk']}," if opponent_boosts['atk']==0 else f"Attack:{round(opponent_stats['atk'] * sim.boost_multiplier('atk', opponent_boosts['atk']))}({opponent_boosts['atk']} stage boosted),") +
            (f"Defense:{opponent_stats['def']}," if opponent_boosts['def']==0 else f"Defense:{round(opponent_stats['def'] * sim.boost_multiplier('def', opponent_boosts['def']))}({opponent_boosts['def']} stage boosted),") +
            (f"Special attack:{opponent_stats['spa']}," if opponent_boosts['spa']==0 else f"Special attack:{round(opponent_stats['spa'] * sim.boost_multiplier('spa', opponent_boosts['spa']))}({opponent_boosts['spa']} stage boosted),") +
            (f"Special defense:{opponent_stats['spd']}," if opponent_boosts['spd']==0 else f"Special defense:{round(opponent_stats['spd'] * sim.boost_multiplier('spd', opponent_boosts['spd']))}({opponent_boosts['spd']} stage boosted),") +
            (f"Speed:{opponent_stats['spe']}," if opponent_boosts['spe'] == 0 else f"Speed:{round(opponent_stats['spe'] * sim.boost_multiplier('spe', opponent_boosts['spe']))}({opponent_boosts['spe']} stage boosted),") +
            (f"Ability:{opponent_ability}" if opponent_ability else "")
    )
    opponent_speed = round(opponent_stats['spe'] * sim.boost_multiplier('spe', opponent_boosts['spe']))

    team_move_type = []
    for move in battle.available_moves:
        if move.base_power > 0:
            team_move_type.append(move.type.name)

    for pokemon in battle.available_switches:
        for move in pokemon.moves.values():
            if move.base_power > 0:
                team_move_type.append(move.type.name)

    opponent_move_type_damage_prompt = move_type_damage_wrapper(battle.opponent_active_pokemon, sim.gen.type_chart, team_move_type)

    if opponent_move_type_damage_prompt:
        opponent_prompt = opponent_prompt + opponent_move_type_damage_prompt + "\n"

    # Opponent active pokemon move
    opponent_move_prompt = ""
    if battle.opponent_active_pokemon.moves:
        for move_id, opponent_move in battle.opponent_active_pokemon.moves.items():
            if opponent_move.base_power == 0:
                continue # only count attack move

            if opponent_move.category.name == "SPECIAL":
                opponent_spa = opponent_stats['spa'] * sim.boost_multiplier('spa', opponent_boosts['spa'])
                active_spd = active_stats['spd'] * sim.boost_multiplier('spd', active_boosts['spd'])
                power = round(opponent_spa / active_spd * opponent_move.base_power)

            elif opponent_move.category.name == "PHYSICAL":
                opponent_atk = opponent_stats['atk'] * sim.boost_multiplier('atk', opponent_boosts['atk'])
                active_def = active_stats['atk'] * sim.boost_multiplier('atk', active_boosts['atk'])
                power = round(opponent_atk/active_def * opponent_move.base_power)
            else:
                power = 0

            opponent_move_prompt += f"[{opponent_move.id},{opponent_move.type.name.capitalize()},Power:{power}],"
            opponent_type_list.append(opponent_move.type.name)

    if opponent_move_prompt:
        opponent_prompt = opponent_prompt + f"{battle.opponent_active_pokemon.species} used moves:" + opponent_move_prompt

    possible_move_prompt = ""
    try:
        if species in sim.pokemon_move_dict:
            possible_move_list = list(sim.pokemon_move_dict[species].values())
            # possible_move_list.sort(key=lambda x: x[3], reverse=True)
            for move in possible_move_list:
                # if move[2]>0:
                possible_move_prompt = possible_move_prompt + f"[{move[0]},{move[1].lower()},Power:{move[2]}],"
                opponent_type_list.append(move[1].upper())
        else:
            possible_move_prompt = ""
    except:
        possible_move_prompt = ""

    if possible_move_prompt:
        opponent_prompt = opponent_prompt + f"{battle.opponent_active_pokemon.species}'s all the possible attacks:" + possible_move_prompt

    opponent_side_condition_list = [] # I should add the description for the side condition. and the status.
    for side_condition in battle.opponent_side_conditions:
        opponent_side_condition_list.append(" ".join(side_condition.name.lower().split("_")))

    opponent_side_condition = ",".join(opponent_side_condition_list)
    if opponent_side_condition:
        opponent_prompt = opponent_prompt + "Opponent team's side condition: " + opponent_side_condition

    opponent_prompt += "\n"

    # The active pokemon
    active_hp_fraction = round(battle.active_pokemon.current_hp / battle.active_pokemon.max_hp * 100)
    active_status = battle.active_pokemon.status

    active_type = ""
    if battle.active_pokemon.type_1:
        active_type += battle.active_pokemon.type_1.name.capitalize()
        if battle.active_pokemon.type_2:
            active_type = active_type + " and " + battle.active_pokemon.type_2.name.capitalize()

    active_move_type_damage_prompt = move_type_damage_wrapper(battle.active_pokemon, sim.gen.type_chart, opponent_type_list)
    speed_active_stats = active_stats['spe']
    if speed_active_stats == None: speed_active_stats = 0
    active_speed = round(speed_active_stats*sim.boost_multiplier('spe', active_boosts['spe']))

    try:
        active_ability = sim.ability_effect[battle.active_pokemon.ability]["name"]
        ability_effect = sim.ability_effect[battle.active_pokemon.ability]["effect"]
    except:
        active_ability = battle.active_pokemon.ability
        ability_effect = ""

    # item
    if battle.active_pokemon.item:
        try:
            active_item = sim.item_effect[battle.active_pokemon.item]["name"]
            item_effect = sim.item_effect[battle.active_pokemon.item]["effect"]
            active_item = f"{active_item}({item_effect})"
        except:
            active_item = battle.active_pokemon.item
    else:
        active_item = ""

    active_pokemon_prompt = (
        f"Your current pokemon:{battle.active_pokemon.species},Type:{active_type},HP:{active_hp_fraction}%" +
        (f"Status:{sim.check_status(active_status)}," if sim.check_status(active_status) else "" ) +
        (f"Attack:{active_stats['atk']}," if active_boosts['atk']==0 else f"Attack:{round(active_stats['atk']*sim.boost_multiplier('atk', active_boosts['atk']))}({active_boosts['atk']} stage boosted),") +
        (f"Defense:{active_stats['def']}," if active_boosts['def']==0 else f"Defense:{round(active_stats['def']*sim.boost_multiplier('def', active_boosts['def']))}({active_boosts['def']} stage boosted),") +
        (f"Special attack:{active_stats['spa']}," if active_boosts['spa']==0 else f"Special attack:{round(active_stats['spa']*sim.boost_multiplier('spa', active_boosts['spa']))}({active_boosts['spa']} stage boosted),") +
        (f"Special defense:{active_stats['spd']}," if active_boosts['spd']==0 else f"Special defense:{round(active_stats['spd']*sim.boost_multiplier('spd', active_boosts['spd']))}({active_boosts['spd']} stage boosted),") +
        (f"Speed:{active_stats['spe']}" if active_boosts['spe']==0 else f"Speed:{round(active_stats['spe']*sim.boost_multiplier('spe', active_boosts['spe']))}({active_boosts['spe']} stage boosted),") +
        (f"(slower than {battle.opponent_active_pokemon.species})." if active_speed < opponent_speed else f"(faster than {battle.opponent_active_pokemon.species}).") +
        (f"Ability:{active_ability}({ability_effect})," if ability_effect else f"Ability:{active_ability},") +
        (f"Item:{active_item}" if active_item else "")
    )

    if active_move_type_damage_prompt:
        active_pokemon_prompt = active_pokemon_prompt + active_move_type_damage_prompt + "\n"

    side_condition_list = []
    for side_condition in battle.side_conditions:

        side_condition_name = " ".join(side_condition.name.lower().split("_"))
        if side_condition == SideCondition.SPIKES:
            effect = " (cause damage to your pokémon when switch in except flying type)"
        elif side_condition == SideCondition.STEALTH_ROCK:
            effect = " (cause rock-type damage to your pokémon when switch in)"
        elif side_condition == SideCondition.STICKY_WEB:
            effect = " (reduce the speed stat of your pokémon when switch in)"
        elif side_condition == SideCondition.TOXIC_SPIKES:
            effect = " (cause your pokémon toxic when switch in)"
        else:
            effect = ""

        side_condition_name = side_condition_name + effect
        side_condition_list.append(side_condition_name)

    side_condition_prompt = ",".join(side_condition_list)

    if side_condition_prompt:
        active_pokemon_prompt = active_pokemon_prompt + "Your team's side condition: " + side_condition_prompt + "\n"

    # Move
    move_prompt = f"Your {battle.active_pokemon.species} has {len(battle.available_moves)} moves:\n"
    for i, move in enumerate(battle.available_moves):
        try:
            effect = sim.move_effect[move.id]
        except:
            effect = ""

        if move.category.name == "SPECIAL":
            active_spa = active_stats["spa"] * sim.boost_multiplier("spa", active_boosts["spa"])
            opponent_spd = opponent_stats["spd"] * sim.boost_multiplier("spd", active_boosts["spd"])
            power = round(active_spa / opponent_spd * move.base_power)
            move_category = ""
        elif move.category.name == "PHYSICAL":
            active_atk = active_stats["atk"] * sim.boost_multiplier("atk", active_boosts["atk"])
            opponent_def = opponent_stats["def"] * sim.boost_multiplier("def", active_boosts["def"])
            power = round(active_atk / opponent_def * move.base_power)
            move_category = ""
        else:
            move_category = move.category.name.capitalize()
            power = 0

        move_prompt += (f"Move:{move.id},Type:{move.type.name.capitalize()}," +
                        (f"{move_category}-move," if move_category else "") +
                        f"Power:{power},Acc:{round(move.accuracy * sim.boost_multiplier('accuracy', active_boosts['accuracy'])*100)}%"
                        )

        if effect:
            move_prompt += f",Effect:{effect}"
        # whether is effective to the target.
        move_type_damage_prompt = move_type_damage_wrapper(battle.opponent_active_pokemon, sim.gen.type_chart, [move.type.name])
        if move_type_damage_prompt and move.base_power:
            move_prompt += f'({move_type_damage_prompt.split("is ")[-1][:-1]})\n'
        else:
            move_prompt += "\n"

    move_choices = [move.id for move in battle.available_moves]
    action_prompt = f' Your current Pokemon: {battle.active_pokemon.species}.\nChoose only from the following action choices:\n'
    action_prompt_move = ''
    if len(move_choices) > 0:
        action_prompt_move = f"[<move_name>] = {move_choices}\n"

    # Switch
    switch_prompt = f"You have {len(battle.available_switches)} pokemons:\n"

    for i, pokemon in enumerate(battle.available_switches):

        type = ""
        if pokemon.type_1:
            type_1 = pokemon.type_1.name
            type += type_1.capitalize()
            if pokemon.type_2:
                type_2 = pokemon.type_2.name
                type = type + " and " + type_2.capitalize()
        if pokemon.max_hp == 0:
            pokemon._max_hp = 1
        hp_fraction = round(pokemon.current_hp / pokemon.max_hp * 100)

        stats = pokemon.stats
        if stats['atk'] is None:
            stats = pokemon.base_stats
        switch_move_prompt = f" Moves:"
        for _, move in pokemon.moves.items():
            if move.base_power == 0:
                continue # only output attack move
            move_type_damage_prompt = move_type_damage_wrapper(battle.opponent_active_pokemon, sim.gen.type_chart, [move.type.name])
            if "2x" in move_type_damage_prompt:
                damage_multiplier = "2"
            elif "4x" in move_type_damage_prompt:
                damage_multiplier = "4"
            elif "0.5x" in move_type_damage_prompt:
                damage_multiplier = "0.5"
            elif "0.25x" in move_type_damage_prompt:
                damage_multiplier = "0.25"
            elif "0x" in move_type_damage_prompt:
                damage_multiplier = "0"
            else:
                damage_multiplier = "1"

            switch_move_prompt += f"[{move.id},{move.type.name.capitalize()},{damage_multiplier}x damage],"

        if stats['spe'] < opponent_speed:
            speed_prompt = f"(slower than {battle.opponent_active_pokemon.species})."
        else:
            speed_prompt = f"(faster than {battle.opponent_active_pokemon.species})."

        switch_prompt += (
                    f"Pokemon:{pokemon.species},Type:{type},HP:{hp_fraction}%," +
                    (f"Status:{sim.check_status(pokemon.status)}, " if sim.check_status(pokemon.status) else "") +
                    f"Attack:{stats['atk']},Defense:{stats['def']},Special attack:{stats['spa']},Special defense:{stats['spd']},Speed:{stats['spe']}"
                    + speed_prompt
                    + switch_move_prompt)

        pokemon_move_type_damage_prompt = move_type_damage_wrapper(pokemon, sim.gen.type_chart, opponent_type_list) # for defense

        if pokemon_move_type_damage_prompt:
            switch_prompt = switch_prompt + pokemon_move_type_damage_prompt + "\n"
        else:
            switch_prompt += "\n"

    switch_choices = [pokemon.species for pokemon in battle.available_switches]
    action_prompt_switch = ''
    if len(switch_choices) > 0:
        action_prompt_switch = f"[<switch_pokemon_name>] = {switch_choices}\n"

    if battle.active_pokemon.fainted: # passive switching

        system_prompt = (
            f"You are a pokemon battler that targets to win the pokemon battle. Your {battle.active_pokemon.species} just fainted. Choose a suitable pokemon to continue the battle. Here are some tips:"
            " Compare the speeds of your pokemon to the opposing pokemon, which determines who take the move first."
            " Consider the defense state and type-resistance of your pokemon when its speed is lower than the opposing pokemon."
            " Consider the move-type advantage of your pokemon pokemon when its speed is higher than the opposing pokemon.")

        state_prompt = battle_prompt + opponent_prompt + switch_prompt
        state_action_prompt = action_prompt + action_prompt_switch

        return system_prompt, state_prompt, state_action_prompt

    else: # take a move or active switch

        system_prompt = (
            "You are a pokemon battler that targets to win the pokemon battle. You can choose to take a move or switch in another pokemon. Here are some battle tips:"
            " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
            " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
            " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
            " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
            " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
            )

        system_prompt = system_prompt + sim.strategy

        state_prompt = battle_prompt + opponent_prompt + active_pokemon_prompt + move_prompt + switch_prompt
        state_action_prompt = action_prompt + action_prompt_move + action_prompt_switch

        return system_prompt, state_prompt, state_action_prompt
    
def get_opp_move_summary(pokemon: Pokemon, seen_moves: list[Move], potential_moves: list[Move], battle: Battle, sim: LocalSim, is_active: bool=False):
    switch_move_prompt = f' Seen Moves:'
    for move in seen_moves:
        if move.base_power == 0:
            switch_move_prompt += f"[{move.id},{move.type.name.capitalize()}],"
        #     continue # only output attack move
        else:
            move_type_damage_prompt = move_type_damage_wrapper(battle.active_pokemon, sim.gen.type_chart, [move.type.name])
            if "2x" in move_type_damage_prompt:
                damage_multiplier = "2"
            elif "4x" in move_type_damage_prompt:
                damage_multiplier = "4"
            elif "0.5x" in move_type_damage_prompt:
                damage_multiplier = "0.5"
            elif "0.25x" in move_type_damage_prompt:
                damage_multiplier = "0.25"
            elif "0x" in move_type_damage_prompt:
                damage_multiplier = "0"
            else:
                damage_multiplier = "1"

            switch_move_prompt += f"[{move.id},{move.type.name.capitalize()},{damage_multiplier}x damage],"
    switch_move_prompt += f' Potential Moves:'
    for move in potential_moves:
        if move.base_power == 0:
            switch_move_prompt += f"[{move.id},{move.type.name.capitalize()}],"
        else:
            move_type_damage_prompt = move_type_damage_wrapper(battle.active_pokemon, sim.gen.type_chart, [move.type.name])
            if "2x" in move_type_damage_prompt:
                damage_multiplier = "2"
            elif "4x" in move_type_damage_prompt:
                damage_multiplier = "4"
            elif "0.5x" in move_type_damage_prompt:
                damage_multiplier = "0.5"
            elif "0.25x" in move_type_damage_prompt:
                damage_multiplier = "0.25"
            elif "0x" in move_type_damage_prompt:
                damage_multiplier = "0"
            else:
                damage_multiplier = "1"

            switch_move_prompt += f"[{move.id},{move.type.name.capitalize()},{damage_multiplier}x damage],"
    # print(switch_move_prompt)
    stats = pokemon.calculate_stats(battle_format=sim.format)
    active_stats = battle.active_pokemon.stats
    active_boosts = battle.active_pokemon.boosts
    speed_active_stats = active_stats['spe']
    if speed_active_stats == None: speed_active_stats = 0
    active_speed = round(speed_active_stats*sim.boost_multiplier('spe', active_boosts['spe']))
    speed_prompt = ''
    if stats['spe'] is not None:
        if stats['spe'] < active_speed:
            speed_prompt = f"(slower than {battle.active_pokemon.species})."
        else:
            speed_prompt = f"(faster than {battle.active_pokemon.species})."
        
    hp_fraction = round(pokemon.current_hp / pokemon.max_hp * 100)
    # Type information
    opponent_type = ""
    if pokemon.type_1:
        type_1 = pokemon.type_1.name
        opponent_type += type_1.capitalize()
        if pokemon.type_2:
            type_2 = pokemon.type_2.name
            opponent_type = opponent_type + " and " + type_2.capitalize()
            
    if pokemon.ability:
        opponent_ability = pokemon.ability
    elif pokemon.species in sim.pokemon_ability_dict:
        opponent_ability = sim.pokemon_ability_dict[pokemon.species][0]
    else:
        opponent_ability = ""

    if opponent_ability:
        try:
            ability_name = sim.ability_effect[opponent_ability]["name"]
            ability_effect = sim.ability_effect[opponent_ability]["effect"]
            opponent_ability = f"{ability_name}({ability_effect})"
        except:
            pass
            
    switch_prompt = (
                f"Pokemon:{pokemon.species},Type:{opponent_type},HP:{hp_fraction}%," +
                (f"Status:{sim.check_status(pokemon.status)}, " if sim.check_status(pokemon.status) else "") +
                f"Attack:{stats['atk']},Defense:{stats['def']},Special attack:{stats['spa']},Special defense:{stats['spd']},Speed:{stats['spe']}"
                + (f"Ability:{opponent_ability}" if opponent_ability else "")
                + speed_prompt
                + switch_move_prompt)
    
    if is_active:
        opponent_status = pokemon.status
        opponent_stats = stats
        opponent_boosts = pokemon._boosts
        opponent_prompt = (
                f"Opposing active pokemon:{battle.opponent_active_pokemon.species},Type:{opponent_type},HP:{hp_fraction}%" +
                (f"Status:{sim.check_status(opponent_status)}," if sim.check_status(opponent_status) else "") +
                (f"Attack:{opponent_stats['atk']}," if opponent_boosts['atk']==0 else f"Attack:{round(opponent_stats['atk'] * sim.boost_multiplier('atk', opponent_boosts['atk']))}({opponent_boosts['atk']} stage boosted),") +
                (f"Defense:{opponent_stats['def']}," if opponent_boosts['def']==0 else f"Defense:{round(opponent_stats['def'] * sim.boost_multiplier('def', opponent_boosts['def']))}({opponent_boosts['def']} stage boosted),") +
                (f"Special attack:{opponent_stats['spa']}," if opponent_boosts['spa']==0 else f"Special attack:{round(opponent_stats['spa'] * sim.boost_multiplier('spa', opponent_boosts['spa']))}({opponent_boosts['spa']} stage boosted),") +
                (f"Special defense:{opponent_stats['spd']}," if opponent_boosts['spd']==0 else f"Special defense:{round(opponent_stats['spd'] * sim.boost_multiplier('spd', opponent_boosts['spd']))}({opponent_boosts['spd']} stage boosted),") +
                (f"Speed:{opponent_stats['spe']}," if opponent_boosts['spe'] == 0 else f"Speed:{round(opponent_stats['spe'] * sim.boost_multiplier('spe', opponent_boosts['spe']))}({opponent_boosts['spe']} stage boosted),") +
                (f"Ability:{opponent_ability}" if opponent_ability else "")
                + speed_prompt
                + switch_move_prompt
        )
        return opponent_prompt + '\n'
    
    return switch_prompt + '\n'

def state_translate2(sim: LocalSim, 
                    battle: Battle,
                    return_actions=False,
                    return_choices=False,
                    ):
    # init default to high elo player
    player_elo, opponent_elo = 1800, 1800
    for player in battle._players:
        if player['rating'] != '':
            if battle._player_role == player['player']:
                player_elo = player['rating']
            else:
                opponent_elo = player['rating']
    player_elo, opponent_elo = 1800, 1200
    
    # get turn history
    battle_prompt = get_turn_summary(sim, battle, n_turn=16)

    # number of fainted pokemon
    opponent_fainted_num = 0
    for _, opponent_pokemon in battle.opponent_team.items():
        if opponent_pokemon.fainted:
            opponent_fainted_num += 1

    opponent_unfainted_num = 6 - opponent_fainted_num
    opponent_hp_fraction = round(battle.opponent_active_pokemon.current_hp / battle.opponent_active_pokemon.max_hp * 100)
    opponent_stats = battle.opponent_active_pokemon.calculate_stats(battle_format=sim.format)
    opponent_boosts = battle.opponent_active_pokemon._boosts
    active_stats = battle.active_pokemon.stats
    if active_stats['atk'] is None:
        active_stats = battle.active_pokemon.base_stats
    active_boosts = battle.active_pokemon._boosts
    opponent_status = battle.opponent_active_pokemon.status
    opponent_is_dynamax = battle.opponent_active_pokemon.is_dynamaxed

    # Type information
    opponent_type = ""

    opponent_type_list = []
    if battle.opponent_active_pokemon.type_1:
        type_1 = battle.opponent_active_pokemon.type_1.name
        opponent_type += type_1.capitalize()
        opponent_type_list.append(type_1)

        if battle.opponent_active_pokemon.type_2:
            type_2 = battle.opponent_active_pokemon.type_2.name
            opponent_type = opponent_type + " and " + type_2.capitalize()
            opponent_type_list.append(type_2)
    # species = battle.opponent_active_pokemon.species
    # if species == 'polteageistantique':
    #     species = 'polteageist'
    # if battle.opponent_active_pokemon.ability:
    #     opponent_ability = battle.opponent_active_pokemon.ability
    # elif species in sim.pokemon_ability_dict:
    #     opponent_ability = sim.pokemon_ability_dict[species][0]
    # else:
    #     opponent_ability = ""

    # if opponent_ability:
    #     try:
    #         ability_name = sim.ability_effect[opponent_ability]["name"]
    #         ability_effect = sim.ability_effect[opponent_ability]["effect"]
    #         opponent_ability = f"{ability_name}({ability_effect})"
    #     except:
    #         pass

    # opponent_prompt = (
    #         f"Opposing active pokemon:{battle.opponent_active_pokemon.species},Type:{opponent_type},HP:{opponent_hp_fraction}%" +
    #         (f"Status:{sim.check_status(opponent_status)}," if sim.check_status(opponent_status) else "") +
    #         (f"Attack:{opponent_stats['atk']}," if opponent_boosts['atk']==0 else f"Attack:{round(opponent_stats['atk'] * sim.boost_multiplier('atk', opponent_boosts['atk']))}({opponent_boosts['atk']} stage boosted),") +
    #         (f"Defense:{opponent_stats['def']}," if opponent_boosts['def']==0 else f"Defense:{round(opponent_stats['def'] * sim.boost_multiplier('def', opponent_boosts['def']))}({opponent_boosts['def']} stage boosted),") +
    #         (f"Special attack:{opponent_stats['spa']}," if opponent_boosts['spa']==0 else f"Special attack:{round(opponent_stats['spa'] * sim.boost_multiplier('spa', opponent_boosts['spa']))}({opponent_boosts['spa']} stage boosted),") +
    #         (f"Special defense:{opponent_stats['spd']}," if opponent_boosts['spd']==0 else f"Special defense:{round(opponent_stats['spd'] * sim.boost_multiplier('spd', opponent_boosts['spd']))}({opponent_boosts['spd']} stage boosted),") +
    #         (f"Speed:{opponent_stats['spe']}," if opponent_boosts['spe'] == 0 else f"Speed:{round(opponent_stats['spe'] * sim.boost_multiplier('spe', opponent_boosts['spe']))}({opponent_boosts['spe']} stage boosted),") +
    #         (f"Ability:{opponent_ability}" if opponent_ability else "")
    # )
    opponent_speed = round(opponent_stats['spe'] * sim.boost_multiplier('spe', opponent_boosts['spe']))

    team_move_type = []
    for move in battle.available_moves:
        if move.base_power > 0:
            team_move_type.append(move.type.name)

    for pokemon in battle.available_switches:
        for move in pokemon.moves.values():
            if move.base_power > 0:
                team_move_type.append(move.type.name)
    
    opponent_prompt = 'Opponent active pokemon:'
    moves_opp_str, moves_opp_possible_str = sim.get_opponent_current_moves(mon=battle.opponent_active_pokemon, return_separate=True)
    moves_opp = [Move(move_opp, sim.gen.gen) for move_opp in moves_opp_str]
    moves_opp_possible = []
    for move_opp in moves_opp_possible_str:
        if move_opp not in moves_opp_str:
            moves_opp_possible.append(Move(move_opp, sim.gen.gen))
    opponent_prompt += get_opp_move_summary(battle.opponent_active_pokemon, moves_opp, moves_opp_possible, battle, sim)

    opponent_move_type_damage_prompt = move_type_damage_wrapper(battle.opponent_active_pokemon, sim.gen.type_chart, team_move_type)

    if opponent_move_type_damage_prompt:
        opponent_prompt = opponent_prompt + opponent_move_type_damage_prompt + "\n"

    # Opponent active pokemon move
    # opponent_move_prompt = ""
    # print('ACTIVE OPPONENT POKEMON FOUND', battle.opponent_active_pokemon, battle.opponent_active_pokemon.moves)
    # if battle.opponent_active_pokemon.moves:
    #     for move_id, opponent_move in battle.opponent_active_pokemon.moves.items():
    #         if opponent_move.base_power == 0:
    #             opponent_move_prompt += f"[{opponent_move.id},{opponent_move.type.name.capitalize()}],"
    #             # continue # only count attack move
    #         else:
    #             if opponent_move.category.name == "SPECIAL":
    #                 opponent_spa = opponent_stats['spa'] * sim.boost_multiplier('spa', opponent_boosts['spa'])
    #                 active_spd = active_stats['spd'] * sim.boost_multiplier('spd', active_boosts['spd'])
    #                 power = round(opponent_spa / active_spd * opponent_move.base_power)

    #             elif opponent_move.category.name == "PHYSICAL":
    #                 opponent_atk = opponent_stats['atk'] * sim.boost_multiplier('atk', opponent_boosts['atk'])
    #                 active_def = active_stats['atk'] * sim.boost_multiplier('atk', active_boosts['atk'])
    #                 power = round(opponent_atk/active_def * opponent_move.base_power)
    #             else:
    #                 power = 0

    #             opponent_move_prompt += f"[{opponent_move.id},{opponent_move.type.name.capitalize()},Power:{power}],"
    #             opponent_type_list.append(opponent_move.type.name)

    # if opponent_move_prompt:
    #     opponent_prompt = opponent_prompt + f"{battle.opponent_active_pokemon.species} used moves:" + opponent_move_prompt

    # possible_move_prompt = ""
    # try:
    #     if species in sim.pokemon_move_dict:
    #         possible_move_list = list(sim.pokemon_move_dict[species].values())
    #         # possible_move_list.sort(key=lambda x: x[3], reverse=True)
    #         for move in possible_move_list:
    #             # if move[2]>0:
    #             possible_move_prompt = possible_move_prompt + f"[{move[0]},{move[1].lower()},Power:{move[2]}],"
    #             opponent_type_list.append(move[1].upper())
    #     else:
    #         possible_move_prompt = ""
    # except:
    #     possible_move_prompt = ""

    # if possible_move_prompt:
    #     opponent_prompt = opponent_prompt + f"{battle.opponent_active_pokemon.species}'s all the possible attacks:" + possible_move_prompt
        
    # opponent possible switches
    opponent_prompt += f"\nOpponent has {opponent_unfainted_num} pokemons left.\n"
    opponent_prompt += 'Seen opponent pokemon:\n'
    for mon_opp in battle.opponent_team.values():
        if mon_opp.fainted or mon_opp.species == battle.opponent_active_pokemon.species: 
            continue
        # moves
        moves_opp_str, moves_opp_possible_str = sim.get_opponent_current_moves(mon=mon_opp, return_separate=True)
        moves_opp = [Move(move_opp, sim.gen.gen) for move_opp in moves_opp_str]
        moves_opp_possible = []
        for move_opp in moves_opp_possible_str:
            if move_opp not in moves_opp_str:
                moves_opp_possible.append(Move(move_opp, sim.gen.gen))
        opponent_prompt += get_opp_move_summary(mon_opp, moves_opp, moves_opp_possible, battle, sim)


    # opponent side conditions
    opponent_side_condition_list = [] # I should add the description for the side condition. and the status.
    for side_condition in battle.opponent_side_conditions:
        opponent_side_condition_list.append(" ".join(side_condition.name.lower().split("_")))

    opponent_side_condition = ",".join(opponent_side_condition_list)
    if opponent_side_condition:
        opponent_prompt = opponent_prompt + "Opponent team's side condition: " + opponent_side_condition

    opponent_prompt += "\n"

    # The active pokemon
    active_hp_fraction = round(battle.active_pokemon.current_hp / battle.active_pokemon.max_hp * 100)
    active_status = battle.active_pokemon.status

    active_type = ""
    if battle.active_pokemon.type_1:
        active_type += battle.active_pokemon.type_1.name.capitalize()
        if battle.active_pokemon.type_2:
            active_type = active_type + " and " + battle.active_pokemon.type_2.name.capitalize()

    active_move_type_damage_prompt = move_type_damage_wrapper(battle.active_pokemon, sim.gen.type_chart, opponent_type_list)
    speed_active_stats = active_stats['spe']
    if speed_active_stats == None: speed_active_stats = 0
    active_speed = round(speed_active_stats*sim.boost_multiplier('spe', active_boosts['spe']))

    try:
        active_ability = sim.ability_effect[battle.active_pokemon.ability]["name"]
        ability_effect = sim.ability_effect[battle.active_pokemon.ability]["effect"]
    except:
        active_ability = battle.active_pokemon.ability
        ability_effect = ""

    # item
    if battle.active_pokemon.item:
        try:
            active_item = sim.item_effect[battle.active_pokemon.item]["name"]
            item_effect = sim.item_effect[battle.active_pokemon.item]["effect"]
            active_item = f"{active_item}({item_effect})"
        except:
            active_item = battle.active_pokemon.item
    else:
        active_item = ""

    
    active_pokemon_prompt = (
        f"Your current pokemon:{battle.active_pokemon.species},Type:{active_type},HP:{active_hp_fraction}%" +
        (f"Status:{sim.check_status(active_status)}," if sim.check_status(active_status) else "" ) +
        (f"Attack:{active_stats['atk']}," if active_boosts['atk']==0 else f"Attack:{round(active_stats['atk']*sim.boost_multiplier('atk', active_boosts['atk']))}({active_boosts['atk']} stage boosted),") +
        (f"Defense:{active_stats['def']}," if active_boosts['def']==0 else f"Defense:{round(active_stats['def']*sim.boost_multiplier('def', active_boosts['def']))}({active_boosts['def']} stage boosted),") +
        (f"Special attack:{active_stats['spa']}," if active_boosts['spa']==0 else f"Special attack:{round(active_stats['spa']*sim.boost_multiplier('spa', active_boosts['spa']))}({active_boosts['spa']} stage boosted),") +
        (f"Special defense:{active_stats['spd']}," if active_boosts['spd']==0 else f"Special defense:{round(active_stats['spd']*sim.boost_multiplier('spd', active_boosts['spd']))}({active_boosts['spd']} stage boosted),") +
        (f"Speed:{active_stats['spe']}" if active_boosts['spe']==0 else f"Speed:{round(active_stats['spe']*sim.boost_multiplier('spe', active_boosts['spe']))}({active_boosts['spe']} stage boosted),") +
        (f"(slower than {battle.opponent_active_pokemon.species})." if active_speed < opponent_speed else f"(faster than {battle.opponent_active_pokemon.species}).") +
        (f"Ability:{active_ability}({ability_effect})," if ability_effect else f"Ability:{active_ability},") +
        (f"Item:{active_item}" if active_item else "")
    )

    if active_move_type_damage_prompt:
        active_pokemon_prompt = active_pokemon_prompt + active_move_type_damage_prompt + "\n"

    side_condition_list = []
    for side_condition in battle.side_conditions:

        side_condition_name = " ".join(side_condition.name.lower().split("_"))
        if side_condition == SideCondition.SPIKES:
            effect = " (cause damage to your pokémon when switch in except flying type)"
        elif side_condition == SideCondition.STEALTH_ROCK:
            effect = " (cause rock-type damage to your pokémon when switch in)"
        elif side_condition == SideCondition.STICKY_WEB:
            effect = " (reduce the speed stat of your pokémon when switch in)"
        elif side_condition == SideCondition.TOXIC_SPIKES:
            effect = " (cause your pokémon toxic when switch in)"
        else:
            effect = ""

        side_condition_name = side_condition_name + effect
        side_condition_list.append(side_condition_name)

    side_condition_prompt = ",".join(side_condition_list)

    if side_condition_prompt:
        active_pokemon_prompt = active_pokemon_prompt + "Your team's side condition: " + side_condition_prompt + "\n"

    # Move
    move_prompt = f"Your {battle.active_pokemon.species} has {len(battle.available_moves)} moves:\n"
    for i, move in enumerate(battle.available_moves):
        try:
            effect = sim.move_effect[move.id]
        except:
            effect = ""

        if move.category.name == "SPECIAL":
            active_spa = active_stats["spa"] * sim.boost_multiplier("spa", active_boosts["spa"])
            opponent_spd = opponent_stats["spd"] * sim.boost_multiplier("spd", active_boosts["spd"])
            power = round(active_spa / opponent_spd * move.base_power)
            move_category = ""
        elif move.category.name == "PHYSICAL":
            active_atk = active_stats["atk"] * sim.boost_multiplier("atk", active_boosts["atk"])
            opponent_def = opponent_stats["def"] * sim.boost_multiplier("def", active_boosts["def"])
            power = round(active_atk / opponent_def * move.base_power)
            move_category = ""
        else:
            move_category = move.category.name.capitalize()
            power = 0

        move_prompt += (f"Move:{move.id},Type:{move.type.name.capitalize()}," +
                        (f"{move_category}-move," if move_category else "") +
                        f"Power:{power},Acc:{round(move.accuracy * sim.boost_multiplier('accuracy', active_boosts['accuracy'])*100)}%"
                        )

        if effect:
            move_prompt += f",Effect:{effect}"
        # whether is effective to the target.
        move_type_damage_prompt = move_type_damage_wrapper(battle.opponent_active_pokemon, sim.gen.type_chart, [move.type.name])
        if move_type_damage_prompt and move.base_power:
            move_prompt += f'({move_type_damage_prompt.split("is ")[-1][:-1]})\n'
        else:
            move_prompt += "\n"

    move_choices = [move.id for move in battle.available_moves]
    action_prompt = f' Your current Pokemon: {battle.active_pokemon.species}.\nChoose only from the following action choices:\n'
    action_prompt_move = ''
    if len(move_choices) > 0:
        action_prompt_move = f"[<move_name>] = {move_choices}\n"

    # TODO: add dynamax option to move_choices, max moves need to be translated


    # Switch
    switch_prompt = f"You have {len(battle.available_switches)} pokemons:\n"

    for i, pokemon in enumerate(battle.available_switches):
        if battle.active_pokemon.species == pokemon.species:
            continue

        type = ""
        if pokemon.type_1:
            type_1 = pokemon.type_1.name
            type += type_1.capitalize()
            if pokemon.type_2:
                type_2 = pokemon.type_2.name
                type = type + " and " + type_2.capitalize()
        if pokemon.max_hp == 0:
            pokemon._max_hp = 1
        hp_fraction = round(pokemon.current_hp / pokemon.max_hp * 100)

        stats = pokemon.stats
        if stats['atk'] is None:
            stats = pokemon.base_stats
        switch_move_prompt = f" Moves:"
        for _, move in pokemon.moves.items():
            if move.base_power == 0:
                switch_move_prompt += f"[{move.id},{move.type.name.capitalize()}],"
            #     continue # only output attack move
            else:
                move_type_damage_prompt = move_type_damage_wrapper(battle.opponent_active_pokemon, sim.gen.type_chart, [move.type.name])
                if "2x" in move_type_damage_prompt:
                    damage_multiplier = "2"
                elif "4x" in move_type_damage_prompt:
                    damage_multiplier = "4"
                elif "0.5x" in move_type_damage_prompt:
                    damage_multiplier = "0.5"
                elif "0.25x" in move_type_damage_prompt:
                    damage_multiplier = "0.25"
                elif "0x" in move_type_damage_prompt:
                    damage_multiplier = "0"
                else:
                    damage_multiplier = "1"

                switch_move_prompt += f"[{move.id},{move.type.name.capitalize()},{damage_multiplier}x damage],"
        # print(switch_move_prompt)
        if stats['spe'] < opponent_speed:
            speed_prompt = f"(slower than {battle.opponent_active_pokemon.species})."
        else:
            speed_prompt = f"(faster than {battle.opponent_active_pokemon.species})."

        switch_prompt += (
                    f"Pokemon:{pokemon.species},Type:{type},HP:{hp_fraction}%," +
                    (f"Status:{sim.check_status(pokemon.status)}, " if sim.check_status(pokemon.status) else "") +
                    f"Attack:{stats['atk']},Defense:{stats['def']},Special attack:{stats['spa']},Special defense:{stats['spd']},Speed:{stats['spe']}"
                    + speed_prompt
                    + switch_move_prompt)
        # print(switch_prompt)
        pokemon_move_type_damage_prompt = move_type_damage_wrapper(pokemon, sim.gen.type_chart, opponent_type_list) # for defense

        if pokemon_move_type_damage_prompt:
            switch_prompt += pokemon_move_type_damage_prompt + "\n"
        else:
            switch_prompt += "\n"

    switch_choices = [pokemon.species for pokemon in battle.available_switches]
    if battle.active_pokemon.species in switch_choices:
        switch_choices.remove(battle.active_pokemon.species)
    action_prompt_switch = ''
    if len(switch_choices) > 0:
        action_prompt_switch = f"[<switch_pokemon_name>] = {switch_choices}\n"

    if battle.active_pokemon.fainted: # passive switching

        system_prompt = (
            f"You are a pokemon battler that targets to win the pokemon battle. Your {battle.active_pokemon.species} just fainted. Choose a suitable pokemon to continue the battle. Here are some tips:"
            " Compare the speeds of your pokemon to the opposing pokemon, which determines who take the move first."
            " Consider the defense state and type-resistance of your pokemon when its speed is lower than the opposing pokemon."
            " Consider the move-type advantage of your pokemon pokemon when its speed is higher than the opposing pokemon."
            f" Player elo is {player_elo}. Player is P2. Opponent elo is {opponent_elo}. Opponent is P1.\n"
            )

        state_prompt = battle_prompt + opponent_prompt + switch_prompt
        state_action_prompt = action_prompt + action_prompt_switch

    else: # take a move or active switch

        system_prompt = (
            "You are a pokemon battler that targets to win the pokemon battle. You can choose to take a move or switch in another pokemon. Here are some battle tips:"
            " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
            " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
            " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
            " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
            " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
            f" Player elo is {player_elo}. Player is P2. Opponent elo is {opponent_elo}. Opponent is P1.\n"
            )

        system_prompt = system_prompt + sim.strategy

        state_prompt = battle_prompt + opponent_prompt + active_pokemon_prompt + move_prompt + switch_prompt
        state_action_prompt = action_prompt + action_prompt_move + action_prompt_switch

    
    if return_actions:
        return system_prompt, state_prompt, action_prompt, action_prompt_switch, action_prompt_move
    if return_choices:
        return system_prompt, state_prompt, action_prompt, switch_choices, move_choices
    
    return system_prompt, state_prompt, state_action_prompt
