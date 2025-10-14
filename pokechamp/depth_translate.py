import json
import string

import numpy as np

from poke_env.environment.move import Move
from pokechamp.llm_player import LLMPlayer
from poke_env.player.local_simulation import LocalSim
from pokechamp.prompts import get_avail_actions, prompt_translate as pt
from poke_env.ps_client.account_configuration import AccountConfiguration
import ast

def recursive_nick_removal(text, start=0):
    for i in range(start, len(text)):
        line = text[i]
        if '|switch|' in line or '|drag|' in line:
            mons = line.split('a: ')[1].split(', ')[0]
            # print(mons, line.strip())
            nickname, mon = mons.split('|')[:2]
            if nickname != mon and len(nickname) > 1:
                player_id = ': '
                text = [t.replace(player_id + nickname, player_id + mon) for t in text]
                return recursive_nick_removal(text, start=i+1)
            elif nickname != mon:
                player_id = line.split('|')[2][:3]
                text = [t.replace(player_id + nickname, player_id + mon) for t in text]
                return recursive_nick_removal(text, start=i+1)
    return text

def get_player_team(battle_turns_text, player_id):
    # reverse engineer player team
    team_player = []
    team_mons = []
    for turn in battle_turns_text:
        split_messages = [m.replace('\n','').split("|") for m in turn]
        for m in split_messages:
            # find any switches for player
            if len(m) < 2: continue
            if m[1] == '': continue
            if 'switch' in m[1] and f'{player_id}a' in m[2]:
                mon = m[3]
                if mon not in team_mons:
                    team_mons.append(mon)
                    # |poke|p1|Dragapult, M|
                    mon_str = f'|poke|{player_id}|{mon}'
                    team_player.append(mon_str)
    return team_player, team_mons


async def data_battle(llm: LLMPlayer, file, format, battle_id, json_text, gen, prompt_translate=pt):
    # load data file and split data into turns
    # with open('scraper/data_all/gen8randombattle/1044473102.txt', 'r') as f:
    print(f'adding {file}')
    with open(file, 'r') as f:
        battle_text = f.readlines()
    for line in battle_text:
        if 'stellar' in line:   # this should be normal type for terapagos
            return False
    # type: null correction (parser cant read : in middle of pokemon name)
    for line in battle_text:
        if 'type: null' in line:
            battle_text = [t.replace('type: null', 'typenull') for t in battle_text]
            break
    battle_text = recursive_nick_removal(battle_text)
    # print(battle_text)

    battle_turns_text = []
    turn_text = []
    p1_username = ''
    p2_username = ''
    winner_username = ''
    for line in battle_text:
        if 'zoroark' in line.lower(): return False
        if '|c|' in line: continue
        if '|raw|' in line: continue
        
        # parse p1, p2 username
        if 'player|p1' in line:
            if p1_username != '': continue
            line_split = line.split('|')
            p1_username = line_split[3]
        if 'player|p2' in line:
            if p2_username != '': continue
            line_split = line.split('|')
            p2_username = line_split[3]

        if '|faint|' in line:
            battle_turns_text.append(turn_text)
            turn_text = []

        turn_text.append(line)
        if '|turn|' in line:
            battle_turns_text.append(turn_text)
            turn_text = []

        if '|win|' in line:
            line_split = line.split('|')
            winner_username = line_split[2].rstrip()
            # print(f'winner {winner_username}')
            # print(winner_username==p1_username, winner_username==p2_username)
            # print(p1_username, p2_username)\
            assert winner_username==p1_username or winner_username==p2_username
    # do not use tie data
    if len(battle_turns_text) < 2: return False
    if not (winner_username==p1_username or winner_username==p2_username): return False
    # set username to be p1 or p2 username
    for username in [winner_username]:
        if username == p1_username:
            player_id = 'p1'
            opponent_id = 'p2'
            winner = 'p1'
        else:
            player_id = 'p2'
            opponent_id = 'p1'
            winner = 'p2'
        # if np.random.random() < 0.5:
        #     player_id, opponent_id = 'p1', 'p2'
        # else:
        #     player_id, opponent_id = 'p2', 'p1'
        print(f'player is {username}')

        # reverse engineer player team
        team_player, team_mons = get_player_team(battle_turns_text, player_id)

        # reverse engineer pokemon moves on player's team
        moves = {}
        for mon in team_mons:
            mon = mon.split(',')[0]
            moves[mon] = []
            for turn in battle_turns_text:
                split_messages = [m.replace('\n','').split("|") for m in turn]
                for m in split_messages:
                    if len(m) < 2: continue
                    if m[1] == '': continue
                    if 'move' not in m[1]: continue
                    if f'{player_id}a' not in m[2]: continue
                    if mon not in m[2]: continue
                    move = m[3]
                    if move not in moves[mon]:
                        moves[mon] = moves[mon] + [move]
        moves_parsed = {}
        for mon in moves.keys():
            moves_parsed[mon.replace(' ', '').translate(str.maketrans('', '', string.punctuation))] = [move.replace(' ', '').translate(str.maketrans('', '', string.punctuation)) for move in moves[mon]]
        # print(moves_parsed)
        if 'random' in format:
            start_line = -1
            for j, line in enumerate(battle_turns_text[0]):
                if '|start' in line:
                    start_line = j
            assert start_line != -1
            for species, mon in zip(team_mons, team_player):
                mon_species = species.split(',')[0]
                for move in moves[mon_species]:
                    move_str = f'|premove|p1a: {mon_species}|{move}'
                    battle_turns_text[0].insert(start_line, move_str)
                battle_turns_text[0].insert(start_line, mon)
            battle_turns_text[0].insert(start_line, '|teampreview')

        # load battle sim
        # create player
        llm_player = LLMPlayer(battle_format=format,
                                    backend='gpt',
                                    temperature=1.0,
                                    prompt_algo='io',
                                    account_configuration=AccountConfiguration(username, ''),
                                    prompt_translate=prompt_translate,
                                    save_replays=False,
                                    )
        llm_player._dynamax_disable = False
        # create battle object
        battle_info = f'>battle-{format}-{battle_id}'.split("-")
        battle = await llm_player._create_battle(battle_info)

        # create simulator
        sim = LocalSim(battle,
                    llm_player.move_effect,
                    llm_player.pokemon_move_dict,
                    llm_player.ability_effect,
                    llm_player.pokemon_ability_dict,
                    llm_player.item_effect,
                    llm_player.pokemon_item_dict,
                    llm_player.gen,
                    llm_player._dynamax_disable,
                    format=llm_player.format,
                    prompt_translate=llm_player.prompt_translate)

        # send turn to battle sim: update simulator with each message from the data
        # [['>battle-gen8randombattle-54418'], ['', 'init', 'battle'], ['', 'title', 'SimpleHeuristics 7 vs. llamaio6'], ['', 'j', '☆SimpleHeuristics 7'], ['']]
        # ['>battle', 'gen8randombattle', '54418']
        player_prompts = []
        player_actions = []
        opponent_prompts = []
        opponent_actions = []
        both_players_actions = []
        world_models = []
        correct_player = 0
        correct_opponent = 0
        total_player = 0
        total_opponent = 0
        for turn, message in enumerate(battle_turns_text):
            split_messages = [m.replace('\n','').split("|") for m in message]
            if turn == 0:
                for msg in split_messages:
                    if len(msg) < 2: continue
                    if msg[1] == '': continue
                    try: 
                        try:
                            sim._handle_battle_message(msg)
                        except (KeyError, ValueError):
                            return False
                    except NotImplementedError:
                        continue
                continue

            # grab player and opponent action
            player_action = None
            opponent_action = None
            opponent_action_unformatted = None
            turn_zero_flag = False
            for m in split_messages:
                if len(m) < 2: continue
                if 'move' in m[1]:
                    move = m[3].replace(' ', '').translate(str.maketrans('', '', string.punctuation))
                    if f'{player_id}a' in m[2]:
                        player_action = f'{{"move":"{move}"}}'
                    elif f'{opponent_id}a' in m[2]:
                        opponent_action = f'{{"move":"{move}"}}'
                        opponent_action_unformatted = move
                elif 'switch' in m[1]:
                    move = m[2].split(':')[1].replace(' ', '').translate(str.maketrans('', '', string.punctuation))
                    if f'{player_id}a' in m[2]:
                        player_action = f'{{"switch":"{move}"}}'
                    elif f'{opponent_id}a' in m[2]:
                        opponent_action = f'{{"switch":"{move}"}}'
                        opponent_action_unformatted = move
            # 'cant' when both are affected by status
            # assert player_action is not None or opponent_action is not None, print(split_messages)    
            if player_action == None: 
                for msg in split_messages:
                    if len(msg) < 2: continue
                    if msg[1] == '': continue
                    try: 
                        try:
                            sim._handle_battle_message(msg)
                        except (KeyError, ValueError):
                            return False
                    except NotImplementedError:
                        continue
                continue
            if opponent_action == None: opponent_action = 'None'
            player_actions.append(player_action)
            opponent_actions.append(opponent_action)
            both_players_action = ''
            if opponent_action == 'None':
                both_players_action = player_action[:-1] + ',"opponent":"None"}'      
            else:
                both_players_action = player_action[:-1] + ',"opponent":"' + opponent_action_unformatted + '"}'
            both_players_actions.append(both_players_action)
            
            # add available moves to current pokemon
            # print(player_action)
            # print(sim.battle.active_pokemon.is_dynamaxed)
            # if 'move' in player_action:
            # current_moves = [move.id for move in sim.battle.available_moves]
            if sim.battle.active_pokemon.species in moves_parsed.keys():
                move_list = moves_parsed[sim.battle.active_pokemon.species]
                sim.battle._available_moves = [Move(move, gen=llm_player.gen.gen) for move in move_list]

            # create player prompt from battle sim
            # print(split_messages)
            try:
                system_prompt, state_prompt, state_action_prompt = sim.state_translate(sim.battle)
            except Exception as e:
                print("Battle not saved", e)
                return False
            
            # print(sim.battle.active_pokemon.fainted, len(sim.battle.available_moves))
            if sim.battle.active_pokemon.fainted or len(sim.battle.available_moves) == 0:
                constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            elif len(sim.battle.available_switches) == 0:
                constraint_prompt_io = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"}\n'''
            else:
                # print('correct move')
                constraint_prompt_io = '''Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"} or {"switch":"<switch_pokemon_name>"}\n'''
            # constraint_prompt_io = 'Choose the top 2 actions. Use the JSON format: {\"action_1\": \"<action>\", \"action_2\": \"<action>\"} and replace \"<action>\" with your action choice from the list [<action>].'
            # constraint_prompt_io = 'Choose the top 2 actions (action_1 AND action_2). Your output MUST be a JSON in the format: {"action_1": "<move_or_switch_pokemon_name>", "action_2": "<move_or_switch_pokemon_name>"}'
            # print(state_prompt, constraint_prompt_io)
            state_prompt_io = system_prompt + state_prompt + state_action_prompt + constraint_prompt_io
            player_prompts.append(state_prompt_io)

            # create opponent prompt from battle sim
            _, state_prompt_o, _, constraint_prompt_io_o, state_action_prompt_o = sim.get_opponent_prompt(state_prompt)
            state_prompt_o = system_prompt + state_prompt_o + state_action_prompt_o + constraint_prompt_io_o
            opponent_prompts.append(state_prompt_o)
            
            
            ###############################################
            # handle llm output to see if we find a match #
            ###############################################
            output, output_opp = llm.tree_search_minimax(5, sim.battle, sim=sim, return_opp=True)
            print(output, output_opp)
            player_action = json.loads(player_action)
            if 'move' in player_action.keys():
                player_action = player_action['move']
            elif 'switch' in player_action.keys():
                player_action = player_action['switch']
            if opponent_action is not None and opponent_action != 'None':
                opponent_action = json.loads(opponent_action)
                if 'move' in opponent_action.keys():
                    opponent_action = opponent_action['move']
                elif 'switch' in opponent_action.keys():
                    opponent_action = opponent_action['switch']
            print(output.message, 'p vs', player_action)
            print(output_opp.message, 'o vs', opponent_action)
            if output.message.split(' ')[-1] == player_action:
                correct_player += 1
            if output_opp.message.split(' ')[-1] == opponent_action:
                correct_opponent += 1
                total_opponent += 1
            elif 'None' not in opponent_action and opponent_action is not None:
                total_opponent += 1
            total_player += 1
            print(f'Player: {correct_player/max(1,total_player)*100:.2f}%')
            print(f'Opponent: {correct_opponent/max(1,total_opponent)*100:.2f}%')

            for msg in split_messages:
                if len(msg) < 2: continue
                if msg[1] == '': continue
                try: 
                    try:
                        sim._handle_battle_message(msg)
                    except (KeyError, ValueError) as e:
                        return False
                except NotImplementedError:
                    continue
            
            # update health AFTER action taken
            world_models.append(sim.get_all_hp())

    return correct_player, total_player, correct_opponent, total_opponent
