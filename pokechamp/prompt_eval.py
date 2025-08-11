import asyncio
import json
import os
import orjson
import numpy as np
from tqdm import tqdm

from poke_env.player.baselines import MaxBasePowerPlayer, Human, OneStepPlayer, AbyssalPlayer
from pokechamp.depth_translate import data_battle
from pokechamp.llm_player import LLMPlayer
from poke_env.player.player import Player
from pokechamp.prompts import prompt_translate, state_translate
# from pokechamp.translate import add_battle
from poke_env.ps_client.account_configuration import AccountConfiguration

system_prompt = (
                "You are a pokemon battler that targets to win the pokemon battle. You can choose to take a move or switch in another pokemon. Here are some battle tips:"
                " Use status-boosting moves like swordsdance, calmmind, dragondance, nastyplot strategically. The boosting will be reset when pokemon switch out."
                " Set traps like stickyweb, spikes, toxicspikes, stealthrock strategically."
                " When face to a opponent is boosting or has already boosted its attack/special attack/speed, knock it out as soon as possible, even sacrificing your pokemon."
                " if choose to switch, you forfeit to take a move this turn and the opposing pokemon will definitely move first. Therefore, you should pay attention to speed, type-resistance and defense of your switch-in pokemon to bear the damage from the opposing pokemon."
                " And If the switch-in pokemon has a slower speed then the opposing pokemon, the opposing pokemon will move twice continuously."
                )


def eval_action_player(llm: LLMPlayer, gen: int=9, format: str='randombattle'):
    '''
    Action Evaluation: Player
    '''
    format_str = 'ou'
    if 'randombattle' in format:
        format_str = 'rb'
    file = f'train/data/augmented_train_gen{gen}{format_str}_singles_1800_state1.json'
    with open(file, 'r') as f:
        data = orjson.loads(f.read())
    num_points = len(data)
    random_ints = np.random.randint(0, num_points, size=1000)
    correct_moves = 0
    total_moves = 0
    pbar = tqdm(random_ints, desc=f"Correct moves: {correct_moves}/{total_moves}")
    for i in pbar:
        prompt = data[i]['instruction']
        action = orjson.loads(data[i]['output'])
        if 'move' in action.keys():
            action = action['move']
        else:
            action = action['switch']
        moves = []
        switches = []
        for line in prompt.split('\n'):
            if '[<move_name>]' in line:
                # Extract the part of the string that contains the moves
                moves_str = line.split(" = ")[1]
                # Remove the brackets and split the string to get the list of moves
                moves = moves_str.strip("[]").replace("'", "").split(", ")
            elif '[<switch_pokemon_name>]' in line:
                # Extract the part of the string that contains the switches
                switch_str = line.split(" = ")[1]
                # Remove the brackets and split the string to get the list of switches
                switches = switch_str.strip("[]").replace("'", "").split(", ")
        action_choices = [moves, switches]

        # get llm action
        output = llm.get_LLM_action(system_prompt, prompt, '', temperature=0.7, json_format=True, seed=None, stop=[], max_tokens=20, actions=action_choices)
        try:
            output = orjson.loads(output)
        except:
            output = llm.get_LLM_action(system_prompt, prompt, '', temperature=0.7, json_format=True, seed=None, stop=[], max_tokens=20, actions=action_choices)
            try:
                output = orjson.loads(output)
            except:
                print("invalid json", output)
                output = {}
                continue
        if 'move' in output.keys():
            output = output['move'].strip()
        elif 'switch' in output.keys():
            output = output['switch'].strip()
        else:
            print("action not found", output)
            output = None
            continue
        # check correctness
        if output == action:
            correct_moves += 1
        total_moves += 1

        # Update tqdm description
        pbar.set_description(f"Correct moves: {correct_moves/total_moves*100:.2f}% = {correct_moves}/{total_moves}. Expected: {action} vs. Output: {output}")
    return


def get_loadout_str(data, mon):
    '''
    Creates a random loadout based on statistical sets.
    '''    
    def get_weighted_choice(category, id, size=1):
        category_dict = data[mon][category]
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
    
    output = f"{mon.lower()} @ {get_weighted_choice('items', 'name')}\n"
    output += f"Ability: {get_weighted_choice('abilities', 'name')}\n"
    output += f"Tera Type: Normal\n"    # @TODO: add this to sets
    spread, nature = get_weighted_choice('spreads', 'stats')
    labels = ['HP', 'Atk', 'Def', 'SpA', 'SpD', 'Spe']
    evs = []
    for j in range(len(spread)):
        if spread[j] != 0:
            evs.append(f'{spread[j]} {labels[j]}')
    output += f'EVs: '
    for j in range(len(evs)):
        output += f'{evs[j]}'
        if j != len(evs)-1:
            output += ' / '
    output += '\n'
    output += f'{nature} Nature\n'
    moves = get_weighted_choice('moves', 'name', size=4)
    for i in range(len(moves)):
        if len(moves[i]) != 0:
            output += f'- {moves[i]}\n'
    return output

async def hand_benchmark(args, PNUMBER1, total=1):
    # find 1v1 mons
    file = f'poke_env/data/static/gen9/ou/sets_1500.json'
    with open(file, 'r') as f:
        data = orjson.loads(f.read())
    available_mons = list(data.keys())
    total_mons = len(available_mons)
    
    t = 0
    i = 0
    output = 'one_vs_one.json'
    player_wins = 0
    bot_wins = 0
    opp_indices = np.random.randint(0, total_mons, size=10000)
    player_indices = np.random.randint(0, total_mons, size=10000)
    teams = []
    while t < 500 and i < 10000:
        # p1
        opponent = available_mons[opp_indices[i]]
        team1_str = get_loadout_str(data, opponent)
        # p2
        player_team = available_mons[player_indices[i]]
        team2_str = get_loadout_str(data, player_team)
        if i == 0:
            bot = MaxBasePowerPlayer(battle_format=args.battle_format,
                                    account_configuration=AccountConfiguration(f'bot2000{PNUMBER1}', ''),
                                    team=team1_str
                                    )
            player = AbyssalPlayer(battle_format=args.battle_format,
                            account_configuration=AccountConfiguration(f'ChiJin{PNUMBER1}', ''),
                            team=team2_str
                            )
             # dynamax is disabled for local battles.
            bot._dynamax_disable = True
            player._dynamax_disable = True
        
        bot.update_team(team1_str)
        player.update_team(team2_str)
        await player.battle_against(bot, n_battles=1)
        if player.n_won_battles > player_wins:
            # save team
            teams.append([team2_str, team1_str])
            # teams = team2_str + '---\n' + team1_str + '+++\n'
            # with open(output, 'a+') as f:
            #     f.write(teams)
            t += 1
            print(f'wins {t}')
            player_wins = player.n_won_battles
        with open(output, 'wb') as f:
                f.write(orjson.dumps(teams, option=orjson.OPT_INDENT_2))
        i += 1
        
async def offline_one_vs_one(args, PNUMBER1,):
    print('classical prompts')
    file = f'one_vs_one.json'
    with open(file, 'r') as f:
        data = orjson.loads(f.read())
    p2_wins = 0
    pbar = tqdm(range(len(data)), desc=f"Winrate: ")
    for i in pbar:
        team1_str = data[i][1]
        team2_str = data[i][0]
        if i == 0:
            # create player if init
            p1 = MaxBasePowerPlayer(battle_format=args.battle_format,
                                account_configuration=AccountConfiguration(f'bot2000{PNUMBER1}', ''),
                                team=team1_str
                                )
            p2 = LLMPlayer(battle_format=args.battle_format,
                           backend=args.backend,
                           temperature=args.temperature,
                           prompt_algo=args.prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration(f'llama_strat{PNUMBER1}', ''),
                           save_replays=args.log_dir,
                           team=team2_str,
                           _use_strat_prompt=False,
                           port=args.port,
                        #    prompt_translate=prompt_translate,
                           device=args.device,
                           )
            # p2 = MaxBasePowerPlayer(battle_format=args.battle_format,
            #                     account_configuration=AccountConfiguration(f'bot2001{PNUMBER1}', ''),
            #                     team=team2_str
            #                     )
            # p2 = MaxDmgCalcPlayer(battle_format=args.battle_format,
            #                     account_configuration=AccountConfiguration(f'dmgcalc{PNUMBER1}', ''),
            #                     team=team2_str
            #                     )
            
            p1._dynamax_disable = True
            p2._dynamax_disable = True
        else:
            # load new teams
            p1.update_team(team1_str)
            p2.update_team(team2_str)
        await p2.battle_against(p1, n_battles=1)
        pbar.set_description(f"Winrate: {p2.win_rate*100:.2f}.")
        # update counts
        if p2.n_won_battles > p2_wins:
            p2_wins = p2.n_won_battles
        

async def one_vs_one(args, PNUMBER1, total=1, compare_bots=False):
    '''
    1v1 Eval
    '''
    # find 1v1 mons
    file = f'poke_env/data/static/gen9/ou/sets_1500.json'
    with open(file, 'r') as f:
        data = orjson.loads(f.read())
    available_mons = list(data.keys())
    total_mons = len(available_mons)
    for i in range(total_mons):
        assert len(get_loadout_str(data, available_mons[i])) != 0
    pbar = tqdm(range(total*2), desc=f"Winrate: ")
    opp_indices = np.random.randint(0, total_mons, size=total*2)
    player_indices = np.random.randint(0, total_mons, size=total*2)
    missed_wins = 0
    p2_bot_wins = 0
    p2_wins = 0
    p3_wins = 0
    missed_wins_p3 = 0
    for i in pbar:
        # create loadouts
        # p1
        opponent = available_mons[opp_indices[i]]
        team1_str = get_loadout_str(data, opponent)
        print('opp mon:', opponent)
        print(team1_str)
        # p2
        player = available_mons[player_indices[i]]
        # player = 'breloom'
        team2_str = get_loadout_str(data, player)
        print('player mon:', player)
        print(team2_str)
        if i == 0:
            # create player if init
            p1 = MaxBasePowerPlayer(battle_format=args.battle_format,
                                account_configuration=AccountConfiguration(f'bot2000{PNUMBER1}', ''),
                                team=team1_str
                                )
            p2_bot = MaxBasePowerPlayer(battle_format=args.battle_format,
                                account_configuration=AccountConfiguration(f'bot3000{PNUMBER1}', ''),
                                team=team2_str
                                )
            p2 = LLMPlayer(battle_format=args.battle_format,
                           backend=args.backend,
                           temperature=args.temperature,
                           prompt_algo=args.prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration(f'llama_strat{PNUMBER1}', ''),
                           save_replays=args.log_dir,
                           team=team2_str,
                           _use_strat_prompt=False,
                           port=args.port,
                           prompt_translate=prompt_translate,
                           device=args.device,
                           )
            if compare_bots:
                p3 = AbyssalPlayer(battle_format=args.battle_format,
                            account_configuration=AccountConfiguration(f'heuristic{PNUMBER1}', ''),
                            team=team2_str
                            )
                # p3 = LLMPlayer(battle_format=args.battle_format,
                #             backend=args.backend,
                #             temperature=args.temperature,
                #             prompt_algo=args.prompt_algo,
                #             log_dir=args.log_dir,
                #             account_configuration=AccountConfiguration(f'pokellmon{PNUMBER1}', ''),
                #             save_replays=args.log_dir,
                #             team=team2_str,
                #             _use_strat_prompt=False,
                #             port=args.port,
                #             device=args.device+1,
                #             )
             # dynamax is disabled for local battles.
            p1._dynamax_disable = True
            p2._dynamax_disable = True
            p2_bot._dynamax_disable = True
            if compare_bots:
                p3._dynamax_disable = True
        else:
            # load new teams
            p1.update_team(team1_str)
            p2.update_team(team2_str)
            p2_bot.update_team(team2_str)
            if compare_bots:
                p3.update_team(team2_str)
        # battle
        if compare_bots:
            await p3.battle_against(p1, n_battles=1)
            if not p3.n_won_battles > p3_wins:
                # if heuristic bot doesnt win, dont play
                p3_wins = p3.n_won_battles
                continue
            if p2_bot.n_won_battles > p2_bot_wins and not p3.n_won_battles > p3_wins:
                missed_wins_p3 += 1
                
        await p2.battle_against(p1, n_battles=1)
        await p2_bot.battle_against(p1, n_battles=1)
        # collect winner stats
        if p2_bot.n_won_battles > p2_bot_wins and not p2.n_won_battles > p2_wins:
            missed_wins += 1
        if compare_bots:
            pbar.set_description(f"Winrate: {p2.win_rate*100:.2f}. Missed Wins: {missed_wins}. Bot wins {p2_bot.win_rate*100:.2f} heuristic: {p3.win_rate*100:.2f} m: {missed_wins_p3}")
        else:
            pbar.set_description(f"Winrate: {p2.win_rate*100:.2f}. Missed Wins: {missed_wins}. Bot wins {p2_bot.win_rate*100:.2f}")
        # update counts
        if p2_bot.n_won_battles > p2_bot_wins:
            p2_bot_wins = p2_bot.n_won_battles
        if p2.n_won_battles > p2_wins:
            p2_wins = p2.n_won_battles
        if compare_bots:
            if p3.n_won_battles > p3_wins:
                p3_wins = p3.n_won_battles
    return p2

def send_to_llm2(llm: LLMPlayer, 
                system_prompt: str, 
                prompt: str, 
                action: str
                ) -> int:
    # get llm action
    # print(prompt)
    output = llm.get_LLM_action(system_prompt, prompt, '', temperature=0.3, json_format=True, seed=None, stop=[], max_tokens=100)
    print(output)
    try:
        output = orjson.loads(output)
    except:
        output = llm.get_LLM_action(system_prompt, prompt, '', temperature=0.3, json_format=True, seed=None, stop=[], max_tokens=100)
        try:
            output = orjson.loads(output)
        except:
            print("invalid json", output)
            output = {}
            return 0, 0
    if 'move' in output.keys():
        output_player = output['move'].strip()
    elif 'switch' in output.keys():
        output_player = output['switch'].strip()
    else:
        print("action not found", output)
        output_player = None
        return 0, 0
    if 'opponent' in output.keys():
        output2 = output['opponent'].strip()
    # check correctness
    print(output_player, action, 'this is the action')
    correct_player = 0
    correct_opponent = 0
    action_ground_truth = list(orjson.loads(action).values())
    print(action_ground_truth)
    if output_player == action_ground_truth[0]:
        correct_player = 1
    if output2 == action_ground_truth[1]:
        correct_opponent = 1
    return correct_player, correct_opponent

def send_to_llm(llm: LLMPlayer, 
                system_prompt: str, 
                prompt: str, 
                action: str,
                action_list: list[str],
                name: str,
                K: int=5,
                ) -> int:
    outputs = llm.llm.get_LLM_action_topK(system_prompt, prompt, '', actions=action_list, temperature=0.3, json_format=True, seed=None, stop=[], max_tokens=10, top_k=K)
    print(name, 'this is the action', action)
    print(outputs)
    for i, output in enumerate(outputs):
        try:
            output = orjson.loads(output)
        except:
            print("invalid json", output)
            output = {}
            continue
        if 'move' in output.keys():
            output = output['move'].strip()
        elif 'switch' in output.keys():
            output = output['switch'].strip()
        else:
            output = None
            continue
        # check correctness
        if output in orjson.loads(action).values():
            print('Found correct')
            return [0] * (i) + [1] * (K-i)
        print(output, orjson.loads(action).values())
    return [0] * K

def send_to_llm_po(llm: LLMPlayer, 
                system_prompt: str, 
                prompt: str, 
                action: str,
                action_opp: str,
                action_list: list[str],
                action_opp_list: list[str],
                name: str,
                K: int=5,
                ) -> int:
    opponent_action_text = json.loads(action_opp)['opponent']
    action_opp_list.append(opponent_action_text)
    outputs, outputs_opp = llm.llm.get_LLM_action_topK(system_prompt, prompt, '', actions=action_list, actions_opp=action_opp_list, temperature=0.3, json_format=True, seed=None, stop=[], max_tokens=100, top_k=K)
    print(name, 'this is the action', action, action_opp)
    print(outputs, outputs_opp)
    
    # check correctness
    correct_player = K
    correct_opponent = K
    for i in range(len(outputs)):
        output = outputs[i]
        try:
            output = orjson.loads(output)
        except:
            print("invalid json", output)
            output = {}
            continue
        if 'move' in output.keys():
            output_player = output['move'].strip()
        elif 'switch' in output.keys():
            output_player = output['switch'].strip()
        else:
            print("action not found", output)
            output_player = ModuleNotFoundError
        action_ground_truth = list(orjson.loads(action).values())
        if output_player == action_ground_truth[0] and correct_player == K:
            correct_player = i
    if 'None' in action_opp:
        return [0] * (correct_player) + [1] * (K-correct_player), None
    for i in range(len(outputs_opp)):
        output_opp = outputs_opp[i]
        try:
            output_opp = orjson.loads(output_opp)
        except:
            print("invalid json", output)
            output = {}
            continue
        if 'opponent' in output_opp.keys():
            output2 = output_opp['opponent'].strip()
        else:
            output2 = None
            # print('error')
        # print(action_opp)
        action_ground_truth_opp = list(orjson.loads(action_opp).values())
        if output2 == action_ground_truth_opp[0] and correct_opponent == K:
            correct_opponent = i
    return [0] * (correct_player) + [1] * (K-correct_player), [0] * (correct_opponent) + [1] * (K-correct_opponent)

async def eval_tree(llm: LLMPlayer, gen: int=9, format: str='ou', elo_str: str='1800+'):
    '''
    Eval tree: determine if llm tree search predicted action (and opponent action at d=0) matches human actions.
    '''
    base_dir = 'filter/data'
    format_dir = f'{base_dir}/gen{gen}{format}/{elo_str}'.lower().replace(' ', '')
    battles = os.listdir(format_dir)
    np.random.shuffle(battles)
    pbar = tqdm(battles[:10], desc="")
    correct_player, total_player, correct_opponent, total_opponent = 0,0,0,0
    for format_path, _ in zip(pbar, range(10)):
        file = f'{format_dir}/{format_path}'
        battle_id = format_path.split('/')[-1].split('.')[0]
        format_name = f'gen{gen}{format}'.lower().replace(' ','')
        json_text = []
        out = await data_battle(llm, file, format_name, battle_id, json_text, gen, prompt_translate=llm.prompt_translate)
        if out:
            _correct_player, _total_player, _correct_opponent, _total_opponent = out
            correct_player += _correct_player
            total_player += _total_player
            correct_opponent += _correct_opponent
            total_opponent += _total_opponent
        pbar.set_description(f"player moves:{correct_player/max(1,total_player)*100:.2f}. Opponent moves: {correct_opponent/max(1,total_opponent)*100:.2f}")
    return

async def eval_actions(llm: LLMPlayer, gen: int=9, format: str='ou', elo_str: str='1800+'):
    '''
    Eval actions: performs player and opponent move prediction based on historical data at an elo.
    '''
    base_dir = 'filter/data'
    format_dir = f'{base_dir}/gen{gen}{format}/{elo_str}'.lower().replace(' ', '')
    K=5
    total_moves = 0
    correct_moves_player = [0]*K
    correct_moves_opponent = [0]*K
    total_moves_opp = 0
    battles = os.listdir(format_dir)
    np.random.shuffle(battles)
    pbar = tqdm(battles[:10], desc="")
    for format_path, _ in zip(pbar, range(10)):
        file = f'{format_dir}/{format_path}'
        battle_id = format_path.split('/')[-1].split('.')[0]
        format_name = f'gen{gen}{format}'.lower().replace(' ','')
        json_text = []
        out = await add_battle(file, format_name, battle_id, json_text, gen, prompt_translate=llm.prompt_translate)
        if out:
            for player_text in json_text:
                total_moves += 1
                player_prompt = player_text['instruction']
                # opponent_prompt = player_text['instruction_opp']
                player_action = player_text['output']
                # opponent_action = player_text['output_opp']
                opponent_action_text = json.loads(player_text['output'])['opponent']
                opponent_action = '{"opponent":"' + opponent_action_text + '"}'
                # get llm action - player
                correct_ply, correct_opp = send_to_llm_po(llm, '', player_prompt, player_action, opponent_action, player_text['actions'], player_text['actions_opp'], 'player', K=K)
                # correct_ply = send_to_llm(llm, '', player_prompt, player_action, player_text['actions'], 'player', K=K)
                for i in range(K):
                    correct_moves_player[i] += correct_ply[i]
                if correct_opp is not None:
                    for i in range(K):
                        correct_moves_opponent[i] += correct_opp[i]
                    total_moves_opp += 1
                # correct_moves_opponent += correct_opp           
                # get llm action - player
                # if 'None' not in opponent_action:
                #     total_moves_opp += 1
                #     correct_opp = send_to_llm(llm, '', opponent_prompt, opponent_action, player_text['actions_opp'], 'opponent', K=K)       
                #     for i in range(K):
                #         correct_moves_opponent[i] += correct_opp[i]
                percent_ply = [round(correct / max(total_moves,1), 2) for correct in correct_moves_player]
                percent_opp = [round(correct / max(total_moves_opp,1), 2) for correct in correct_moves_opponent]
                pbar.set_description(f"player moves:{percent_ply}. Opponent moves: {percent_opp}")
                # pbar.set_description(f"player moves: {correct_moves_player}/{total_moves}={correct_moves_player/max(total_moves,1)*100:.2f}%. Opponent moves: {correct_moves_opponent}/{total_moves_opp}={correct_moves_opponent/max(total_moves_opp,1)*100:.2f}%")
    return


async def eval_actions_tree(llm: LLMPlayer, gen: int=9, format: str='ou', elo_str: str='1800+'):
    '''
    Eval actions: performs player and opponent move prediction based on historical data at an elo.
    '''
    base_dir = 'filter/data'
    format_dir = f'{base_dir}/gen{gen}{format}/{elo_str}'.lower().replace(' ', '')
    K=5
    total_moves = 0
    correct_moves_player = [0]*K
    correct_moves_opponent = [0]*K
    total_moves_opp = 0
    battles = os.listdir(format_dir)
    np.random.shuffle(battles)
    # pbar = tqdm(battles[:10], desc=f"player moves: {correct_moves_player}/{total_moves}={correct_moves_player/max(total_moves,1)*100:.2f}%. Opponent moves: {correct_moves_opponent}/{total_moves_opp}={correct_moves_opponent/max(total_moves_opp,1)*100:.2f}%")
    pbar = tqdm(battles[:10], desc='')
    for format_path, _ in zip(pbar, range(10)):
        file = f'{format_dir}/{format_path}'
        battle_id = format_path.split('/')[-1].split('.')[0]
        format_name = f'gen{gen}{format}'.lower().replace(' ','')
        json_text = []
        out = await add_battle(file, format_name, battle_id, json_text, gen, prompt_translate=llm.prompt_translate)
        if out:
            for player_text in json_text:
                total_moves += 1
                player_prompt = player_text['instruction']
                opponent_prompt = player_text['instruction_opp']
                player_action = player_text['output']
                opponent_action = player_text['output_opp']
                # get llm action - player
                # correct_ply, correct_opp = send_to_llm2(llm, '', player_prompt, player_action)
                correct_ply = send_to_llm(llm, '', player_prompt, player_action, player_text['actions'], 'player', K=K)
                for i in range(K):
                    correct_moves_player[i] += correct_ply[i]
                # correct_moves_player += correct_ply
                # correct_moves_opponent += correct_opp           
                # total_moves_opp += 1
                # get llm action - player
                if 'None' not in opponent_action:
                    total_moves_opp += 1
                    correct_opp = send_to_llm(llm, '', opponent_prompt, opponent_action, player_text['actions_opp'], 'opponent', K=K)       
                    for i in range(K):
                        correct_moves_opponent[i] += correct_opp[i]
                percent_ply = [round(correct / max(total_moves,1) * 100, 2) for correct in correct_moves_player]
                percent_opp = [round(correct / max(total_moves_opp,1) * 100, 2) for correct in correct_moves_opponent]
                pbar.set_description(f"player moves:{percent_ply}. Opponent moves: {percent_opp}")
                # pbar.set_description(f"player moves: {correct_moves_player}/{total_moves}={correct_moves_player/max(total_moves,1)*100:.2f}%. Opponent moves: {correct_moves_opponent}/{total_moves_opp}={correct_moves_opponent/max(total_moves_opp,1)*100:.2f}%")
    return

'''
State Evaluation: HP
'''
def eval_state_hp():
    raise NotImplementedError

'''
State Evaluation: Status
'''
def eval_state_status():
    raise NotImplementedError

'''
State Evaluation: Move Typing
'''
def eval_state_type():
    raise NotImplementedError

'''
General Knowledge Evaluation: Typing
'''
def eval_state_status():
    raise NotImplementedError