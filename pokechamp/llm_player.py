import ast
import asyncio
from copy import copy, deepcopy
import datetime
import json
import os
import random
import sys

import numpy as np
from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.environment.battle import Battle
from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move_category import MoveCategory
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.side_condition import SideCondition
from poke_env.player.player import Player, BattleOrder
from typing import Callable, Dict, List, Optional, Tuple, Union
from poke_env.environment.move import Move
import time
import json
from poke_env.data.gen_data import GenData
from pokechamp.gpt_player import GPTPlayer
from pokechamp.llama_player import LLAMAPlayer
from pokechamp.openrouter_player import OpenRouterPlayer
from pokechamp.gemini_player import GeminiPlayer
from pokechamp.ollama_player import OllamaPlayer
from pokechamp.data_cache import (
    get_cached_move_effect,
    get_cached_pokemon_move_dict,
    get_cached_ability_effect,
    get_cached_pokemon_ability_dict,
    get_cached_item_effect,
    get_cached_pokemon_item_dict,
    get_cached_pokedex
)
from poke_env.concurrency import POKE_LOOP
from pokechamp.minimax_optimizer import (
    get_minimax_optimizer,
    initialize_minimax_optimization,
    fast_battle_evaluation,
    create_battle_state_hash,
    OptimizedSimNode
)
from poke_env.player.local_simulation import LocalSim, SimNode
from difflib import get_close_matches
from pokechamp.prompts import get_number_turns_faint, get_status_num_turns_fnt, state_translate, get_gimmick_motivation

# Visual effects import (optional)
try:
    from pokechamp.visual_effects import visual, print_banner, print_status
    VISUAL_EFFECTS = True
except ImportError:
    VISUAL_EFFECTS = False

DEBUG=False

class LLMPlayer(Player):
    def __init__(self,
                 battle_format,
                 api_key="",
                 backend="gpt-4-1106-preview",
                 temperature=1.0,
                 prompt_algo="io",
                 log_dir=None,
                 team=None,
                 save_replays=None,
                 account_configuration=None,
                 server_configuration=None,
                 K=2,
                 _use_strat_prompt=False,
                 prompt_translate: Callable=state_translate,
                 device=0,
                 llm_backend=None
                 ):

        super().__init__(battle_format=battle_format,
                         team=team,
                         save_replays=save_replays,
                         account_configuration=account_configuration,
                         server_configuration=server_configuration)

        self._reward_buffer: Dict[AbstractBattle, float] = {}
        self._battle_last_action : Dict[AbstractBattle, Dict] = {}
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.backend = backend
        self.temperature = temperature
        self.log_dir = log_dir
        self.api_key = api_key
        self.prompt_algo = prompt_algo
        self.gen = GenData.from_format(battle_format)
        self.genNum = self.gen.gen
        self.prompt_translate = prompt_translate

        self.strategy_prompt = ""
        self.team_str = team
        self.use_strat_prompt = _use_strat_prompt
        
        # Use cached data instead of loading files repeatedly
        self.move_effect = get_cached_move_effect()
        # only used in old prompting method, replaced by statistcal sets data
        self.pokemon_move_dict = get_cached_pokemon_move_dict()
        self.ability_effect = get_cached_ability_effect()
        # only used is old prompting method
        self.pokemon_ability_dict = get_cached_pokemon_ability_dict()
        self.item_effect = get_cached_item_effect()
        # unused
        # with open(f"./poke_env/data/static/items/gen8pokemon_item_dict.json", "r") as f:
        #     self.pokemon_item_dict = json.load(f)
        self.pokemon_item_dict = get_cached_pokemon_item_dict()
        self._pokemon_dict = get_cached_pokedex(self.gen.gen)

        self.last_plan = ""

        if llm_backend is None:
            print(f"Initializing backend: {backend}")  # Debug logging
            if backend.startswith('ollama/'):
                # Ollama models - extract model name after 'ollama/'
                model_name = backend.replace('ollama/', '')
                print(f"Using Ollama with model: {model_name}")
                self.llm = OllamaPlayer(model=model_name, device=device)
            elif 'gpt' in backend and not backend.startswith('openai/'):
                self.llm = GPTPlayer(self.api_key)
            elif 'llama' == backend:
                self.llm = LLAMAPlayer(device=device)
            elif 'gemini' in backend:
                self.llm = GeminiPlayer(self.api_key)
            elif backend.startswith(('openai/', 'anthropic/', 'google/', 'meta/', 'mistral/', 'cohere/', 'perplexity/', 'deepseek/', 'microsoft/', 'nvidia/', 'huggingface/', 'together/', 'replicate/', 'fireworks/', 'localai/', 'vllm/', 'sagemaker/', 'vertex/', 'bedrock/', 'azure/', 'custom/')):
                # OpenRouter supports hundreds of models from various providers
                self.llm = OpenRouterPlayer(self.api_key)
            else:
                raise NotImplementedError('LLM type not implemented:', backend)
        else:
            self.llm = llm_backend
        self.llm_value = self.llm
        self.K = K      # for minimax, SC, ToT
        self.use_optimized_minimax = True  # Enable optimized minimax by default
        self._minimax_initialized = False
        # Configuration for time optimization
        self.use_damage_calc_early_exit = True  # Use damage calculator to exit early when advantageous
        self.use_llm_value_function = True  # Use LLM for leaf node evaluation (vs fast heuristic)
        self.max_depth_for_llm_eval = 2  # Only use LLM evaluation for shallow depths to save time
        
        # Warm-up flag to track if pre-initialization is complete
        self._warmed_up = False

    def warm_up(self, dummy_battle=None):
        """
        Pre-initialize all expensive components to avoid delays during first battle turn.
        This should be called after player creation but before battles start.
        """
        if self._warmed_up:
            return
            
        if VISUAL_EFFECTS:
            print_banner("WARMUP", "water")
            print("Starting player component initialization...")
        else:
            print("[WARMUP] Starting player component initialization...")
        
        # 1. Data cache is already loaded during __init__
        
        # 2. Pre-initialize minimax optimizer if using minimax
        if self.prompt_algo == "minimax" and self.use_optimized_minimax:
            # Skip minimax warm-up for now - it requires a real battle state
            # The optimizer will initialize on first actual battle turn
            print("   [INFO] Minimax optimizer will initialize on first battle turn")
        
        # 3. Pre-load team predictor (triggers Bayesian model loading)
        try:
            from bayesian.predictor_singleton import get_pokemon_predictor
            predictor = get_pokemon_predictor()
            # Trigger model loading if it has training methods
            if hasattr(predictor, 'load_and_train'):
                predictor.load_and_train()
            elif hasattr(predictor, 'team_predictor') and hasattr(predictor.team_predictor, 'load_and_train'):
                predictor.team_predictor.load_and_train()
            print("   [OK] Pokemon predictor pre-loaded")
        except Exception as e:
            print(f"   [WARN] Pokemon predictor warm-up failed: {e}")
        
        # 4. Pre-load move set data
        try:
            from pokechamp.data_cache import get_cached_moves_set
            get_cached_moves_set('gen9ou')
            print("   [OK] Move set data pre-loaded")
        except Exception as e:
            print(f"   [WARN] Move set data warm-up failed: {e}")
        
        self._warmed_up = True
        if VISUAL_EFFECTS:
            print(visual.create_banner("READY", font="small", style="greenblue"))
            print("Player warm-up complete!")
        else:
            print("[WARMUP] Player warm-up complete!")

    def _send_thinking_message(self, battle: AbstractBattle, message: str):
        """
        Send LLM thinking as chat messages during battle in 1000-character chunks.
        Based on TimeoutLLMPlayer._send_chat_message implementation.
        """
        try:
            # Split message into 1000-character chunks
            max_chunk_size = 950  # Leave room for turn prefix
            chunks = []
            
            for i in range(0, len(message), max_chunk_size):
                chunk = message[i:i + max_chunk_size]
                chunks.append(chunk)
            
            # Create an async function to send all message chunks
            async def send_message_async():
                try:
                    print(f"   Sending thinking to battle chat ({len(chunks)} parts)...")
                    
                    for part_num, chunk in enumerate(chunks, 1):
                        if len(chunks) == 1:
                            # Single message
                            chat_message = f"Turn #{battle.turn} thinking: {chunk}"
                        else:
                            # Multiple parts
                            chat_message = f"Turn #{battle.turn} thinking ({part_num}/{len(chunks)}): {chunk}"
                        
                        await self.ps_client.send_message(chat_message, room=battle.battle_tag)
                        
                        # Small delay between multiple messages
                        if part_num < len(chunks):
                            await asyncio.sleep(0.25)
                    
                    # Send fast mode command once after all thinking
                    # await self.ps_client.send_message("/timer off", room=battle.battle_tag)
                    print(f"   All thinking sent to {battle.battle_tag}")
                    
                except Exception as e:
                    print(f"   Failed to send thinking message: {e}")
            
            # Submit to the poke loop for execution
            try:
                future = asyncio.run_coroutine_threadsafe(send_message_async(), POKE_LOOP)
                # Don't wait for completion to avoid blocking
            except Exception as e:
                print(f"   Could not schedule thinking message: {e}")
                
        except Exception as e:
            print(f"Failed to send thinking message: {e}")

    def get_LLM_action(self, system_prompt, user_prompt, model, temperature=0.7, json_format=False, seed=None, stop=[], max_tokens=200, actions=None, llm=None, battle=None) -> str:
        if llm is None:
            output, _, raw_message = self.llm.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions, battle=battle, ps_client=self.ps_client)
        else:
            output, _, raw_message = llm.get_LLM_action(system_prompt, user_prompt, model, temperature, True, seed, stop, max_tokens=max_tokens, actions=actions, battle=battle, ps_client=self.ps_client)
        
        # Send thinking message if battle is provided
        if battle is not None and raw_message:
            try:
                                # Always show LLM reasoning in chat
                print(f"LLM [{self.ps_client.account_configuration.username}]: {raw_message}")
                
                self._send_thinking_message(battle, raw_message)
            except Exception as e:
                print(f"Failed to send thinking message: {e}")
        
        return output
    
    def check_all_pokemon(self, pokemon_str: str) -> Pokemon:
        valid_pokemon = None
        if pokemon_str in self._pokemon_dict:
            valid_pokemon = pokemon_str
        else:
            closest = get_close_matches(pokemon_str, self._pokemon_dict.keys(), n=1, cutoff=0.8)
            if len(closest) > 0:
                valid_pokemon = closest[0]
        if valid_pokemon is None:
            return None
        pokemon = Pokemon(species=pokemon_str, gen=self.genNum)
        return pokemon

    def choose_move(self, battle: AbstractBattle):
        sim = LocalSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    self.strategy_prompt,
                    format=self.format,
                    prompt_translate=self.prompt_translate
        )
        if battle.turn <=1 and self.use_strat_prompt:
            self.strategy_prompt = sim.get_llm_system_prompt(self.format, self.llm, team_str=self.team_str, model='gpt-4o-2024-05-13')
        
        if battle.active_pokemon:
            if battle.active_pokemon.fainted and len(battle.available_switches) == 1:
                next_action = BattleOrder(battle.available_switches[0])
                return next_action
            elif not battle.active_pokemon.fainted and len(battle.available_moves) == 1 and len(battle.available_switches) == 0:
                return self.choose_max_damage_move(battle)
        elif len(battle.available_moves) <= 1 and len(battle.available_switches) == 0:
            return self.choose_max_damage_move(battle)

        system_prompt, state_prompt, state_action_prompt = sim.state_translate(battle) # add lower case
        moves = [move.id for move in battle.available_moves]
        switches = [pokemon.species for pokemon in battle.available_switches]
        actions = [moves, switches]
        

        gimmick_output_format = ''
        if 'pokellmon' not in self.ps_client.account_configuration.username: # make sure we dont mess with pokellmon original strat
            dynamax_option = ' or {"dynamax":"<move_name>"}' if battle.can_dynamax else ''
            tera_option = ' or {"terastallize":"<move_name>"}' if battle.can_tera else ''
            gimmick_output_format = f'{dynamax_option}{tera_option}'

        if battle.active_pokemon.fainted or len(battle.available_moves) == 0:

            constraint_prompt_io = '''Choose the most suitable pokemon to switch. Your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_cot = '''Choose the most suitable pokemon to switch by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best switch options. Your output MUST be a JSON like:{"option_1":{"action":"switch","target":"<switch_pokemon_name>"}, ..., "option_k":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best option from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:{"decision":{"action":"switch","target":"<switch_pokemon_name>"}}\n'''
        elif len(battle.available_switches) == 0:
            constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>"}}{gimmick_output_format}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move>", "target":"<move_name>"}, ..., "option_k":{"action":"<move>", "target":"<move_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move>", "target":"<move_name>"}\n'''
        else:
            constraint_prompt_io = f'''Choose the best action and your output MUST be a JSON like: {{"move":"<move_name>"}}{gimmick_output_format} or {{"switch":"<switch_pokemon_name>"}}\n'''
            constraint_prompt_cot = '''Choose the best action by thinking step by step. Your thought should no more than 4 sentences. Your output MUST be a JSON like: {"thought":"<step-by-step-thinking>", "move":"<move_name>"} or {"thought":"<step-by-step-thinking>", "switch":"<switch_pokemon_name>"}\n'''
            constraint_prompt_tot_1 = '''Generate top-k (k<=3) best action options. Your output MUST be a JSON like: {"option_1":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}, ..., "option_k":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}}\n'''
            constraint_prompt_tot_2 = '''Select the best action from the following choices by considering their consequences: [OPTIONS]. Your output MUST be a JSON like:"decision":{"action":"<move_or_switch>", "target":"<move_name_or_switch_pokemon_name>"}\n'''

        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io
        state_prompt_cot = state_prompt + state_action_prompt + constraint_prompt_cot
        state_prompt_tot_1 = state_prompt + state_action_prompt + constraint_prompt_tot_1
        state_prompt_tot_2 = state_prompt + state_action_prompt + constraint_prompt_tot_2

        retries = 10
        # Chain-of-thought
        if self.prompt_algo == "io":
            return self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim, actions=actions)

        # Self-consistency with k = 3
        elif self.prompt_algo == "sc":
            return self.sc(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim)

        # Tree of thought, k = 3
        elif self.prompt_algo == "tot":
            llm_output1 = ""
            next_action = None
            for i in range(retries):
                try:
                    llm_output1 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_1,
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=200,
                                               json_format=True,
                                               battle=battle)
                    break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if llm_output1 == "":
                return self.choose_max_damage_move(battle)

            for i in range(retries):
                try:
                    llm_output2 = self.get_LLM_action(system_prompt=system_prompt,
                                               user_prompt=state_prompt_tot_2.replace("[OPTIONS]", llm_output1),
                                               model=self.backend,
                                               temperature=self.temperature,
                                               max_tokens=100,
                                               json_format=True,
                                               battle=battle)

                    next_action = self.parse_new(llm_output2, battle, sim)
                    with open(f"{self.log_dir}/output.jsonl", "a") as f:
                        f.write(json.dumps({"turn": battle.turn,
                                            "system_prompt": system_prompt,
                                            "user_prompt1": state_prompt_tot_1,
                                            "user_prompt2": state_prompt_tot_2,
                                            "llm_output1": llm_output1,
                                            "llm_output2": llm_output2,
                                            "battle_tag": battle.battle_tag
                                            }) + "\n")
                    if next_action is not None:     break
                except:
                    raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
                    continue

            if next_action is None:
                next_action = self.choose_max_damage_move(battle)
            return next_action

        elif self.prompt_algo == "minimax":
            try:
                # Initialize minimax optimizer if not already done
                if self.use_optimized_minimax and not self._minimax_initialized:
                    self._initialize_minimax_optimizer(battle)
                    
                if self.use_optimized_minimax:
                    return self.tree_search_optimized(retries, battle)
                else:
                    return self.tree_search(retries, battle)
            except Exception as e:
                print(f'minimax step failed ({e}). Using dmg calc')
                print(f'Exception: {e}', 'passed')
                return self.choose_max_damage_move(battle)

        
    def io(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle: Battle, sim, dont_verify=False, actions=None):
        next_action = None
        cot_prompt = 'In fewer than 3 sentences, let\'s think step by step:'
        state_prompt_io = state_prompt + state_action_prompt + constraint_prompt_io + cot_prompt

        for i in range(retries):
            try:
                llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                            user_prompt=state_prompt_io,
                                            model=self.backend,
                                            temperature=self.temperature,
                                            max_tokens=300,
                                            # stop=["reason"],
                                            json_format=True,
                                            actions=actions,
                                            battle=battle)

                # load when llm does heavylifting for parsing
                if DEBUG:
                    print(f"Raw LLM output: {llm_output}")
                

                llm_action_json = json.loads(llm_output)
                if DEBUG:
                    print(f"Parsed JSON: {llm_action_json}")
                next_action = None

                dynamax = "dynamax" in llm_action_json.keys()
                tera = "terastallize" in llm_action_json.keys()
                is_a_move = dynamax or tera

                if "move" in llm_action_json.keys() or is_a_move:
                    if dynamax:
                        llm_move_id = llm_action_json["dynamax"].strip()
                    elif tera:
                        llm_move_id = llm_action_json["terastallize"].strip()
                    else:
                        llm_move_id = llm_action_json["move"].strip()
                    move_list = battle.available_moves
                    if dont_verify: # opponent
                        move_list = battle.opponent_active_pokemon.moves.values()
                    
                    # Debug: print available moves
                    if DEBUG:
                        print(f"LLM requested move: '{llm_move_id}'")
                        print(f"Available moves: {[move.id for move in move_list]}")
                    
                    for i, move in enumerate(move_list):
                        if move.id.lower().replace(' ', '') == llm_move_id.lower().replace(' ', ''):
                            #next_action = self.create_order(move, dynamax=sim._should_dynamax(battle), terastallize=sim._should_terastallize(battle))
                            next_action = self.create_order(move, dynamax=dynamax, terastallize=tera)
                            if DEBUG:
                                print(f"Move match found: {move.id}")
                            break
                    
                    if next_action is None and dont_verify:
                        # unseen move so just check if it is in the action prompt
                        if llm_move_id.lower().replace(' ', '') in state_action_prompt:
                            next_action = self.create_order(Move(llm_move_id.lower().replace(' ', ''), self.gen.gen), dynamax=dynamax, terastallize=tera)
                    
                    if next_action is None and DEBUG:
                        print(f"No move match found for '{llm_move_id}'")
                elif "switch" in llm_action_json.keys():
                    llm_switch_species = llm_action_json["switch"].strip()
                    switch_list = battle.available_switches
                    if dont_verify: # opponent prediction
                        observable_switches = []
                        for _, opponent_pokemon in battle.opponent_team.items():
                            if not opponent_pokemon.active:
                                observable_switches.append(opponent_pokemon)
                        switch_list = observable_switches
                    
                    # Debug: print available switches
                    if DEBUG:
                        print(f"LLM requested switch: '{llm_switch_species}'")
                        print(f"Available switches: {[pokemon.species for pokemon in switch_list]}")
                    
                    for i, pokemon in enumerate(switch_list):
                        if pokemon.species.lower().replace(' ', '') == llm_switch_species.lower().replace(' ', ''):
                            next_action = self.create_order(pokemon)
                            if DEBUG:
                                print(f"Switch match found: {pokemon.species}")
                            break
                    
                else:
                    raise ValueError('No valid action')
                
                # with open(f"{self.log_dir}/output.jsonl", "a") as f:
                #     f.write(json.dumps({"turn": battle.turn,
                #                         "system_prompt": system_prompt,
                #                         "user_prompt": state_prompt_io,
                #                         "llm_output": llm_output,
                #                         "battle_tag": battle.battle_tag
                #                         }) + "\n")
                
                if next_action is not None:
                    break
            except Exception as e:
                print(f'Exception: {e}', 'passed')
                continue
        if next_action is None:
            print('No action found. Choosing max damage move')
            try:
                print('No action found', llm_action_json, actions, dont_verify)
            except:
                pass
            print()
            # raise ValueError('No valid move', battle.active_pokemon.fainted, len(battle.available_switches))
            next_action = self.choose_max_damage_move(battle)
        return next_action

    def sc(self, retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim):
        actions = [self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, battle, sim) for i in range(self.K)]
        action_message = [action.message for action in actions]
        _, counts = np.unique(action_message, return_counts=True)
        index = np.argmax(counts)
        return actions[index]
    
    def estimate_matchup(self, sim: LocalSim, battle: Battle, mon: Pokemon, mon_opp: Pokemon, is_opp: bool=False) -> Tuple[Move, int]:
        hp_remaining = []
        moves = list(mon.moves.keys())
        if is_opp:
            moves = sim.get_opponent_current_moves(mon=mon)
        if battle.active_pokemon.species == mon.species and not is_opp:
            moves = [move.id for move in battle.available_moves]
        for move_id in moves:
            move = Move(move_id, gen=sim.gen.gen)
            t = np.inf
            if move.category == MoveCategory.STATUS:
                # apply stat boosting effects to see if it will KO in fewer turns
                t = get_status_num_turns_fnt(mon, move, mon_opp, sim, boosts=mon._boosts.copy())
            else:
                t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
            hp_remaining.append(t)
            # _, hp2, _, _ = sim.calculate_remaining_hp(battle.active_pokemon, battle.opponent_active_pokemon, move, None)
            # hp_remaining.append(hp2)
        hp_best_index = np.argmin(hp_remaining)
        best_move = moves[hp_best_index]
        best_move_turns = hp_remaining[hp_best_index]
        best_move = Move(best_move, gen=sim.gen.gen)
        best_move = self.create_order(best_move)
        # check special moves: tera/dyna
        # dyna for gen 8
        if sim.battle._data.gen == 8 and sim.battle.can_dynamax:
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen).dynamaxed
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, dynamax=True)
                        best_move_turns = t
        # tera for gen 9
        elif sim.battle._data.gen == 9 and sim.battle.can_tera:
            mon.terastallize()
            for move_id in moves:
                move = Move(move_id, gen=sim.gen.gen)
                if move.category != MoveCategory.STATUS:
                    t = get_number_turns_faint(mon, move, mon_opp, sim, boosts1=mon._boosts.copy(), boosts2=mon_opp.boosts.copy())
                    if t < best_move_turns:
                        best_move = self.create_order(move, terastallize=True)
                        best_move_turns = t
            mon.unterastallize()
            
        return best_move, best_move_turns

    def dmg_calc_move(self, battle: AbstractBattle, return_move: bool=False):
        sim = LocalSim(battle, 
                    self.move_effect,
                    self.pokemon_move_dict,
                    self.ability_effect,
                    self.pokemon_ability_dict,
                    self.item_effect,
                    self.pokemon_item_dict,
                    self.gen,
                    self._dynamax_disable,
                    format=self.format
        )
        best_action = None
        best_action_turns = np.inf
        if battle.available_moves and not battle.active_pokemon.fainted:
            # try moves and find hp remaining for opponent
            mon = battle.active_pokemon
            mon_opp = battle.opponent_active_pokemon
            best_action, best_action_turns = self.estimate_matchup(sim, battle, mon, mon_opp)
        if return_move:
            if best_action is None:
                return None, best_action_turns
            return best_action.order, best_action_turns
        if best_action_turns > 4:
            return None, np.inf
        if best_action is not None:
            return best_action, best_action_turns
        return self.choose_random_move(battle), 1
    
    
    SPEED_TIER_COEFICIENT = 0.1
    HP_FRACTION_COEFICIENT = 0.4

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
    
    def _get_fast_heuristic_evaluation(self, battle_state):
        """Fast heuristic evaluation for leaf nodes when LLM is not used."""
        try:
            player_hp = int(battle_state.active_pokemon.current_hp_fraction * 100) if battle_state.active_pokemon else 0
            opp_hp = int(battle_state.opponent_active_pokemon.current_hp_fraction * 100) if battle_state.opponent_active_pokemon else 0
            player_remaining = len([p for p in battle_state.team.values() if not p.fainted])
            opp_remaining = len([p for p in battle_state.opponent_team.values() if not p.fainted])
            
            # Use cached fast evaluation
            return fast_battle_evaluation(
                player_hp, opp_hp, 
                player_remaining, opp_remaining,
                battle_state.turn
            )
        except:
            # Ultimate fallback to basic hp difference
            try:
                from poke_env.player.local_simulation import LocalSim
                sim = LocalSim(battle_state, 
                            self.move_effect,
                            self.pokemon_move_dict,
                            self.ability_effect,
                            self.pokemon_ability_dict,
                            self.item_effect,
                            self.pokemon_item_dict,
                            self.gen,
                            self._dynamax_disable,
                            format=self.format
                )
                return sim.get_hp_diff()
            except:
                return 50  # Neutral fallback score
    
    def _initialize_minimax_optimizer(self, battle):
        """Initialize the minimax optimizer with current battle state."""
        try:
            initialize_minimax_optimization(
                battle=battle,
                move_effect=self.move_effect,
                pokemon_move_dict=self.pokemon_move_dict,
                ability_effect=self.ability_effect,
                pokemon_ability_dict=self.pokemon_ability_dict,
                item_effect=self.item_effect,
                pokemon_item_dict=self.pokemon_item_dict,
                gen=self.gen,
                _dynamax_disable=self._dynamax_disable,
                format=self.format,
                prompt_translate=self.prompt_translate
            )
            self._minimax_initialized = True
            if VISUAL_EFFECTS:
                print_status("Minimax optimizer initialized", "success")
            else:
                print("[INIT] Minimax optimizer initialized")
        except Exception as e:
            print(f"[WARN] Failed to initialize minimax optimizer: {e}")
            self.use_optimized_minimax = False  # Fallback to original

    def check_timeout(self, start_time, battle):
        if time.time() - start_time > 30:
            print('default due to time')
            move, _ = self.dmg_calc_move(battle)
            return move
        else:
            return None
    
    def tree_search(self, retries, battle, sim=None, return_opp = False) -> BattleOrder:
        # generate local simulation
        root = SimNode(battle, 
                        self.move_effect,
                        self.pokemon_move_dict,
                        self.ability_effect,
                        self.pokemon_ability_dict,
                        self.item_effect,
                        self.pokemon_item_dict,
                        self.gen,
                        self._dynamax_disable,
                        depth=1,
                        format=self.format,
                        prompt_translate=self.prompt_translate,
                        sim=sim
                        ) 
        q = [
                root
            ]
        leaf_nodes = []
        # create node and add to q B times
        start_time = time.time()
        while len(q) != 0:
            node = q.pop(0)
            # choose node for expansion
            # generate B actions
            player_actions = []
            system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # end if terminal
            if node.simulation.is_terminal() or node.depth == self.K:
                try:
                    # value estimation for leaf nodes
                    value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                    'Add points based on the effectiveness of current available moves.' +\
                                    'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                    'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                    'Subtract points for excessive switching.' +\
                                    'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                    'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                    cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                    state_prompt_io = state_prompt + value_prompt + cot_prompt
                    llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                    user_prompt=state_prompt_io,
                                                    model=self.backend,
                                                    temperature=self.temperature,
                                                    max_tokens=500,
                                                    json_format=True,
                                                    llm=self.llm_value,
                                                    battle=battle
                                                    )
                    # load when llm does heavylifting for parsing
                    llm_action_json = json.loads(llm_output)
                    node.hp_diff = int(llm_action_json['score'])
                except Exception as e:
                    node.hp_diff = node.simulation.get_hp_diff()                    
                    print(e)
                
                leaf_nodes.append(node)
                continue
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # estimate opp
            try:
                action_opp, opp_turns = self.estimate_matchup(node.simulation, node.simulation.battle, node.simulation.battle.opponent_active_pokemon, node.simulation.battle.active_pokemon, is_opp=True)
            except:
                action_opp = None
                opp_turns = np.inf
            ##############################
            # generate players's action  #
            ##############################
            if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                # get dmg calc move
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                if dmg_calc_out is not None:
                    if dmg_calc_turns <= opp_turns:
                        try:
                            # ask LLM to use heuristic tool or minimax search
                            tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                                1. Damage calculator advantages:
                                - Quick and efficient for finding optimal damaging moves
                                - Useful when a clear type advantage or high-power move is available
                                - Effective when the opponent's is not switching and current pokemon is likely to KO opponent

                                2. Minimax tree search advantages:
                                - Can model opponent behavior and predict future moves
                                - Useful in complex situations with multiple viable options
                                - Effective when long-term strategy is crucial

                                3. Current battle state:
                                - Remaining Pokémon on each side
                                - Health of active Pokémon
                                - Type matchups
                                - Available moves and their effects
                                - Presence of status conditions or field effects

                                4. Uncertainty level:
                                - How predictable is the opponent's next move?
                                - Are there multiple equally viable options for your next move?

                                Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                                {"choice":"damage calculator"} or {"choice":"minimax"}'''

                            state_prompt_io = state_prompt + tool_prompt
                            llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                            user_prompt=state_prompt_io,
                                                            model=self.backend,
                                                            temperature=0.6,
                                                            max_tokens=100,
                                                            json_format=True,
                                                            battle=battle
                                                            )
                            # load when llm does heavylifting for parsing
                            llm_action_json = json.loads(llm_output)
                            if 'choice' in llm_action_json.keys():
                                if llm_action_json['choice']  != 'minimax':
                                    if return_opp:
                                        # use tool to save time and llm when move makes bigger difference
                                        return dmg_calc_out, action_opp
                                    return dmg_calc_out
                        except:
                            print('defaulting to minimax')
                    player_actions.append(dmg_calc_out)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # get llm switch
            if len(node.simulation.battle.available_switches) != 0:# or opp_turns < dmg_calc_turns):
                state_action_prompt_switch = state_action_prompt + action_prompt_switch + '\nYou can only choose to switch this turn.\n'
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"switch":"<switch_pokemon_name>"}.\n'
                for i in range(2):
                    action_llm_switch = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt_switch, node.simulation.battle, node.simulation)
                    if len(player_actions) == 0:
                        player_actions.append(action_llm_switch)
                    elif action_llm_switch.message != player_actions[-1].message:
                        player_actions.append(action_llm_switch)

            if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:# and not opp_turns < dmg_calc_turns:
                # get llm move
                state_action_prompt_move = state_action_prompt + action_prompt_move + '\nYou can only choose to move this turn.\n'
                constraint_prompt_io = 'Choose the best action and your output MUST be a JSON like: {"move":"<move_name>"}.\n'
                action_llm_move = self.io(retries, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt_move, node.simulation.battle, node.simulation)
                if len(player_actions) == 0:
                    player_actions.append(action_llm_move)
                elif action_llm_move.message != player_actions[0].message:
                    player_actions.append(action_llm_move)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            ##############################
            # generate opponent's action #
            ##############################
            opponent_actions = []
            tool_is_optimal = False
            # dmg calc suggestion
            # action_opp, opp_turns = self.estimate_matchup(node.simulation, node.simulation.battle, node.simulation.battle.opponent_active_pokemon, node.simulation.battle.active_pokemon, is_opp=True)
            if action_opp is not None:
                tool_is_optimal = True
                opponent_actions.append(self.create_order(action_opp))
            # heuristic matchup switch action
            best_score = np.inf
            best_action = None
            for mon in node.simulation.battle.opponent_team.values():
                if mon.species == node.simulation.battle.opponent_active_pokemon.species:
                    continue
                score = self._estimate_matchup(mon, node.simulation.battle.active_pokemon)
                if score < best_score:
                    best_score = score
                    best_action = mon
            if best_action is not None:
                opponent_actions.append(self.create_order(best_action))
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # create opponent prompt from battle sim
            system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(system_prompt)
            action_o = self.io(2, system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o, node.simulation.battle, node.simulation, dont_verify=True)
            is_repeat_action_o = np.array([action_o.message == opponent_action.message for opponent_action in opponent_actions]).any()
            if not is_repeat_action_o:
                opponent_actions.append(action_o)
            # panic_move = self.check_timeout(start_time, battle)
            # if panic_move is not None:
            #     return panic_move
            # simulate outcome
            if node.depth < self.K:
                for action_p in player_actions:
                    for action_o in opponent_actions:
                        node_new = copy(node)
                        node_new.simulation.battle = copy(node.simulation.battle)
                        # if not tool_is_optimal:
                        node_new.children = []
                        node_new.depth = node.depth + 1
                        node_new.action = action_p
                        node_new.action_opp = action_o
                        node_new.parent_node = node
                        node_new.parent_action = node.action
                        node.children.append(node_new)
                        node_new.simulation.step(action_p, action_o)
                        q.append(node_new)

        # choose best action according to max or min rule
        def get_tree_action(root: SimNode):
            if len(root.children) == 0:
                return root.action, root.hp_diff, root.action_opp
            score_dict = {}
            action_dict = {}
            opp_dict = {}
            for child in root.children:
                action = str(child.action.order)
                _, score, _ = get_tree_action(child)
                if action in score_dict.keys():
                    # imitation
                    # score_dict[action] = score + score_dict[action]
                    # minimax
                    score_dict[action] = min(score, score_dict[action])
                else:
                    score_dict[action] = score
                    action_dict[action] = child.action
                    opp_dict[action] = child.action_opp
            scores = list(score_dict.values())
            best_action_str = list(action_dict.keys())[np.argmax(scores)]
            return action_dict[best_action_str], score_dict[best_action_str], opp_dict[best_action_str]
        
        action, _, action_opp = get_tree_action(root)
        end_time = time.time()
        if return_opp:
            return action, action_opp
        return action

    def tree_search_optimized(self, retries, battle, sim=None, return_opp=False) -> BattleOrder:
        """
        Optimized version of tree_search using object pooling and caching.
        
        This version provides significant performance improvements for minimax:
        - Object pooling for LocalSim instances
        - LLM choice between damage calculator and minimax upfront
        - Battle state caching to avoid repeated computations
        """
        optimizer = get_minimax_optimizer()
        start_time = time.time()
        
        try:
            # Create optimized root node
            root = optimizer.create_optimized_root(battle)
            
            # Get battle state information for LLM decision
            system_prompt, state_prompt, _, _, _, _, _ = root.simulation.get_player_prompt(return_actions=True)
            
            # Ask LLM upfront whether to use minimax or damage calculator
            if not battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                # Get dmg calc move for potential early return
                dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(battle)
                if dmg_calc_out is not None:
                    try:
                        # Ask LLM to choose between damage calculator tool or minimax search upfront
                        tool_prompt = '''Based on the current battle state, evaluate whether to use the damage calculator tool or the minimax tree search method. Consider the following factors:

                        1. Damage calculator advantages:
                        - Quick and efficient for finding optimal damaging moves
                        - Useful when a clear type advantage or high-power move is available
                        - Effective when the opponent is not switching and current pokemon is likely to KO opponent

                        2. Minimax tree search advantages:
                        - Can model opponent behavior and predict future moves
                        - Useful in complex situations with multiple viable options
                        - Effective when long-term strategy is crucial

                        3. Current battle state:
                        - Remaining Pokémon on each side
                        - Health of active Pokémon
                        - Type matchups
                        - Available moves and their effects
                        - Presence of status conditions or field effects

                        4. Uncertainty level:
                        - How predictable is the opponent's next move?
                        - Are there multiple equally viable options for your next move?

                        Evaluate these factors and decide which method would be more beneficial in the current situation. Output your choice in the following JSON format:

                        {"choice":"damage calculator"} or {"choice":"minimax"}'''

                        state_prompt_io = state_prompt + tool_prompt
                        llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                        user_prompt=state_prompt_io,
                                                        model=self.backend,
                                                        temperature=0.6,
                                                        max_tokens=100,
                                                        json_format=True,
                                                        battle=battle
                                                        )
                        # Load when llm does heavylifting for parsing
                        llm_action_json = json.loads(llm_output)
                        if 'choice' in llm_action_json.keys():
                            if llm_action_json['choice'] != 'minimax':
                                # LLM chose damage calculator - return it directly
                                print("LLM chose damage calculator over minimax")
                                if return_opp:
                                    try:
                                        action_opp, _ = self.estimate_matchup(root.simulation, battle, 
                                                                           battle.opponent_active_pokemon, 
                                                                           battle.active_pokemon, is_opp=True)
                                        return dmg_calc_out, self.create_order(action_opp) if action_opp else None
                                    except:
                                        return dmg_calc_out, None
                                return dmg_calc_out
                    except Exception as e:
                        print(f'LLM choice failed ({e}), defaulting to minimax')
            
            print("Using minimax tree search")
            
            q = [root]
            leaf_nodes = []
            
            while len(q) != 0:
                node = q.pop(0)
                
                # Get available actions efficiently 
                player_actions = []
                system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, action_prompt_switch, action_prompt_move = node.simulation.get_player_prompt(return_actions=True)
                
                # Check if terminal node or reached depth limit
                if node.simulation.is_terminal() or node.depth == self.K:
                    try:
                        # Use LLM value function for leaf nodes evaluation
                        value_prompt = 'Evaluate the score from 1-100 based on how likely the player is to win. Higher is better. Start at 50 points.' +\
                                        'Add points based on the effectiveness of current available moves.' +\
                                        'Award points for each pokemon remaining on the player\'s team, weighted by their strength' +\
                                        'Add points for boosted status and opponent entry hazards and subtract points for status effects and player entry hazards. ' +\
                                        'Subtract points for excessive switching.' +\
                                        'Subtract points based on the effectiveness of the opponent\'s current moves, especially if they have a faster speed.' +\
                                        'Remove points for each pokemon remaining on the opponent\'s team, weighted by their strength.\n'
                        cot_prompt = 'Briefly justify your total score, up to 100 words. Then, conclude with the score in the JSON format: {"score": <total_points>}. '
                        state_prompt_io = state_prompt + value_prompt + cot_prompt
                        llm_output = self.get_LLM_action(system_prompt=system_prompt,
                                                        user_prompt=state_prompt_io,
                                                        model=self.backend,
                                                        temperature=self.temperature,
                                                        max_tokens=500,
                                                        json_format=True,
                                                        llm=self.llm_value,
                                                        battle=battle
                                                        )
                        # Load when llm does heavylifting for parsing
                        llm_action_json = json.loads(llm_output)
                        node.hp_diff = int(llm_action_json['score'])
                    except Exception as e:
                        # Fallback to damage calculator based evaluation
                        try:
                            damage_calc_move, damage_calc_turns = self.dmg_calc_move(node.simulation.battle)
                            if damage_calc_turns < float('inf'):
                                # Score based on how many turns to KO opponent vs how many they need to KO us
                                try:
                                    opp_action, opp_turns = self.estimate_matchup(
                                        node.simulation, node.simulation.battle,
                                        node.simulation.battle.opponent_active_pokemon,
                                        node.simulation.battle.active_pokemon,
                                        is_opp=True
                                    )
                                    # Higher score if we can KO faster than opponent
                                    if opp_turns > damage_calc_turns:
                                        node.hp_diff = 75  # We have advantage
                                    elif opp_turns == damage_calc_turns:
                                        node.hp_diff = 50  # Even
                                    else:
                                        node.hp_diff = 25  # Opponent has advantage
                                except:
                                    node.hp_diff = 50  # Neutral if opponent estimation fails
                            else:
                                # Use basic hp difference if damage calc fails
                                node.hp_diff = node.simulation.get_hp_diff()
                        except:
                            # Ultimate fallback to basic hp difference
                            node.hp_diff = node.simulation.get_hp_diff()
                        print(f"LLM value function failed, using damage calculator fallback: {e}")
                    
                    leaf_nodes.append(node)
                    continue
                
                # Estimate opponent action (reuse existing logic)
                try:
                    action_opp, opp_turns = self.estimate_matchup(
                        node.simulation, node.simulation.battle, 
                        node.simulation.battle.opponent_active_pokemon, 
                        node.simulation.battle.active_pokemon, 
                        is_opp=True
                    )
                except:
                    action_opp = None
                    opp_turns = float('inf')
                
                # Get player actions - damage calculator move
                if not node.simulation.battle.active_pokemon.fainted and len(battle.available_moves) > 0:
                    # Get dmg calc move
                    dmg_calc_out, dmg_calc_turns = self.dmg_calc_move(node.simulation.battle)
                    if dmg_calc_out is not None:
                        player_actions.append(dmg_calc_out)

                # Generate opponent actions (reuse existing logic)
                opponent_actions = []
                if action_opp is not None:
                    opponent_actions.append(self.create_order(action_opp))
                
                # Get more opponent actions via LLM (simplified)
                try:
                    system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o = node.simulation.get_opponent_prompt(system_prompt)
                    action_o = self.io(2, system_prompt_o, state_prompt_o, constraint_prompt_cot_o, constraint_prompt_io_o, state_action_prompt_o, node.simulation.battle, node.simulation, dont_verify=True)
                    if action_o not in opponent_actions:
                        opponent_actions.append(action_o)
                except:
                    pass  # Use what we have
                
                # Generate a few additional actions
                try:
                    action_io = self.io(2, system_prompt, state_prompt, constraint_prompt_cot, constraint_prompt_io, state_action_prompt, node.simulation.battle, node.simulation, actions=player_actions)
                    if action_io not in player_actions:
                        player_actions.append(action_io)
                except:
                    pass
                
                # Create child nodes efficiently (if not at depth limit)
                if node.depth < self.K and player_actions and opponent_actions:
                    for action_p in player_actions[:2]:  # Limit to 2 player actions for performance
                        for action_o in opponent_actions[:2]:  # Limit to 2 opponent actions for performance
                            try:
                                child_node = node.create_child_node(action_p, action_o)
                                q.append(child_node)
                            except Exception as e:
                                print(f"Failed to create child node: {e}")
                                continue
            
            # Choose best action using original logic
            def get_tree_action(root_node):
                if len(root_node.children) == 0:
                    return root_node.action, root_node.hp_diff, root_node.action_opp
                    
                score_dict = {}
                action_dict = {}
                opp_dict = {}
                
                for child in root_node.children:
                    action = str(child.action.order)
                    if action not in score_dict:
                        score_dict[action] = []
                        action_dict[action] = child.action
                        opp_dict[action] = child.action_opp
                    score_dict[action].append(child.hp_diff)
                
                # Use max score for each action
                for action in score_dict:
                    score_dict[action] = max(score_dict[action])
                
                best_action_str = max(score_dict, key=score_dict.get)
                return action_dict[best_action_str], score_dict[best_action_str], opp_dict[best_action_str]
            
            action, _, action_opp = get_tree_action(root)
            
            # Cleanup resources
            optimizer.cleanup_tree(root)
            
            # Log performance stats
            end_time = time.time()
            stats = optimizer.get_performance_stats()
            if VISUAL_EFFECTS:
                print(visual.minimax_progress(depth, evaluated_nodes, end_time - start_time))
            else:
                print(f"[PERF] Optimized minimax: {end_time - start_time:.2f}s, nodes: {evaluated_nodes}, "
                      f"Pool reuse: {stats['pool_stats']['reuse_rate']:.2f}, "
                      f"Cache hit rate: {stats['cache_stats']['hit_rate']:.2f}")
            
            if return_opp:
                return action, action_opp
            return action
            
        except Exception as e:
            print(f"Optimized minimax failed: {e}, falling back to damage calculator")
            # Cleanup any resources
            try:
                optimizer.cleanup_tree(root)
            except:
                pass
            # Fallback to damage calculator instead of original tree search
            try:
                dmg_calc_move, _ = self.dmg_calc_move(battle)
                if dmg_calc_move is not None:
                    if return_opp:
                        try:
                            action_opp, _ = self.estimate_matchup(None, battle, 
                                                               battle.opponent_active_pokemon, 
                                                               battle.active_pokemon, is_opp=True)
                            return dmg_calc_move, self.create_order(action_opp) if action_opp else None
                        except:
                            return dmg_calc_move, None
                    return dmg_calc_move
            except:
                pass
            # Ultimate fallback to max damage move
            return self.choose_max_damage_move(battle)
 
    def battle_summary(self):

        beat_list = []
        remain_list = []
        win_list = []
        tag_list = []
        for tag, battle in self.battles.items():
            beat_score = 0
            for mon in battle.opponent_team.values():
                beat_score += (1-mon.current_hp_fraction)

            beat_list.append(beat_score)

            remain_score = 0
            for mon in battle.team.values():
                remain_score += mon.current_hp_fraction

            remain_list.append(remain_score)
            if battle.won:
                win_list.append(1)

            tag_list.append(tag)

        return beat_list, remain_list, win_list, tag_list

    def reward_computing_helper(
        self,
        battle: AbstractBattle,
        *,
        fainted_value: float = 0.0,
        hp_value: float = 0.0,
        number_of_pokemons: int = 6,
        starting_value: float = 0.0,
        status_value: float = 0.0,
        victory_value: float = 1.0,
    ) -> float:
        """A helper function to compute rewards."""

        if battle not in self._reward_buffer:
            self._reward_buffer[battle] = starting_value
        current_value = 0

        for mon in battle.team.values():
            current_value += mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value -= fainted_value
            elif mon.status is not None:
                current_value -= status_value

        current_value += (number_of_pokemons - len(battle.team)) * hp_value

        for mon in battle.opponent_team.values():
            current_value -= mon.current_hp_fraction * hp_value
            if mon.fainted:
                current_value += fainted_value
            elif mon.status is not None:
                current_value += status_value

        current_value -= (number_of_pokemons - len(battle.opponent_team)) * hp_value

        if battle.won:
            current_value += victory_value
        elif battle.lost:
            current_value -= victory_value

        to_return = current_value - self._reward_buffer[battle] # the return value is the delta
        self._reward_buffer[battle] = current_value

        return to_return

    def choose_max_damage_move(self, battle: Battle):
        if battle.available_moves:
            best_move = max(battle.available_moves, key=lambda move: move.base_power)
            return self.create_order(best_move)
        return self.choose_random_move(battle)
