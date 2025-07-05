from poke_env.player.player import Player
from poke_env.player.baselines import AbyssalPlayer, MaxBasePowerPlayer, OneStepPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.llm_player import LLMPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration

from poke_env.player.prompts import prompt_translate, state_translate2
from numpy.random import randint
import importlib
import inspect
import os

def load_random_team(id=None):
    if id == None:
        team_id = randint(1, 14)
    else:
        team_id = id
    with open(f'poke_env/data/static/teams/gen9ou{team_id}.txt', 'r') as f:
        team = f.read()
    return team

def get_custom_bot_class(bot_name: str):
    """
    Get a custom bot class by name from the bots folder.
    
    Args:
        bot_name: The name of the bot (without _bot suffix)
        
    Returns:
        The bot class if found, None otherwise
    """
    try:
        # Import the bot module
        module_name = f"bots.{bot_name}_bot"
        module = importlib.import_module(module_name)
        
        # Find the bot class in the module
        for name, obj in inspect.getmembers(module):
            if (inspect.isclass(obj) and 
                issubclass(obj, LLMPlayer) and 
                obj != LLMPlayer):
                return obj
        
        return None
    except ImportError:
        return None

def get_llm_player(args, 
                   backend: str, 
                   prompt_algo: str, 
                   name: str, 
                   KEY: str='', 
                   battle_format='gen9ou',
                   llm_backend=None, 
                   device=0,
                   PNUMBER1: str='', 
                   USERNAME: str='', 
                   PASSWORD: str='', 
                   online: bool=False) -> Player:
    server_config = None
    if online:
        server_config = ShowdownServerConfiguration
    if USERNAME == '':
        USERNAME = name
    if name == 'abyssal':
        return AbyssalPlayer(battle_format=battle_format,
                            account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                            server_configuration=server_config
                            )
    elif name == 'max_power':
        return MaxBasePowerPlayer(battle_format=battle_format,
                            account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                            server_configuration=server_config
                            )
    elif name == 'random':
        return RandomPlayer(battle_format=battle_format,
                            account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                            server_configuration=server_config
                            )
    elif name == 'one_step':
        return OneStepPlayer(battle_format=battle_format,
                            account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                            server_configuration=server_config
                            )
    elif 'pokellmon' in name:
        return LLMPlayer(battle_format=battle_format,
                       api_key=KEY,
                       backend=backend,
                       temperature=args.temperature,
                       prompt_algo=prompt_algo,
                       log_dir=args.log_dir,
                       account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                       server_configuration=server_config,
                       save_replays=args.log_dir,
                       device=device,
                       llm_backend=llm_backend)
    elif 'pokechamp' in name:
        return LLMPlayer(battle_format=battle_format,
                       api_key=KEY,
                       backend=backend,
                       temperature=args.temperature,
                       prompt_algo="minimax",
                       log_dir=args.log_dir,
                       account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                       server_configuration=server_config,
                       save_replays=args.log_dir,
                       prompt_translate=prompt_translate,
                    #    prompt_translate=state_translate2,
                       device=device,
                       llm_backend=llm_backend)
    else:
        # Try to find a custom bot in the bots folder
        custom_bot_class = get_custom_bot_class(name)
        if custom_bot_class:
            return custom_bot_class(
                battle_format=battle_format,
                api_key=KEY,
                backend=backend,
                temperature=args.temperature,
                log_dir=args.log_dir,
                account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                server_configuration=server_config,
                save_replays=args.log_dir,
                device=device,
                llm_backend=llm_backend
            )
        else:
            raise ValueError(f'Bot not found: {name}')