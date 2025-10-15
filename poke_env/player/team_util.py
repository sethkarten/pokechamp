from poke_env.data.download import download_teams
from poke_env.player.player import Player
from poke_env.player.baselines import AbyssalPlayer, MaxBasePowerPlayer, OneStepPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
from poke_env.teambuilder import Teambuilder
from numpy.random import randint
import importlib
import inspect
import os
import random
from pokechamp.llm_vgc_player import LLMVGCPlayer

class TeamSet(Teambuilder):
    """Sample from a directory of Showdown team files.

    A simple wrapper around poke-env's Teambuilder that randomly samples a team from a
    directory of team files.

    Args:
        team_file_dir: The directory containing the team files (searched recursively).
            Team files are just text files in the standard Showdown export format. See
            https://pokepast.es/syntax.html for details.
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers",
            etc.). Note that we assume files have a matching extension (e.g.
            "any_name.gen1ou_team").
    """

    def __init__(self, team_file_dir: str, battle_format: str):
        super().__init__()
        self.team_file_dir = team_file_dir
        self.battle_format = battle_format
        self.team_files = self._find_team_files()

    def _find_team_files(self):
        team_files = []
        for root, _, files in os.walk(self.team_file_dir):
            for file in files:
                if file.endswith(f".{self.battle_format}_team"):
                    team_files.append(os.path.join(root, file))
        return team_files

    def yield_team(self):
        file = random.choice(self.team_files)
        with open(file, "r") as f:
            team_data = f.read()
        team = self.parse_showdown_team(team_data)
        print(team)
        for mon in team:
            if mon.species is not None:
                mon.nickname = mon.species
        return self.join_team(team)

def get_metamon_teams(battle_format: str, set_name: str) -> TeamSet:
    """
    Download a set of teams from huggingface (if necessary) and return a TeamSet.

    Args:
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers", etc.).
        set_name: The name of the set of teams to download. See the README for options.
    """
    if set_name not in {
        "competitive",
        "paper_replays",
        "paper_variety",
        "modern_replays",
        "pokeagent_modern_replays",
    }:
        raise ValueError(
            f"Invalid set name: {set_name}. Must be one of: competitive, paper_replays, paper_variety, modern_replays"
        )
    path = download_teams(battle_format, set_name=set_name)
    if not os.path.exists(path):
        raise ValueError(
            f"Cannot locate valid team directory for format {battle_format} at path {path}"
        )
    return TeamSet(path, battle_format)

class TeamSet(Teambuilder):
    """Sample from a directory of Showdown team files.

    A simple wrapper around poke-env's Teambuilder that randomly samples a team from a
    directory of team files.

    Args:
        team_file_dir: The directory containing the team files (searched recursively).
            Team files are just text files in the standard Showdown export format. See
            https://pokepast.es/syntax.html for details.
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers",
            etc.). Note that we assume files have a matching extension (e.g.
            "any_name.gen1ou_team").
    """

    def __init__(self, team_file_dir: str, battle_format: str):
        super().__init__()
        self.team_file_dir = team_file_dir
        self.battle_format = battle_format
        self.team_files = self._find_team_files()

    def _find_team_files(self):
        team_files = []
        for root, _, files in os.walk(self.team_file_dir):
            for file in files:
                if file.endswith(f".{self.battle_format}_team"):
                    team_files.append(os.path.join(root, file))
        return team_files

    def yield_team(self):
        file = random.choice(self.team_files)
        with open(file, "r") as f:
            team_data = f.read()
        team = self.parse_showdown_team(team_data)
        print(team)
        for mon in team:
            if mon.species is not None:
                mon.nickname = mon.species
        return self.join_team(team)

def get_metamon_teams(battle_format: str, set_name: str) -> TeamSet:
    """
    Download a set of teams from huggingface (if necessary) and return a TeamSet.

    Args:
        battle_format: The battle format of the team files (e.g. "gen1ou", "gen2ubers", etc.).
        set_name: The name of the set of teams to download. See the README for options.
    """
    if set_name not in {
        "competitive",
        "paper_replays",
        "paper_variety",
        "modern_replays",
        "pokeagent_modern_replays",
    }:
        raise ValueError(
            f"Invalid set name: {set_name}. Must be one of: competitive, paper_replays, paper_variety, modern_replays"
        )
    if battle_format == "gen9vgc2025regi":
        path = 'bayesian_dataset'
    else:
        path = download_teams(battle_format, set_name=set_name)
    if not os.path.exists(path):
        raise ValueError(
            f"Cannot locate valid team directory for format {battle_format} at path {path}"
        )
    
    # Check if team files exist for this format
    team_set = TeamSet(path, battle_format)
    if not team_set.team_files:
        raise ValueError(
            f"No team files found for format {battle_format} in {path}. "
            f"Expected files with extension '.{battle_format}_team'"
        )
    
    return team_set

def load_random_team(id=None, vgc=False):
    if id == None:
        team_id = randint(1, 14)
    else:
        team_id = id
    if vgc is True:
        print(f'Loading VGC team {team_id}')
        with open(f'poke_env/data/static/teams/gen9vgc2025regi/gen9vgc2025regi{team_id}.txt', 'r') as f:
            team = f.read()
    else:
        with open(f'poke_env/data/static/teams/gen9ou/gen9ou{team_id}.txt', 'r') as f:
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
    from pokechamp.llm_player import LLMPlayer
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
                   online: bool=False,
                   use_timeout: bool=True,
                   timeout_seconds: int=90) -> Player:
    from pokechamp.llm_player import LLMPlayer
    from pokechamp.prompts import prompt_translate, state_translate2, state_translate3
    
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
        if use_timeout and online:
            from pokechamp.timeout_llm_player import PokellmonTimeoutLLMPlayer
            return PokellmonTimeoutLLMPlayer(battle_format=battle_format,
                           api_key=KEY,
                           backend=backend,
                           temperature=args.temperature,
                           prompt_algo=prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                           server_configuration=server_config,
                           save_replays=args.log_dir,
                           device=device,
                           llm_backend=llm_backend,
                           timeout_seconds=timeout_seconds)
        else:
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
        # Use VGC player for VGC formats, timeout player for online mode, regular player for others
        if 'vgc' in battle_format:
            return LLMVGCPlayer(battle_format=battle_format,
                           api_key=KEY,
                           backend=backend,
                           temperature=args.temperature,
                           prompt_algo=prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                           server_configuration=server_config,
                           save_replays=args.log_dir,
                           prompt_translate=state_translate3,
                           device=device,
                           llm_backend=llm_backend)
        elif use_timeout and online:
            from pokechamp.timeout_llm_player import TimeoutLLMPlayer
            return TimeoutLLMPlayer(battle_format=battle_format,
                           api_key=KEY,
                           backend=backend,
                           temperature=args.temperature,
                           prompt_algo=prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                           server_configuration=server_config,
                           save_replays=args.log_dir,
                           prompt_translate=state_translate2,
                           device=device,
                           llm_backend=llm_backend,
                           timeout_seconds=timeout_seconds)
        else:
            return LLMPlayer(battle_format=battle_format,
                           api_key=KEY,
                           backend=backend,
                           temperature=args.temperature,
                           prompt_algo=prompt_algo,
                           log_dir=args.log_dir,
                           account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                           server_configuration=server_config,
                           save_replays=args.log_dir,
                           prompt_translate=state_translate2,
                           device=device,
                           llm_backend=llm_backend)
    elif 'vgc' in name:
        return LLMVGCPlayer(battle_format=battle_format,
                       api_key=KEY,
                       backend=backend,
                       temperature=args.temperature,
                       prompt_algo=prompt_algo,
                       log_dir=args.log_dir,
                       account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                       server_configuration=server_config,
                       save_replays=args.log_dir,
                       # Use state_translate3 for VGC formats, state_translate2 for others
                       prompt_translate=state_translate3 if "vgc" in battle_format.lower() else state_translate2,
                       device=device,
                       llm_backend=llm_backend)
    elif 'pokechamp' in name:
        return LLMPlayer(battle_format=battle_format,
                       api_key=KEY,
                       backend=backend,
                       temperature=args.temperature,
                       prompt_algo=prompt_algo,
                    #    prompt_algo="minimax",
                    #    prompt_algo="io",
                       log_dir=args.log_dir,
                       account_configuration=AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
                       server_configuration=server_config,
                       save_replays=args.log_dir,
                    #    prompt_translate=prompt_translate,
                       prompt_translate=state_translate3 if "vgc" in battle_format.lower() else state_translate2,
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