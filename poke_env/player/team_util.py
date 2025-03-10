import random
import os

from poke_env.player.player import Player
from poke_env.player.baselines import AbyssalPlayer, MaxBasePowerPlayer, OneStepPlayer
from poke_env.player.random_player import RandomPlayer
from poke_env.player.llm_player import LLMPlayer
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.server_configuration import ShowdownServerConfiguration, LocalhostServerConfiguration
from poke_env.teambuilder import Teambuilder
from poke_env.data import DATA_PATH

from poke_env.player.prompts import prompt_translate

def load_random_team(battle_format: str) -> str:
    battle_format = battle_format.lower()
    path = os.path.join(DATA_PATH, "static", "teams", f"{battle_format}_sample_teams")
    if not os.path.exists(path):
        raise ValueError(
            f"Cannot locate valid team directory for format {battle_format}"
        )
    print(path)
    choice = random.choice(os.listdir(path))
    path_to_choice = os.path.join(path, choice)
    return path_to_choice


class UniformRandomTeambuilder(Teambuilder):
    def __init__(self, battle_format: str):
        self.battle_format = battle_format

    def yield_team(self):
        team = load_random_team(self.battle_format)
        with open(os.path.join(team), "r", encoding="utf-8") as f:
            team_data = f.read()
        self.team_name = os.path.basename(team)
        return self.join_team(self.parse_showdown_team(team_data))


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
                   ladder: str = "local") -> Player:

    if ladder == "online":
        server_config = ShowdownServerConfiguration
    elif ladder == "local":
        server_config = LocalhostServerConfiguration
    else:
        raise ValueError(f"Invalid ladder type: {ladder}")

    player_kwargs = {
        "battle_format": battle_format,
        "account_configuration": AccountConfiguration(f'{USERNAME}{PNUMBER1}', PASSWORD),
        "server_configuration": server_config,
    }

    if "random" not in battle_format:
        player_kwargs["team"] = UniformRandomTeambuilder(battle_format)
    
    if ladder == "local":
        player_kwargs.update({
            # prevent losses on time (as long the opponent does the same)
            "ping_timeout" : 1000,
            "start_timer_on_battle_start" : False,
        })

    if USERNAME == '':
        USERNAME = name
    if prompt_algo == 'abyssal':
        return AbyssalPlayer(**player_kwargs)
    elif prompt_algo == 'max_power':
        return MaxBasePowerPlayer(**player_kwargs)
    elif prompt_algo == 'random':
        return RandomPlayer(**player_kwargs)
    elif prompt_algo == 'one_step':
        return OneStepPlayer(**player_kwargs)
    elif 'pokellmon' in name:
        return LLMPlayer(api_key=KEY,
                       backend=backend,
                       temperature=args.temperature,
                       prompt_algo=prompt_algo,
                       log_dir=args.log_dir,
                       save_replays=args.log_dir,
                       device=device,
                       llm_backend=llm_backend,
                       **player_kwargs)
    elif 'pokechamp' in name:
        return LLMPlayer(api_key=KEY,
                       backend=backend,
                       temperature=args.temperature,
                       prompt_algo="minimax",
                       log_dir=args.log_dir,
                       save_replays=args.log_dir,
                       prompt_translate=prompt_translate,
                       device=device,
                       llm_backend=llm_backend,
                       **player_kwargs)
    else:
        raise ValueError('Bot not found')