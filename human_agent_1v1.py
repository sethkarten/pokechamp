import asyncio
from tqdm import tqdm
import argparse

from common import *
from poke_env.player.team_util import get_llm_player, load_random_team

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle", "gen9vgc2025regi"])
parser.add_argument("--backend", type=str, default="gpt-4o", choices=["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "llama", 'None'])
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random'])
args = parser.parse_args()

async def main():

    opponent = get_llm_player(args, 
                            args.backend, 
                            args.prompt_algo, 
                            args.name, 
                            PNUMBER1=PNUMBER1,
                            battle_format=args.battle_format)
    if not 'random' in args.battle_format:
        if 'vgc' in args.battle_format:
            opponent.update_team(load_random_team(id=1, vgc=True))
        else: 
            opponent.update_team(load_random_team())                     
    
    # Playing 5 games on local
    for i in tqdm(range(1)):
        await opponent.ladder(1)
        if not 'random' in args.battle_format:
            opponent.update_team(load_random_team())

if __name__ == "__main__":
    asyncio.run(main())
