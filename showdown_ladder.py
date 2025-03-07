import asyncio
from time import sleep
from tqdm import tqdm
import argparse

from common import *
from poke_env.player.team_util import get_llm_player, load_random_team

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"])
parser.add_argument("--backend", type=str, default="gpt-4o", choices=["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "llama", 'None'])
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random'])
parser.add_argument("--USERNAME", type=str, default='')
parser.add_argument("--PASSWORD", type=str, default='')
args = parser.parse_args()
    
async def main():
    player = get_llm_player(args, 
                            args.backend, 
                            args.prompt_algo, 
                            args.name, 
                            battle_format=args.battle_format, 
                            online=True, 
                            USERNAME=args.USERNAME, 
                            PASSWORD=args.PASSWORD)
    if not 'random' in args.battle_format:
        player.update_team(load_random_team(1))

    # Playing n_challenges games on the ladder
    n_challenges = 5
    pbar = tqdm(total=n_challenges)
    wins = 0
    for i in range(n_challenges):
        print('starting ladder')
        await player.ladder(1)
        winner = 'opponent'
        if player.win_rate > 0: 
            winner = args.name
            wins += 1
        if not 'random' in args.battle_format:
            player.update_team(load_random_team())
        sleep(30)
        pbar.set_description(f"{wins/(i+1)*100:.2f}%")
        pbar.update(1)
        print(winner)
        player.reset_battles()
    print(f'player 2 winrate: {wins/n_challenges*100}')

if __name__ == "__main__":
    asyncio.run(main())