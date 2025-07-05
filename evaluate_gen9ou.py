import asyncio
import pandas as pd
from argparse import ArgumentParser
from poke_env.player.utils import cross_evaluate
from common import PNUMBER1
from poke_env.player.llama_player import LLAMAPlayer
from whr import whr as whole_history_rating
from poke_env.player.team_util import get_llm_player, load_random_team

parser = ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--log_dir", type=str, default="./battle_log/gen9ou")
parser.add_argument("--device", type=int, default=0)
args = parser.parse_args()

async def evaluate_gen9ou():
    file = f'battle_log/gen9ou_{PNUMBER1}.csv'
    n_battles = 1
    players = []
    
    combos = [
        ('gpt-4o', 'pokechamp', 'minimax'),
        ('gpt-4o', 'pokellmon', 'io'),
        ('abyssal', 'abyssal', 'abyssal'),
        ('one_step', 'one_step', 'one_step'),
        ('random', 'random', 'random'),
    ]

   
    llm = None
    device = args.device
    for i, (backend, name, algo) in enumerate(combos):
        if 'llama' in backend:
            llm = LLAMAPlayer(device=device)
            device += 1
        llm_backend = llm if 'llama' in backend else None
        print(backend, algo, name)
        player = get_llm_player(args, backend, algo, name, llm_backend=llm_backend, PNUMBER1=PNUMBER1)
        player.team = load_random_team(i + 1)
        players.append(player)
    
    await cross_evaluate(players, n_challenges=n_battles, file=file)
    
    # Calculate Elo ratings
    df = pd.read_csv(file)
    elo_ratings = whole_history_rating(df)
    
    print("Win rates:")
    for player in players:
        print(f"{player.username}:")
        for opponent in players:
            if player != opponent:
                wins = df[(df['model_a'] == player.username) & (df['model_b'] == opponent.username) & (df['winner'] == 'model_a')].shape[0]
                total = df[(df['model_a'] == player.username) & (df['model_b'] == opponent.username)].shape[0]
                win_rate = wins / total if total > 0 else 0
                print(f"  vs {opponent.username}: {win_rate:.2%}")

    print("\nElo ratings:")
    for player, rating in elo_ratings.items():
        print(f"{player}: {rating:.2f}")

    # Save Elo ratings to CSV
    elo_ratings.to_csv('gen9ou_elo_ratings.csv')

if __name__ == "__main__":
    asyncio.run(evaluate_gen9ou())
