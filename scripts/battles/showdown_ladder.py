import asyncio
from time import sleep
from tqdm import tqdm
import argparse
import os, sys

# Add the current directory to Python path (since we're in project root)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from common import *
from poke_env.player.team_util import get_llm_player, get_metamon_teams, load_random_team

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.5)
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"])
parser.add_argument("--backend", type=str, default="gpt-4o", choices=[
    # OpenAI models
    "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite",
    # Meta models
    "meta-llama/llama-3.1-70b-instruct", "meta-llama/llama-3.1-8b-instruct",
    # Mistral models
    "mistralai/mistral-7b-instruct", "mistralai/mixtral-8x7b-instruct",
    # Cohere models
    "cohere/command-r-plus", "cohere/command-r",
    # Perplexity models
    "perplexity/llama-3.1-sonar-small-128k", "perplexity/llama-3.1-sonar-large-128k",
    # DeepSeek models
    "deepseek-ai/deepseek-coder-33b-instruct", "deepseek-ai/deepseek-llm-67b-chat",
    # Microsoft models
    "microsoft/wizardlm-2-8x22b", "microsoft/phi-3-medium-128k-instruct",
    # Local models (via OpenRouter)
    "llama", 'None'
])
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--USERNAME", type=str, default='')
parser.add_argument("--PASSWORD", type=str, default='')
parser.add_argument("--N", type=int, default=1)
args = parser.parse_args()
    
async def main():
    player = get_llm_player(args, 
                            args.backend, 
                            args.prompt_algo, 
                            args.name, 
                            device=args.device,
                            battle_format=args.battle_format, 
                            online=True, 
                            USERNAME=args.USERNAME, 
                            PASSWORD=args.PASSWORD)
    
    teamloader = get_metamon_teams(args.battle_format, "modern_replays")
    
    if not 'random' in args.battle_format:
        # Set teamloader on player for rejection recovery
        player.set_teamloader(teamloader)
        player.update_team(teamloader.yield_team())

    # Playing n_challenges games on the ladder
    n_challenges = args.N
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
            player.update_team(teamloader.yield_team())
        sleep(30)
        pbar.set_description(f"{wins/(i+1)*100:.2f}%")
        pbar.update(1)
        print(winner)
        player.reset_battles()
    print(f'player 2 winrate: {wins/n_challenges*100}')

if __name__ == "__main__":
    asyncio.run(main())