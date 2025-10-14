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
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle", "gen9vgc2025regi"])
parser.add_argument("--backend", type=str, default="gemini-2.5-flash", choices=[
    # OpenAI models
    "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
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
    # Ollama models
    "ollama/gpt-oss:20b", "ollama/llama3.1:8b", "ollama/llama3.1:8b-instruct-q4_K_M", 
    "ollama/mistral", "ollama/qwen2.5:32b", "ollama/qwen3:30b", "ollama/qwen3:14b", 
    "ollama/qwen3:8b", "ollama/qwen3:4b", "ollama/gemma3:4b", "ollama/gemma3:4b-it-qat", 
    "ollama/gemma3:1b-it-qat", "ollama/gemma3:27b", "ollama/gemma3:27b-it-qat", 
    "ollama/gemma3:12b-it-qat", "ollama/gpt-oss:20b", "ollama/llama3.1:8b",
    # Local models (via OpenRouter)
    "llama", 'None'
])
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--USERNAME", type=str, default='')
parser.add_argument("--PASSWORD", type=str, default='')
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--timeout", type=int, default=90, help="LLM timeout in seconds (0 to disable)")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
args = parser.parse_args()

# Set random seed if provided
if args.seed is not None:
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"Using random seed: {args.seed}")
    
async def main():
    player = get_llm_player(args, 
                            args.backend, 
                            args.prompt_algo, 
                            args.name, 
                            device=args.device,
                            battle_format=args.battle_format, 
                            online=True, 
                            USERNAME=args.USERNAME, 
                            PASSWORD=args.PASSWORD,
                            use_timeout=(args.timeout > 0),
                            timeout_seconds=args.timeout)
    team_list = "modern_replays"
    if args.name == 'pokechamp':
        team_list = "competitive"
    print(f"Using {team_list} team list")
    teamloader = get_metamon_teams(args.battle_format, team_list)
    
    if not 'random' in args.battle_format:
        # Set teamloader on player for rejection recovery
        player.set_teamloader(teamloader)
        player.update_team(teamloader.yield_team())

    # Warm up player components before battles to avoid turn-time delays
    print("[WARMUP] Warming up player before battles...")
    if hasattr(player, 'warm_up'):
        player.warm_up()
    print("[WARMUP] Player warm-up complete!")

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
    
    # Print timeout statistics if using timeout player
    if hasattr(player, 'get_timeout_stats'):
        stats = player.get_timeout_stats()
        print(f"\nTimeout Statistics:")
        print(f"  Total moves: {stats['total_moves']}")
        print(f"  Timeouts: {stats['timeouts']}")
        print(f"  Timeout rate: {stats['timeout_rate']:.1f}%")

if __name__ == "__main__":
    asyncio.run(main())