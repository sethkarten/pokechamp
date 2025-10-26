import asyncio
from tqdm import tqdm
import os
import sys
import argparse

# Import visual effects early
try:
    from pokechamp.visual_effects import visual, print_banner, print_status
    VISUAL_EFFECTS = True
except ImportError:
    VISUAL_EFFECTS = False

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from common import *
from poke_env.player.team_util import get_llm_player, get_metamon_teams, load_random_team

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument("--player_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--player_backend", type=str, default="gemini-2.5-flash", choices=[
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
    "ollama/gpt-oss:20b", "ollama/llama3.1:8b", "ollama/mistral", "ollama/qwen2.5", "ollama/gemma3:4b",
    # Local models (via OpenRouter)
    "llama", 'None'
])
parser.add_argument("--player_name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--player_device", type=int, default=0)

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--opponent_backend", type=str, default="gemini-2.5-pro", choices=[
    # OpenAI models
    "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro",
    # Ollama models
    "ollama/gpt-oss:20b", "ollama/llama3.1:8b", "ollama/mistral", "ollama/qwen2.5", "ollama/gemma3:4b",
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
parser.add_argument("--opponent_name", type=str, default='pokellmon', choices=bot_choices)
parser.add_argument("--opponent_device", type=int, default=0)

# Shared arguments
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle", "gen9vgc2025regi"])
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")
parser.add_argument("--N", type=int, default=25)
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
    # Visual banner for local battles
    if VISUAL_EFFECTS:
        print_banner("LOCAL", "fire")
        print_banner("BATTLE", "water")
        print(f"Player: {args.player_name} ({args.player_backend})")
        print(f"Opponent: {args.opponent_name} ({args.opponent_backend})")
        print(f"Format: {args.battle_format}")
        print("=" * 50)
    else:
        print(f"\n=== LOCAL BATTLE ===")
        print(f"Player: {args.player_name} vs Opponent: {args.opponent_name}")
        print(f"Format: {args.battle_format}\n")
    player = get_llm_player(args, 
                            args.player_backend, 
                            args.player_prompt_algo, 
                            args.player_name, 
                            device=args.player_device,
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format)
    
    opponent = get_llm_player(args, 
                            args.opponent_backend, 
                            args.opponent_prompt_algo, 
                            args.opponent_name, 
                            device=args.opponent_device,
                            PNUMBER1=PNUMBER1 + '2',  # for name uniqueness locally
                            battle_format=args.battle_format)

    # Try to use metamon teams, fallback to static teams if not available
    player_teamloader = None
    opponent_teamloader = None
    
    try:
        player_teamloader = get_metamon_teams(args.battle_format, "competitive")
        opponent_teamloader = get_metamon_teams(args.battle_format, "modern_replays")
    except (ValueError, Exception) as e:
        if VISUAL_EFFECTS:
            print_status(f"Metamon teams not available for {args.battle_format}: {e}", "warning")
            print_status("Falling back to static teams...", "info")
        else:
            print(f"Metamon teams not available for {args.battle_format}: {e}")
            print(f"Falling back to static teams...")
    
    if not 'random' in args.battle_format:
        if player_teamloader is None or opponent_teamloader is None:
            # Fallback to static teams when metamon teams not available
            player.update_team(load_random_team(id=None, vgc=False))
            opponent.update_team(load_random_team(id=None, vgc=False))
        else:
            # Use metamon teams if available
            player.set_teamloader(player_teamloader)
            opponent.set_teamloader(opponent_teamloader)
            
            player.update_team(player_teamloader.yield_team())
            opponent.update_team(opponent_teamloader.yield_team())

    # play against bot for five battles
    N = args.N
    pbar = tqdm(total=N)
    for i in range(N):
        x = np.random.randint(0, 100)
        if x > 50:
            await player.battle_against(opponent, n_battles=1)
        else:
            await opponent.battle_against(player, n_battles=1)
        
        if not 'random' in args.battle_format:
            if 'vgc' in args.battle_format:
                player.update_team(load_random_team(id=None, vgc=True))
                opponent.update_team(load_random_team(id=None, vgc=True))
            elif player_teamloader is None or opponent_teamloader is None:
                # Fallback to static teams when metamon teams not available
                player.update_team(load_random_team(id=None, vgc=False))
                opponent.update_team(load_random_team(id=None, vgc=False))
            else:
                # Use metamon teams if available
                player.update_team(player_teamloader.yield_team())
                opponent.update_team(opponent_teamloader.yield_team())
        pbar.set_description(f"{player.win_rate*100:.2f}%")
        pbar.update(1)
    print(f'player winrate: {player.win_rate*100}')


if __name__ == "__main__":
    asyncio.run(main())
