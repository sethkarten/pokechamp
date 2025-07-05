import asyncio
from tqdm import tqdm
import os
import argparse

from common import *
from poke_env.player.team_util import get_llm_player, load_random_team

parser = argparse.ArgumentParser()

# Player arguments
parser.add_argument("--player_prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--player_backend", type=str, default="gpt-4o", choices=[
    # OpenAI models
    "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "google/gemini-flash-1.5",
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
parser.add_argument("--player_name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--player_device", type=int, default=0)

# Opponent arguments
parser.add_argument("--opponent_prompt_algo", default="io", choices=prompt_algos)
parser.add_argument("--opponent_backend", type=str, default="gpt-4o", choices=[
    # OpenAI models
    "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "google/gemini-flash-1.5",
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
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"])
parser.add_argument("--log_dir", type=str, default="./battle_log/one_vs_one")

args = parser.parse_args()

async def main():
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
                            PNUMBER1=PNUMBER1,  # for name uniqueness locally
                            battle_format=args.battle_format)

    if not 'random' in args.battle_format:
        player.update_team(load_random_team())
        opponent.update_team(load_random_team())

    # play against bot for five battles
    N = 5
    pbar = tqdm(total=N)
    for i in range(N):
        x = np.random.randint(0, 100)
        if x > 50:
            await player.battle_against(opponent, n_battles=1)
        else:
            await opponent.battle_against(player, n_battles=1)
        if not 'random' in args.battle_format:
            player.update_team(load_random_team())
            opponent.update_team(load_random_team())
        pbar.set_description(f"{player.win_rate*100:.2f}%")
        pbar.update(1)
    print(f'player winrate: {player.win_rate*100}')


if __name__ == "__main__":
    asyncio.run(main())
