import asyncio
from tqdm import tqdm
import argparse

from common import *
from poke_env.player.team_util import get_llm_player, load_random_team

parser = argparse.ArgumentParser()
parser.add_argument("--temperature", type=float, default=0.3)
parser.add_argument("--prompt_algo", default="minimax", choices=prompt_algos)
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle"])
parser.add_argument("--backend", type=str, default="gpt-4o", choices=[
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
