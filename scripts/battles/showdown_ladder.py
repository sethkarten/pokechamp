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
parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9oulongtimer", "gen9randombattle", "gen9vgc2025regi"])
parser.add_argument("--backend", type=str, default="gemini-2.5-flash", choices=[
    # OpenAI models
    "gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo",
    # Anthropic models
    "anthropic/claude-3.5-sonnet", "anthropic/claude-3-opus", "anthropic/claude-3-haiku",
    # Google models
    "google/gemini-pro", "gemini-2.0-flash", "gemini-2.0-pro", "gemini-2.0-flash-lite", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite", 
    "gemini-3-pro-preview", "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview",
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
    "llama", 'None',
    # Frontier models (via OpenRouter)
    "minimax/minimax-m2.5", "qwen/qwen3.5-plus-02-15", "deepseek/deepseek-chat",
    "x-ai/grok-3-mini", "x-ai/grok-3", "z-ai/glm-5", "moonshotai/kimi-k2.5",
    "openai/gpt-5", "anthropic/claude-sonnet-4.6", "anthropic/claude-opus-4.6",
])
parser.add_argument("--log_dir", type=str, default="./battle_log/ladder")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--name", type=str, default='pokechamp', choices=bot_choices)
parser.add_argument("--USERNAME", type=str, default='')
parser.add_argument("--PASSWORD", type=str, default='')
parser.add_argument("--N", type=int, default=1)
parser.add_argument("--timeout", type=int, default=90, help="LLM timeout in seconds (0 to disable)")
parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
parser.add_argument("--min_sleep", type=int, default=30, help="Minimum sleep between games in seconds")
parser.add_argument("--max_sleep", type=int, default=120, help="Maximum sleep between games in seconds")
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
    # Try to use metamon teams, fallback to static teams if not available
    teamloader = None
    
    try:
        team_list = "modern_replays"
        # if args.name == 'pokechamp':
        #     team_list = "modern_replays"
        print(f"Using {team_list} team list")
        teamloader = get_metamon_teams(args.battle_format, team_list)
    except Exception as e:
        print(f"Metamon teams not available for {args.battle_format}: {e}")
        print(f"Falling back to static teams...")
    
    if not 'random' in args.battle_format:
        if teamloader is None:
            # Fallback to static teams when metamon teams not available
            player.update_team(load_random_team(id=None, vgc=False))
        else:
            # Use metamon teams if available
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
        print(f'Starting ladder game {i+1}/{n_challenges}')
        
        # Handle connection drops and retries
        max_retries = 3
        game_completed = False
        
        for retry in range(max_retries):
            try:
                # Ensure connection is established before starting game
                connection_ok = False
                try:
                    if hasattr(player.ps_client, 'websocket') and player.ps_client.websocket and not player.ps_client.websocket.closed:
                        connection_ok = True
                except Exception:
                    pass
                
                if not connection_ok:
                    print(f'Connection lost, waiting for reconnection before game {i+1}, attempt {retry+1}')
                    # Wait for connection to be re-established
                    reconnect_attempts = 0
                    max_reconnect_attempts = 5
                    while reconnect_attempts < max_reconnect_attempts:
                        try:
                            print(f'   Attempting to reconnect ({reconnect_attempts+1}/{max_reconnect_attempts})...')
                            await player.ps_client.wait_for_login(wait_for=10)
                            print(f'   ✅ Reconnection successful!')
                            break
                        except Exception as reconnect_error:
                            reconnect_attempts += 1
                            print(f'   ❌ Reconnection attempt {reconnect_attempts} failed: {reconnect_error}')
                            if reconnect_attempts < max_reconnect_attempts:
                                await asyncio.sleep(5)  # Wait before retry
                    
                    if reconnect_attempts >= max_reconnect_attempts:
                        print(f'   💀 Failed to reconnect after {max_reconnect_attempts} attempts')
                        raise Exception("Could not reconnect to server")
                
                await player.ladder(1)
                game_completed = True
                break  # Success, exit retry loop
                
            except Exception as e:
                print(f'Error in game {i+1}, attempt {retry+1}/{max_retries}: {e}')
                if retry == max_retries - 1:
                    print(f'Failed to complete game {i+1} after {max_retries} attempts, skipping')
                else:
                    print(f'Retrying game {i+1} in 30 seconds...')
                    sleep(30)  # Longer wait before retry
                    # Reset the player state before retrying
                    try:
                        player.reset_battles()
                    except Exception:
                        pass
        
        # Skip win calculation if game didn't complete
        if not game_completed:
            print(f'Game {i+1} did not complete, continuing to next game')
            pbar.update(1)
            continue
        
        winner = 'opponent'
        # Check win rate safely to avoid division by zero
        try:
            if player.n_finished_battles > 0 and player.win_rate > 0: 
                winner = args.name
                wins += 1
        except (ZeroDivisionError, AttributeError):
            # If no battles finished or win_rate unavailable, assume loss
            pass
        if not 'random' in args.battle_format:
            player.update_team(teamloader.yield_team())
        pbar.set_description(f"{wins/(i+1)*100:.2f}%")
        pbar.update(1)
        print(winner)
        
        # Reset battles after each game
        try:
            player.reset_battles()
        except Exception as e:
            print(f'Warning: Could not reset battles: {e}')
        
        # Use configurable sleep between games instead of hard-coded 10-60s
        sleep_duration = random.randint(args.min_sleep, args.max_sleep)
        print(f'Sleeping {sleep_duration}s before next game')
        sleep(sleep_duration)
    print(f'player 2 winrate: {wins/n_challenges*100}')
    
    # Print timeout statistics if using timeout player
    if hasattr(player, 'get_timeout_stats'):
        stats = player.get_timeout_stats()
        print(f"\nTimeout Statistics:")
        print(f"  Total moves: {stats['total_moves']}")
        print(f"  Timeouts: {stats['timeouts']}")
        print(f"  Timeout rate: {stats['timeout_rate']:.1f}%")

if __name__ == "__main__":
    # Proper asyncio event loop management to ensure all tasks complete
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        # Run main function
        loop.run_until_complete(main())
        
        # Wait for any remaining tasks to complete
        pending_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        if pending_tasks:
            print(f"Waiting for {len(pending_tasks)} remaining tasks to complete...")
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Proper cleanup
        print("Cleaning up async resources...")
        
        # Cancel any remaining tasks
        remaining_tasks = [task for task in asyncio.all_tasks(loop) if not task.done()]
        for task in remaining_tasks:
            task.cancel()
        
        # Wait for cancelled tasks to finish
        if remaining_tasks:
            loop.run_until_complete(asyncio.gather(*remaining_tasks, return_exceptions=True))
        
        # Shutdown async generators
        loop.run_until_complete(loop.shutdown_asyncgens())
        
        # Close the loop
        loop.close()
        
        print("Ladder session completely finished.")