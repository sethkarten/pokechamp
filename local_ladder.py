import argparse
import asyncio

from common import *
from poke_env.player.team_util import get_llm_player
    
async def main(args):
    player = get_llm_player(args, 
                            args.backend, 
                            prompt_algo="minimax" if args.method == "pokechamp" else "io", 
                            name=args.method,
                            battle_format=args.battle_format, 
                            ladder="local",
                            USERNAME=args.USERNAME, 
                            PASSWORD=args.PASSWORD)
    await player.ladder(args.n_challenges)
    win_loss = [b.won for b in player.battles.values()]
    rating = [b.rating for b in player.battles.values()]
    print(f"Win Rate: {sum(win_loss) / len(win_loss)}")
    print(f"Rating History : {rating}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--method", type=str, default="pokellmon", choices=["pokellmon", "pokechamp"])
    parser.add_argument("--backend", type=str, default="gpt-4o-mini", choices=["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-05-13", "llama", 'None'])
    parser.add_argument("--battle_format", default="gen9ou", choices=["gen8randombattle", "gen8ou", "gen9ou", "gen9randombattle", "gen1ou", "gen2ou", "gen3ou", "gen4ou"])
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--USERNAME", type=str, default='')
    parser.add_argument("--PASSWORD", type=str, default=None)
    parser.add_argument("--n_challenges", type=int, default=5)
    args = parser.parse_args()

    asyncio.get_event_loop().run_until_complete(main(args))
