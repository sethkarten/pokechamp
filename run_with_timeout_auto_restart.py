#!/usr/bin/env python3
"""
Auto-restart version of run_with_timeout.py that immediately restarts individual agents 
when they finish their battles instead of waiting for all agents to complete.
"""

import subprocess
import time
import sys
import json
from datetime import datetime
from itertools import combinations
from threading import Lock, Thread
from queue import Queue, Empty
import os
import asyncio
import re

# Configuration
TIMEOUT_SECONDS = 2 * 60 * 60  # 2 hours per matchup
RESTART_DELAY = 30  # 30 seconds between checking for new matchups
GAMES_PER_MATCHUP = 1  # Number of games per matchup (configurable)
MAX_CONCURRENT_BATTLES = 3  # Maximum number of concurrent battles
LLM_TIMEOUT_SECONDS = 90  # LLM timeout per move (configurable)

def load_passwords():
    """Load passwords from a secure JSON file."""
    password_file = "passwords.json"
    try:
        with open(password_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {password_file} not found. Please create it from passwords.json.example")
        print("Copy passwords.json.example to passwords.json and fill in your actual passwords.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {password_file}: {e}")
        sys.exit(1)

# Track which agents are currently in battle
agents_in_battle = set()
battle_lock = Lock()

# Load passwords once at startup
PASSWORDS = load_passwords()

# Define the agents and their configurations
AGENTS = [
    # PokeChamp Harness LLM-based agents
    {
        "name": "pokechamp",
        "backend": "ollama/llama3.1:8b-instruct-q4_K_M",
        "prompt_algo": "minimax",
        "device": 6,
        "username": "PAC-PC-llama31-8b",
        "password": PASSWORDS["PAC-PC-llama31-8b"],
    },
    {
        "name": "pokechamp",
        "backend": "ollama/gemma3:4b-it-qat",
        "prompt_algo": "minimax",
        "device": 6,
        "username": "PAC-PC-gemma3-4b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/gemma3:1b-it-qat",
        "prompt_algo": "minimax",
        "device": 6,
        "username": "PAC-PC-gemma3-1b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/qwen3:8b",
        "prompt_algo": "minimax",
        "device": 6,
        "username": "PAC-PC-qwen3-8b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/qwen3:4b",
        "prompt_algo": "minimax",
        "device": 6,
        "username": "PAC-PC-qwen3-4b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    
    # LLM-based agents
    {
        "name": "pokechamp",
        "backend": "ollama/llama3.1:8b-instruct-q4_K_M",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-llama31-8b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/gemma3:12b-it-qat",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-gemma3-12b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/gemma3:4b-it-qat",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-gemma3-4b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/gemma3:1b-it-qat",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-gemma3-1b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/qwen3:14b",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-qwen3-14b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/qwen3:8b",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-qwen3-8b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/qwen3:4b",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-qwen3-4b",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "ollama/gpt-oss:20b",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-gpt-oss",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "gemini-2.5-flash",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-gem25f",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "pokechamp",
        "backend": "gemini-2.5-flash-lite",
        "prompt_algo": "io",
        "device": 6,
        "username": "PAC-LLM-gem25fl",
        "password": None,  # Will be loaded from PASSWORDS
    },
    
    # PokeLLMon baseline agents
    {
        "name": "pokellmon",
        "backend": "ollama/llama3.1:8b-instruct-q4_K_M",
        "prompt_algo": "io",
        "device": 5,
        "username": "PAC-PC-pokellmon",
        "password": None,  # Will be loaded from PASSWORDS
    },
    
    # Non-LLM baseline agents (active by default)
    {
        "name": "one_step",
        "backend": "None",
        "prompt_algo": "one_step",
        "device": 0,
        "username": "PAC-PC-DC",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "abyssal",
        "backend": "None",
        "prompt_algo": "heuristic",
        "device": 0,
        "username": "PAC-PC-ABYSSAL",
        "password": None,  # Will be loaded from PASSWORDS
    },
    {
        "name": "max_power",
        "backend": "None",
        "prompt_algo": "max_power",
        "device": 0,
        "username": "PAC-PC-MAX-POWER",
        "password": None,  # Will be loaded from PASSWORDS
    },
]

def initialize_agent_passwords():
    """Initialize agent passwords from the password file."""
    for agent in AGENTS:
        username = agent["username"]
        if username not in PASSWORDS:
            print(f"Error: Password not found for username '{username}' in passwords.json")
            print(f"Available usernames: {list(PASSWORDS.keys())}")
            sys.exit(1)
        agent["password"] = PASSWORDS[username]
    print(f"Successfully loaded passwords for {len(AGENTS)} agents")

# Initialize passwords for all agents
initialize_agent_passwords()

def parse_ladder_results(log_file, agent):
    """Parse ladder battle log to extract win/loss results."""
    try:
        with open(log_file, 'r') as f:
            content = f.read()
        
        # Look for win rate pattern from showdown_ladder.py
        win_rate_pattern = r"player 2 winrate: ([\d.]+)"
        timeout_pattern = r"Timeout rate: ([\d.]+)%"
        
        win_rate_match = re.search(win_rate_pattern, content)
        timeout_match = re.search(timeout_pattern, content)
        
        if win_rate_match:
            win_rate = float(win_rate_match.group(1))
            wins = int(win_rate * GAMES_PER_MATCHUP / 100)
            losses = GAMES_PER_MATCHUP - wins
            
            result = {
                "agent_name": agent["name"],
                "wins": wins,
                "losses": losses,
                "win_rate": win_rate,
                "total_games": GAMES_PER_MATCHUP,
                "parsed_successfully": True
            }
            
            if timeout_match:
                result["timeout_rate"] = float(timeout_match.group(1))
            
            return result
        
        return None
        
    except Exception as e:
        print(f"Warning: Could not parse ladder results from {log_file}: {e}")
        return None

def get_results_filename():
    """Get the results JSON filename."""
    return "battle_results_ladder_auto_restart.json"

def load_results():
    """Load existing results from file."""
    try:
        with open(get_results_filename(), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_results(results):
    """Save results to file."""
    with open(get_results_filename(), 'w') as f:
        json.dump(results, f, indent=2)

def run_ladder_session(agent, session_id):
    """Run a single agent on the ladder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"battle_log_gen9ou/ladder_auto_restart/{agent['username']}_{timestamp}.txt"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Generate unique seed for this session
    # Use session_id and current time to ensure uniqueness
    unique_seed = (session_id * 1000 + int(time.time() * 1000)) % (2**31 - 1)
    
    # Build command for showdown_ladder.py (single agent vs ladder)
    cmd = [
        sys.executable,
        "scripts/battles/showdown_ladder.py",
        # Player configuration
        "--name", agent["name"],
        "--backend", agent["backend"],
        "--prompt_algo", agent["prompt_algo"],
        "--device", str(agent["device"]),
        "--USERNAME", agent["username"],
        "--PASSWORD", agent["password"],
        # Number of games
        "--N", str(GAMES_PER_MATCHUP),
        # Timeout protection
        "--timeout", str(LLM_TIMEOUT_SECONDS),
        # Unique seed for this process
        "--seed", str(unique_seed),
    ]
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting auto-restart ladder session #{session_id}: {agent['username']}")
    print(f"  Games: {GAMES_PER_MATCHUP} | Timeout: {TIMEOUT_SECONDS}s | LLM Timeout: {LLM_TIMEOUT_SECONDS}s | Seed: {unique_seed}")
    print(f"  Log: {log_file}")
    from pokechamp.visual_effects import visual, print_banner
    print_banner("AUTO-RESTART LADDER", "fire")
    print("Agent vs random opponents on Pokemon Showdown - IMMEDIATE RESTART")
    
    try:
        with open(log_file, "w") as f:
            f.write(f"===== Auto-restart Ladder Session #{session_id}: {agent['username']} =====\n")
            f.write(f"Started at: {datetime.now()}\n")
            f.write(f"Command: {' '.join(cmd)}\n\n")
            f.flush()
            
            # Run the ladder session with timeout
            subprocess.run(
                cmd, 
                timeout=TIMEOUT_SECONDS, 
                stdout=f, 
                stderr=subprocess.STDOUT,
                text=True,
                capture_output=False
            )
            
            f.write(f"\n===== Completed at: {datetime.now()} =====\n")
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed auto-restart ladder session #{session_id}: {agent['username']}")
        
        # Parse the log file to extract ladder results
        battle_results = parse_ladder_results(log_file, agent)
        
        result = {
            "status": "completed",
            "log_file": log_file,
            "timestamp": datetime.now().isoformat(),
            "agent": agent["username"]
        }
        
        # Add detailed results if parsing was successful
        if battle_results:
            result.update(battle_results)
            print(f"  Results: {agent['username']} - {battle_results.get('wins', 0)}W-{battle_results.get('losses', 0)}L ({battle_results.get('win_rate', 0):.1f}%)")
        
        return result
        
    except subprocess.TimeoutExpired:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout in auto-restart ladder session #{session_id}: {agent['username']}")
        return {
            "status": "timeout",
            "log_file": log_file,
            "timestamp": datetime.now().isoformat(),
            "agent": agent["username"]
        }
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in auto-restart ladder session #{session_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "log_file": log_file,
            "timestamp": datetime.now().isoformat(),
            "agent": agent["username"]
        }

def agent_runner(agent, session_queue, results_dict, agent_games_completed, continuous_mode, games_per_agent_target):
    """
    Dedicated thread for a single agent that runs sessions and immediately restarts.
    This is the key difference from the original - each agent runs independently.
    """
    import threading
    thread_name = threading.current_thread().name
    agent_username = agent["username"]
    session_counter = 0
    
    print(f"[Agent Runner {agent_username}] Started dedicated thread")
    
    while True:
        try:
            # Check if we should stop (only in non-continuous mode when target reached)
            if not continuous_mode and agent_games_completed[agent_username] >= games_per_agent_target:
                print(f"[Agent Runner {agent_username}] Reached target {games_per_agent_target} games - stopping")
                break
            
            # Reset games counter for continuous mode
            if continuous_mode and agent_games_completed[agent_username] >= games_per_agent_target:
                print(f"[Agent Runner {agent_username}] Resetting games counter for continuous mode")
                agent_games_completed[agent_username] = 0
            
            # Mark agent as busy and run session
            with battle_lock:
                agents_in_battle.add(agent_username)
                session_counter += 1
                current_session = session_counter
                
            print(f"[Agent Runner {agent_username}] Starting session #{current_session}")
            
            # Run the ladder session
            result = run_ladder_session(agent, current_session)
            
            # Update completed games count
            if result.get('status') == 'completed':
                games_completed = result.get('total_games', GAMES_PER_MATCHUP)
                agent_games_completed[agent_username] += games_completed
                print(f"[Agent Runner {agent_username}] Completed session #{current_session} - Total games: {agent_games_completed[agent_username]}")
            else:
                print(f"[Agent Runner {agent_username}] Session #{current_session} failed with status: {result.get('status')}")
            
            # Store result
            session_key = f"{agent_username}_auto_restart_{current_session}"
            results_dict[session_key] = result
            save_results(results_dict)
            
            # Mark agent as free
            with battle_lock:
                agents_in_battle.discard(agent_username)
                print(f"[Agent Runner {agent_username}] Freed after session #{current_session}. Currently in battle: {agents_in_battle}")
            
            # Immediate restart - no waiting for other agents!
            print(f"[Agent Runner {agent_username}] Immediately restarting (auto-restart mode)")
            time.sleep(1)  # Brief pause to avoid overwhelming the system
            
        except KeyboardInterrupt:
            print(f"[Agent Runner {agent_username}] Stopping due to keyboard interrupt")
            break
        except Exception as e:
            print(f"[Agent Runner {agent_username}] Error in session: {e}")
            # Mark agent as free on error
            with battle_lock:
                agents_in_battle.discard(agent_username)
            
            if continuous_mode:
                print(f"[Agent Runner {agent_username}] Continuing despite error in continuous mode")
                time.sleep(10)  # Wait before retrying
            else:
                print(f"[Agent Runner {agent_username}] Stopping due to error")
                break
    
    print(f"[Agent Runner {agent_username}] Thread exiting")

def forfeit_existing_battles():
    """Forfeit any existing battles for all agents to ensure clean start."""
    print(f"\n{'='*60}")
    print("FORFEITING EXISTING BATTLES")
    print(f"{'='*60}")
    print("Note: Attempting to connect briefly and send forfeit commands")
    
    forfeit_count = 0
    
    for agent in AGENTS:
        print(f"Checking agent: {agent['name']} ({agent['username']})")
        
        try:
            # Create a temporary player to connect and forfeit
            from poke_env.ps_client.account_configuration import AccountConfiguration
            from poke_env.ps_client.server_configuration import ShowdownServerConfiguration
            from poke_env.player.random_player import RandomPlayer
            
            # Create a minimal player just for forfeiting
            temp_player = RandomPlayer(
                battle_format="gen9ou",
                account_configuration=AccountConfiguration(agent["username"], agent["password"]),
                server_configuration=ShowdownServerConfiguration,
                start_listening=False  # Don't auto-start listening
            )
            
            # Connect and forfeit any existing battles
            
            async def forfeit_agent_battles():
                listen_task = None
                try:
                    # Start listening connection in background
                    listen_task = asyncio.create_task(temp_player.ps_client.listen())
                    
                    # Wait for connection to establish
                    await asyncio.sleep(3)
                    
                    # Wait for login completion
                    await temp_player.ps_client.wait_for_login(wait_for=5)
                    
                    # Send universal forfeit command - works for any active battles
                    await temp_player.ps_client.send_message("/forfeit")
                    from pokechamp.visual_effects import print_status
                    print_status("Sent forfeit command", "success")
                    
                    # Brief delay to let forfeit process
                    await asyncio.sleep(2)
                    
                    nonlocal forfeit_count
                    forfeit_count += 1
                    
                except Exception as e:
                    print_status(f"Error with {agent['username']}: {e}", "error")
                finally:
                    # Ensure clean shutdown regardless of errors
                    try:
                        if listen_task and not listen_task.done():
                            listen_task.cancel()
                            try:
                                await asyncio.wait_for(listen_task, timeout=2)
                            except (asyncio.CancelledError, asyncio.TimeoutError):
                                pass
                        
                        # Force close websocket if still open
                        if hasattr(temp_player.ps_client, 'websocket'):
                            try:
                                await temp_player.ps_client.websocket.close()
                            except Exception:
                                pass
                                
                    except Exception as cleanup_error:
                        print(f"  [WARN] Cleanup error: {cleanup_error}")
            
            # Run the forfeit operation with timeout
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(asyncio.wait_for(forfeit_agent_battles(), timeout=15))
                loop.close()
            except asyncio.TimeoutError:
                print(f"  [WARN] Timeout checking {agent['username']} (15s)")
            except Exception as e:
                print(f"  [WARN] Error during forfeit for {agent['username']}: {e}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to setup forfeit for {agent['username']}: {e}")
        
        time.sleep(2)  # Small delay between agents
    
    print(f"\n{'='*60}")
    print(f"FORFEIT COMPLETE")
    print(f"Forfeit commands sent: {forfeit_count}")
    print(f"All agents should be ready for tournament")
    print(f"{'='*60}\n")
    
    return forfeit_count

def run_auto_restart_ladder(skip_forfeit=False, games_per_agent_target=1000, continuous_mode=False):
    """
    Run auto-restart ladder sessions where each agent restarts immediately after finishing.
    No waiting for other agents - each runs independently.
    """
    # First, forfeit any existing battles (unless skipped)
    if not skip_forfeit:
        forfeit_existing_battles()
    else:
        print("Skipping forfeit step as requested")
    
    results = load_results()
    
    print(f"\n{'='*60}")
    print(f"AUTO-RESTART LADDER TOURNAMENT")
    print(f"Each agent restarts IMMEDIATELY after finishing - no sync waiting!")
    print(f"Agents: {[agent['username'] for agent in AGENTS]}")
    print(f"Total agents: {len(AGENTS)}")
    print(f"Games per agent per session: {GAMES_PER_MATCHUP}")
    print(f"Target games per agent: {games_per_agent_target}")
    print(f"Continuous mode: {continuous_mode}")
    print(f"{'='*60}\n")
    
    # Track games completed per agent
    agent_games_completed = {agent["username"]: 0 for agent in AGENTS}
    
    # Start dedicated thread for each agent
    agent_threads = []
    
    for agent in AGENTS:
        thread = Thread(
            target=agent_runner, 
            args=(agent, None, results, agent_games_completed, continuous_mode, games_per_agent_target),
            name=f"Agent-{agent['username']}"
        )
        thread.daemon = True
        thread.start()
        agent_threads.append(thread)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Started dedicated thread for {agent['username']}")
        time.sleep(0.5)  # Stagger thread starts slightly
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] All {len(AGENTS)} agent threads started!")
    
    # Monitor and wait for completion (or run forever in continuous mode)
    try:
        if continuous_mode:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Running in continuous mode - press Ctrl+C to stop")
            while True:
                time.sleep(10)
                # Print periodic status
                with battle_lock:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Status - Agents in battle: {len(agents_in_battle)}")
                    for agent in AGENTS:
                        username = agent["username"]
                        completed = agent_games_completed[username]
                        status = "BATTLING" if username in agents_in_battle else "FREE"
                        print(f"  {username}: {completed} games, {status}")
        else:
            # Wait for all agents to reach target
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for all agents to reach {games_per_agent_target} games...")
            while True:
                time.sleep(5)
                
                all_complete = all(
                    agent_games_completed[agent["username"]] >= games_per_agent_target 
                    for agent in AGENTS
                )
                
                if all_complete:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents have reached target games!")
                    break
                
                # Print progress
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Progress:")
                for agent in AGENTS:
                    username = agent["username"]
                    completed = agent_games_completed[username]
                    remaining = max(0, games_per_agent_target - completed)
                    status = "BATTLING" if username in agents_in_battle else "FREE"
                    print(f"  {username}: {completed}/{games_per_agent_target} games, {remaining} remaining, {status}")
    
    except KeyboardInterrupt:
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stopping auto-restart ladder tournament...")
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"AUTO-RESTART LADDER SESSIONS COMPLETED")
    print(f"\nFinal games per agent:")
    total_games = 0
    for agent in AGENTS:
        username = agent["username"]
        completed = agent_games_completed[username]
        total_games += completed
        print(f"  {username}: {completed} games")
    print(f"\nTotal games completed: {total_games}")
    print(f"Results saved to: {get_results_filename()}")
    print(f"{'='*60}\n")

def run_continuous(skip_forfeit=False, games_per_agent_target=1000):
    """Run auto-restart ladder sessions continuously in a loop."""
    try:
        # Check battle state before starting continuous mode
        with battle_lock:
            if agents_in_battle:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] WARNING: Detected agents stuck in battle state: {agents_in_battle}")
                print("Force clearing battle state to allow continuous running...")
                agents_in_battle.clear()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents forcibly freed for clean restart")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Battle state is clean - starting continuous mode")
        
        # Run in continuous mode - this will never exit unless interrupted or errors
        run_auto_restart_ladder(skip_forfeit=skip_forfeit, games_per_agent_target=games_per_agent_target, continuous_mode=True)
        
    except KeyboardInterrupt:
        print("\nStopping auto-restart ladder tournament...")
    except Exception as e:
        print(f"\nError in auto-restart ladder tournament: {e}")
        print("Stopping continuous mode.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run auto-restart Pokemon ladder tournament")
    parser.add_argument("--games-per-session", type=int, default=1, help="Number of games per agent per session")
    parser.add_argument("--target-games", type=int, default=1000, help="Total target games per agent before stopping")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds per ladder session")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--llm-timeout", type=int, default=90, help="LLM timeout per move in seconds")
    parser.add_argument("--skip-forfeit", action="store_true", help="Skip forfeiting existing battles at startup")
    
    args = parser.parse_args()
    
    # Update global configuration
    GAMES_PER_MATCHUP = args.games_per_session
    TIMEOUT_SECONDS = args.timeout
    LLM_TIMEOUT_SECONDS = args.llm_timeout
    
    try:
        if args.continuous:
            print("Running in continuous auto-restart mode. Press Ctrl+C to stop.")
            run_continuous(skip_forfeit=args.skip_forfeit, games_per_agent_target=args.target_games)
        else:
            run_auto_restart_ladder(skip_forfeit=args.skip_forfeit, games_per_agent_target=args.target_games, continuous_mode=False)
    except KeyboardInterrupt:
        print("\nExiting run_with_timeout_auto_restart.py")