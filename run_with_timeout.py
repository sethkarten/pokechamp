#!/usr/bin/env python3
"""
Concurrent version of run_with_timeout.py that can run multiple non-conflicting battles simultaneously.
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
# results_lock = Lock()

# Queue to trigger immediate restart when all agents are freed
restart_queue = Queue()
restart_lock = Lock()

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
    # {
    #     "name": "pokechamp",
    #     "backend": "ollama/gemma3:12b-it-qat",
    #     "prompt_algo": "minimax",
    #     "device": 6,
    #     "username": "PAC-PC-gemma3-12b",
    #     "password": None,  # Will be loaded from PASSWORDS
    # },
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
    # {
    #     "name": "pokechamp",
    #     "backend": "ollama/qwen3:14b",
    #     "prompt_algo": "minimax",
    #     "device": 6,
    #     "username": "PAC-PC-qwen3-14b",
    #     "password": None,  # Will be loaded from PASSWORDS
    # },
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
    # {
    #     "name": "pokechamp",
    #     "backend": "ollama/gpt-oss:20b",
    #     "prompt_algo": "minimax",
    #     "device": 6,
    #     "username": "PAC-PC-gpt-oss",
    #     "password": None,  # Will be loaded from PASSWORDS
    # },
    # {
    #     "name": "pokechamp",
    #     "backend": "gemini-2.5-flash",
    #     "prompt_algo": "minimax",
    #     "device": 6,
    #     "username": "PAC-PC-gem25f",
    #     "password": None,  # Will be loaded from PASSWORDS
    # },
    # {
    #     "name": "pokechamp",
    #     "backend": "gemini-2.5-pro",
    #     "prompt_algo": "minimax",
    #     "device": 6,
    #     "username": "PAC-PC-gem25p",
    #     "password": None,  # Will be loaded from PASSWORDS
    # },
    
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
    # {
    #     "name": "pokechamp",
    #     "backend": "gemini-2.5-pro",
    #     "prompt_algo": "io",
    #     "device": 6,
    #     "username": "PAC-LLM-gem25p",
    #     "password": None,  # Will be loaded from PASSWORDS
    # },
    
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

def mark_agent_free(agent):
    """Mark single agent as free after ladder session."""
    with battle_lock:
        agents_in_battle.discard(agent["username"])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agent freed: {agent['username']}. Currently in battle: {agents_in_battle}")
        
        # If no agents are left in battle, trigger immediate restart
        if len(agents_in_battle) == 0:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents freed - triggering immediate restart!")
            restart_queue.put("restart")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Restart signal queued!")

def get_log_filename(agent1, agent2):
    """Generate a log filename for the matchup."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"battle_log/vs_battles/{agent1['username']}_vs_{agent2['username']}_{timestamp}.txt"

def get_results_filename():
    """Get the results JSON filename."""
    return "battle_results_ladder_concurrent.json"

def load_results():
    """Load existing results from file."""
    try:
        with open(get_results_filename(), 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_results(results):
    """Save results to file."""
    # with results_lock:
    with open(get_results_filename(), 'w') as f:
        json.dump(results, f, indent=2)

def are_agents_available(agent1, agent2):
    """Check if both agents are available for battle."""
    with battle_lock:
        agent1_username = agent1["username"]
        agent2_username = agent2["username"]
        return agent1_username not in agents_in_battle and agent2_username not in agents_in_battle

def try_reserve_agents(agent1, agent2):
    """Atomically check and reserve agents for battle. Returns True if successful."""
    with battle_lock:
        agent1_username = agent1["username"]
        agent2_username = agent2["username"]
        
        # Check if both agents are available
        if agent1_username in agents_in_battle or agent2_username in agents_in_battle:
            return False
        
        # Reserve both agents atomically
        agents_in_battle.add(agent1_username)
        agents_in_battle.add(agent2_username)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agents reserved for battle: {agents_in_battle}")
        return True

def mark_agents_busy(agent1, agent2):
    """Mark agents as busy in battle (deprecated - use try_reserve_agents instead)."""
    with battle_lock:
        agents_in_battle.add(agent1["username"])
        agents_in_battle.add(agent2["username"])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agents now in battle: {agents_in_battle}")

def mark_agents_free(agent1, agent2):
    """Mark agents as free after battle."""
    with battle_lock:
        agents_in_battle.discard(agent1["username"])
        agents_in_battle.discard(agent2["username"])
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Agents freed. Currently in battle: {agents_in_battle}")

def parse_separate_process_results(agent1_results_file, agent2_results_file, agent1, agent2):
    """Parse results from separate agent process result files."""
    try:
        import json
        
        # Load results from both agents
        agent1_data = None
        agent2_data = None
        
        try:
            with open(agent1_results_file, 'r') as f:
                agent1_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load agent1 results: {e}")
            
        try:
            with open(agent2_results_file, 'r') as f:
                agent2_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load agent2 results: {e}")
        
        if not agent1_data or not agent2_data:
            return None
        
        # Combine results from both agents
        result = {
            "agent1_name": agent1_data.get("agent_name", agent1["name"]),
            "agent1_wins": agent1_data.get("wins", 0),
            "agent1_losses": agent1_data.get("losses", 0),
            "agent1_win_rate": agent1_data.get("win_rate", 0.0),
            "agent2_name": agent2_data.get("agent_name", agent2["name"]),
            "agent2_wins": agent2_data.get("wins", 0),
            "agent2_losses": agent2_data.get("losses", 0),
            "agent2_win_rate": agent2_data.get("win_rate", 0.0),
            "total_games": agent1_data.get("games_completed", 0),
            "parsed_successfully": True,
            "separate_process_results": True
        }
        
        # Add timeout statistics if available
        if "timeout_stats" in agent1_data:
            result["agent1_timeout_stats"] = agent1_data["timeout_stats"]
            
        if "timeout_stats" in agent2_data:
            result["agent2_timeout_stats"] = agent2_data["timeout_stats"]
        
        return result
        
    except Exception as e:
        print(f"Warning: Could not parse separate process results: {e}")
        return None

def run_ladder_session(agent, session_id):
    """Run a single agent on the ladder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"battle_log/ladder/{agent['username']}_{timestamp}.txt"
    
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
    
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Starting ladder session #{session_id}: {agent['username']}")
    print(f"  Games: {GAMES_PER_MATCHUP} | Timeout: {TIMEOUT_SECONDS}s | LLM Timeout: {LLM_TIMEOUT_SECONDS}s | Seed: {unique_seed}")
    print(f"  Log: {log_file}")
    print(f"  🎯 LADDER BATTLES: Agent vs random opponents on Pokemon Showdown")
    
    try:
        with open(log_file, "w") as f:
            f.write(f"===== Ladder Session #{session_id}: {agent['username']} =====\n")
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
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Completed ladder session #{session_id}: {agent['username']}")
        
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
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout in ladder session #{session_id}: {agent['username']}")
        return {
            "status": "timeout",
            "log_file": log_file,
            "timestamp": datetime.now().isoformat(),
            "agent": agent["username"]
        }
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in ladder session #{session_id}: {e}")
        return {
            "status": "error",
            "error": str(e),
            "log_file": log_file,
            "timestamp": datetime.now().isoformat(),
            "agent": agent["username"]
        }
    finally:
        # Mark agent as free after session completes or fails
        mark_agent_free(agent)

def ladder_worker(session_queue, results_dict, agent_games_completed):
    """Worker thread that processes ladder sessions from the queue."""
    import threading
    worker_name = threading.current_thread().name
    print(f"[Worker {worker_name}] Started")
    
    while True:
        try:
            # Get next session from queue
            session_data = session_queue.get(timeout=5)
            if session_data is None:  # Poison pill to stop worker
                print(f"[Worker {worker_name}] Received stop signal")
                break
                
            agent, session_id, session_key = session_data
            print(f"[Worker {worker_name}] Got session #{session_id} for {agent['username']}")
            
            # Run the ladder session
            result = run_ladder_session(agent, session_id)
            
            # Update completed games count (simplified, no results lock)
            if result.get('status') == 'completed':
                games_completed = result.get('total_games', GAMES_PER_MATCHUP)
                agent_games_completed[agent['username']] += games_completed
                print(f"[Worker {worker_name}] Updated {agent['username']} games: {agent_games_completed[agent['username']]}")
            
            # Skip results storage for now to avoid lock issues
            # with results_lock:
            #     if session_key not in results_dict:
            #         results_dict[session_key] = []
            #     results_dict[session_key].append(result)
            #     save_results(results_dict)
            
            session_queue.task_done()
            
        except Empty:
            print(f"[Worker {worker_name}] Queue timeout - still alive, waiting for work...")
            continue
        except Exception as e:
            print(f"[Worker {worker_name}] Error: {e}")
            import traceback
            traceback.print_exc()
            session_queue.task_done()

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
                    print(f"  ✅ Sent forfeit command")
                    
                    # Brief delay to let forfeit process
                    await asyncio.sleep(2)
                    
                    nonlocal forfeit_count
                    forfeit_count += 1
                    
                except Exception as e:
                    print(f"  ❌ Error with {agent['username']}: {e}")
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
                        print(f"  ⚠️  Cleanup error: {cleanup_error}")
            
            # Run the forfeit operation with timeout
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(asyncio.wait_for(forfeit_agent_battles(), timeout=15))
                loop.close()
            except asyncio.TimeoutError:
                print(f"  ⚠️  Timeout checking {agent['username']} (15s)")
            except Exception as e:
                print(f"  ⚠️  Error during forfeit for {agent['username']}: {e}")
            
        except Exception as e:
            print(f"  ❌ Failed to setup forfeit for {agent['username']}: {e}")
        
        time.sleep(2)  # Small delay between agents
    
    print(f"\n{'='*60}")
    print(f"FORFEIT COMPLETE")
    print(f"Forfeit commands sent: {forfeit_count}")
    print(f"All agents should be ready for tournament")
    print(f"{'='*60}\n")
    
    return forfeit_count

def run_concurrent_ladder(skip_forfeit=False, games_per_agent_target=1000, continuous_mode=False):
    """Run concurrent ladder sessions for all agents."""
    # First, forfeit any existing battles (unless skipped)
    if not skip_forfeit:
        forfeit_existing_battles()
    else:
        print("Skipping forfeit step as requested")
    
    results = load_results()
    
    print(f"\n{'='*60}")
    print(f"CONCURRENT LADDER TOURNAMENT")
    print(f"Agents: {[agent['username'] for agent in AGENTS]}")
    print(f"Total agents: {len(AGENTS)}")
    print(f"Games per agent per session: {GAMES_PER_MATCHUP}")
    print(f"Max concurrent sessions: {MAX_CONCURRENT_BATTLES}")
    print(f"Target games per agent: {games_per_agent_target}")
    print(f"Total target games: {games_per_agent_target * len(AGENTS)}")
    print(f"{'='*60}\n")
    
    # Create queue for ladder sessions
    session_queue = Queue()
    
    # Track sessions and games completed per agent
    session_counter = 0
    agent_games_completed = {agent["username"]: 0 for agent in AGENTS}
    agent_games_queued = {agent["username"]: 0 for agent in AGENTS}
    games_per_session = GAMES_PER_MATCHUP
    
    # Start worker threads
    workers = []
    for i in range(MAX_CONCURRENT_BATTLES):
        worker = Thread(target=ladder_worker, args=(session_queue, results, agent_games_completed))
        worker.daemon = True
        worker.start()
        workers.append(worker)
    
    # Initial queue of all agents
    for agent in AGENTS:
        username = agent["username"]
        with battle_lock:
            agents_in_battle.add(username)
            session_counter += 1
            session_key = f"{username}_ladder_{session_counter}"
            session_queue.put((agent, session_counter, session_key))
            agent_games_queued[username] += games_per_session
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Queued ladder session #{session_counter}: {username} "
                  f"(target: {games_per_agent_target} total games)")
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial {len(AGENTS)} agents queued. Starting continuous queueing...")
    
    # Continuously check and requeue agents until they reach target games
    all_agents_complete = False
    loop_counter = 0
    while not all_agents_complete:
        try:
            loop_counter += 1
            if loop_counter % 5 == 0:  # Print every 10 seconds
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Main loop iteration {loop_counter}")
            time.sleep(2)  # Check every 2 seconds for faster response
            
            # Debug: Show current battle state
            with battle_lock:
                if len(agents_in_battle) == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: No agents currently in battle")
            
            # Check for restart signal OR if all agents are free (belt and suspenders approach)
            restart_needed = False
            
            # First check the restart queue
            try:
                restart_signal = restart_queue.get(block=False)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: Got restart signal: {restart_signal}")
                if restart_signal == "restart":
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Got restart signal from queue")
                    restart_needed = True
                    restart_queue.task_done()
            except Empty:
                # Only print this occasionally to avoid spam
                if session_counter % 10 == 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: No restart signal in queue")
            
            # Also check if all agents are free (backup check)
            with battle_lock:
                if len(agents_in_battle) == 0:
                    # In continuous mode, always restart when all agents are free
                    # In non-continuous mode, only restart if agents haven't reached target
                    if continuous_mode:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents free in continuous mode - forcing restart")
                        restart_needed = True
                    else:
                        # Check if any agent needs more games
                        any_need_games = False
                        for agent in AGENTS:
                            if agent_games_completed[agent["username"]] < games_per_agent_target:
                                any_need_games = True
                                break
                        
                        if any_need_games:
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents free and need more games - forcing restart")
                            restart_needed = True
            
            # Process restart if needed
            if restart_needed:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Processing restart - requeuing all agents")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: session_queue size before: {session_queue.qsize()}")
                
                # Clear battle state
                with battle_lock:
                    agents_in_battle.clear()
                
                # In continuous mode, reset counters for infinite restart
                if continuous_mode:
                    for agent in AGENTS:
                        username = agent["username"]
                        agent_games_completed[username] = 0
                        agent_games_queued[username] = 0
                
                # Requeue all agents that need more games
                agents_requeued = 0
                for agent in AGENTS:
                    username = agent["username"]
                    if continuous_mode or agent_games_completed[username] < games_per_agent_target:
                        # Add to battle tracking
                        with battle_lock:
                            agents_in_battle.add(username)
                        
                        # Queue the session
                        session_counter += 1
                        session_key = f"{username}_ladder_{session_counter}"
                        session_queue.put((agent, session_counter, session_key))
                        agent_games_queued[username] += games_per_session
                        agents_requeued += 1
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Queued ladder session #{session_counter}: {username}")
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Restart complete: {agents_requeued} agents requeued")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] DEBUG: session_queue size after: {session_queue.qsize()}")
                continue  # Skip normal requeue logic
            
            # Check if we need to queue more sessions
            agents_to_requeue = []
            
            # Synchronized restart: Wait for ALL agents to complete their current round
            with battle_lock:
                # Check how many agents have completed their current round
                agents_ready_for_next_round = []
                agents_still_battling = []
                
                for agent in AGENTS:
                    username = agent["username"]
                    completed = agent_games_completed[username]
                    queued = agent_games_queued[username]
                    
                    if username in agents_in_battle:
                        agents_still_battling.append(username)
                    elif completed < games_per_agent_target and queued <= completed:
                        agents_ready_for_next_round.append(agent)
                
                # Only restart when ALL agents are done with current round
                if len(agents_still_battling) == 0 and len(agents_ready_for_next_round) == len(AGENTS):
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] All {len(AGENTS)} agents completed round - synchronized restart!")
                    print(f"  Agents completed: {[a['username'] for a in agents_ready_for_next_round]}")
                    agents_to_requeue = agents_ready_for_next_round
                elif len(agents_still_battling) > 0:
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for {len(agents_still_battling)} agents to complete: {agents_still_battling}")
                
                # Special case: all agents timed out or errored (battle set empty but not all reached target)
                if len(agents_in_battle) == 0 and len(agents_ready_for_next_round) < len(AGENTS):
                    all_stuck = True
                    for agent in AGENTS:
                        username = agent["username"]
                        if agent_games_queued[username] > agent_games_completed[username]:
                            all_stuck = False  # Some agent still has pending games
                            break
                    
                    if all_stuck:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents appear stuck - forcing synchronized restart!")
                        for agent in AGENTS:
                            username = agent["username"]
                            if agent_games_completed[username] < games_per_agent_target:
                                agents_to_requeue.append(agent)
            
            # Requeue agents that need more games
            for agent in agents_to_requeue:
                username = agent["username"]
                with battle_lock:
                    agents_in_battle.add(username)
                    session_counter += 1
                    session_key = f"{username}_ladder_{session_counter}"
                    session_queue.put((agent, session_counter, session_key))
                    agent_games_queued[username] += games_per_session
                    
                    completed = agent_games_completed[username]
                    remaining = games_per_agent_target - completed
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Requeued ladder session #{session_counter}: {username} "
                          f"(completed: {completed}/{games_per_agent_target}, remaining: {remaining})")
            
            # Check if all agents have completed their target
            # In continuous mode, never exit - keep restarting agents
            # with results_lock:
            #     all_complete = all(
            #         agent_games_completed[agent["username"]] >= games_per_agent_target 
            #         for agent in AGENTS
            #     )
            #     if all_complete and not continuous_mode:
            #         all_agents_complete = True
            #         print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents have reached target games!")
            #     elif all_complete and continuous_mode:
            #         print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents reached target - resetting for continuous mode")
            #         # Reset counters to continue indefinitely
            #         for agent in AGENTS:
            #             username = agent["username"]
            #             agent_games_completed[username] = 0
            #             agent_games_queued[username] = 0
                        
        except KeyboardInterrupt:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Stopping ladder tournament due to keyboard interrupt...")
            all_agents_complete = True
            break
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error in main loop: {e}")
            if continuous_mode:
                print("Continuing in continuous mode despite error...")
                # Force clear battle state and continue
                force_clear_battle_state()
                time.sleep(10)  # Wait a bit before retrying
            else:
                raise
    
    # Wait for all remaining sessions to complete
    print(f"[{datetime.now().strftime('%H:%M:%S')}] All targets reached. Waiting for remaining sessions to complete...")
    session_queue.join()
    
    # Stop workers
    for _ in workers:
        session_queue.put(None)
    for worker in workers:
        worker.join()
    
    # Print final statistics
    print(f"\n{'='*60}")
    print(f"ALL LADDER SESSIONS COMPLETED")
    print(f"Total sessions run: {session_counter}")
    print(f"\nGames per agent:")
    for agent in AGENTS:
        username = agent["username"]
        completed = agent_games_completed[username]
        queued = agent_games_queued[username]
        print(f"  {username}: {completed} completed / {queued} queued")
    print(f"\nTotal games completed: {sum(agent_games_completed.values())}")
    print(f"Total games queued: {sum(agent_games_queued.values())}")
    print(f"Results saved to: {get_results_filename()}")
    print(f"{'='*60}\n")
    
    # Print summary
    print_summary(results)

def print_summary(results):
    """Print a summary of all results."""
    print("\nRESULTS SUMMARY:")
    print("-" * 40)
    
    total_completed = 0
    total_timeout = 0
    total_error = 0
    
    for matchup, runs in results.items():
        print(f"\n{matchup}:")
        for run in runs:
            status = run.get('status', 'unknown')
            timestamp = run.get('timestamp', 'unknown')
            print(f"  - {status} at {timestamp}")
            
            if status == 'completed':
                total_completed += 1
            elif status == 'timeout':
                total_timeout += 1
            elif status == 'error':
                total_error += 1
    
    print(f"\nTotals: {total_completed} completed, {total_timeout} timeouts, {total_error} errors")

def force_clear_battle_state():
    """Force clear all agents from battle state and restart clean."""
    global agents_in_battle
    with battle_lock:
        if agents_in_battle:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Force clearing stuck battle state: {agents_in_battle}")
            agents_in_battle.clear()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] All agents forcibly freed for clean restart")
    
    # Also clear any restart signals
    try:
        while True:
            restart_queue.get(block=False)
            restart_queue.task_done()
    except Empty:
        pass  # Queue is empty

def run_continuous(skip_forfeit=False, games_per_agent_target=1000):
    """Run ladder sessions continuously in a loop."""
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
        run_concurrent_ladder(skip_forfeit=skip_forfeit, games_per_agent_target=games_per_agent_target, continuous_mode=True)
        
    except KeyboardInterrupt:
        print("\nStopping ladder tournament...")
    except Exception as e:
        print(f"\nError in ladder tournament: {e}")
        print("Stopping continuous mode.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run concurrent Pokemon ladder tournament")
    parser.add_argument("--games-per-round", type=int, default=1, help="Number of games per agent per round (before sync)")
    parser.add_argument("--target-games", type=int, default=1000, help="Total target games per agent before full restart")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds per ladder session")
    parser.add_argument("--max-concurrent", type=int, default=16, help="Maximum concurrent ladder sessions")
    parser.add_argument("--continuous", action="store_true", help="Run continuously in a loop")
    parser.add_argument("--delay", type=int, default=30, help="Delay between rounds in seconds")
    parser.add_argument("--llm-timeout", type=int, default=90, help="LLM timeout per move in seconds")
    parser.add_argument("--skip-forfeit", action="store_true", help="Skip forfeiting existing battles at startup")
    
    args = parser.parse_args()
    
    # Update global configuration
    GAMES_PER_MATCHUP = args.games_per_round  # Number of games each agent runs before sync restart
    TIMEOUT_SECONDS = args.timeout
    MAX_CONCURRENT_BATTLES = args.max_concurrent
    RESTART_DELAY = args.delay
    LLM_TIMEOUT_SECONDS = args.llm_timeout
    
    try:
        if args.continuous:
            print("Running in continuous mode. Press Ctrl+C to stop.")
            run_continuous(skip_forfeit=args.skip_forfeit, games_per_agent_target=args.target_games)
        else:
            run_concurrent_ladder(skip_forfeit=args.skip_forfeit, games_per_agent_target=args.target_games, continuous_mode=False)
    except KeyboardInterrupt:
        print("\nExiting run_with_timeout.py")