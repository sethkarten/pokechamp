#!/usr/bin/env python3
"""
Host Gemini 3 Flash and Gemini 3.1 Flash Lite on pokeagentchallenge.com.
Each bot accepts challenges from any user.
A shared daily budget of $100 is enforced — when hit, no new challenges are accepted.
Budget resets at midnight UTC.
"""

import asyncio
import json
import os
import sys
import argparse
from datetime import date, datetime
from threading import Lock, Thread
from time import sleep

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
sys.path.insert(0, project_root)

from common import get_metamon_teams
from poke_env.player.team_util import get_llm_player

# ── Config ────────────────────────────────────────────────────────────────────

DAILY_BUDGET_USD = 100.0
BUDGET_FILE = os.path.join(project_root, "pac_budget.json")
BATTLE_FORMAT = "gen9oulongtimer"
LLM_TIMEOUT = 90

# USD per token: (prompt, completion)
PRICING = {
    "gemini-3-flash-preview":        (0.0000005,  0.000003),
    "gemini-3.1-flash-lite-preview": (0.00000025, 0.000001),
}

AGENTS = [
    {
        "username":  "PAC-LLM-gem3f",
        "backend":   "gemini-3-flash-preview",
    },
    {
        "username":  "PAC-LLM-gem31fl",
        "backend":   "gemini-3.1-flash-lite-preview",
    },
]

# ── Budget tracker ─────────────────────────────────────────────────────────────

_budget_lock = Lock()

def _load_budget() -> dict:
    today = str(date.today())
    try:
        with open(BUDGET_FILE) as f:
            data = json.load(f)
        if data.get("date") != today:
            data = {"date": today, "spent_usd": 0.0}
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"date": today, "spent_usd": 0.0}
    return data

def _save_budget(data: dict):
    with open(BUDGET_FILE, "w") as f:
        json.dump(data, f)

def add_cost(prompt_tokens: int, completion_tokens: int, backend: str) -> float:
    """Add cost for a completed battle; return new daily total."""
    pp, cp = PRICING.get(backend, (0.0, 0.0))
    cost = prompt_tokens * pp + completion_tokens * cp
    with _budget_lock:
        data = _load_budget()
        data["spent_usd"] += cost
        _save_budget(data)
        total = data["spent_usd"]
    print(f"[budget] +${cost:.4f} ({backend}) → daily total ${total:.4f}/{DAILY_BUDGET_USD}")
    return total

def budget_remaining() -> float:
    with _budget_lock:
        data = _load_budget()
    return DAILY_BUDGET_USD - data["spent_usd"]

# ── Agent loop ─────────────────────────────────────────────────────────────────

def run_agent(agent: dict, passwords: dict):
    username = agent["username"]
    backend  = agent["backend"]
    password = passwords.get(username, "")

    print(f"[{username}] starting, backend={backend}")

    class _Args:
        temperature = 0.7
        prompt_algo = "io"
        log_dir     = os.path.join(project_root, "battle_log", "pac")
        battle_format = BATTLE_FORMAT

    teamloader = None
    try:
        teamloader = get_metamon_teams(BATTLE_FORMAT, "modern_replays")
    except Exception as e:
        print(f"[{username}] metamon teams unavailable: {e}")

    player = get_llm_player(
        _Args(),
        backend,
        "io",
        "pokechamp",
        battle_format=BATTLE_FORMAT,
        online=True,
        server='pac',
        USERNAME=username,
        PASSWORD=password,
        use_timeout=True,
        timeout_seconds=LLM_TIMEOUT,
    )

    if teamloader:
        player.set_teamloader(teamloader)
        player.update_team(teamloader.yield_team())

    if hasattr(player, "warm_up"):
        player.warm_up()

    game_num = 0
    while True:
        remaining = budget_remaining()
        if remaining <= 0:
            print(f"[{username}] daily budget exhausted (${DAILY_BUDGET_USD}), sleeping until midnight UTC")
            now = datetime.utcnow()
            secs_until_midnight = (24 - now.hour) * 3600 - now.minute * 60 - now.second
            sleep(secs_until_midnight + 5)
            continue

        game_num += 1
        print(f"[{username}] waiting for challenge (game #{game_num}, budget remaining ${remaining:.2f})")

        # snapshot token counts before battle
        pt_before = getattr(player, "prompt_tokens", 0)
        ct_before = getattr(player, "completion_tokens", 0)

        try:
            asyncio.run(player.accept_challenges(None, 1))
        except Exception as e:
            print(f"[{username}] error accepting challenge: {e}")
            sleep(5)
            try:
                player.reset_battles()
            except Exception:
                pass
            continue

        # compute cost from tokens used this battle
        pt_used = getattr(player, "prompt_tokens", 0) - pt_before
        ct_used = getattr(player, "completion_tokens", 0) - ct_before
        add_cost(pt_used, ct_used, backend)

        # update team for next battle
        try:
            if teamloader:
                player.update_team(teamloader.yield_team())
        except Exception:
            pass

        try:
            player.reset_battles()
        except Exception:
            pass

# ── Entry point ────────────────────────────────────────────────────────────────

def load_passwords() -> dict:
    path = os.path.join(project_root, "passwords.json")
    with open(path) as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Host PAC bots on pokeagentchallenge.com")
    parser.add_argument("--budget", type=float, default=DAILY_BUDGET_USD,
                        help="Daily USD budget cap (default 100)")
    args = parser.parse_args()
    DAILY_BUDGET_USD = args.budget

    passwords = load_passwords()
    os.makedirs(os.path.join(project_root, "battle_log", "pac"), exist_ok=True)

    threads = []
    for agent in AGENTS:
        t = Thread(target=run_agent, args=(agent, passwords), daemon=True)
        t.start()
        threads.append(t)
        sleep(2)  # stagger startup

    print(f"Hosting {len(AGENTS)} agents on pokeagentchallenge.com, daily budget ${DAILY_BUDGET_USD}")
    try:
        while True:
            sleep(60)
            print(f"[status] budget remaining: ${budget_remaining():.2f}")
    except KeyboardInterrupt:
        print("Shutting down.")
