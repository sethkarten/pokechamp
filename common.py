import random
import numpy as np
import os
import importlib
import inspect

prompt_algos = [
    "io", 
    "sc", 
    "cot", 
    "tot", 
    "minimax", 
    "heuristic", 
    'max_power',
    'dmg_calc',
    'random'
    ]

def get_available_bots():
    """Get a list of all available bot names from the bots folder."""
    bot_names = []
    bots_dir = os.path.join(os.path.dirname(__file__), 'bots')
    
    if not os.path.exists(bots_dir):
        return bot_names
    
    # Look for Python files in the bots directory
    for filename in os.listdir(bots_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            # Remove .py extension and _bot suffix to get the bot name
            bot_name = filename[:-3]  # Remove .py
            if bot_name.endswith('_bot'):
                bot_name = bot_name[:-4]  # Remove _bot suffix
            bot_names.append(bot_name)
    
    return bot_names

# Get available bot names from the bots folder
available_bots = get_available_bots()

# Combine built-in bots with custom bots
bot_choices = ['pokechamp', 'pokellmon', 'one_step', 'abyssal', 'max_power', 'random'] + available_bots

PNUMBER1 = str(np.random.randint(0,10000))
print(PNUMBER1)
seed = 100
random.seed(seed)
np.random.seed(seed)