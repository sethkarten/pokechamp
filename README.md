# Pokémon Champion

This is the implementation for the paper "PokéChamp: an Expert-level Minimax Language Agent for Competitive Pokémon"

<div align="center">
  <img src="./resource/method.png" alt="PokemonChamp">
</div>

## Requirements:

```sh
conda create -n pokechamp python=3.12
conda activate pokechamp
pip install -r requirements.txt
```

## Configuration 

### Configuring a Local Pokémon Showdown Server (Battle Engine)

1. Install Node.js v10+.
2. Clone the Pokémon Showdown repository and set it up:

```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
# Optional: All repo features were tested with the following showdown version.
# git reset --hard dd4b004e54d4ef8c66c8b583a8fa64b020574727
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

Enter "http://localhost:8000/" in your browser.


## Reproducing Paper Results

### Gen 9 OU Battles

To evaluate a group of agents on Gen 9 OU format:

```sh
python evaluate_gen9ou.py
```

This script will run battles between PokéChamp and the baseline bots, including PokéLLMon, Abyssal Bot, and others. It will output win rates, Elo ratings, and average number of turns per battle.

## Additional Experiments

### Battle Any Agent Against Any Agent Locally
```sh
python local_1v1.py 
```

### Battle Against Any Agent Locally

First, log into your other account manually on the local server, choosing "[Gen 9] Random Battle".

```sh
python human_agent_1v1.py 
```

### Battle Against Ladder Players on Pokémon Showdown

Register an account on https://play.pokemonshowdown.com/ and get your password.

Open and log in: https://play.pokemonshowdown.com/

```sh
python showdown_ladder.py --USERNAME $USERNAME --PASSWORD $PASSWORD # fill in your username and password for PokéChamp, no need to set up local server.
```

### Benchmark Puzzles - Coming Soon with dataset release!

This requires download of our dataset (release TBD).
To reproduce the action prediction results:

```sh
python evaluate_action_prediction.py
```

This script will analyze the dataset and output prediction accuracies for player and opponent actions across different Elo ratings.


## Acknowledgement

The environment is implemented based on [PokeLLMon](https://github.com/git-disl/PokeLLMon) and [Poke Env](https://github.com/hsahovic/poke-env). This work provides an implementation of the PokéChamp paper:

```
@article{karten2025pokechamp,
  title={PokéChamp: an Expert-level Minimax Language Agent for Competitive Pokémon},
  author={Karten, Seth and Nguyen, Andy Luu and Jin, Chi},
  journal={arXiv preprint arXiv:2503.04094},
  year={2025}
}
```