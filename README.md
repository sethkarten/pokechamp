```
██████╗  ██████╗ ██╗  ██╗███████╗ ██████╗██╗  ██╗ █████╗ ███╗   ███╗██████╗ 
██╔══██╗██╔═══██╗██║ ██╔╝██╔════╝██╔════╝██║  ██║██╔══██╗████╗ ████║██╔══██╗
██████╔╝██║   ██║█████╔╝ █████╗  ██║     ███████║███████║██╔████╔██║██████╔╝
██╔═══╝ ██║   ██║██╔═██╗ ██╔══╝  ██║     ██╔══██║██╔══██║██║╚██╔╝██║██╔═══╝ 
██║     ╚██████╔╝██║  ██╗███████╗╚██████╗██║  ██║██║  ██║██║ ╚═╝ ██║██║     
╚═╝      ╚═════╝ ╚═╝  ╚═╝╚══════╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     
```
# Pokémon Champion
<!-- project badges -->
[![Paper (ICML '25)](https://img.shields.io/badge/Paper-ICML-blue?style=flat)](https://openreview.net/pdf?id=SnZ7SKykHh)
[![Dataset on HuggingFace](https://img.shields.io/badge/Dataset-HuggingFace-brightgreen?logo=huggingface&logoColor=white&style=flat)](https://huggingface.co/datasets/milkkarten/pokechamp)
[![Source Code](https://img.shields.io/badge/Code-GitHub-black?logo=github&logoColor=white&style=flat)](https://github.com/sethkarten/pokechamp)

This is the implementation for the paper "PokéChamp: an Expert-level Minimax Language Agent"

<div align="center">
  <img src="./resource/method.png" alt="PokemonChamp">
</div>

## Architecture

The codebase is organized into several clean modules:

```
pokechamp/
├── pokechamp/           # [CORE] LLM player implementation
│   ├── llm_player.py    # Core LLM player class
│   ├── mcp_player.py    # MCP protocol support
│   ├── llm_vgc_player.py # VGC doubles support
│   ├── gpt_player.py    # OpenAI GPT backend
│   ├── llama_player.py  # Meta LLaMA backend  
│   ├── gemini_player.py # Google Gemini backend
│   ├── openrouter_player.py # OpenRouter API backend
│   ├── prompts.py       # Battle prompts & algorithms
│   └── translate.py     # Battle translation utilities
├── bayesian/            # [PREDICT] Bayesian prediction system
│   ├── pokemon_predictor.py    # Pokemon team predictions
│   ├── team_predictor.py       # Bayesian team predictor
│   └── live_battle_predictor.py # Live battle predictions
├── scripts/             # [SCRIPTS] Battle execution scripts
│   ├── battles/         # Battle runners (local_1v1.py, etc.)
│   ├── evaluation/      # Evaluation tools
│   └── training/        # Dataset processing
├── poke_env/            # [ENGINE] Core battle engine (LLM-independent)
├── bots/                # [BOTS] Custom bot implementations
└── tests/               # [TESTS] Comprehensive test suite
```

**Key Benefits:**
- **Clean separation**: Core battle engine (`poke_env`) is independent of LLM code
- **Modular design**: Each component has clear responsibilities
- **Extensible**: Easy to add new LLM backends or battle algorithms
- **Testable**: Comprehensive test coverage for all functionality

## Quick Start

### Requirements

```sh
# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/sethkarten/pokechamp.git
cd pokechamp
uv sync
```

### Battle Any Agent Against Any Agent
```sh
# Basic battle
uv run python local_1v1.py --player_name pokechamp --opponent_name abyssal

# Try MCP integration
uv run python local_1v1.py --player_prompt_algo mcp --player_backend gemini-2.5-flash --opponent_name abyssal

# VGC double battles
uv run python run_with_timeout_vgc.py --continuous --max-concurrent 2
```

### Evaluation
```sh
uv run python scripts/evaluation/evaluate_gen9ou.py
```

## Battle Configuration

### Local Pokémon Showdown Server Setup

1. Install Node.js v10+
2. Set up the battle server:

```sh
git clone https://github.com/smogon/pokemon-showdown.git
cd pokemon-showdown
npm install
cp config/config-example.js config/config.js
node pokemon-showdown start --no-security
```

3. Open http://localhost:8000/ in your browser

## Available Bots

### Built-in Bots
- `pokechamp` - Main PokéChamp agent using minimax algorithm
- `pokellmon` - LLM-based agent with various prompt algorithms
- `abyssal` - Abyssal Bot baseline
- `max_power` - Maximum base power move selection
- `one_step` - One-step lookahead agent
- `random` - Random move selection
- `vgc` - VGC-specialized agent for double battles

### Custom Bots
- `starter_kit` - Example LLM-based bot for creating custom implementations

### Prompt Algorithms
Available prompt algorithms for LLM-based bots:
- `io` - Input/Output prompting (default)
- `sc` - Self-consistency prompting
- `cot` - Chain-of-thought prompting
- `tot` - Tree-of-thought prompting
- `minimax` - Minimax algorithm with LLM evaluation
- `heuristic` - Heuristic-based decisions
- `max_power` - Maximum base power move selection
- `one_step` - One-step lookahead
- `random` - Random move selection
- `mcp` - Model Context Protocol integration

### Creating Custom Bots

1. Create `bots/my_bot_bot.py`
2. Inherit from `LLMPlayer`:

```python
from pokechamp.llm_player import LLMPlayer

class MyCustomBot(LLMPlayer):
    def choose_move(self, battle):
        # Implement your strategy
        return self.choose_random_move(battle)
```

3. Your bot automatically becomes available in battle scripts

## LLM Backend Support

The system supports multiple LLM backends through OpenRouter, providing access to hundreds of models:

### Supported Providers
- **OpenAI**: `gpt-4o`, `gpt-4o-mini`, `gpt-4-turbo`, `gpt-4`, `gpt-3.5-turbo`
- **Anthropic**: `anthropic/claude-3.5-sonnet`, `anthropic/claude-3-opus`, `anthropic/claude-3-haiku`
- **Google**: `google/gemini-pro`, `gemini-2.0-flash`, `gemini-2.0-pro`, `gemini-2.5-flash`, `gemini-2.5-pro`
- **Meta**: `meta-llama/llama-3.1-70b-instruct`, `meta-llama/llama-3.1-8b-instruct`
- **Mistral**: `mistralai/mistral-7b-instruct`, `mistralai/mixtral-8x7b-instruct`
- **Cohere**: `cohere/command-r-plus`, `cohere/command-r`
- **Perplexity**: `perplexity/llama-3.1-sonar-small-128k`, `perplexity/llama-3.1-sonar-large-128k`
- **DeepSeek**: `deepseek-ai/deepseek-coder-33b-instruct`, `deepseek-ai/deepseek-llm-67b-chat`
- **Microsoft**: `microsoft/wizardlm-2-8x22b`, `microsoft/phi-3-medium-128k-instruct`
- **Local via Ollama**: `ollama/llama3.1:8b`, `ollama/mistral`, `ollama/qwen2.5`, `ollama/gemma3:4b`, `ollama/gpt-oss:20b`

### Setup
1. Get your API key from [OpenRouter](https://openrouter.ai/keys)
2. `export OPENROUTER_API_KEY='your-api-key-here'`
3. Use any supported model:

```sh
# Claude vs Gemini battle
uv run python local_1v1.py --player_backend anthropic/claude-3-haiku --opponent_backend gemini-2.5-flash

# Test different models
uv run python local_1v1.py --player_backend mistralai/mixtral-8x7b-instruct --opponent_backend gpt-4o

# Local models (no API key needed)
uv run python local_1v1.py --player_backend ollama/llama3.1:8b --opponent_name abyssal
```

## Bayesian Prediction System

The codebase includes a sophisticated Bayesian predictor for real-time battle analysis:

### Features
- **Team Prediction**: Predict unrevealed opponent Pokemon
- **Move Prediction**: Predict opponent moves and items
- **Stats Prediction**: Predict EVs, natures, and hidden stats
- **Live Integration**: Real-time predictions during battles

### Usage
```python
from bayesian.pokemon_predictor import PokemonPredictor

predictor = PokemonPredictor()
predictions = predictor.predict_teammates(
    revealed_pokemon=["Kingambit", "Gholdengo"],
    max_predictions=5
)
```

### Live Battle Predictions
```sh
uv run python bayesian/live_battle_predictor.py
```

Shows turn-by-turn Bayesian predictions with probabilities for unrevealed Pokemon, predicted moves, items, and EVs.

## Battle Execution

### Local 1v1 Battles
```sh
# Basic battle
uv run python scripts/battles/local_1v1.py --player_name pokechamp --opponent_name abyssal

# Custom backends
uv run python scripts/battles/local_1v1.py --player_name starter_kit --player_backend gpt-4o

# MCP integration
uv run python local_1v1.py --player_prompt_algo mcp --player_backend gemini-2.5-flash --opponent_name abyssal
```

### VGC Double Battles
```sh
# VGC tournament
uv run python run_with_timeout_vgc.py --continuous --max-concurrent 2

# Single VGC battle
uv run python local_1v1.py --battle_format gen9vgc2025regi --player_name pokechamp --opponent_name abyssal
```

### Human vs Agent
```sh
uv run python scripts/battles/human_agent_1v1.py
```

### Ladder Battles
```sh
uv run python scripts/battles/showdown_ladder.py --USERNAME $USERNAME --PASSWORD $PASSWORD
```

## Evaluation & Analysis

### Cross-Evaluation
```sh
uv run python scripts/evaluation/evaluate_gen9ou.py
```

Runs battles between all agents and outputs:
- Win rates matrix
- Elo ratings
- Average turns per battle

### Dataset Processing
```sh
uv run python scripts/training/battle_translate.py --output data/battles.json --limit 5000 --gamemode gen9ou
```

## Dataset

The PokéChamp dataset contains over 2 million competitive Pokémon battles across 37+ formats.

### Dataset Features
- **Size**: 2M clean battles (1.9M train, 213K test)
- **Formats**: Gen 1-9 competitive formats
- **Skill Range**: All Elo ranges (1000-1800+)
- **Time Period**: Multiple months (2024-2025)

### Usage
```python
from datasets import load_dataset
from scripts.training.battle_translate import load_filtered_dataset

# Load filtered dataset
filtered_dataset = load_filtered_dataset(
    min_month="January2025",
    max_month="March2025", 
    elo_ranges=["1600-1799", "1800+"],
    split="train",
    gamemode="gen9ou"
)
```

## Testing

Run the comprehensive test suite:

```sh
# All tests
uv run pytest tests/

# Specific test categories  
uv run pytest tests/ -m bayesian      # Bayesian functionality
uv run pytest tests/ -m moves         # Move normalization
uv run pytest tests/ -m teamloader    # Team loading
```

The test suite includes:
- [OK] Bayesian prediction accuracy (100% success rate)
- [OK] Move normalization (284 unique moves tested)
- [OK] Team loading and rejection handling
- [OK] Bot system integration
- [OK] Core battle engine functionality

## Reproducing Paper Results

### Gen 9 OU Evaluation
```sh
uv run python scripts/evaluation/evaluate_gen9ou.py
```

This runs the full cross-evaluation between PokéChamp and baseline bots, outputting win rates, Elo ratings, and turn statistics as reported in the paper.

### Action Prediction Benchmark (Coming Soon)
```sh
uv run python evaluate_action_prediction.py
```

## Acknowledgments

## Citation

```bibtex
@article{karten2025pokechamp,
  title={PokéChamp: an Expert-level Minimax Language Agent},
  author={Karten, Seth and Nguyen, Andy Luu and Jin, Chi},
  journal={arXiv preprint arXiv:2503.04094},
  year={2025}
}

@inproceedings{karten2025pokeagent,
  title        = {The PokeAgent Challenge: Competitive and Long-Context Learning at Scale},
  author       = {Karten, Seth and Grigsby, Jake and Milani, Stephanie and Vodrahalli, Kiran
                  and Zhang, Amy and Fang, Fei and Zhu, Yuke and Jin, Chi},
  booktitle    = {NeurIPS Competition Track},
  year         = {2025},
  month        = apr,
}
```