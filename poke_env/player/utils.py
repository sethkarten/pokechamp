"""This module contains utility functions and objects related to Player classes.
"""

import asyncio
import math
from concurrent.futures import Future
from typing import Dict, List, Optional, Tuple

from poke_env.concurrency import POKE_LOOP
from poke_env.data import to_id_str
from poke_env.player.baselines import MaxBasePowerPlayer, AbyssalPlayer
from poke_env.player.player import Player
from poke_env.player.random_player import RandomPlayer
from tqdm import tqdm
import numpy as np
import pandas as pd

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.player.team_util import load_random_team

_EVALUATION_RATINGS = {
    # RandomPlayer: 1,
    MaxBasePowerPlayer: 7.665994,
    # AbyssalPlayer: 128.757145,
}

def write_stats(model_As: list[str], model_Bs: list[str], winners: list[str], turns: list[str], file='whole_history.csv') -> None:
    model_As = np.array(model_As)
    model_Bs = np.array(model_Bs)
    winners = np.array(winners)
    turns = np.array(turns)
    info_dict = {'model_a': model_As,
            'model_b': model_Bs,
            'winner': winners,
            'turns': turns
            }
    df = pd.DataFrame(info_dict)
    df.to_csv(file, mode='w')
    return


def background_cross_evaluate(
    players: List[Player], n_challenges: int
) -> "Future[Dict[str, Dict[str, Optional[float]]]]":
    return asyncio.run_coroutine_threadsafe(
        cross_evaluate(players, n_challenges), POKE_LOOP
    )

async def cross_evaluate(
    players: List[Player], n_challenges: int, file: str
) -> Dict[str, Dict[str, Optional[float]]]:
    results: Dict[str, Dict[str, Optional[float]]] = {
        p_1.username: {p_2.username: None for p_2 in players} for p_1 in players
    }
    total = 0
    # for k in range(1, len(players)):
    #     total += k * n_challenges
    total = n_challenges * (len(players)-1)
    pbar = tqdm(total=total)
    model_As = []
    model_Bs = []
    winners = []
    turns = []
    for i, p_1 in enumerate(players):
        for j, p_2 in enumerate(players):
            if j <= i:
                continue
            for team_p1_i in range(1, 1+int(math.sqrt(n_challenges))):
                if 'ou' in p_1._format:
                    # load new p1 team
                    p_1.update_team(load_random_team())
                for team_p2_i in range(1, 1+int(math.sqrt(n_challenges))):
                    if 'ou' in p_2._format:
                        # load new p2 team
                        p_2.update_team(load_random_team())
                    await p_1.battle_against(p_2)
                    pbar.update(1)
                    # results[p_1.username][p_2.username] = p_1.win_rate
                    # results[p_2.username][p_1.username] = p_2.win_rate

                    model_As.append(p_1.username)
                    model_Bs.append(p_2.username)
                    if p_1.win_rate > p_2.win_rate: winner = 'model_a'
                    else: winner = 'model_b'
                    winners.append(winner)
                    num_turns = str(list(p_1.battles.values())[0].turn)
                    turns.append(num_turns)
                    write_stats(model_As, model_Bs, winners, turns, file=file)
                    # reset stats to verse another player
                    p_1.reset_battles()
                    p_2.reset_battles()
    write_stats(model_As, model_Bs, winners, turns, file=file)
    return


def _estimate_strength_from_results(
    number_of_games: int, number_of_wins: int, opponent_rating: float
) -> Tuple[float, Tuple[float, float]]:
    """Estimate player strength based on game results and opponent rating.

    :param number_of_games: Number of performance games for evaluation.
    :type number_of_games: int
    :param number_of_wins: Number of won evaluation games.
    :type number_of_wins: int
    :param opponent_rating: The opponent's rating.
    :type opponent_rating: float
    :raises: ValueError if the results are too extreme to be interpreted.
    :return: A tuple containing the estimated player strength and a 95% confidence
        interval
    :rtype: tuple of float and tuple of floats
    """
    print(f'wins {number_of_wins}\nnumber of games {number_of_games}')
    n, p = number_of_games, number_of_wins / number_of_games
    q = 1 - p

    if n * p * q < 9:  # Cannot apply normal approximation of binomial distribution
        raise ValueError(
            "The results obtained in evaluate_player are too extreme to obtain an "
            "accurate player evaluation. You can try to solve this issue by increasing"
            " the total number of battles. Obtained results: %d victories out of %d"
            " games." % (p * n, n)
        )

    estimate = opponent_rating * p / q
    error = (
        math.sqrt(n * p * q) / n * 1.96
    )  # 95% confidence interval for normal distribution

    lower_bound = max(0, p - error)
    lower_bound = opponent_rating * lower_bound / (1 - lower_bound)

    higher_bound = min(1, p + error)

    if higher_bound == 1:
        higher_bound = math.inf
    else:
        higher_bound = opponent_rating * higher_bound / (1 - higher_bound)

    return estimate, (lower_bound, higher_bound)


def background_evaluate_player(
    player: Player, n_battles: int = 1000, n_placement_battles: int = 30
) -> "Future[Tuple[float, Tuple[float, float]]]":
    return asyncio.run_coroutine_threadsafe(
        evaluate_player(player, n_battles, n_placement_battles), POKE_LOOP
    )


async def evaluate_player(
    player: Player, n_battles: int = 1000, n_placement_battles: int = 30, battle_format: str="gen8randombattle"
) -> Tuple[float, Tuple[float, float]]:
    """Estimate player strength.

    This functions calculates an estimate of a player's strength, measured as its
    expected performance against a random opponent in a gen 8 random battle. The
    returned number can be interpreted as follows: a strength of k means that the
    probability of the player winning a gen 8 random battle against a random player is k
    times higher than the probability of the random player winning.

    The function returns a tuple containing the best guess based on the played games
    as well as a tuple describing a 95% confidence interval for that estimated strength.

    The actual evaluation can be performed against any baseline player for which an
    accurate strength estimate is available. This baseline is determined at the start of
    the process, by playing a limited number of placement battles and choosing the
    opponent closest to the player in terms of performance.

    :param player: The player to evaluate.
    :type player: Player
    :param n_battles: The total number of battle to perform, including placement
        battles.
    :type n_battles: int
    :param n_placement_battles: Number of placement battles to perform per baseline
        player.
    :type n_placement_battles: int
    :raises: ValueError if the results are too extreme to be interpreted.
    :raises: AssertionError if the player is not configured to play gen8battles or the
        selected number of games to play it too small.
    :return: A tuple containing the estimated player strength and a 95% confidence
        interval
    :rtype: tuple of float and tuple of floats
    """
    # Input checks
    assert player.format == battle_format, (
        "Player %s can not be evaluated as its current format (%s) is not "
        f"{battle_format}." % (player, player.format)
    )

    if n_placement_battles * len(_EVALUATION_RATINGS) > n_battles // 2:
        player.logger.warning(
            "Number of placement battles reduced from %d to %d due to limited number of"
            " battles (%d). A more accurate evaluation can be performed by increasing "
            "the total number of players.",
            n_placement_battles,
            n_battles // len(_EVALUATION_RATINGS) // 2,
            n_battles,
        )
        n_placement_battles = n_battles // len(_EVALUATION_RATINGS) // 2

    assert (
        n_placement_battles > 0
    ), "Not enough battles to perform placement battles. Please increase the number of "
    "battles to perform to evaluate the player."

    # Initial placement battles
    baselines = [p(max_concurrent_battles=n_battles, battle_format=battle_format, account_configuration=AccountConfiguration('bot'+str(np.random.randint(0,10000)), '')) for p in _EVALUATION_RATINGS]
    pbar = tqdm(total=n_battles)
    for p in baselines:
        p._dynamax_disable = True
        await p.battle_against(player, n_battles=n_placement_battles)
        pbar.update(n_placement_battles)

    # Select the best opponent for evaluation
    best_opp = min(
        baselines, key=lambda p: (abs(p.win_rate - 0.5), -_EVALUATION_RATINGS[type(p)])
    )

    # Performing the main evaluation
    remaining_battles = n_battles - len(_EVALUATION_RATINGS) * n_placement_battles
    for i in range(remaining_battles):
        await best_opp.battle_against(player, 1)
        pbar.update(1)
        pbar.set_description(f"Wins: {best_opp.n_lost_battles}/{best_opp.n_finished_battles}")

    return _estimate_strength_from_results(
        best_opp.n_finished_battles,
        best_opp.n_lost_battles,
        _EVALUATION_RATINGS[type(best_opp)],
    )
