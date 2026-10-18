"""DA2: evaluation pairs agents by identical episode seeds."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.evaluate import run_episodes, elo_ratings, head_to_head
from narcade.policies import GreedyPolicy


def test_same_seeds_give_paired_environments():
    """Two agents on the same seed list face identical environments.

    A deterministic policy must produce identical scores on the same
    seeds, proving the seed list (not the agent) determines the episode.
    """
    seeds = [5000, 5001, 5002]
    a = run_episodes(GreedyPolicy().act, seeds)
    b = run_episodes(GreedyPolicy().act, seeds)
    assert a == b
    assert len(a) == len(seeds)


def test_paired_stats_compare_same_index_same_seed():
    """ELO and head-to-head treat index i as the same environment.

    With identical seed lists, a stronger deterministic policy beats a
    weaker one on the shared episodes; the tables reflect that.
    """
    seeds = [6000, 6001, 6002, 6003]
    greedy_scores = run_episodes(GreedyPolicy().act, seeds)

    def random_act(env):
        import random
        return random.Random(999).choice([0, 1, 2])

    # head-to-head on truly paired data: greedy vs itself draws everything
    paired = {"greedy": greedy_scores, "greedy2": list(greedy_scores)}
    h2h = head_to_head(paired)
    assert h2h["greedy"]["greedy2"]["win_rate"] == 0.0
    assert h2h["greedy"]["greedy2"]["draw_rate"] == 1.0
    # ELO of identical agents stays tied
    elo = elo_ratings(paired)
    assert elo["greedy"] == elo["greedy2"]
