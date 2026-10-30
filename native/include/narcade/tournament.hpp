// Tournament runner: repeated round-robin rounds of paired episodes
// between the five evaluation players, with Elo ratings carried
// across rounds and a standings table at the end.
//
// Relationship to scripts/evaluate.py (the semantics authority where
// the two overlap): evaluate.py computes Elo and head-to-head over
// ONE paired pool (its "paired subset"). This runner keeps every
// shared rule and adds the tournament structure on top:
//
//   - Roster order is evaluate.py main()'s order: random, greedy,
//     rookie, pro, grandmaster.
//   - A match is a paired episode on a shared seed: both players play
//     the same seed, the higher score wins, equal scores draw. That
//     is the comparison _elo_from_aligned and head_to_head use.
//   - Elo starts at 1200, K = 32, expected score
//     1 / (1 + 10 ** ((elo_b - elo_a) / 400)), with sequential
//     in-place updates in episode-major, pair order — the exact
//     update rule of _elo_from_aligned, applied round by round with
//     ratings carried across rounds.
//   - Players are single instances for the whole tournament, as in
//     evaluate.py (one RandomPolicy(999), one MCTSPlayer(31337),
//     one DQN policy). Rounds use disjoint seed blocks starting at
//     the evaluation seed base (5000).
//
// What is new, and claimed as nothing else: the multi-round
// structure, the per-pair win/draw/loss records aggregated over the
// whole tournament, the points column (win 1, draw 0.5), and the
// standings order (final Elo, then points, then mean score, then
// roster order). No Python tournament exists, so no parity with one
// is claimed; parity for the shared pieces is checked against
// runs/eval.json by the tournament_regression CTest.
//
// Scope: all five players are covered. "pro" plays through the
// native DQN inference module (CPP-06d) over the committed
// runs/dqn_weights.json — the module the CPP-05 evaluator's scope
// note was waiting for. DQN training stays in Python; this is
// inference only, like every other native DQN consumer.
#ifndef NARCADE_TOURNAMENT_HPP
#define NARCADE_TOURNAMENT_HPP

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "narcade/dqn_infer.hpp"
#include "narcade/evaluate.hpp"

namespace narcade {

// The report scope declaration, shared by the CLI and the report so
// the two can never drift apart.
const char* tournament_scope();

// evaluate.py's fifth player: DQNPolicy over the committed JSON
// weights export (evaluate.py lines 105-110 in main(), the class at
// lines 20-38). act() mirrors DQNPolicy.act: encode the env, take
// the argmax action (first-max tie, as torch.argmax).
class DqnPlayer final : public Player {
public:
    explicit DqnPlayer(DqnWeights weights) : net_(std::move(weights)) {}
    int act(SnakeEnv& env) override;

private:
    DqnNet net_;
};

// make_player (evaluate.hpp) plus "pro": the four non-neural names
// delegate to make_player; "pro" requires the DQN weights and
// ignores the Q-table / MCTS arguments.
std::unique_ptr<Player> make_tournament_player(
    const std::string& canonical_name, int mcts_simulations,
    const QTablePolicy::Table* qtable, const DqnWeights* dqn_weights);

// Expected score of a against b: evaluate.py line 71 (and
// scripts/train.py's expected_score): 1 / (1 + 10 ** ((b - a) / 400)).
double elo_expected(double elo_a, double elo_b);

// Match outcome from a's perspective: 1.0 win, 0.5 draw, 0.0 loss
// (evaluate.py line 73).
double match_result(int score_a, int score_b);

// One side's record against one opponent (or, aggregated, against
// the field). Points follow the match convention: win 1, draw 0.5.
struct MatchRecord {
    std::int64_t wins = 0;
    std::int64_t draws = 0;
    std::int64_t losses = 0;
    double points() const {
        return static_cast<double>(wins) + 0.5 * static_cast<double>(draws);
    }
    std::int64_t games() const { return wins + draws + losses; }
};

// Sequential Elo engine over a fixed roster: ratings plus per-pair
// and per-player records, updated one match at a time in the order
// matches are applied. apply_match performs exactly the in-place
// two-sided update of evaluate.py lines 74-75 (with line 75's exact
// expression), so feeding an aligned score block through
// play_score_block reproduces elo_ratings_raw bit for bit.
class EloTable {
public:
    EloTable(std::vector<std::string> names, double k = 32.0,
             double initial = 1200.0);

    void apply_match(std::size_t a, std::size_t b, int score_a, int score_b);

    const std::vector<std::string>& names() const { return names_; }
    const std::vector<double>& ratings() const { return ratings_; }
    // Python-rounded ratings (round-half-even), evaluate.py line 78.
    std::vector<std::int64_t> ratings_rounded() const;
    // pair_records()[a][b] is a's record against b; the diagonal is
    // an always-empty record.
    const std::vector<std::vector<MatchRecord>>& pair_records() const {
        return pair_records_;
    }
    const std::vector<MatchRecord>& totals() const { return totals_; }

private:
    std::vector<std::string> names_;
    double k_;
    std::vector<double> ratings_;
    std::vector<std::vector<MatchRecord>> pair_records_;
    std::vector<MatchRecord> totals_;
};

// One aligned score block (every player's scores over the same seed
// list, in roster order) processed in _elo_from_aligned's order:
// episode-major, then pairs (a, b) with a before b. Throws
// std::runtime_error on a ragged block.
void play_score_block(EloTable& table, const OrderedResults& scores);

struct TournamentConfig {
    int rounds = 3;
    int seeds_per_round = 10;
    std::int64_t seed0 = 5000;
    int mcts_sims = 80;
    int width = 12;
    int height = 12;
    double k = 32.0;
    double initial_rating = 1200.0;
};

struct RoundResult {
    std::vector<std::int64_t> seeds;
    OrderedResults scores;  // roster order, aligned with seeds
    std::vector<double> elo_after_raw;
    std::vector<std::int64_t> elo_after;
};

struct Standing {
    std::string name;
    std::size_t roster_index = 0;
    double elo_raw = 0.0;
    std::int64_t elo = 0;
    MatchRecord record;
    double mean_score = 0.0;
};

// Standings from a finished table: final Elo (unrounded) desc, then
// points desc, then mean score desc, then roster order. Mean scores
// are per player over every episode they played.
std::vector<Standing> compute_standings(const EloTable& table,
                                        const std::vector<double>& mean_scores);

struct TournamentResult {
    TournamentConfig config;
    std::vector<std::string> roster;
    std::vector<RoundResult> rounds;
    std::vector<Standing> standings;
    EloTable table;  // final ratings and records, roster order
};

// Runs the tournament. `players` are the roster's instances, created
// once by the caller and reused for every round (evaluate.py's
// lifecycle). Each round, every player plays the round's seed block
// via run_episodes; the block then goes through play_score_block.
TournamentResult run_tournament(std::vector<std::unique_ptr<Player>>& players,
                                const std::vector<std::string>& roster,
                                const TournamentConfig& config);

// The standings report as a single JSON document (schema
// "tournament-v1"): scope, protocol, per-round seeds/scores/Elo,
// standings, and aggregated head-to-head counts. Deterministic:
// the same tournament renders byte-identically.
std::string tournament_report_json(const TournamentResult& result);

}  // namespace narcade

#endif  // NARCADE_TOURNAMENT_HPP
