// Tournament runner implementation. The shared rules mirror
// scripts/evaluate.py operation by operation (see tournament.hpp);
// the tournament structure itself is new.
#include "narcade/tournament.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace narcade {

const char* tournament_scope() {
    return "native tournament (five players): repeated round-robin "
           "rounds of paired episodes with Elo carried across "
           "rounds. Match, Elo and seed rules mirror "
           "scripts/evaluate.py where the two overlap; the "
           "multi-round structure, records and standings are new "
           "(no Python tournament exists). pro plays through the "
           "native DQN inference module over runs/dqn_weights.json; "
           "DQN training stays in Python.";
}

int DqnPlayer::act(SnakeEnv& env) {
    // DQNPolicy.act (evaluate.py): encode, forward, argmax.
    return net_.act(encode_dqn(env));
}

std::unique_ptr<Player> make_tournament_player(
    const std::string& canonical_name, int mcts_simulations,
    const QTablePolicy::Table* qtable, const DqnWeights* dqn_weights) {
    if (canonical_name == "pro") {
        if (dqn_weights == nullptr) {
            throw std::runtime_error("pro player requires DQN weights");
        }
        return std::make_unique<DqnPlayer>(*dqn_weights);
    }
    return make_player(canonical_name, mcts_simulations, qtable);
}

double elo_expected(double elo_a, double elo_b) {
    return 1.0 / (1.0 + std::pow(10.0, (elo_b - elo_a) / 400.0));
}

double match_result(int score_a, int score_b) {
    return score_a > score_b ? 1.0 : (score_a == score_b ? 0.5 : 0.0);
}

EloTable::EloTable(std::vector<std::string> names, double k, double initial)
    : names_(std::move(names)),
      k_(k),
      ratings_(names_.size(), initial),
      pair_records_(names_.size(),
                    std::vector<MatchRecord>(names_.size())),
      totals_(names_.size()) {}

void EloTable::apply_match(std::size_t a, std::size_t b, int score_a,
                           int score_b) {
    const double ra = match_result(score_a, score_b);
    const double ea = elo_expected(ratings_[a], ratings_[b]);
    ratings_[a] += k_ * (ra - ea);
    ratings_[b] += k_ * ((1.0 - ra) - (1.0 - ea));
    MatchRecord& ab = pair_records_[a][b];
    MatchRecord& ba = pair_records_[b][a];
    if (ra == 1.0) {
        ++ab.wins;
        ++ba.losses;
        ++totals_[a].wins;
        ++totals_[b].losses;
    } else if (ra == 0.5) {
        ++ab.draws;
        ++ba.draws;
        ++totals_[a].draws;
        ++totals_[b].draws;
    } else {
        ++ab.losses;
        ++ba.wins;
        ++totals_[a].losses;
        ++totals_[b].wins;
    }
}

std::vector<std::int64_t> EloTable::ratings_rounded() const {
    std::vector<std::int64_t> out;
    out.reserve(ratings_.size());
    for (const double r : ratings_) {
        out.push_back(py_round_int(r));
    }
    return out;
}

void play_score_block(EloTable& table, const OrderedResults& scores) {
    if (scores.empty()) {
        throw std::runtime_error("play_score_block: no players");
    }
    const std::size_t n = scores[0].second.size();
    for (const auto& kv : scores) {
        if (kv.second.size() != n) {
            throw std::runtime_error(
                "play_score_block: ragged score block for " + kv.first);
        }
    }
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t a = 0; a < scores.size(); ++a) {
            for (std::size_t b = a + 1; b < scores.size(); ++b) {
                table.apply_match(a, b, scores[a].second[i],
                                  scores[b].second[i]);
            }
        }
    }
}

std::vector<Standing> compute_standings(
    const EloTable& table, const std::vector<double>& mean_scores) {
    std::vector<Standing> out;
    const auto rounded = table.ratings_rounded();
    for (std::size_t i = 0; i < table.names().size(); ++i) {
        Standing s;
        s.name = table.names()[i];
        s.roster_index = i;
        s.elo_raw = table.ratings()[i];
        s.elo = rounded[i];
        s.record = table.totals()[i];
        s.mean_score = mean_scores[i];
        out.push_back(s);
    }
    std::sort(out.begin(), out.end(), [](const Standing& x, const Standing& y) {
        if (x.elo_raw != y.elo_raw) return x.elo_raw > y.elo_raw;
        const double px = x.record.points();
        const double py = y.record.points();
        if (px != py) return px > py;
        if (x.mean_score != y.mean_score) return x.mean_score > y.mean_score;
        return x.roster_index < y.roster_index;
    });
    return out;
}

TournamentResult run_tournament(std::vector<std::unique_ptr<Player>>& players,
                                const std::vector<std::string>& roster,
                                const TournamentConfig& config) {
    if (players.size() != roster.size() || roster.empty()) {
        throw std::runtime_error(
            "run_tournament: players and roster disagree or are empty");
    }
    if (config.rounds < 1 || config.seeds_per_round < 1) {
        throw std::runtime_error(
            "run_tournament: rounds and seeds_per_round must be >= 1");
    }
    TournamentResult result{config, roster, {},
                            {}, EloTable(roster, config.k, config.initial_rating)};
    std::vector<std::int64_t> score_sums(roster.size(), 0);
    std::int64_t episodes_each = 0;
    for (int round = 0; round < config.rounds; ++round) {
        RoundResult rr;
        for (int i = 0; i < config.seeds_per_round; ++i) {
            rr.seeds.push_back(config.seed0 +
                               static_cast<std::int64_t>(round) *
                                   config.seeds_per_round +
                               i);
        }
        for (std::size_t p = 0; p < roster.size(); ++p) {
            const auto eps = run_episodes(*players[p], rr.seeds, config.width,
                                          config.height);
            std::vector<int> scores;
            scores.reserve(eps.size());
            for (const auto& ep : eps) {
                scores.push_back(ep.score);
                score_sums[p] += ep.score;
            }
            rr.scores.emplace_back(roster[p], std::move(scores));
        }
        episodes_each += config.seeds_per_round;
        play_score_block(result.table, rr.scores);
        rr.elo_after_raw = result.table.ratings();
        rr.elo_after = result.table.ratings_rounded();
        result.rounds.push_back(std::move(rr));
    }
    std::vector<double> means(roster.size(), 0.0);
    for (std::size_t p = 0; p < roster.size(); ++p) {
        means[p] = episodes_each > 0
                       ? static_cast<double>(score_sums[p]) /
                             static_cast<double>(episodes_each)
                       : 0.0;
    }
    result.standings = compute_standings(result.table, means);
    return result;
}

namespace {

// %.17g always round-trips a double exactly (the evaluate CLI's
// convention for JSON numbers).
std::string json_number(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}

std::string json_escape(const std::string& s) {
    std::string out;
    for (const char ch : s) {
        switch (ch) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            default: out += ch; break;
        }
    }
    return out;
}

void append_scores(std::string& json, const OrderedResults& scores) {
    json += "{";
    for (std::size_t p = 0; p < scores.size(); ++p) {
        if (p != 0) json += ", ";
        json += "\"" + scores[p].first + "\": [";
        for (std::size_t i = 0; i < scores[p].second.size(); ++i) {
            if (i != 0) json += ", ";
            json += std::to_string(scores[p].second[i]);
        }
        json += "]";
    }
    json += "}";
}

}  // namespace

std::string tournament_report_json(const TournamentResult& result) {
    const TournamentConfig& cfg = result.config;
    const EloTable& table = result.table;
    std::string json = "{";
    json += "\"schema\": \"tournament-v1\", ";
    json += "\"scope\": \"" + json_escape(tournament_scope()) + "\", ";
    json += "\"protocol\": {\"players\": [";
    for (std::size_t i = 0; i < result.roster.size(); ++i) {
        if (i != 0) json += ", ";
        json += "\"" + result.roster[i] + "\"";
    }
    json += "], \"rounds\": " + std::to_string(cfg.rounds) +
            ", \"seeds_per_round\": " + std::to_string(cfg.seeds_per_round) +
            ", \"seed0\": " + std::to_string(cfg.seed0) +
            ", \"mcts_sims\": " + std::to_string(cfg.mcts_sims) +
            ", \"width\": " + std::to_string(cfg.width) +
            ", \"height\": " + std::to_string(cfg.height) +
            ", \"k_factor\": " + json_number(cfg.k) +
            ", \"initial_rating\": " + json_number(cfg.initial_rating) +
            ", \"match\": \"paired episode on a shared seed; higher "
            "score wins, equal scores draw\"" +
            ", \"standings_order\": \"final Elo, then points, then "
            "mean score, then roster order\"}, ";
    json += "\"rounds\": [";
    for (std::size_t r = 0; r < result.rounds.size(); ++r) {
        const RoundResult& rr = result.rounds[r];
        if (r != 0) json += ", ";
        json += "{\"round\": " + std::to_string(r + 1) + ", \"seeds\": [";
        for (std::size_t i = 0; i < rr.seeds.size(); ++i) {
            if (i != 0) json += ", ";
            json += std::to_string(rr.seeds[i]);
        }
        json += "], \"scores\": ";
        append_scores(json, rr.scores);
        json += ", \"elo_after\": {";
        for (std::size_t i = 0; i < result.roster.size(); ++i) {
            if (i != 0) json += ", ";
            json += "\"" + result.roster[i] +
                    "\": " + std::to_string(rr.elo_after[i]);
        }
        json += "}}";
    }
    json += "], \"standings\": [";
    for (std::size_t i = 0; i < result.standings.size(); ++i) {
        const Standing& s = result.standings[i];
        if (i != 0) json += ", ";
        json += "{\"player\": \"" + s.name +
                "\", \"elo\": " + std::to_string(s.elo) +
                ", \"elo_raw\": " + json_number(s.elo_raw) +
                ", \"points\": " + json_number(s.record.points()) +
                ", \"wins\": " + std::to_string(s.record.wins) +
                ", \"draws\": " + std::to_string(s.record.draws) +
                ", \"losses\": " + std::to_string(s.record.losses) +
                ", \"mean_score\": " + json_number(s.mean_score) + "}";
    }
    json += "], \"head_to_head\": {";
    const auto& pairs = table.pair_records();
    for (std::size_t a = 0; a < result.roster.size(); ++a) {
        if (a != 0) json += ", ";
        json += "\"" + result.roster[a] + "\": {";
        for (std::size_t b = 0; b < result.roster.size(); ++b) {
            if (b != 0) json += ", ";
            json += "\"" + result.roster[b] + "\": ";
            if (a == b) {
                json += "null";
            } else {
                const MatchRecord& rec = pairs[a][b];
                json += "{\"wins\": " + std::to_string(rec.wins) +
                        ", \"draws\": " + std::to_string(rec.draws) +
                        ", \"losses\": " + std::to_string(rec.losses) + "}";
            }
        }
        json += "}";
    }
    json += "}}";
    return json;
}

}  // namespace narcade
