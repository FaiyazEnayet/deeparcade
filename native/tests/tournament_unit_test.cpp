// tournament_unit_test: unit tests for the tournament Elo engine
// (native/src/tournament.cpp) against hand-computed values.
//
// Every expected rating below was computed from the defining
// formulas (expected score 1 / (1 + 10 ** ((b - a) / 400)), K = 32,
// sequential two-sided update) independently of this code, and the
// scripted-sequence values were cross-checked against
// scripts/evaluate.py's _elo_from_aligned arithmetic on the same
// score lists. Tolerances are 1e-9 on raw ratings (the arithmetic is
// exact double math in the same order, so agreement is far tighter
// than the tolerance); rounded ratings and records are exact.
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "narcade/tournament.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cout << "FAIL: " << what << '\n';
        ++failures;
    }
}

void check_near(double got, double want, const std::string& what) {
    if (std::fabs(got - want) > 1e-9) {
        std::cout << "FAIL: " << what << ": got " << got << ", want " << want
                  << '\n';
        ++failures;
    }
}

void check_record(const narcade::MatchRecord& rec, std::int64_t w, std::int64_t d,
                  std::int64_t l, const std::string& who) {
    check(rec.wins == w && rec.draws == d && rec.losses == l,
          who + " record");
}

}  // namespace

int main() {
    using narcade::EloTable;
    using narcade::OrderedResults;

    // Expected scores: even match, and a 400-point gap (the
    // favorite's expectation is exactly 10/11).
    check_near(narcade::elo_expected(1200.0, 1200.0), 0.5, "E even");
    check_near(narcade::elo_expected(1600.0, 1200.0), 0.9090909090909091,
               "E favorite");
    check_near(narcade::elo_expected(1200.0, 1600.0), 0.09090909090909091,
               "E underdog");

    // Single matches from known ratings.
    {
        EloTable t({"a", "b"});
        t.apply_match(0, 1, 5, 3);  // even-rating win: +-16 exactly
        check_near(t.ratings()[0], 1216.0, "win at 1200 (a)");
        check_near(t.ratings()[1], 1184.0, "win at 1200 (b)");
        check_record(t.totals()[0], 1, 0, 0, "win at 1200 a");
        check_record(t.totals()[1], 0, 0, 1, "win at 1200 b");
        check(t.ratings_rounded()[0] == 1216 &&
                  t.ratings_rounded()[1] == 1184,
              "win at 1200 rounded");
    }
    {
        EloTable t({"a", "b"});
        t.apply_match(0, 1, 4, 4);  // draw at even ratings: no change
        check_near(t.ratings()[0], 1200.0, "draw at 1200 (a)");
        check_near(t.ratings()[1], 1200.0, "draw at 1200 (b)");
        check_record(t.totals()[0], 0, 1, 0, "draw at 1200 a");
        check_near(t.totals()[0].points(), 0.5, "draw points");
    }
    {
        // Favorite-win delta on a two-step sequence: after a wins
        // once (1216 vs 1184), a second win gains exactly
        // 32 * (1 - E(1216, 1184)). (EloTable starts every player at
        // one initial rating by design, so unequal-rating single
        // matches are exercised through sequences like this one and
        // the scripted block below, whose episode 1 contains an
        // underdog win — C over the higher-rated A — and a favorite
        // draw, A vs B.)
        EloTable t({"a", "b"});
        t.apply_match(0, 1, 5, 3);
        const double e = narcade::elo_expected(1216.0, 1184.0);
        t.apply_match(0, 1, 5, 3);
        check_near(t.ratings()[0], 1216.0 + 32.0 * (1.0 - e),
                   "second win gain");
        check_record(t.totals()[0], 2, 0, 0, "two wins a");
    }

    // Scripted sequence over an aligned block (the protocol order of
    // _elo_from_aligned): players A, B, C; episode 0 scores
    // A=10, B=7, C=7; episode 1 scores A=3, B=3, C=9. Expected
    // finals were computed independently (see header comment):
    //   A 1211.0683623404832, B 1170.956759348506,
    //   C 1217.9748783110108; records A 2/1/1, B 0/2/2, C 2/1/1.
    {
        EloTable t({"A", "B", "C"});
        const OrderedResults scores = {
            {"A", {10, 3}}, {"B", {7, 3}}, {"C", {7, 9}}};
        narcade::play_score_block(t, scores);
        check_near(t.ratings()[0], 1211.0683623404832, "scripted A");
        check_near(t.ratings()[1], 1170.956759348506, "scripted B");
        check_near(t.ratings()[2], 1217.9748783110108, "scripted C");
        const auto rounded = t.ratings_rounded();
        check(rounded[0] == 1211 && rounded[1] == 1171 && rounded[2] == 1218,
              "scripted rounded");
        check_record(t.totals()[0], 2, 1, 1, "scripted A");
        check_record(t.totals()[1], 0, 2, 2, "scripted B");
        check_record(t.totals()[2], 2, 1, 1, "scripted C");
        // Pair records mirror each other.
        check_record(t.pair_records()[0][2], 1, 0, 1, "scripted A vs C");
        check_record(t.pair_records()[2][0], 1, 0, 1, "scripted C vs A");
        check_record(t.pair_records()[0][1], 1, 1, 0, "scripted A vs B");
        check_near(t.totals()[0].points(), 2.5, "scripted A points");
        check_near(t.totals()[1].points(), 1.0, "scripted B points");

        // The same block through evaluate.cpp's elo_ratings_raw must
        // agree bit for bit (same update order, same expressions).
        const auto raw = narcade::elo_ratings_raw(scores);
        for (std::size_t i = 0; i < 3; ++i) {
            check(t.ratings()[i] == raw[i].second,
                  "scripted agreement with elo_ratings_raw");
        }

        // Standings: C first on Elo, then A, then B (A and C tie on
        // points, Elo decides before the tie-breaks are needed).
        const auto standings =
            narcade::compute_standings(t, {6.5, 5.0, 8.0});
        check(standings.size() == 3 && standings[0].name == "C" &&
                  standings[1].name == "A" && standings[2].name == "B",
              "scripted standings order");
    }

    // Standings tie-breaks: identical Elo and points -> mean score;
    // identical everything -> roster order.
    {
        EloTable t({"x", "y", "z"});
        const auto by_mean = narcade::compute_standings(t, {1.0, 9.0, 5.0});
        check(by_mean[0].name == "y" && by_mean[1].name == "z" &&
                  by_mean[2].name == "x",
              "standings mean-score tie-break");
        const auto by_roster = narcade::compute_standings(t, {5.0, 5.0, 5.0});
        check(by_roster[0].name == "x" && by_roster[1].name == "y" &&
                  by_roster[2].name == "z",
              "standings roster tie-break");
    }

    // A ragged block is a hard error, never a silent miscomparison.
    {
        EloTable t({"a", "b"});
        bool threw = false;
        try {
            narcade::play_score_block(t, {{"a", {1, 2}}, {"b", {1}}});
        } catch (const std::runtime_error&) {
            threw = true;
        }
        check(threw, "ragged block throws");
    }

    if (failures != 0) {
        std::cout << "tournament_unit_test: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "tournament_unit_test: PASS\n";
    return 0;
}
