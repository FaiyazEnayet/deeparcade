// bindings.cpp: the pybind11 module `narcade_native` (CPP-04).
//
// A thin conversion layer over the verified C++ core in
// native/include/narcade/*.hpp (PythonRandom, SnakeEnv, QLearningAgent,
// policies, MCTSPlayer). It contains NO game or search logic: every method
// converts arguments, calls the C++ API, and converts the result back.
// Behavioral parity with the Python package is inherited from the core,
// whose parity is gated on the golden fixtures and differential fuzzers
// under native/golden/ and tools/native_*_fuzz.py.
//
// Documented API gaps bridged here (core headers are not modified):
//  1. narcade::MCTSPlayer::search takes a lightweight MctsState snapshot,
//     not a SnakeEnv. The binding snapshots any env-like object (the
//     Python SnakeEnv, this module's SnakeEnv, or the optional_backend
//     wrapper) by reading its width/height/body/food/heading attributes,
//     exactly the attributes narcade.mcts.MCTSPlayer.search reads.
//     A food of None (terminal full board) maps to the (-1, -1) sentinel
//     the C++ search uses internally.
//  2. The C++ classes require an explicit integer seed; CPython's
//     seed=None (OS entropy) has no native counterpart. The bindings
//     default the seed to 0; callers wanting Python's unseeded behavior
//     should stay on the Python classes (see
//     src/narcade/optional_backend.py, which draws an entropy seed before
//     delegating when seed=None is passed with the native backend on).
//  3. PythonRandom::getrandbits is limited to k <= 64 (uint64_t result);
//     the core throws std::invalid_argument outside that range, which
//     pybind11 surfaces as ValueError, matching CPython's exception type
//     for negative k.

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "narcade/mcts.hpp"
#include "narcade/policies.hpp"
#include "narcade/qlearn.hpp"
#include "narcade/rng.hpp"
#include "narcade/snake.hpp"

namespace py = pybind11;

namespace {

using namespace narcade;

// (row, col) tuple for one cell, mirroring the Python body/food shape.
py::tuple cell_tuple(const Cell& cell) { return py::make_tuple(cell.r, cell.c); }

// env.food: a tuple, or None exactly when the Python env's food is None.
py::object food_object(const SnakeEnv& env) {
    const std::optional<Cell> food = env.food();
    if (!food.has_value()) {
        return py::none();
    }
    return cell_tuple(*food);
}

// env.body: a tuple of (row, col) tuples, tail first, like tuple(env.body).
py::tuple body_tuple(const SnakeEnv& env) {
    const std::deque<Cell>& body = env.body();
    py::tuple out(static_cast<py::ssize_t>(body.size()));
    py::ssize_t i = 0;
    for (const Cell& cell : body) {
        out[i++] = cell_tuple(cell);
    }
    return out;
}

// SnakeEnv.step's Python contract: (state, reward, done, info) where info
// is {"reason": ...} or {}. The core returns the reason as an optional
// string instead, so the binding returns it in info's place: reason string
// or None. Callers wanting the exact dict shape use optional_backend.
py::tuple step_tuple(const StepResult& result) {
    py::object reason = py::none();
    if (result.reason.has_value()) {
        reason = py::str(*result.reason);
    }
    return py::make_tuple(result.state, result.reward, result.done, reason);
}

// Read one (row, col) pair out of any Python 2-sequence of ints.
Cell parse_cell(const py::handle& item) {
    const auto pair = py::cast<std::pair<int, int>>(item);
    return Cell{pair.first, pair.second};
}

// Snapshot the attributes MCTSPlayer.search reads into an MctsState.
MctsState snapshot_state(const py::object& env) {
    MctsState state;
    state.width = py::cast<int>(env.attr("width"));
    state.height = py::cast<int>(env.attr("height"));
    state.heading = py::cast<int>(env.attr("heading"));
    for (const py::handle item : env.attr("body")) {
        state.body.push_back(parse_cell(item));
    }
    const py::object food = env.attr("food");
    state.food = food.is_none() ? Cell{-1, -1} : parse_cell(food);
    return state;
}

// Walls: an iterable of (row, col) pairs (a Python frozenset by default).
MctsWalls parse_walls(const py::object& walls) {
    MctsWalls out;
    if (walls.is_none()) {
        return out;
    }
    for (const py::handle item : walls) {
        const Cell cell = parse_cell(item);
        out.insert(mcts_cell_key(cell.r, cell.c));
    }
    return out;
}

int mcts_search(MCTSPlayer& player, const py::object& env,
                const py::object& walls) {
    return player.search(snapshot_state(env), parse_walls(walls));
}

}  // namespace

PYBIND11_MODULE(narcade_native, m) {
    m.doc() = "Native DeepArcade core (C++20) exposed via pybind11. "
              "Thin bindings over the fixture-verified implementations "
              "in native/; the Python package remains the default.";

    py::class_<PythonRandom>(m, "PythonRandom")
        .def(py::init<std::int64_t>(), py::arg("seed") = 0)
        .def("random", &PythonRandom::random)
        .def("getrandbits", &PythonRandom::getrandbits, py::arg("k"))
        .def("randrange", &PythonRandom::randrange, py::arg("n"))
        .def("randint", &PythonRandom::randint, py::arg("a"), py::arg("b"))
        .def("choice_index", &PythonRandom::choice_index, py::arg("len"))
        .def("randbelow", &PythonRandom::randbelow, py::arg("n"));

    py::class_<SnakeEnv>(m, "SnakeEnv")
        .def(py::init<int, int, std::int64_t>(), py::arg("width") = 12,
             py::arg("height") = 12, py::arg("seed") = 0)
        .def(py::init<int, int, std::int64_t, std::int64_t>(),
             py::arg("width"), py::arg("height"), py::arg("seed"),
             py::arg("max_steps"))
        .def("reset", &SnakeEnv::reset)
        .def("step",
             [](SnakeEnv& env, int action) { return step_tuple(env.step(action)); },
             py::arg("action"))
        .def("encode_state", &SnakeEnv::encode_state)
        .def_property_readonly("width", &SnakeEnv::width)
        .def_property_readonly("height", &SnakeEnv::height)
        .def_property_readonly("body", &body_tuple)
        .def_property_readonly("food", &food_object)
        .def_property_readonly("heading", &SnakeEnv::heading)
        .def_property_readonly("score", &SnakeEnv::score)
        .def_property_readonly("steps", &SnakeEnv::steps)
        .def_property_readonly("done", &SnakeEnv::done)
        .def_property_readonly("max_steps", &SnakeEnv::max_steps);

    py::class_<QLearningAgent>(m, "QLearningAgent")
        .def(py::init<int, int, double, double, double, std::int64_t>(),
             py::arg("n_states") = 32, py::arg("n_actions") = 3,
             py::arg("alpha") = 0.15, py::arg("gamma") = 0.9,
             py::arg("epsilon") = 1.0, py::arg("seed") = 0)
        .def("act", &QLearningAgent::act, py::arg("state"))
        .def("greedy", &QLearningAgent::greedy, py::arg("state"))
        .def("update", &QLearningAgent::update, py::arg("s"), py::arg("a"),
             py::arg("r"), py::arg("s_next"), py::arg("done"))
        .def_property(
            "q",
            [](const QLearningAgent& agent) { return agent.q(); },
            [](QLearningAgent& agent, const QLearningAgent::Table& q) {
                agent.set_q(q);
            })
        .def_property_readonly("n_states", &QLearningAgent::n_states)
        .def_property_readonly("n_actions", &QLearningAgent::n_actions)
        .def_property_readonly("alpha", &QLearningAgent::alpha)
        .def_property_readonly("gamma", &QLearningAgent::gamma)
        .def_property_readonly("epsilon", &QLearningAgent::epsilon);

    py::class_<RandomPolicy>(m, "RandomPolicy")
        .def(py::init<std::int64_t>(), py::arg("seed") = 0)
        // The Python RandomPolicy.act(env) ignores the env; the native
        // signature takes no env at all.
        .def("act", [](RandomPolicy& policy) { return policy.act(); });

    py::class_<GreedyPolicy>(m, "GreedyPolicy")
        .def(py::init<>())
        .def_static("act",
                    [](const SnakeEnv& env) { return GreedyPolicy::act(env); },
                    py::arg("env"));

    py::class_<QTablePolicy>(m, "QTablePolicy")
        .def(py::init<QTablePolicy::Table>(), py::arg("qtable"))
        .def("act", &QTablePolicy::act, py::arg("env"))
        .def("best_action", &QTablePolicy::best_action)
        .def_property_readonly("qtable", &QTablePolicy::qtable);

    // Free-function forms of the policy actions, for callers that hold a
    // bare Q-table instead of a QTablePolicy object.
    m.def("greedy_policy_act",
          [](const SnakeEnv& env) { return GreedyPolicy::act(env); },
          py::arg("env"));
    m.def("qtable_policy_act",
          [](const QTablePolicy::Table& qtable, const SnakeEnv& env) {
              return QTablePolicy(qtable).act(env);
          },
          py::arg("qtable"), py::arg("env"));

    py::class_<MCTSPlayer>(m, "MCTSPlayer")
        .def(py::init<int, int, double, std::int64_t>(),
             py::arg("simulations") = 300, py::arg("rollout_steps") = 100,
             py::arg("c") = 1.0, py::arg("seed") = 0)
        .def("search", &mcts_search, py::arg("env"),
             py::arg("walls") = py::none())
        .def("act", &mcts_search, py::arg("env"),
             py::arg("walls") = py::none());
}
