cmake_minimum_required(VERSION 3.16)
project(narcade_native CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_library(narcade_rng src/rng.cpp)
target_include_directories(narcade_rng PUBLIC include)
target_compile_options(narcade_rng PRIVATE -Wall -Wextra)

add_executable(rng_trace_test tests/rng_trace_test.cpp)
target_link_libraries(rng_trace_test PRIVATE narcade_rng)
target_compile_options(rng_trace_test PRIVATE -Wall -Wextra)

add_library(narcade_snake src/snake.cpp)
target_include_directories(narcade_snake PUBLIC include)
target_link_libraries(narcade_snake PUBLIC narcade_rng)
target_compile_options(narcade_snake PRIVATE -Wall -Wextra)

add_executable(snake_diff_runner tests/snake_diff_runner.cpp)
target_link_libraries(snake_diff_runner PRIVATE narcade_snake)
target_compile_options(snake_diff_runner PRIVATE -Wall -Wextra)

add_library(narcade_qlearn src/qlearn.cpp)
target_include_directories(narcade_qlearn PUBLIC include)
target_link_libraries(narcade_qlearn PUBLIC narcade_rng)
target_compile_options(narcade_qlearn PRIVATE -Wall -Wextra)

add_library(narcade_policies src/policies.cpp)
target_include_directories(narcade_policies PUBLIC include)
target_link_libraries(narcade_policies PUBLIC narcade_snake)
target_compile_options(narcade_policies PRIVATE -Wall -Wextra)

add_executable(policies_diff_runner tests/policies_diff_runner.cpp)
target_link_libraries(policies_diff_runner PRIVATE narcade_qlearn
                      narcade_policies narcade_snake)
target_compile_options(policies_diff_runner PRIVATE -Wall -Wextra)

add_library(narcade_mcts src/mcts.cpp)
target_include_directories(narcade_mcts PUBLIC include)
target_link_libraries(narcade_mcts PUBLIC narcade_rng)
target_compile_options(narcade_mcts PRIVATE -Wall -Wextra)

add_executable(mcts_diff_runner tests/mcts_diff_runner.cpp)
target_link_libraries(mcts_diff_runner PRIVATE narcade_mcts narcade_snake)
target_compile_options(mcts_diff_runner PRIVATE -Wall -Wextra)

enable_testing()
add_test(NAME rng_trace_test
         COMMAND rng_trace_test ${CMAKE_CURRENT_SOURCE_DIR}/golden/rng_traces.txt)

# Frozen differential corpora (CPP-07): the runners replay the golden
# session files in batch mode and compare every output field in-process;
# no Python driver runs at test time. policies_fixture / policies_fuzz
# and mcts_oracle / mcts_fuzz share one corpus file each (the fixture /
# oracle session is the file's first section).
add_test(NAME snake_replay
         COMMAND snake_diff_runner
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/snake_corpus.txt)
add_test(NAME policies_fixture
         COMMAND policies_diff_runner
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/policies_corpus.txt)
add_test(NAME policies_fuzz
         COMMAND policies_diff_runner
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/policies_corpus.txt)
add_test(NAME mcts_oracle
         COMMAND mcts_diff_runner
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/mcts_cases.txt)
add_test(NAME mcts_fuzz
         COMMAND mcts_diff_runner
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/mcts_cases.txt)
# Maze-mode (wall) MCTS corpus (golden-v3): same runner, WALLS
# protocol overlay; thresholds sit just under the frozen counts
# (404 searches / 977 root children) so a truncated corpus fails.
add_test(NAME mcts_wall
         COMMAND mcts_diff_runner
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/mcts_wall_cases.txt
                 400 900)

find_program(NARCADE_PYTHON3 NAMES python3)
if(NARCADE_PYTHON3)
  # Live-reference guard: regenerating the goldens from the current
  # Python sources must reproduce every committed golden data file
  # byte-for-byte (meta.json is excluded; it records the HEAD sha at
  # generation time).
  add_test(NAME regen_check
           COMMAND ${CMAKE_COMMAND}
                   -DREGEN_PYTHON=${NARCADE_PYTHON3}
                   -DGENERATOR=${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_reference.py
                   -DGOLDEN_DIR=${CMAKE_CURRENT_SOURCE_DIR}/golden
                   -DSCRATCH=${CMAKE_CURRENT_BINARY_DIR}/regen_scratch
                   -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/regen_check.cmake)
endif()

# meta_check: integrity of the golden metadata regen_check excludes.
# Verifies the fixture version, the recorded counts (exactly the
# expected key set, every value positive), and each golden file's
# SHA-256 against native/golden/meta.json. Pure CMake; no Python
# needed, so it is registered unconditionally.
add_test(NAME meta_check
         COMMAND ${CMAKE_COMMAND}
                 -DGOLDEN_DIR=${CMAKE_CURRENT_SOURCE_DIR}/golden
                 -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/meta_check.cmake)

# corpus_stats_check: independent recount of the golden corpora.
# meta_check validates the recorded counts structurally; this check
# re-derives every count from the corpus files themselves (same record
# definitions the C++ runners parse) and compares against meta.json,
# so a wrong-but-positive count fails. Native tool; no Python needed,
# so it is registered unconditionally.
add_executable(corpus_stats_check tests/corpus_stats_check.cpp)
target_compile_options(corpus_stats_check PRIVATE -Wall -Wextra)

add_test(NAME corpus_stats_check
         COMMAND corpus_stats_check ${CMAKE_CURRENT_SOURCE_DIR}/golden)

# ---- CPP-04: pybind11 bindings (optional) ----
# Builds the `narcade_native` Python extension from bindings/bindings.cpp.
# Entirely optional: if pybind11 is not installed the native libraries and
# runners still configure and build exactly as before. Locate pybind11 via
# the interpreter, per the documented `python3 -m pybind11 --cmakedir`.
find_program(NARCADE_BINDINGS_PYTHON3 NAMES python3)
if(NARCADE_BINDINGS_PYTHON3)
  execute_process(
    COMMAND ${NARCADE_BINDINGS_PYTHON3} -m pybind11 --cmakedir
    OUTPUT_VARIABLE NARCADE_PYBIND11_CMAKEDIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
if(NARCADE_PYBIND11_CMAKEDIR)
  list(APPEND CMAKE_PREFIX_PATH "${NARCADE_PYBIND11_CMAKEDIR}")
  set(PYBIND11_FINDPYTHON ON)
  find_package(pybind11 CONFIG QUIET)
endif()
if(pybind11_FOUND)
  # The core libraries above are static; a Python extension is a shared
  # object, so their objects must be position-independent. Setting the
  # property here keeps the append self-contained (the runners that also
  # link these libraries are unaffected semantically).
  set_target_properties(narcade_rng narcade_snake narcade_qlearn
                        narcade_policies narcade_mcts
                        PROPERTIES POSITION_INDEPENDENT_CODE ON)
  pybind11_add_module(narcade_native bindings/bindings.cpp)
  target_link_libraries(narcade_native PRIVATE narcade_mcts narcade_policies
                        narcade_qlearn narcade_snake narcade_rng)
  target_compile_options(narcade_native PRIVATE -Wall -Wextra)
else()
  message(STATUS "pybind11 not found: skipping the narcade_native extension")
endif()

# --- CPP-06c: native benchmark suite -------------------------------------
# Fixed deterministic workloads (env steps/s, RNG draws/s, MCTS
# searches/s at 80 sims, full evaluate-protocol episodes/s). Numbers
# are for the record only; no claim is published from them.
add_executable(narcade_bench apps/bench_main.cpp)
target_link_libraries(narcade_bench PRIVATE narcade_mcts narcade_policies
                      narcade_snake)
target_compile_options(narcade_bench PRIVATE -Wall -Wextra)

add_test(NAME bench_smoke
         COMMAND narcade_bench all --steps 2000 --draws 10000 --searches 3
                 --episodes 3
                 --qtable ${CMAKE_CURRENT_SOURCE_DIR}/../runs/qtable.json)

# --- CPP-06d: native DQN inference ----------------------------------------
# Float32 inference over the exported runs/dqn_weights.json (inference
# only; training stays in PyTorch). Gated on tools/native_dqn_check.py
# against the PyTorch reference.
add_library(narcade_dqn src/dqn_infer.cpp)
target_include_directories(narcade_dqn PUBLIC include)
target_link_libraries(narcade_dqn PUBLIC narcade_snake)
target_compile_options(narcade_dqn PRIVATE -Wall -Wextra)

add_executable(dqn_diff_runner tests/dqn_diff_runner.cpp)
target_link_libraries(dqn_diff_runner PRIVATE narcade_dqn)
target_compile_options(dqn_diff_runner PRIVATE -Wall -Wextra)

# The check driver needs torch for the reference; register it only
# when a Python interpreter with torch is available (it is run
# manually regardless as the packet gate).
find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_FOUND)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch"
    RESULT_VARIABLE NARCADE_DQN_TORCH_RC
    OUTPUT_QUIET ERROR_QUIET)
  if(NARCADE_DQN_TORCH_RC EQUAL 0)
    add_test(NAME dqn_check
             COMMAND "${Python3_EXECUTABLE}"
                     ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_dqn_check.py
                     --runner $<TARGET_FILE:dqn_diff_runner> --states 3000)
  else()
    message(STATUS
      "dqn_check skipped: Python3 interpreter has no torch available")
  endif()
else()
  message(STATUS "dqn_check skipped: Python3 not found")
endif()

# --- CPP-05: native evaluation engine (appended; self-contained) ---
add_library(narcade_evaluate src/evaluate.cpp)
target_include_directories(narcade_evaluate PUBLIC include)
target_link_libraries(narcade_evaluate PUBLIC narcade_snake narcade_policies
                      narcade_mcts)
target_compile_options(narcade_evaluate PRIVATE -Wall -Wextra)

add_executable(narcade_evaluate_app apps/evaluate_main.cpp)
set_target_properties(narcade_evaluate_app PROPERTIES
                      OUTPUT_NAME narcade_evaluate)
target_link_libraries(narcade_evaluate_app PRIVATE narcade_evaluate)
target_compile_options(narcade_evaluate_app PRIVATE -Wall -Wextra)

if(NARCADE_PYTHON3)
  add_test(NAME eval_fuzz
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_eval_fuzz.py
                   --runner $<TARGET_FILE:narcade_evaluate_app>)
endif()

# --- FIX-J: frozen evaluator regression for hosted CI ------------------
# eval_fuzz above is a live differential check against the Python
# reference over 200 paired episodes; its Python pass takes tens of
# minutes, so hosted CI excludes it (ctest -E eval_fuzz, see the Native
# workflow) and it remains the local gate for native evaluation
# changes. eval_regression freezes the Python reference results for a
# 12-episode subset (seeds 5000-5002, all four players) in
# golden/eval_regression.json and replays them through the native
# evaluator, so hosted CI still guards evaluation parity on every run.
# The frozen file is deliberately outside meta.json's corpus
# accounting (meta_check / corpus_stats_check cover the eight corpus
# files) and outside regen_check (its regeneration needs the
# evaluation protocol's Python pass); it is regenerated manually with
# tools/native_reference.py --eval-regression. Native tool; no Python
# needed, so it is registered unconditionally.
add_executable(eval_regression_check tests/eval_regression_check.cpp)
target_link_libraries(eval_regression_check PRIVATE narcade_evaluate)
target_compile_options(eval_regression_check PRIVATE -Wall -Wextra)

add_test(NAME eval_regression
         COMMAND eval_regression_check
                 ${CMAKE_CURRENT_SOURCE_DIR}/golden/eval_regression.json
                 ${CMAKE_CURRENT_SOURCE_DIR}/../runs/qtable.json)

# --- CPP-06a: native Q-learning trainer (scripts/train.py port). ---
add_library(narcade_trainer src/trainer.cpp)
target_include_directories(narcade_trainer PUBLIC include)
target_link_libraries(narcade_trainer PUBLIC narcade_qlearn narcade_policies
                      narcade_snake)
target_compile_options(narcade_trainer PRIVATE -Wall -Wextra)

add_executable(train_main apps/train_main.cpp)
target_link_libraries(train_main PRIVATE narcade_trainer)
target_compile_options(train_main PRIVATE -Wall -Wextra)

if(NARCADE_PYTHON3)
  add_test(NAME trainer_check
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_trainer_check.py
                   --trainer $<TARGET_FILE:train_main> --small-only)
endif()

# --- CPP-06b: native move-level trace instrumentation (analyze.py). ---
add_library(narcade_trace src/trace.cpp)
target_include_directories(narcade_trace PUBLIC include)
target_link_libraries(narcade_trace PUBLIC narcade_snake)
target_compile_options(narcade_trace PRIVATE -Wall -Wextra)

add_executable(analyze_main apps/analyze_main.cpp)
target_link_libraries(analyze_main PRIVATE narcade_trace narcade_policies
                      narcade_mcts)
target_compile_options(analyze_main PRIVATE -Wall -Wextra)

if(NARCADE_PYTHON3)
  add_test(NAME analyze_check
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_analyze_check.py
                   --analyzer $<TARGET_FILE:analyze_main> --fast)
endif()
