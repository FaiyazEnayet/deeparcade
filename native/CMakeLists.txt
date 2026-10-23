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

find_program(NARCADE_PYTHON3 NAMES python3)
if(NARCADE_PYTHON3)
  add_test(NAME snake_replay
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_fuzz.py
                   --runner $<TARGET_FILE:snake_diff_runner> --replay-only)
  add_test(NAME policies_fixture
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_policies_fuzz.py
                   --runner $<TARGET_FILE:policies_diff_runner> --fixture-only)
  add_test(NAME policies_fuzz
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_policies_fuzz.py
                   --runner $<TARGET_FILE:policies_diff_runner>)
  add_test(NAME mcts_oracle
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_mcts_fuzz.py
                   --runner $<TARGET_FILE:mcts_diff_runner> --oracle-only)
  add_test(NAME mcts_fuzz
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_mcts_fuzz.py
                   --runner $<TARGET_FILE:mcts_diff_runner>)
endif()

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
# when the known torch venv is present (it is run manually regardless
# as the packet gate).
if(EXISTS "$ENV{HOME}/workspace/flowforge/.venv/bin/python")
  add_test(NAME dqn_check
           COMMAND "$ENV{HOME}/workspace/flowforge/.venv/bin/python"
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_dqn_check.py
                   --runner $<TARGET_FILE:dqn_diff_runner> --states 3000)
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
