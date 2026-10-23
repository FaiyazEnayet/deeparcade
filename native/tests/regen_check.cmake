# regen_check (CPP-07): live-reference guard for the frozen goldens.
#
# Runs tools/native_reference.py into a scratch directory and
# byte-compares every golden data file against the committed copy.
# meta.json is excluded: it records the HEAD sha at generation time,
# so it legitimately differs after any commit.
#
# Required -D definitions: REGEN_PYTHON, GENERATOR, GOLDEN_DIR, SCRATCH.
foreach(var REGEN_PYTHON GENERATOR GOLDEN_DIR SCRATCH)
  if(NOT DEFINED ${var})
    message(FATAL_ERROR "regen_check: missing -D${var}")
  endif()
endforeach()

file(REMOVE_RECURSE "${SCRATCH}")
file(MAKE_DIRECTORY "${SCRATCH}")
execute_process(
  COMMAND "${REGEN_PYTHON}" "${GENERATOR}" --out-dir "${SCRATCH}"
  RESULT_VARIABLE rc
  OUTPUT_VARIABLE gen_out
  ERROR_VARIABLE gen_err)
if(NOT rc EQUAL 0)
  message(FATAL_ERROR "regen_check: generator failed (${rc})\n"
                      "${gen_out}\n${gen_err}")
endif()

foreach(f rng_traces.txt transitions.json mcts_oracle.json q_updates.json
          snake_corpus.txt policies_corpus.txt mcts_cases.txt
          mcts_wall_cases.txt)
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -E compare_files
            "${GOLDEN_DIR}/${f}" "${SCRATCH}/${f}"
    RESULT_VARIABLE crc
    OUTPUT_QUIET ERROR_QUIET)
  if(NOT crc EQUAL 0)
    message(FATAL_ERROR
      "regen_check: ${f} differs from the committed golden "
      "(Python reference drift or generator nondeterminism)")
  endif()
endforeach()
message(STATUS "regen_check: all golden files reproduce byte-identically")
