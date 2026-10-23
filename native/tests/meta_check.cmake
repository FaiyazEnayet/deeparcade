# meta_check: integrity check for the golden metadata.
#
# regen_check excludes meta.json (it records the HEAD sha at
# generation time, so it legitimately differs after any commit).
# This check validates what meta.json records: the fixture version,
# that the counts object carries exactly the expected key set with
# every value a positive whole number, and that each golden file's
# SHA-256 equals the value in the file_sha256 map. (The count values
# themselves are recounted from the corpora by corpus_stats_check.)
#
# Required -D definitions: GOLDEN_DIR.
if(NOT DEFINED GOLDEN_DIR)
  message(FATAL_ERROR "meta_check: missing -DGOLDEN_DIR")
endif()

file(READ "${GOLDEN_DIR}/meta.json" META_JSON)

string(JSON FIXTURE_VERSION GET "${META_JSON}" fixture_version)
if(NOT FIXTURE_VERSION STREQUAL "golden-v2")
  message(FATAL_ERROR
    "meta_check: fixture_version is '${FIXTURE_VERSION}', "
    "expected 'golden-v2'")
endif()

string(JSON N_COUNTS ERROR_VARIABLE COUNTS_ERR
       LENGTH "${META_JSON}" counts)
if(COUNTS_ERR)
  message(FATAL_ERROR "meta_check: meta.json records no counts object")
endif()
if(N_COUNTS EQUAL 0)
  message(FATAL_ERROR "meta_check: meta.json records zero counts")
endif()
set(IDX 0)
set(ACTUAL_COUNT_KEYS "")
while(IDX LESS N_COUNTS)
  string(JSON COUNT_KEY MEMBER "${META_JSON}" counts ${IDX})
  list(APPEND ACTUAL_COUNT_KEYS "${COUNT_KEY}")
  string(JSON COUNT_VAL GET "${META_JSON}" counts ${COUNT_KEY})
  if(NOT COUNT_VAL MATCHES "^[0-9]+$")
    message(FATAL_ERROR
      "meta_check: count '${COUNT_KEY}' is not a whole number: "
      "'${COUNT_VAL}'")
  endif()
  if(COUNT_VAL EQUAL 0)
    message(FATAL_ERROR
      "meta_check: count '${COUNT_KEY}' is not positive")
  endif()
  math(EXPR IDX "${IDX} + 1")
endwhile()

# The counts object must carry exactly the expected key set: a count
# dropped or renamed by a future regeneration must fail here rather
# than pass silently now that every value is recounted elsewhere.
set(EXPECTED_COUNT_KEYS
    mcts_oracle_cases mcts_root_children mcts_searches
    policies_episodes policies_exceptions policies_rand_draws
    policies_states q_act_stream q_updates snake_deaths snake_eats
    snake_episodes snake_exceptions snake_tail_vacations snake_timeouts
    snake_transitions snake_wins transitions_episodes transitions_steps)
set(ACT_SORTED ${ACTUAL_COUNT_KEYS})
list(SORT ACT_SORTED)
set(EXP_SORTED ${EXPECTED_COUNT_KEYS})
list(SORT EXP_SORTED)
if(NOT ACT_SORTED STREQUAL EXP_SORTED)
  set(MISSING "")
  foreach(k ${EXPECTED_COUNT_KEYS})
    list(FIND ACTUAL_COUNT_KEYS "${k}" _idx)
    if(_idx EQUAL -1)
      string(APPEND MISSING " ${k}")
    endif()
  endforeach()
  set(EXTRA "")
  foreach(k ${ACTUAL_COUNT_KEYS})
    list(FIND EXPECTED_COUNT_KEYS "${k}" _idx)
    if(_idx EQUAL -1)
      string(APPEND EXTRA " ${k}")
    endif()
  endforeach()
  message(FATAL_ERROR
    "meta_check: counts key set mismatch\n"
    "  missing:${MISSING}\n"
    "  unexpected:${EXTRA}")
endif()

set(N_FILES 0)
foreach(f rng_traces.txt transitions.json mcts_oracle.json
          q_updates.json snake_corpus.txt policies_corpus.txt
          mcts_cases.txt)
  if(NOT EXISTS "${GOLDEN_DIR}/${f}")
    message(FATAL_ERROR "meta_check: golden file missing: ${f}")
  endif()
  string(JSON RECORDED ERROR_VARIABLE HASH_ERR
         GET "${META_JSON}" file_sha256 ${f})
  if(HASH_ERR)
    message(FATAL_ERROR
      "meta_check: meta.json records no SHA-256 for ${f}")
  endif()
  file(SHA256 "${GOLDEN_DIR}/${f}" ACTUAL)
  if(NOT ACTUAL STREQUAL RECORDED)
    message(FATAL_ERROR
      "meta_check: ${f} SHA-256 mismatch\n"
      "  recorded: ${RECORDED}\n"
      "  actual:   ${ACTUAL}")
  endif()
  math(EXPR N_FILES "${N_FILES} + 1")
endforeach()
message(STATUS
  "meta_check: golden-v2 metadata OK (${N_FILES} file hashes "
  "verified, ${N_COUNTS} counts positive, exact key set)")
