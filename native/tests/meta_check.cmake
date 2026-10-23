# meta_check: integrity check for the golden metadata.
#
# regen_check excludes meta.json (it records the HEAD sha at
# generation time, so it legitimately differs after any commit).
# This check validates what meta.json records: the fixture version,
# that every recorded count is present and positive, and that each
# golden file's SHA-256 equals the value in the file_sha256 map.
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
while(IDX LESS N_COUNTS)
  string(JSON COUNT_KEY MEMBER "${META_JSON}" counts ${IDX})
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
  "verified, ${N_COUNTS} counts positive)")
