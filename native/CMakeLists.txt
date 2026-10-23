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

enable_testing()
add_test(NAME rng_trace_test
         COMMAND rng_trace_test ${CMAKE_CURRENT_SOURCE_DIR}/golden/rng_traces.txt)

find_program(NARCADE_PYTHON3 NAMES python3)
if(NARCADE_PYTHON3)
  add_test(NAME snake_replay
           COMMAND ${NARCADE_PYTHON3}
                   ${CMAKE_CURRENT_SOURCE_DIR}/../tools/native_fuzz.py
                   --runner $<TARGET_FILE:snake_diff_runner> --replay-only)
endif()
