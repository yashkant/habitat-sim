#!/bin/sh
C_COMPILER_FLAGS="-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls"
export CXXFLAGS="${C_COMPILER_FLAGS}"
export CFLAGS="${C_COMPILER_FLAGS}"
GIT_ROOT_DIR="$(git rev-parse --show-toplevel)"
export ASAN_OPTIONS="fast_unwind_on_malloc=0:suppressions=${GIT_ROOT_DIR}/tools/asan_suppressions.txt"
export LSAN_OPTIONS="suppressions=${GIT_ROOT_DIR}/tools/lsan_suppressions.txt"
export CORRADE_TEST_COLOR=ON
export GTEST_COLOR=yes
"${GIT_ROOT_DIR}/build.sh" --headless --with-cuda --bullet --run-tests --debug --cmake
#TODO Add Python Tests for ASAN
