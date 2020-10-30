#!/bin/bash
# export LD_PRELOAD="$(gcc -print-file-name=libasan.so):$LD_PRELOAD"
export MAGNUM_GPU_VALIDATION=ON
export MAGNUM_LOG=verbose
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvidia-opengl/:$LD_LIBRARY_PATH
C_COMPILER_FLAGS="-fsanitize=address -fno-omit-frame-pointer -fno-optimize-sibling-calls -g -O1"
export CXXFLAGS="${C_COMPILER_FLAGS}"
export CFLAGS="${C_COMPILER_FLAGS}"
GIT_ROOT_DIR="$(git rev-parse --show-toplevel)"
export ASAN_OPTIONS="fast_unwind_on_malloc=0:suppressions=${GIT_ROOT_DIR}/tools/asan_suppressions.txt"
export LSAN_OPTIONS="suppressions=${GIT_ROOT_DIR}/tools/lsan_suppressions.txt"
export CORRADE_TEST_COLOR=ON
export GTEST_COLOR=yes
#build/tests/SimTest
"${GIT_ROOT_DIR}/build.sh" --with-cuda --bullet --build-datatool  --debug --run-tests --cmake --cmake-args="-Dgtest_disable_pthreads=ON"
#TODO Add Python Tests for ASAN
