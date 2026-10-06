# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20.0)

find_package(Zephyr REQUIRED HINTS $ENV{ZEPHYR_BASE})
project(torus_early_exit)

target_sources(app PRIVATE src/main.c src/ee_math.c src/ee_pipeline.cpp src/ee_tflm_compat.cpp)
# ee_tflm_compat.cpp: int16 TRANSPOSE for TFLM 48613f7 (used only by int16 builds, i.e. TinierHAR; the linker drops
# it from the int8 builds).

# ee_math.c implements the arithmetic shared with the Python host reference
# (EE_INTERFACE.md, including TinierHAR's float32 C tail): keep multiply-adds unfused so its rounding matches the host.
set_source_files_properties(src/ee_math.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# As in Zephyr's own TFLM samples: Zephyr does not support the C++ ABI for thread-safe
# initialisation of function-local statics, so turn the guards off (ee_pipeline.cpp has
# no function-local statics; this keeps libsupc++'s guard / exception code out of the image).
set(NO_THREADSAFE_STATICS $<TARGET_PROPERTY:compiler-cpp,no_threadsafe_statics>)
zephyr_compile_options($<$<COMPILE_LANGUAGE:CXX>:${NO_THREADSAFE_STATICS}>)

# Backbone / option selection from the west command line (always build pristine, -p):
#   west build -b nrf54l15dk/nrf54l15/cpuapp -p -- -DEE_BACKBONE=DCL -DEE_FORCE_EXIT=2
# EE_BACKBONE is TCN (default), TINIERHAR or DCL; the other options are the macros
# documented in src/ee_select.h. To hard-wire a backbone here instead, use e.g.
#   target_compile_definitions(app PRIVATE EE_BACKBONE_DCL)
# west builds with sysbuild by default (NCS 3.x). A -D after "--" then goes to the sysbuild
# cache, not to this application's CMake cache, so a plain if(DEFINED EE_BACKBONE) would
# never see it and the build would silently keep the defaults. zephyr_get() reads the
# sysbuild cache (EE_X or <image>_EE_X), then the CMake cache (--no-sysbuild), then the
# environment.
zephyr_get(EE_BACKBONE)
if(DEFINED EE_BACKBONE)
  if(NOT EE_BACKBONE MATCHES "^(TCN|TINIERHAR|DCL)$")
    message(FATAL_ERROR "EE_BACKBONE must be TCN, TINIERHAR or DCL (got '${EE_BACKBONE}')")
  endif()
  target_compile_definitions(app PRIVATE EE_BACKBONE_${EE_BACKBONE})
  message(STATUS "Early-exit firmware: EE_BACKBONE=${EE_BACKBONE} (from the command line)")
else()
  message(STATUS "Early-exit firmware: EE_BACKBONE not given, src/ee_select.h decides (TCN unless edited)")
endif()
foreach(opt EE_FORCE_EXIT EE_SELF_TEST EE_INFERENCE_IDLE_MS EE_WARMUP_ITERATIONS
            EE_USE_WINDOW_MARKER EE_LOG_BREAKDOWN EE_PART_TIMER_DWT)
  zephyr_get(${opt})
  if(DEFINED ${opt})
    target_compile_definitions(app PRIVATE ${opt}=${${opt}})
    message(STATUS "Early-exit firmware: ${opt}=${${opt}} (from the command line)")
  endif()
endforeach()
