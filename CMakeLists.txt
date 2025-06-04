cmake_minimum_required(VERSION 3.15)
project(alphazero_chess)

# 1) Force a Release build by default
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build (Debug, Release, RelWithDebInfo, MinSizeRel)"
      FORCE)
endif()

# 2) Base C++ standard
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 3) Define your Release flags
#    -O3            : highest safe optimization
#    -march=native : tune to your CPU
#    -flto          : link-time optimization
#    -funroll-loops: loop unrolling
#    -DNDEBUG       : disable asserts
set(CMAKE_CXX_FLAGS_RELEASE
    "${CMAKE_CXX_FLAGS_RELEASE} -Ofast -march=native -flto -funroll-loops -DNDEBUG")


# 5) GPU architecture for LibTorch
# Try to use environment variable `TORCH_CUDA_ARCH_LIST` if provided.
# Otherwise attempt to detect the compute capability of the first CUDA device
# via a short Python snippet. When detection fails, leave the variable unset
# so LibTorch can decide a suitable default.
if(DEFINED ENV{TORCH_CUDA_ARCH_LIST})
  set(TORCH_CUDA_ARCH_LIST "$ENV{TORCH_CUDA_ARCH_LIST}")
else()
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch,sys; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')"
    RESULT_VARIABLE TORCH_CC_RES
    OUTPUT_VARIABLE TORCH_CC_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(TORCH_CC_RES EQUAL 0)
    set(TORCH_CUDA_ARCH_LIST "${TORCH_CC_OUT}")
  endif()
endif()

# Pull in Python & PyBind11 & Torch
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
find_package(Torch REQUIRED)

# Includes
include_directories(
  ${CMAKE_SOURCE_DIR}/src/cpp/include
  ${CMAKE_SOURCE_DIR}/lib/chess-library/include
  ${TORCH_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
)

# Build your module
add_library(chess_engine MODULE
  src/cpp/src/chess_game.cpp
  src/cpp/src/mcts.cpp
  src/cpp/src/batch_manager.cpp
  src/cpp/src/simulate.cpp
)

target_link_libraries(chess_engine PRIVATE
  pybind11::embed
  ${Python3_LIBRARIES}
  ${TORCH_LIBRARIES}
)

# 6) Enable LTO/IPO on this target (requires CMake ≥3.9)
include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED)
if(IPO_SUPPORTED)
  set_property(TARGET chess_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# 7) Static‐link the C++ runtime if you really need it
set_target_properties(chess_engine PROPERTIES
  LINK_FLAGS "-static-libstdc++ -static-libgcc"
  PREFIX ""
)

# Visibility settings (you can leave these as-is)
set_property(TARGET chess_engine PROPERTY CXX_VISIBILITY_PRESET default)
set_property(TARGET chess_engine PROPERTY VISIBILITY_INLINES_HIDDEN OFF)

# Optional install
install(TARGETS chess_engine LIBRARY DESTINATION .)
