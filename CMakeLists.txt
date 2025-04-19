cmake_minimum_required(VERSION 3.15)
project(alphazero_chess)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/cpp/include
    ${CMAKE_SOURCE_DIR}/lib/chess-library
)

# Build the simulate module as a shared library
add_library(chess_engine MODULE
    src/cpp/src/chess_game.cpp
    src/cpp/src/mcts.cpp
    src/cpp/src/batch_manager.cpp
    src/cpp/src/simulate.cpp
)

target_link_libraries(chess_engine PRIVATE pybind11::embed)
set_target_properties(chess_engine PROPERTIES
    PREFIX ""   # produce chess_engine.so
)

# Optionally, install the module into your Python path
install(TARGETS chess_engine LIBRARY DESTINATION .)
