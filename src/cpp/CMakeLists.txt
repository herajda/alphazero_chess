cmake_minimum_required(VERSION 3.15)
project(alphazero_chess)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../../lib/chess-library)

# Source files
set(SOURCES
    src/chess_game.cpp
    src/mcts.cpp
    src/batch_manager.cpp
    src/bindings.cpp
)

# Create the pybind11 module
pybind11_add_module(chess_engine ${SOURCES})

# Link libraries (if chess-library requires any)
# target_link_libraries(chess_engine PRIVATE <chess-lib>)
