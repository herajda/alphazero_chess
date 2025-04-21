cmake_minimum_required(VERSION 3.15)
project(alphazero_chess)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(TORCH_CUDA_ARCH_LIST "8.9")  

# pull in the python headers so <Python.h> is found
include_directories(${Python3_INCLUDE_DIRS})
# Find PyTorch C++ (LibTorch)
find_package(Torch REQUIRED)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/cpp/include
    ${CMAKE_SOURCE_DIR}/lib/chess-library
    ${TORCH_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
)

# Build the simulate module as a shared library
add_library(chess_engine MODULE
    src/cpp/src/chess_game.cpp
    src/cpp/src/mcts.cpp
    src/cpp/src/batch_manager.cpp
    src/cpp/src/simulate.cpp
)

target_link_libraries(chess_engine PRIVATE 
    pybind11::embed      
    ${Python3_LIBRARIES} 
    ${TORCH_LIBRARIES})
set_target_properties(chess_engine PROPERTIES
    LINK_FLAGS "-static-libstdc++ -static-libgcc"
)
set_target_properties(chess_engine PROPERTIES
    PREFIX ""   # produce chess_engine.so

)
set_property(TARGET chess_engine PROPERTY CXX_VISIBILITY_PRESET default)
set_property(TARGET chess_engine PROPERTY VISIBILITY_INLINES_HIDDEN OFF)

# Optionally, install the module into your Python path
install(TARGETS chess_engine LIBRARY DESTINATION .)
