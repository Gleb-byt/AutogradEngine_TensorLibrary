cmake_minimum_required(VERSION 3.14)
project(AutogradEngine)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
add_library(autograd_lib 
    src/tensor.cpp
    src/autograd.cpp 
    include/tensor.hpp
    include/autograd.hpp
)
target_include_directories(autograd_lib PUBLIC include)
enable_testing()
add_subdirectory(tests)