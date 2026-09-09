cmake_minimum_required(VERSION 3.14)
project(AutogradEngine)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  )

  

FetchContent_MakeAvailable(googletest)
FetchContent_MakeAvailable(pybind11)

add_library(autograd_lib 
    src/tensor.cpp
    src/autograd.cpp 
    src/flatten.cpp
    src/linear.cpp
    src/loss_functions.cpp
    src/module.cpp
    src/relu.cpp
    
    include/tensor.hpp
    include/autograd.hpp
    include/flatten.hpp
    include/linear.hpp
    include/loss_functions.hpp
    include/module.hpp
    include/relu.hpp
)

add_library(datasets_lib
    src/data_processing/dataloader.cpp
    src/data_processing/dataset.cpp

    include/dataloader.hpp
    include/dataset.hpp
)



target_include_directories(autograd_lib PUBLIC include)

target_include_directories(datasets_lib PUBLIC include)


target_link_libraries(datasets_lib PUBLIC autograd_lib)


enable_testing()
add_subdirectory(tests)

add_subdirectory(bindings)