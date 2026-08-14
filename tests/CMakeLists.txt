add_executable(run_tests test_tensor.cpp)
target_link_libraries(run_tests 
    GTest::gtest_main 
    autograd_lib
)
include(GoogleTest)
gtest_discover_tests(run_tests)