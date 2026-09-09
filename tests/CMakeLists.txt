add_executable(run_tests test_tensor.cpp)
add_executable(data_tests data_tests.cpp)

target_link_libraries(
    data_tests
    PRIVATE
    datasets_lib
    autograd_lib
)
target_link_libraries(run_tests 
    GTest::gtest_main 
    autograd_lib
)
include(GoogleTest)
gtest_discover_tests(run_tests)