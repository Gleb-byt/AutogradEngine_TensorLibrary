
pybind11_add_module(autograd_engine bindings.cpp)

target_link_options(autograd_engine PRIVATE "-static-libgcc" "-static-libstdc++" "-static")


target_link_libraries(autograd_engine PRIVATE autograd_lib)