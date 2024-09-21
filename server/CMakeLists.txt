add_executable(llam2_7b main.cpp)
target_link_directories(llam2_7b PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(llam2_7b llama)
set_target_properties(llam2_7b PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)