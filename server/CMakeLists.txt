add_executable(qwen2 main.cpp)
target_link_directories(qwen2 PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(qwen2 llama)
set_target_properties(qwen2 PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)