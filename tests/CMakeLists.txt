find_package(glog REQUIRED)
#set(GTEST_INCLUDE_DIRS /usr/src/googletest/googletest/include)
# set(GTEST_INCLUDE_DIRS "/usr/src/googletest/googletest/include")
# set(GTEST_LIBRARIES "/usr/src/googletest/googletest/build/lib")
# 然后使用这些变量来查找GTest
find_package(GTest REQUIRED)

set(link_ext_lib glog::glog GTest::gtest)
#add_executable(test_llm test_lalma_cpu.cpp test_main.cpp)
add_executable(test_llm test_main.cpp test_scale.cpp test_emb.cpp test_swiglu.cpp
    test_rmsnorm.cpp test_mamul.cpp test_buffer.cpp test_tensor.cpp test_add.cpp)
# add_executable(test_llm test_main.cpp test_mamul.cpp)
# add_executable(test_llm test_main.cpp test_rmsnorm.cpp)
# add_executable(test_llm test_main.cpp test_emb.cpp)
# add_executable(test_llm test_main.cpp test_scale.cpp)
# add_executable(test_llm test_main.cpp test_rmsnorm.cpp)

target_link_libraries(test_llm ${link_ext_lib})
target_include_directories(test_llm PUBLIC ${glog_INCLUDE_DIR})
target_include_directories(test_llm PUBLIC ${GTEST_INCLUDE_DIRS})
target_include_directories(test_llm PUBLIC ../kernels)
target_include_directories(test_llm PUBLIC ../include)
target_link_directories(test_llm PUBLIC ${PROJECT_SOURCE_DIR}/lib)
target_link_libraries(test_llm llama)

set_target_properties(test_llm PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})