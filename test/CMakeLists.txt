function(setup_test test_name)
	cuda_add_executable(${test_name} ${test_name}.cu)
	target_link_libraries(${test_name} gtest)
	add_test(${test_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${test_name})
endfunction()


add_subdirectory(cpu)
add_subdirectory(gpu)