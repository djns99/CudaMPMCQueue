function(setup_bench test_name)
	cuda_add_executable(${test_name} ${test_name}.cu)
	target_link_libraries(${test_name} benchmark)
endfunction()

add_subdirectory(gpu)