function(hnw_link_openmp target_name)
    if(NOT HNW_ENABLE_OPENMP)
        return()
    endif()

    find_package(OpenMP REQUIRED COMPONENTS CXX)
    target_link_libraries("${target_name}" PRIVATE OpenMP::OpenMP_CXX)
endfunction()
