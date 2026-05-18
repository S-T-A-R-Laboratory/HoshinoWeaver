function(hnw_link_openmp target_name)
    if(NOT HNW_ENABLE_OPENMP)
        return()
    endif()

    if(APPLE)
        # Apple Clang does not ship OpenMP. Use Homebrew libomp.
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE _LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _BREW_RESULT
        )
        if(NOT _BREW_RESULT EQUAL 0 OR NOT EXISTS "${_LIBOMP_PREFIX}/lib/libomp.a")
            message(FATAL_ERROR
                "OpenMP requested but libomp not found.\n"
                "Install via: brew install libomp")
        endif()

        target_compile_options("${target_name}" PRIVATE -Xpreprocessor -fopenmp)
        target_include_directories("${target_name}" PRIVATE "${_LIBOMP_PREFIX}/include")
        # Static link to avoid runtime dylib dependency
        target_link_libraries("${target_name}" PRIVATE "${_LIBOMP_PREFIX}/lib/libomp.a")
    else()
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries("${target_name}" PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()
