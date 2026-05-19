function(hnw_require_python_and_pybind11)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE hnw_pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE hnw_pybind11_result
    )
    if(NOT hnw_pybind11_result EQUAL 0 OR NOT hnw_pybind11_cmakedir)
        message(FATAL_ERROR "Failed to resolve pybind11 CMake directory via `${Python3_EXECUTABLE} -m pybind11 --cmakedir`.")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${hnw_pybind11_cmakedir}")
    find_package(pybind11 CONFIG REQUIRED)
    set(HNW_PYBIND11_INCLUDE_DIRS "${pybind11_INCLUDE_DIRS}" PARENT_SCOPE)
endfunction()
