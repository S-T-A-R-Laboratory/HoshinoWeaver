function(hnw_require_python_and_pybind11)
    find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

    # Calling get_cmake_dir() directly returns the raw path (avoid double quotes).
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE hnw_pybind11_cmakedir
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE hnw_pybind11_result
    )
    if(NOT hnw_pybind11_result EQUAL 0 OR NOT hnw_pybind11_cmakedir)
        message(FATAL_ERROR "Failed to resolve pybind11 CMake directory via `${Python3_EXECUTABLE} -c 'import pybind11; print(pybind11.get_cmake_dir())'`.")
    endif()

    # Defensive: strip a single pair of surrounding double quotes if present
    # (e.g. older pybind11 CLI output, or any future wrapper that re-quotes).
    string(REGEX REPLACE "^\"(.*)\"$" "\\1" hnw_pybind11_cmakedir "${hnw_pybind11_cmakedir}")

    list(APPEND CMAKE_PREFIX_PATH "${hnw_pybind11_cmakedir}")
    find_package(pybind11 CONFIG REQUIRED)
    set(HNW_PYBIND11_INCLUDE_DIRS "${pybind11_INCLUDE_DIRS}" PARENT_SCOPE)
endfunction()
