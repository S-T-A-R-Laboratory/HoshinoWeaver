include(CheckIPOSupported)

option(HNW_ENABLE_OPENMP "Enable OpenMP for custom-op builds." ON)
option(HNW_ENABLE_OMP_SIMD "Enable explicit OpenMP SIMD pragmas." OFF)
option(HNW_ENABLE_LTO "Enable link-time optimization." OFF)
option(HNW_ENABLE_MARCH_NATIVE "Enable -march=native for local builds." OFF)
option(HNW_ENABLE_CUDA "Enable CUDA native custom-op targets." OFF)

set(HNW_EXTRA_CXX_FLAGS "" CACHE STRING "Extra C++ compiler flags for custom-op builds.")
set(HNW_EXTRA_LINK_FLAGS "" CACHE STRING "Extra linker flags for custom-op builds.")

function(hnw_enable_lto target_name)
    if(NOT HNW_ENABLE_LTO)
        return()
    endif()

    check_ipo_supported(RESULT hnw_ipo_supported OUTPUT hnw_ipo_output)
    if(hnw_ipo_supported)
        set_property(TARGET "${target_name}" PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        return()
    endif()

    message(WARNING "HNW_ENABLE_LTO=ON but IPO/LTO is unsupported: ${hnw_ipo_output}")
endfunction()
