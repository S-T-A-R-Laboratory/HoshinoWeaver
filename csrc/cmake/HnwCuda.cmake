macro(hnw_enable_cuda_language)
    if(HNW_ENABLE_CUDA)
        enable_language(CUDA)
        include(CMakeCUDAInformation)
        find_package(CUDAToolkit REQUIRED)
    endif()
endmacro()

function(hnw_link_cuda_runtime target_name)
    if(NOT HNW_ENABLE_CUDA)
        return()
    endif()

    # Static link to eliminate cudart DLL dependency
    target_link_libraries("${target_name}" PRIVATE CUDA::cudart_static)
endfunction()
