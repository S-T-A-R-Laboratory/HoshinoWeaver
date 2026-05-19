macro(hnw_enable_cuda_language)
    if(HNW_ENABLE_CUDA)
        # Detect toolkit version first to choose architectures
        find_package(CUDAToolkit REQUIRED)

        if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
            # Pascal (60) through latest supported by detected toolkit
            set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86")
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "11.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 89)
            endif()
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 90)
            endif()
            if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.8")
                list(APPEND CMAKE_CUDA_ARCHITECTURES 100)
            endif()
            message(STATUS "CUDA architectures (auto): ${CMAKE_CUDA_ARCHITECTURES}")
        endif()

        enable_language(CUDA)
    endif()
endmacro()

function(hnw_link_cuda_runtime target_name)
    if(NOT HNW_ENABLE_CUDA)
        return()
    endif()

    # Static link to eliminate cudart DLL dependency
    target_link_libraries("${target_name}" PRIVATE CUDA::cudart_static)
endfunction()
