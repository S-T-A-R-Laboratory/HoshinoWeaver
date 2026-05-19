function(hnw_apply_common_compile_options target_name)
    set_target_properties(
        "${target_name}"
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            CXX_EXTENSIONS OFF
    )
    if(HNW_ENABLE_CUDA)
        set_target_properties(
            "${target_name}"
            PROPERTIES
                CUDA_STANDARD 17
                CUDA_STANDARD_REQUIRED ON
                CUDA_EXTENSIONS OFF
        )
    endif()
    target_compile_definitions(
        "${target_name}"
        PRIVATE
            HNW_ENABLE_OMP_SIMD=$<IF:$<BOOL:${HNW_ENABLE_OMP_SIMD}>,1,0>
            HNW_ENABLE_CUDA=$<IF:$<BOOL:${HNW_ENABLE_CUDA}>,1,0>
    )

    if(MSVC)
        target_compile_options(
            "${target_name}"
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/EHsc>
        )
    elseif(HNW_ENABLE_MARCH_NATIVE)
        target_compile_options(
            "${target_name}"
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-march=native>
        )
    endif()

    if(HNW_EXTRA_CXX_FLAGS)
        separate_arguments(hnw_extra_cxx_flags NATIVE_COMMAND "${HNW_EXTRA_CXX_FLAGS}")
        foreach(hnw_extra_cxx_flag IN LISTS hnw_extra_cxx_flags)
            target_compile_options(
                "${target_name}"
                PRIVATE
                    $<$<COMPILE_LANGUAGE:CXX>:${hnw_extra_cxx_flag}>
            )
        endforeach()
    endif()

    if(HNW_EXTRA_LINK_FLAGS)
        separate_arguments(hnw_extra_link_flags NATIVE_COMMAND "${HNW_EXTRA_LINK_FLAGS}")
        target_link_options("${target_name}" PRIVATE ${hnw_extra_link_flags})
    endif()
endfunction()
