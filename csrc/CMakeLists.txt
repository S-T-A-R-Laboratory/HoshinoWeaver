cmake_minimum_required(VERSION 3.23)

project(hnw_custom_ops LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(HnwOptions)
include(HnwCompiler)
include(HnwOpenMP)
include(HnwPython)
include(HnwCuda)

hnw_enable_cuda_language()
hnw_require_python_and_pybind11()

if(HNW_ENABLE_METAL)
    if(NOT APPLE)
        message(FATAL_ERROR "HNW_ENABLE_METAL is only supported on macOS.")
    endif()
    enable_language(OBJCXX)
endif()

set(HNW_PACKAGE_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../hoshicore/_custom_op")

add_library(hnw_ops_common STATIC
    common/backend_info.cpp
    common/host_parallel_copy.cpp
)

add_library(hnw_ops_cpu_calibration STATIC
    ops/cpu/calibration/calibration_ops.cpp
)

add_library(hnw_ops_cpu_detection STATIC
    ops/cpu/detection/gray_ops.cpp
    ops/cpu/detection/median_star_mask_ops.cpp
    ops/cpu/detection/star_detect_fused_pixel_components_ops.cpp
)

add_library(hnw_ops_cpu_alignment STATIC
    ops/cpu/alignment/alignment_ops.cpp
    ops/cpu/alignment/asterism_match_ops.cpp
    ops/cpu/alignment/asterism_mutual_nearest_ops.cpp
    ops/cpu/alignment/matching_bidirectional_nearest_ops.cpp
)

# Token arithmetic must round like NumPy even on targets with FMA.
set_property(SOURCE ops/cpu/alignment/asterism_match_ops.cpp APPEND PROPERTY
    COMPILE_OPTIONS "$<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-ffp-contract=off>"
)

add_library(hnw_ops_cpu_fgp STATIC
    ops/cpu/fgp/fgp_accumulate_ops.cpp
    ops/cpu/fgp/fgp_merge_ops.cpp
    ops/cpu/fgp/fgp_ops.cpp
    ops/cpu/fgp/huber_accumulate_ops.cpp
    ops/cpu/fgp/sigma_clip_merge_ops.cpp
)

add_library(hnw_ops_cpu_filter STATIC
    ops/cpu/filter/filter_ops.cpp
)

add_library(hnw_ops_cpu_max STATIC
    ops/cpu/max/max_ops.cpp
)

add_library(hnw_ops_cpu_median STATIC
    ops/cpu/median/median_ops.cpp
)

add_library(hnw_ops_cpu_noise STATIC
    ops/cpu/noise/noise_ops.cpp
)

add_library(hnw_ops_cpu_remap STATIC
    ops/cpu/remap/remap_ops.cpp
)

add_library(hnw_ops_cpu_sigma_clip STATIC
    ops/cpu/sigma_clip/sigma_clip_chunk_ops.cpp
)

add_library(hnw_ops_cpu_star_shrink STATIC
    ops/cpu/star_shrink/star_mask_dog_ops.cpp
    ops/cpu/star_shrink/star_shrink_ops.cpp
)

add_library(hnw_ops_cpu_wavelet STATIC
    ops/cpu/wavelet/wavelet_ops.cpp
)

set(hnw_cpu_targets
    hnw_ops_common
    hnw_ops_cpu_calibration
    hnw_ops_cpu_detection
    hnw_ops_cpu_alignment
    hnw_ops_cpu_fgp
    hnw_ops_cpu_filter
    hnw_ops_cpu_max
    hnw_ops_cpu_median
    hnw_ops_cpu_noise
    hnw_ops_cpu_remap
    hnw_ops_cpu_sigma_clip
    hnw_ops_cpu_star_shrink
    hnw_ops_cpu_wavelet
)

foreach(target IN LISTS hnw_cpu_targets)
    target_include_directories(
        "${target}"
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            ${HNW_PYBIND11_INCLUDE_DIRS}
    )
    target_link_libraries("${target}" PRIVATE pybind11::headers)
    hnw_apply_common_compile_options("${target}")
    hnw_link_openmp("${target}")
    hnw_enable_lto("${target}")
endforeach()

target_link_libraries(hnw_ops_cpu_detection PRIVATE hnw_ops_cpu_wavelet)

pybind11_add_module(_C MODULE
    NO_EXTRAS
    modules/_C.cpp
)

if(HNW_ENABLE_CUDA)
    target_sources(_C PRIVATE
        common/backend_cuda_info.cu
        # Alignment matching distance and bidirectional nearest-neighbor reduction.
        ops/cuda/alignment/matching_bidirectional_nearest_ops.cpp
        ops/cuda/alignment/matching_bidirectional_nearest_ops.cu
        # Asterism-token mutual nearest pairs within a threshold.
        ops/cuda/alignment/asterism_mutual_nearest_ops.cpp
        ops/cuda/alignment/asterism_mutual_nearest_ops.cu
        # Star-point angular-histogram descriptors.
        ops/cuda/alignment/point_features_ops.cpp
        ops/cuda/alignment/point_features_ops.cu
        # Fused camera-model remap.
        ops/cuda/remap/camera_model_remap_fused_ops.cpp
        ops/cuda/remap/camera_model_remap_fused_ops.cu
        # Wavelet CUDA backend and fused star-detection pixel/component stage.
        ops/cuda/wavelet/wavelet_cuda_ops.cpp
        ops/cuda/wavelet/wavelet_cuda_ops.cu
        ops/cuda/detection/star_detect_fused_pixel_components_ops.cpp
        ops/cuda/detection/star_detect_fused_pixel_components_ops.cu
        # Chunk stack kernels.
        ops/cuda/fgp/huber_weighted_chunk_ops.cpp
        ops/cuda/fgp/huber_weighted_chunk_ops.cu
        ops/cuda/sigma_clip/sigma_clip_fused_chunk_ops.cpp
        ops/cuda/sigma_clip/sigma_clip_fused_chunk_ops.cu
        # Star shrink fused processing.
        ops/cuda/star_shrink/star_mask_ops.cpp
        ops/cuda/star_shrink/star_mask_ops.cu
        ops/cuda/star_shrink/star_shrink_process_ops.cpp
        ops/cuda/star_shrink/star_shrink_process_ops.cu
    )
endif()

target_include_directories(
    _C
    PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}"
        ${HNW_PYBIND11_INCLUDE_DIRS}
)
target_link_libraries(_C PRIVATE
    hnw_ops_common
    hnw_ops_cpu_calibration
    hnw_ops_cpu_detection
    hnw_ops_cpu_alignment
    hnw_ops_cpu_fgp
    hnw_ops_cpu_filter
    hnw_ops_cpu_max
    hnw_ops_cpu_median
    hnw_ops_cpu_noise
    hnw_ops_cpu_remap
    hnw_ops_cpu_sigma_clip
    hnw_ops_cpu_star_shrink
    hnw_ops_cpu_wavelet
)
hnw_apply_common_compile_options(_C)
hnw_link_openmp(_C)
hnw_link_cuda_runtime(_C)
hnw_enable_lto(_C)

set_target_properties(_C PROPERTIES
    OUTPUT_NAME "_C"
    LIBRARY_OUTPUT_DIRECTORY "${HNW_PACKAGE_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${HNW_PACKAGE_OUTPUT_DIR}"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/archive"
)

if(HNW_ENABLE_METAL)
    find_program(HNW_XCRUN_EXECUTABLE xcrun REQUIRED)
    find_library(HNW_METAL_FRAMEWORK Metal REQUIRED)
    find_library(HNW_FOUNDATION_FRAMEWORK Foundation REQUIRED)
    find_library(HNW_COREGRAPHICS_FRAMEWORK CoreGraphics REQUIRED)

    execute_process(
        COMMAND "${HNW_XCRUN_EXECUTABLE}" -sdk macosx -find metal
        RESULT_VARIABLE HNW_METAL_COMPILER_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    execute_process(
        COMMAND "${HNW_XCRUN_EXECUTABLE}" -sdk macosx -find metallib
        RESULT_VARIABLE HNW_METALLIB_COMPILER_RESULT
        OUTPUT_QUIET
        ERROR_QUIET
    )
    if(NOT HNW_METAL_COMPILER_RESULT EQUAL 0 OR
       NOT HNW_METALLIB_COMPILER_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Metal shader tools are unavailable. Install/select a full Xcode "
            "toolchain that provides xcrun metal and metallib."
        )
    endif()

    # Every shader compiles to its own .air; metallib links them into the single
    # library the runtime loads, so kernels can be split per logical op.
    set(HNW_METAL_SHADER_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/metal/filter/median_filter.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/metal/star_shrink/star_shrink_process.metal"
        "${CMAKE_CURRENT_SOURCE_DIR}/ops/metal/star_shrink/star_mask.metal"
    )
    set(HNW_METAL_LIBRARY
        "${HNW_PACKAGE_OUTPUT_DIR}/_metal_kernels.metallib"
    )
    set(HNW_METAL_AIR_FILES "")
    foreach(HNW_METAL_SHADER IN LISTS HNW_METAL_SHADER_SOURCES)
        get_filename_component(HNW_METAL_SHADER_NAME "${HNW_METAL_SHADER}" NAME_WE)
        set(HNW_METAL_AIR
            "${CMAKE_CURRENT_BINARY_DIR}/metal/${HNW_METAL_SHADER_NAME}.air"
        )
        add_custom_command(
            OUTPUT "${HNW_METAL_AIR}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory
                    "${CMAKE_CURRENT_BINARY_DIR}/metal"
            COMMAND "${HNW_XCRUN_EXECUTABLE}" -sdk macosx metal
                    -c "${HNW_METAL_SHADER}" -o "${HNW_METAL_AIR}"
            DEPENDS "${HNW_METAL_SHADER}"
            VERBATIM
        )
        list(APPEND HNW_METAL_AIR_FILES "${HNW_METAL_AIR}")
    endforeach()
    add_custom_command(
        OUTPUT "${HNW_METAL_LIBRARY}"
        COMMAND "${HNW_XCRUN_EXECUTABLE}" -sdk macosx metallib
                ${HNW_METAL_AIR_FILES} -o "${HNW_METAL_LIBRARY}"
        DEPENDS ${HNW_METAL_AIR_FILES}
        VERBATIM
    )
    add_custom_target(hnw_metal_shaders ALL DEPENDS "${HNW_METAL_LIBRARY}")

    pybind11_add_module(_metal MODULE
        NO_EXTRAS
        modules/_metal.mm
        common/metal_runtime.mm
        common/metal_host_io_workspace.mm
        ops/metal/filter/median_filter_ops.mm
        ops/metal/star_shrink/star_shrink_process_ops.mm
        ops/metal/star_shrink/star_shrink_dog_process_ops.mm
        ops/metal/star_shrink/star_mask_ops.mm
    )
    add_dependencies(_metal hnw_metal_shaders)
    target_include_directories(
        _metal
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}"
            ${HNW_PYBIND11_INCLUDE_DIRS}
    )
    target_compile_options(_metal PRIVATE -fobjc-arc)
    target_link_libraries(
        _metal
        PRIVATE
            pybind11::headers
            "${HNW_METAL_FRAMEWORK}"
            "${HNW_FOUNDATION_FRAMEWORK}"
            "${HNW_COREGRAPHICS_FRAMEWORK}"
    )
    set_target_properties(
        _metal
        PROPERTIES
            CXX_STANDARD 17
            CXX_STANDARD_REQUIRED ON
            OBJCXX_STANDARD 17
            OBJCXX_STANDARD_REQUIRED ON
            OUTPUT_NAME "_metal"
            LIBRARY_OUTPUT_DIRECTORY "${HNW_PACKAGE_OUTPUT_DIR}"
            RUNTIME_OUTPUT_DIRECTORY "${HNW_PACKAGE_OUTPUT_DIR}"
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/archive"
    )
endif()
