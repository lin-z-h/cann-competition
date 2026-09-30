cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(batch_mat_mul_max_sum_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 14)

if(DEFINED NPU_ARCH)
    set(SOC_ARCH ${NPU_ARCH})
else()
    set(SOC_ARCH "dav-2201")
endif()

set(CANN_KERNEL "reliable" CACHE STRING "Kernel to build: reliable or new_design")
set_property(CACHE CANN_KERNEL PROPERTY STRINGS reliable new_design)
if(NOT CANN_KERNEL STREQUAL "reliable" AND NOT CANN_KERNEL STREQUAL "new_design")
    message(FATAL_ERROR "CANN_KERNEL must be reliable or new_design")
endif()

add_executable(batch_mat_mul_max_sum_custom main.asc)

target_link_libraries(batch_mat_mul_max_sum_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(batch_mat_mul_max_sum_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

set(CANN_KERNEL_FILE "" CACHE FILEPATH "Explicit experimental kernel source")
option(CANNLAB_TRACE "Print experimental host tiling and partial results" OFF)
set(CANNLAB_MAX_CALL_TILES "4" CACHE STRING "Experimental MatMul call tile limit")
set(CANNLAB_BASE_M "0" CACHE STRING "Experimental baseM override; 0 keeps the original selector")
option(CANNLAB_ALIGNED_FAST_PATH "Use validated consumer for aligned FP16 M stripes" ON)
if(CANN_KERNEL_FILE)
    get_filename_component(CANN_KERNEL_FILE "${CANN_KERNEL_FILE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    if(NOT EXISTS "${CANN_KERNEL_FILE}")
        message(FATAL_ERROR "CANN_KERNEL_FILE does not exist: ${CANN_KERNEL_FILE}")
    endif()
    target_compile_definitions(batch_mat_mul_max_sum_custom PRIVATE CANNLAB_KERNEL_FILE="${CANN_KERNEL_FILE}")
elseif(CANN_KERNEL STREQUAL "new_design")
    target_compile_definitions(batch_mat_mul_max_sum_custom PRIVATE CANNLAB_USE_NEW_DESIGN=1)
endif()
if(CANNLAB_TRACE)
    target_compile_definitions(batch_mat_mul_max_sum_custom PRIVATE CANNLAB_TRACE=1)
endif()
if(CANN_KERNEL_FILE)
    target_compile_definitions(batch_mat_mul_max_sum_custom PRIVATE
        CANNLAB_MAX_CALL_TILES=${CANNLAB_MAX_CALL_TILES}
        CANNLAB_BASE_M=${CANNLAB_BASE_M}
        CANNLAB_ALIGNED_FAST_PATH=$<BOOL:${CANNLAB_ALIGNED_FAST_PATH}>)
endif()

target_compile_options(batch_mat_mul_max_sum_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${SOC_ARCH}>
)
