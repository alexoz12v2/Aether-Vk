file(GLOB_RECURSE AVK_WIN32_COMUTILS_PRIVATE_SOURCES CONFIRURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cxx"
  "${CMAKE_CURRENT_SOURCE_DIR}/private/*.cc"
)
file(GLOB_RECURSE AVK_WIN32_COMUTILS_PUBLIC_HEADERS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.h"
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.hh"
  "${CMAKE_CURRENT_SOURCE_DIR}/public/*.hpp"
)

add_library(avk-win-comutils SHARED)
add_library(avk::win-comutils ALIAS avk-win-comutils)
target_compile_definitions(avk-win-comutils
  PRIVATE 
    "AVK_OS_${AVK_OS}" "AVK_COMPILER_${AVK_COMPILER}" "AVK_ARCH_${AVK_ARCH}" "AVK_${AVK_BUILD_FRAGMENT}"
    "_UNICODE" "UNICODE" "WIN32_LEAN_AND_MEAN" "NOMINMAX"
    "AVK_WIN32_COMUTILS_EXPORT" # necessary for dllexport
)
target_compile_features(avk-win-comutils PRIVATE cxx_std_20)
target_sources(avk-win-comutils
  PUBLIC FILE_SET avk_win_comutils_headers TYPE HEADERS BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/public" FILES
    ${AVK_WIN32_COMUTILS_PUBLIC_HEADERS}
  PRIVATE
    ${AVK_WIN32_COMUTILS_PRIVATE_SOURCES}
)
# we don't care about the install interface, since this is not an installable library
target_include_directories(avk-win-comutils
  INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/public>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/public"
)
# no need to install the header set, since this is not an installable library
# install()
target_link_libraries(avk-win-comutils PUBLIC avk-core)
set_target_properties(avk-win-comutils PROPERTIES
  # name of the interface .lib
  ARCHIVE_OUTPUT_NAME avkcomutils
  # name of dll
  OUTPUT_NAME avkcomutils
)
