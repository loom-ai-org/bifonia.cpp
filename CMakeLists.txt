cmake_minimum_required(VERSION 3.16)

project(bifonia VERSION 0.3.7 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(_bifonia_top ON)
else()
    set(_bifonia_top OFF)
endif()
option(BIFONIA_BUILD_TESTS "Build the parity and unit tests" ${_bifonia_top})

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# The data files are installed here, and the library looks here unless
# set_data_dir() or BIFONIA_DATA_DIR names another directory.
set(BIFONIA_DATA_INSTALL_DIR "${CMAKE_INSTALL_DATADIR}/bifonia")
if(IS_ABSOLUTE "${BIFONIA_DATA_INSTALL_DIR}")
    set(_bifonia_default_data "${BIFONIA_DATA_INSTALL_DIR}")
else()
    set(_bifonia_default_data "${CMAKE_INSTALL_PREFIX}/${BIFONIA_DATA_INSTALL_DIR}")
endif()

add_library(bifonia
    src/unicode.cpp
    src/vocab.cpp
    src/data.cpp
    src/text.cpp
    src/scoring.cpp
    src/features.cpp
    src/model.cpp
    src/infopedia.cpp
    src/bifonia.cpp
)
add_library(bifonia::bifonia ALIAS bifonia)
target_include_directories(bifonia
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_compile_definitions(bifonia PRIVATE BIFONIA_DEFAULT_DATA_DIR="${_bifonia_default_data}")
set_target_properties(bifonia PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(bifonia PRIVATE -Wall -Wextra -Wpedantic)
    # Python rounds each product and each sum. A fused multiply-add rounds once,
    # so the model scores differ in the last bit on CPUs with FMA (aarch64,
    # x86-64-v3). The flag is on the target, so a parent project that builds
    # bifonia from source gets it too.
    target_compile_options(bifonia PRIVATE -ffp-contract=off)
endif()
# On x87 (-m32), GCC in GNU mode keeps intermediates at 80 bits. The standard
# mode rounds each assignment to double, as Python does. GCC supports it for
# C++ from version 13.
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-fexcess-precision=standard BIFONIA_HAS_EXCESS_PRECISION)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND BIFONIA_HAS_EXCESS_PRECISION)
    target_compile_options(bifonia PRIVATE -fexcess-precision=standard)
endif()

install(TARGETS bifonia EXPORT bifoniaTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY data/ DESTINATION ${BIFONIA_DATA_INSTALL_DIR})
install(EXPORT bifoniaTargets
    NAMESPACE bifonia::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bifonia)
configure_package_config_file(cmake/bifoniaConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/bifoniaConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bifonia)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/bifoniaConfigVersion.cmake
    COMPATIBILITY SameMinorVersion)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/bifoniaConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/bifoniaConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/bifonia)

if(BIFONIA_BUILD_TESTS)
    enable_testing()
    find_package(ZLIB REQUIRED)

    add_executable(bifonia_parity tests/parity.cpp)
    target_link_libraries(bifonia_parity PRIVATE bifonia ZLIB::ZLIB)
    add_test(NAME parity COMMAND bifonia_parity ${CMAKE_CURRENT_SOURCE_DIR}/tests/goldens)
    set_tests_properties(parity PROPERTIES ENVIRONMENT "BIFONIA_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")

    add_executable(bifonia_examples tests/examples.cpp)
    target_link_libraries(bifonia_examples PRIVATE bifonia)
    add_test(NAME examples COMMAND bifonia_examples)
    set_tests_properties(examples PROPERTIES ENVIRONMENT "BIFONIA_DATA_DIR=${CMAKE_CURRENT_SOURCE_DIR}/data")
endif()
