# Installs the build into PREFIX, which is not the CMAKE_INSTALL_PREFIX of the
# build, and installs tests/package_consumer there. The consumer runs with no
# BIFONIA_DATA_DIR and must find the data under PREFIX.
if(PREFIX STREQUAL INSTALL_PREFIX)
    message(FATAL_ERROR "PREFIX must differ from CMAKE_INSTALL_PREFIX: ${PREFIX}")
endif()
file(REMOVE_RECURSE "${PREFIX}" "${PREFIX}-consumer")

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        list(JOIN ARGN " " command)
        message(FATAL_ERROR "${command}: ${result}")
    endif()
endfunction()

run("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${PREFIX}")
run("${CMAKE_COMMAND}" -S "${SOURCE_DIR}/tests/package_consumer" -B "${PREFIX}-consumer"
    "-DCMAKE_PREFIX_PATH=${PREFIX}" "-DCMAKE_INSTALL_PREFIX=${PREFIX}" "-DCMAKE_INSTALL_BINDIR=${BINDIR}"
    "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_CXX_FLAGS=${CXX_FLAGS}" "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")
run("${CMAKE_COMMAND}" --build "${PREFIX}-consumer")
run("${CMAKE_COMMAND}" --install "${PREFIX}-consumer")
run("${CMAKE_COMMAND}" -E env --unset=BIFONIA_DATA_DIR
    "${PREFIX}/${BINDIR}/consumer" "${PREFIX}/${DATADIR}/bifonia")
