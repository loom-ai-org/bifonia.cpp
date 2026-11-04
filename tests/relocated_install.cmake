# Installs bifonia into prefixes that are not the CMAKE_INSTALL_PREFIX of the
# build and runs tests/package_consumer with no BIFONIA_DATA_DIR. The consumer
# must find the data of its own prefix:
#   1. this build, the consumer installed beside it, run by absolute path and
#      by bare name through PATH;
#   2. a static build with a two-level CMAKE_INSTALL_LIBDIR, with the data of
#      another install in <prefix>/../share/bifonia;
#   3. a shared build with a two-level CMAKE_INSTALL_LIBDIR, the consumer
#      outside the prefix.
if(PREFIX STREQUAL INSTALL_PREFIX)
    message(FATAL_ERROR "PREFIX must differ from CMAKE_INSTALL_PREFIX: ${PREFIX}")
endif()
file(REMOVE_RECURSE "${PREFIX}" "${PREFIX}-consumer" "${PREFIX}-layouts")

function(run)
    execute_process(COMMAND ${ARGN} RESULT_VARIABLE result)
    if(NOT result EQUAL 0)
        list(JOIN ARGN " " command)
        message(FATAL_ERROR "${command}: ${result}")
    endif()
endfunction()

set(toolchain "-DCMAKE_CXX_COMPILER=${CXX_COMPILER}" "-DCMAKE_CXX_FLAGS=${CXX_FLAGS}"
    "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}")

# prefix, its libdir, consumer build dir, consumer install dir or "" to leave
# it in its build dir.
function(build_consumer prefix libdir build install)
    run("${CMAKE_COMMAND}" -S "${SOURCE_DIR}/tests/package_consumer" -B "${build}" -G "${GENERATOR}"
        "-Dbifonia_DIR=${prefix}/${libdir}/cmake/bifonia" "-DCMAKE_INSTALL_PREFIX=${install}"
        "-DCMAKE_INSTALL_BINDIR=${BINDIR}"
        ${toolchain})
    run("${CMAKE_COMMAND}" --build "${build}")
    if(install)
        run("${CMAKE_COMMAND}" --install "${build}")
    endif()
endfunction()

run("${CMAKE_COMMAND}" --install "${BUILD_DIR}" --prefix "${PREFIX}")
build_consumer("${PREFIX}" "${LIBDIR}" "${PREFIX}-consumer" "${PREFIX}")
run("${CMAKE_COMMAND}" -E env --unset=BIFONIA_DATA_DIR
    "${PREFIX}/${BINDIR}/consumer" "${PREFIX}/${DATADIR}/bifonia")
execute_process(COMMAND "${CMAKE_COMMAND}" -E env --unset=BIFONIA_DATA_DIR
                        "PATH=${PREFIX}/${BINDIR}:$ENV{PATH}" consumer "${PREFIX}/${DATADIR}/bifonia"
                WORKING_DIRECTORY "${PREFIX}/${DATADIR}" RESULT_VARIABLE result)
if(NOT result EQUAL 0)
    message(FATAL_ERROR "consumer through PATH: ${result}")
endif()

set(layouts "${PREFIX}-layouts")
set(libdir "lib/x86_64-linux-gnu")
foreach(shared OFF ON)
    set(build "${layouts}/build-${shared}")
    run("${CMAKE_COMMAND}" -S "${SOURCE_DIR}" -B "${build}" -G "${GENERATOR}" -DBIFONIA_BUILD_TESTS=OFF
        "-DBUILD_SHARED_LIBS=${shared}" "-DCMAKE_INSTALL_LIBDIR=${libdir}" ${toolchain})
    run("${CMAKE_COMMAND}" --build "${build}")
endforeach()

set(static "${layouts}/static/usr/local")
run("${CMAKE_COMMAND}" --install "${layouts}/build-OFF" --prefix "${static}")
file(COPY "${static}/${DATADIR}/bifonia" DESTINATION "${layouts}/static/usr/${DATADIR}")
build_consumer("${static}" "${libdir}" "${layouts}/consumer-static" "${static}")
run("${CMAKE_COMMAND}" -E env --unset=BIFONIA_DATA_DIR
    "${static}/${BINDIR}/consumer" "${static}/${DATADIR}/bifonia")

set(shared "${layouts}/shared")
run("${CMAKE_COMMAND}" --install "${layouts}/build-ON" --prefix "${shared}")
build_consumer("${shared}" "${libdir}" "${layouts}/consumer-shared" "")
run("${CMAKE_COMMAND}" -E env --unset=BIFONIA_DATA_DIR "LD_LIBRARY_PATH=${shared}/${libdir}"
    "DYLD_LIBRARY_PATH=${shared}/${libdir}" "${layouts}/consumer-shared/consumer" "${shared}/${DATADIR}/bifonia")
