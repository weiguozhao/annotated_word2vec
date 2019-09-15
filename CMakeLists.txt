cmake_minimum_required(VERSION 3.4)
project(word2vec)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11 -lm -pthread -march=native -Wall -funroll-loops -Wno-unused-result")

# output dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")

set(CMAKE_DISABLE_IN_SOURCE_BUILD ON)
set(CMAKE_DISABLE_SOURCE_CHANGES  ON)

if ("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
    message(SEND_ERROR "In-source builds are not allowed.")
endif ()

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_COLOR_MAKEFILE   ON)


option(FIX_SKIP_GRAM OFF)

if (FIX_SKIP_GRAM)
    message(STATUS "fix skip-gram on")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DFIX_SKIP_GRAM")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DFIX_SKIP_GRAM")
endif()


set(SOURCE_FILES
    compute-accuracy.c
    distance.c
    word-analogy.c
    word2phrase.c
    word2vec.c)

foreach(EXEC_SOURCE ${SOURCE_FILES})
    get_filename_component(EXEC_NAME ${EXEC_SOURCE} NAME_WE)
    message(STATUS ${EXEC_NAME})
    add_executable(${EXEC_NAME} ${EXEC_SOURCE})
endforeach(EXEC_SOURCE)
