cmake_minimum_required(VERSION 3.4...3.18)
project(HDR2LDR)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED system)
find_package(pybind11 REQUIRED)

execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "NUMPY_INCLUDE: " ${NUMPY_INCLUDE})
include_directories(${NUMPY_INCLUDE})
include_directories(${Boost_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})


pybind11_add_module(random_tone_map random_tone_map.cpp)

target_include_directories(random_tone_map PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIR}
)

target_link_libraries(random_tone_map PRIVATE
    ${Boost_LIBRARIES}
    ${OpenCV_LIBS}
)

target_compile_definitions(random_tone_map PRIVATE)