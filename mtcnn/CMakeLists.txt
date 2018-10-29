project(mtcnn)
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)

if(UNIX AND NOT APPLE)
    set(LINUX TRUE)
endif()
if (LINUX)
	SET(LDFLAGS "-std=c++11 -msse4.1 -fPIC -O3 -ggdb -Wall -finline-functions -Wl,--allow-multiple-definition -Wl,--whole-archive")
else()
	SET(LDFLAGS "-std=c++11 -msse4.1 -fPIC -O3 -ggdb -Wall -finline-functions -undefined dynamic_lookup -all_load")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${LDFLAGS}")

message(STATUS "CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

set(TF_DIR /home/lj/tool/tensorflow)
set(TF_MAKEFILE_DIR ${TF_DIR}/tensorflow/contrib/makefile)

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(SRC_DIR ${CMAKE_SOURCE_DIR}/src)

include_directories(${TF_DIR} 
	${TF_MAKEFILE_DIR}/gen/proto 
	${TF_MAKEFILE_DIR}/gen/protobuf-host/include 
	${TF_MAKEFILE_DIR}/downloads/eigen
	${TF_MAKEFILE_DIR}/downloads/nsync/public)

link_directories(${TF_MAKEFILE_DIR}/gen/lib 
	${TF_MAKEFILE_DIR}/gen/protobuf-host/lib
	${TF_MAKEFILE_DIR}/downloads/nsync/builds/default.linux.c++11)

set(TF_LIBS tensorflow-core protobuf nsync pthread m z dl)

include_directories(${INCLUDE_DIR})


file(GLOB INC ${CMAKE_CURRENT_LIST_DIR}/*.hpp ${CMAKE_CURRENT_LIST_DIR}/*.h ${INCLUDE_DIR}/*.h ${INCLUDE_DIR}/*.hpp)
file(GLOB SRC ${CMAKE_CURRENT_LIST_DIR}/*.cpp ${CMAKE_CURRENT_LIST_DIR}/*.c ${SRC_DIR}/*.cpp ${SRC_DIR}/*.c)

add_executable(mtcnn_detector ${INC} ${SRC})
target_link_libraries(mtcnn_detector ${TF_LIBS} ${OpenCV_LIBS})
