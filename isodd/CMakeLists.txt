project(isodd CXX)

file(GLOB ISODD_HEADERS "include/*.hpp")

file(GLOB ISODD_SOURCES "src/*.cpp")

add_executable(isodd ${ISODD_SOURCES} ${ISODD_HEADERS})

target_link_libraries(isodd neuralnet)
