project(sine CXX)

file(GLOB SINE_HEADERS "include/*.hpp")

file(GLOB SINE_SOURCES "src/*.cpp")

add_executable(sine ${SINE_SOURCES} ${SINE_HEADERS})

target_link_libraries(sine neuralnet)
