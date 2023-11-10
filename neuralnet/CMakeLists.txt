project(neuralnet CXX)

include_directories(include)

file(GLOB NEURALNET_HEADERS "include/*.hpp" "include/.[^.]*.hpp")

file(GLOB NEURALNET_SOURCES "src/*.cpp" "src/.[^.]*.cpp")

add_library(neuralnet SHARED ${NEURALNET_SOURCES} ${NEURALNET_HEADERS})
set_target_properties(neuralnet PROPERTIES LINKER_LANGUAGE CXX)
