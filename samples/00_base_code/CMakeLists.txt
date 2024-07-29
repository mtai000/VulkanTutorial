cmake_minimum_required(VERSION 3.17)

project("00_base_code")

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${GLFW_LIBRARIES} ${GLEW_LIBRARIES} ${VULKAN_LIBRARY})