cmake_minimum_required(VERSION 3.17)
project("05_window_surface")

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL
    ${GLFW_LIBRARIES} 
    ${GLEW_LIBRARIES} 
    ${VULKAN_LIBRARY})