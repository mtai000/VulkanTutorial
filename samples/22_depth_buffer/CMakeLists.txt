cmake_minimum_required(VERSION 3.17)
project("22_depth_buffer")

include_directories(${LOCAL_3RDPARTY}/stb_image)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL
    ${GLFW_LIBRARIES} 
    ${GLEW_LIBRARIES} 
    ${VULKAN_LIBRARY})
