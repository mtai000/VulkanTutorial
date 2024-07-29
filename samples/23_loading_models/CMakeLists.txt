cmake_minimum_required(VERSION 3.17)
project("23_loading_models")

include_directories(${LOCAL_3RDPARTY}/stb_image)
include_directories(${LOCAL_3RDPARTY}/tinyobjloader)

add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL
    ${GLFW_LIBRARIES} 
    ${GLEW_LIBRARIES} 
    ${VULKAN_LIBRARY})
