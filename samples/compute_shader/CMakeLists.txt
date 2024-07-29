cmake_minimum_required(VERSION 3.17)
project("compute_shader")

include_directories(${LOCAL_3RDPARTY}/stb_image)
include_directories(${LOCAL_3RDPARTY}/tinyobjloader)

include_directories(./inc)

set(INC 
    inc/DebugUtils.h
    inc/Utils.h
    inc/ComputeShaderApplication.h
)

set(SRC 
    src/DebugUtils.cpp
    src/Utils.cpp
    src/ComputeShaderApplication.cpp
)

add_executable(${PROJECT_NAME} main.cpp ${INC} ${SRC})

target_link_libraries(${PROJECT_NAME} 
    OpenGL::GL
    ${GLFW_LIBRARIES} 
    ${GLEW_LIBRARIES} 
    ${VULKAN_LIBRARY})
