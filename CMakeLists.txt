cmake_minimum_required (VERSION 3.17)

project ("VulkanTutorial")

set(LOCAL_3RDPARTY "${CMAKE_CURRENT_SOURCE_DIR}/3rdparty")
set(CMAKE_MODULE_PATH "${LOCAL_3RDPARTY}/CMake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")


if(WIN32 AND "${CMAKE_GENERATOR}" MATCHES "^(Visual Studio).*")
  set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$(ConfigurationName)")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
else()
  set(MODULE_TARGET_DIR "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
endif()

find_package(OpenGL REQUIRED)
find_package(GLFW REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(Vulkan)
message(STATUS "Try to found vulkan")

if(VULKAN_FOUND)
    if(NOT Vulkan_dxc_EXECUTABLE)
        find_program(Vulkan_dxc_EXECUTABLE
                    NAMES dxc
                    HINTS
                    "$ENV{VULKAN_SDK}/Bin"
                    "$ENV{VULKAN_SDK}/bin"
                    )
    endif()
    if(DEFINED Vulkan_dxc_EXECUTABLE)
        message("Found DirectX Shader Compiler under ${Vulkan_dxc_EXECUTABLE}")
    elseif()
        message("Couldn't find DirectX Shader Compiler executable, make sure it is present in Vulkan SDK or add it manually via Vulkan_dxc_EXECUTABLE env variable")
    endif()
else()
    message("VULKAN_SDK path not found, HLSL shaders won't be compiled")
endif()

set(Vulkan_INCLUDE_DIRS "$ENV{VULKAN_SDK}/Include")
if (CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(VULKAN_LIBRARY "$ENV{VULKAN_SDK}/Lib/vulkan-1.lib")
    set(VULKAN_STATIC_LIBRARY "$ENV{VULKAN_SDK}/Lib/vkstatic.1.lib")
else()
    set(VULKAN_LIBRARY "$ENV{VULKAN_SDK}/Lib32/vulkan-1.lib")
    set(VULKAN_STATIC_LIBRARY "$ENV{VULKAN_SDK}/Lib32/vkstatic.1.lib")
endif()

list(APPEND LIBS ${VULKAN_LIBRARY})
message(STATUS "Vulkan Include dirs: ${Vulkan_INCLUDE_DIRS} and libs: ${VULKAN_LIBRARY}")

add_subdirectory(samples)
