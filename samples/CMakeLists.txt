cmake_minimum_required(VERSION 3.17)

include_directories(
    ${GLEW_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIR}
    ${Vulkan_INCLUDE_DIRS}
    ${GLM_INCLUDE_DIR}
)

add_subdirectory(00_base_code)
add_subdirectory(01_instance)
add_subdirectory(02_validation_layers)
add_subdirectory(03_physical_device)
add_subdirectory(04_logical_device)
add_subdirectory(05_window_surface)
add_subdirectory(06_swap_chain)
add_subdirectory(07_image_views)
add_subdirectory(08_shader_module)
add_subdirectory(09_fixed_functions)
add_subdirectory(10_render_pass)
add_subdirectory(11_graphics_pipeline)
add_subdirectory(12_command_buffers)
add_subdirectory(13_rendering_and_presentation)
add_subdirectory(14_frames_in_flight)
add_subdirectory(15_swap_chain_recreation)
add_subdirectory(16_vertex_input)
add_subdirectory(17_staging_buffer)
add_subdirectory(18_index_buffer)
add_subdirectory(19_uniform_buffers)
add_subdirectory(20_texture_mapping)
add_subdirectory(21_combine_image_sampler)
add_subdirectory(22_depth_buffer)
add_subdirectory(23_loading_models)
add_subdirectory(25_multisampling)
add_subdirectory(compute_shader)