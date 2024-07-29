#include "ComputeShaderApplication.h"
constexpr int max_frames_in_flight = 2;
constexpr uint32_t PARTICLE_COUNT = 8192;

void ComputeShaderApplication::initWindow()
{
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);

    mWindow = glfwCreateWindow(mWidth,mHeight,"Vulkan - ComputeShader",nullptr,nullptr);
    glfwSetWindowUserPointer(mWindow,this);
    glfwSetFramebufferSizeCallback(mWindow,framebufferResizeCallback);
}

void ComputeShaderApplication::initVulkan()
{
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createRenderPass();
    createComputeDescriptorSetLayout();
    createGraphicsPipeline();
    createComputePipeline();
    createFramebuffers();
    createCommandPool();
    createShaderStorageBuffers();
    createUniformBuffers();
    createDescriptorPool();
    createComputeDescriptorSets();
    createCommandBuffers();
    createComputeCommandBuffers();
    createSyncObjects();
}

void ComputeShaderApplication::mainLoop()
{
    while(!glfwWindowShouldClose(mWindow)){
        glfwPollEvents();
        drawFrame();

        double currentTime = glfwGetTime();
        mLastFrameTime = (currentTime - mLastTime) * 1000.0;
        mLastTime = currentTime;
    }

    vkDeviceWaitIdle(mDevice);
}

void ComputeShaderApplication::cleanup(){
    cleanupSwapChain();
    vkDestroyPipeline(mDevice,mGraphicsPipeline,nullptr);
    vkDestroyPipelineLayout(mDevice,mPipelineLayout,nullptr);

    vkDestroyPipeline(mDevice,mComputePipeline,nullptr);
    vkDestroyPipelineLayout(mDevice,mComputePipelineLayout,nullptr);
    vkDestroyRenderPass(mDevice,mRenderPass,nullptr);
    vkDestroyDescriptorPool(mDevice,mDescriptorPool,nullptr);
    vkDestroyDescriptorSetLayout(mDevice,mComputeDescriptorSetLayout,nullptr);

    for(size_t i = 0; i < max_frames_in_flight; i++){
        
        vkDestroyBuffer(mDevice, mUniformBuffers[i], nullptr);
        vkFreeMemory(mDevice, mUniformBuffersMemory[i], nullptr);

        // destroy buffer memory
        vkDestroyBuffer(mDevice,mShaderStorageBuffers[i],nullptr);
        vkFreeMemory(mDevice,mShaderStorageBuffersMemory[i],nullptr);

        // destroy semaphore
        vkDestroySemaphore(mDevice,mRenderFinishedSemaphores[i],nullptr);
        vkDestroySemaphore(mDevice,mImageAvailableSemaphores[i],nullptr);
        vkDestroySemaphore(mDevice,mComputeFinishedSemaphores[i],nullptr);

        // destroy fence
        vkDestroyFence(mDevice,mInFlightFences[i],nullptr);
        vkDestroyFence(mDevice,mComputeInFlightFences[i],nullptr);
    }

    vkDestroyCommandPool(mDevice,mCommandPool,nullptr);
    vkDestroyDevice(mDevice,nullptr);

    if(DebugUtils::enableValidationLayers){
        DebugUtils::DestroyDebugUtilsMessegnerEXT(mInstance,mDebugMessenger,nullptr);
    }

    vkDestroySurfaceKHR(mInstance,mSurface,nullptr);
    vkDestroyInstance(mInstance,nullptr);
    glfwDestroyWindow(mWindow);
    glfwTerminate();
}

void ComputeShaderApplication::cleanupSwapChain()
{
    for(auto frameBuffer : mSwapChainFramebuffers){
        vkDestroyFramebuffer(mDevice,frameBuffer,nullptr);
    }

    for(auto imageView : mSwapChainImageViews){
        vkDestroyImageView(mDevice,imageView,nullptr);
    }

    vkDestroySwapchainKHR(mDevice,mSwapChain,nullptr);
}

void ComputeShaderApplication::recreateSwapChain(){
    int width = 0, height = 0;
    glfwGetFramebufferSize(mWindow,&width,&height);
    while(width == 0 || height == 0){
        glfwGetFramebufferSize(mWindow,&width,&height);
        glfwWaitEvents();
    }

    vkDeviceWaitIdle(mDevice);

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createFramebuffers();
}

void ComputeShaderApplication::createInstance()
{
    if(DebugUtils::enableValidationLayers && !DebugUtils::checkValidationLayerSupport()){
        throw std::runtime_error("validation layers requested , but not available!");
    }

    VkApplicationInfo appinfo{};
    appinfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appinfo.pApplicationName = "Vulkan Compute Shader";
    appinfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appinfo.pEngineName = "No Engine";
    appinfo.engineVersion = VK_MAKE_VERSION(1,0,0);
    appinfo.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo createinfo{};
    createinfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createinfo.pApplicationInfo = &appinfo;

    auto extensions = DebugUtils::getRequiredExtensions();
    createinfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createinfo.ppEnabledExtensionNames = extensions.data();

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    if(DebugUtils::enableValidationLayers){
        createinfo.enabledLayerCount = static_cast<uint32_t>(DebugUtils::validationLayers.size());
        createinfo.ppEnabledLayerNames = DebugUtils::validationLayers.data();

        DebugUtils::populateDebugMessengerCreateInfo(debugCreateInfo);
        createinfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
    } else {
        createinfo.enabledLayerCount = 0;
        createinfo.pNext = nullptr;
    }
    if(vkCreateInstance(&createinfo,nullptr,&mInstance) != VK_SUCCESS){
        throw std::runtime_error("Failed to create instance");
    }
}

void ComputeShaderApplication::setupDebugMessenger()
{
    if(!DebugUtils::enableValidationLayers) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    DebugUtils::populateDebugMessengerCreateInfo(createInfo);

    if(DebugUtils::CreateDebugUtilsMessengerEXT(mInstance,&createInfo,nullptr,&mDebugMessenger) != VK_SUCCESS){
        throw std::runtime_error("Failed to set up debug messenger!");
    }
}

void ComputeShaderApplication::createSurface()
{
    if(glfwCreateWindowSurface(mInstance,mWindow,nullptr,&mSurface) != VK_SUCCESS){
        throw std::runtime_error("failed to create windows surface!");
    }
}

void ComputeShaderApplication::pickPhysicalDevice()
{
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(mInstance,&deviceCount,nullptr);

    if(deviceCount == 0){
        throw std::runtime_error("failed to find GPUs with Vulkan support!");
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(mInstance,&deviceCount,devices.data());

    for(const auto& device : devices){
        if(isDeviceSuitable(device)){
            mPhysicalDevice = device;
            break;
        }
    }

    if(mPhysicalDevice == VK_NULL_HANDLE){
        throw std::runtime_error("Failed to find a suitable GPU");
    }
}

void ComputeShaderApplication::createLogicalDevice()
{
    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);

    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsAndComputeFamily.value(),indices.presentFamily.value()};

    float queuePriority = 1.0f;
    for(uint32_t queueFamily : uniqueQueueFamilies){
        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = queueFamily;
        queueCreateInfo.queueCount = 1;
        queueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(queueCreateInfo);
    }

    VkPhysicalDeviceFeatures deviceFeatures{};
    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    createInfo.pQueueCreateInfos = queueCreateInfos.data();

    createInfo.pEnabledFeatures = &deviceFeatures;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(DebugUtils::deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = DebugUtils::deviceExtensions.data();

    if(DebugUtils::enableValidationLayers){
        createInfo.enabledLayerCount = static_cast<uint32_t>(DebugUtils::validationLayers.size());
        createInfo.ppEnabledLayerNames = DebugUtils::validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    if(vkCreateDevice(mPhysicalDevice,&createInfo,nullptr,&mDevice) != VK_SUCCESS){
        throw std::runtime_error("Failed to create logical device!");
    }

    vkGetDeviceQueue(mDevice,indices.graphicsAndComputeFamily.value(),0,&mGraphicsQueue);
    vkGetDeviceQueue(mDevice, indices.graphicsAndComputeFamily.value(), 0, &mComputeQueue);
    vkGetDeviceQueue(mDevice, indices.presentFamily.value(), 0, &mPresentQueue);
}

void ComputeShaderApplication::createSwapChain()
{
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(mPhysicalDevice);

    VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if(swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount){
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createinfo{};
    createinfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createinfo.surface = mSurface;
    
    createinfo.minImageCount = imageCount;
    createinfo.imageFormat = surfaceFormat.format;
    createinfo.imageColorSpace = surfaceFormat.colorSpace;
    createinfo.imageExtent = extent;
    createinfo.imageArrayLayers = 1;
    createinfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsAndComputeFamily.value(),indices.presentFamily.value()};

    if(indices.graphicsAndComputeFamily != indices.presentFamily){
        createinfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        createinfo.queueFamilyIndexCount = 2;
        createinfo.pQueueFamilyIndices = queueFamilyIndices;
    } else {
        createinfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    createinfo.preTransform = swapChainSupport.capabilities.currentTransform;
    createinfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createinfo.presentMode = presentMode;
    createinfo.clipped = VK_TRUE;

    if(vkCreateSwapchainKHR(mDevice,&createinfo,nullptr,&mSwapChain) != VK_SUCCESS){
        throw std::runtime_error("failed to create swap chain!");
    }

    vkGetSwapchainImagesKHR(mDevice,mSwapChain,&imageCount,nullptr);
    mSwapChainImages.resize(imageCount);
    vkGetSwapchainImagesKHR(mDevice,mSwapChain,&imageCount,mSwapChainImages.data());

    mSwapChainImageFormat = surfaceFormat.format;
    mSwapChainExtent = extent;
}

void ComputeShaderApplication::createImageViews()
{
    mSwapChainImageViews.resize(mSwapChainImages.size());

    for(size_t i = 0; i < mSwapChainImages.size(); i++){
        VkImageViewCreateInfo createinfo{};
        createinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        createinfo.image = mSwapChainImages[i];
        createinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        createinfo.format = mSwapChainImageFormat;

        createinfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        createinfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        createinfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        createinfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        createinfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        createinfo.subresourceRange.baseMipLevel = 0;
        createinfo.subresourceRange.levelCount = 1;
        createinfo.subresourceRange.baseArrayLayer = 0;
        createinfo.subresourceRange.layerCount = 1;

        if(vkCreateImageView(mDevice,&createinfo,nullptr,&mSwapChainImageViews[i]) != VK_SUCCESS){
            throw std::runtime_error("Failed to create image views!");
        }
    }
}

void ComputeShaderApplication::createRenderPass()
{
    VkAttachmentDescription colorAttachment{};
    colorAttachment.format = mSwapChainImageFormat;
    colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    
    VkAttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    renderPassInfo.attachmentCount = 1;
    renderPassInfo.pAttachments = &colorAttachment;
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    if(vkCreateRenderPass(mDevice,&renderPassInfo,nullptr,&mRenderPass) != VK_SUCCESS){
        throw std::runtime_error("Failed to create render pass!");
    }

}

void ComputeShaderApplication::createComputeDescriptorSetLayout()
{
    std::array<VkDescriptorSetLayoutBinding,3> layoutBindings{};
    layoutBindings[0].binding = 0;
    layoutBindings[0].descriptorCount = 1;
    layoutBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    layoutBindings[0].pImmutableSamplers = nullptr;
    layoutBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[1].binding = 1;
    layoutBindings[1].descriptorCount = 1;
    layoutBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[1].pImmutableSamplers = nullptr;
    layoutBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    layoutBindings[2].binding = 2;
    layoutBindings[2].descriptorCount = 1;
    layoutBindings[2].descriptorType =VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    layoutBindings[2].pImmutableSamplers = nullptr;
    layoutBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutinfo{};
    layoutinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutinfo.bindingCount = 3;
    layoutinfo.pBindings = layoutBindings.data();

    if(vkCreateDescriptorSetLayout(mDevice,&layoutinfo,nullptr,&mComputeDescriptorSetLayout) != VK_SUCCESS){
        throw std::runtime_error("Failed to create compute descriptor set layout!");
    }
}

void ComputeShaderApplication::createGraphicsPipeline()
{
    auto vertShaderCode = Utils::readFile("shaders/csvert.spv");
    auto fragShaderCode = Utils::readFile("shaders/csfrag.spv");

    VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
    VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

    VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertShaderStageInfo.module = vertShaderModule;
    vertShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragShaderStageInfo.module = fragShaderModule;
    fragShaderStageInfo.pName = "main";

    VkPipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,fragShaderStageInfo};

    VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
    vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    auto bindingDescription = Particle::getBindingDescription();
    auto attributeDescriptions = Particle::getAttributeDescriptions();

    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    inputAssembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.lineWidth = 1.0f;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    colorBlendAttachment.blendEnable = VK_TRUE;
    colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
    colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;
    colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;

    VkPipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlending.logicOpEnable = VK_FALSE;
    colorBlending.logicOp = VK_LOGIC_OP_COPY;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;
    colorBlending.blendConstants[0] = 0.0f;
    colorBlending.blendConstants[1] = 0.0f;
    colorBlending.blendConstants[2] = 0.0f;
    colorBlending.blendConstants[3] = 0.0f;
    
    std::vector<VkDynamicState> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicStateCreateInfo{};
    dynamicStateCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicStateCreateInfo.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicStateCreateInfo.pDynamicStates = dynamicStates.data(); 

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.pSetLayouts = nullptr;
    pipelineLayoutInfo.setLayoutCount = 0;

    if (vkCreatePipelineLayout(mDevice, &pipelineLayoutInfo, nullptr, &mPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("failed to create pipeline layout!");
    }

    VkGraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipelineInfo.stageCount = 2;
    pipelineInfo.pStages = shaderStages;
    pipelineInfo.pVertexInputState = &vertexInputInfo;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewportState;
    pipelineInfo.pRasterizationState = &rasterizer;
    pipelineInfo.pMultisampleState = &multisampling;
    pipelineInfo.pColorBlendState = &colorBlending;
    pipelineInfo.pDynamicState = &dynamicStateCreateInfo;
    pipelineInfo.layout = mPipelineLayout;
    pipelineInfo.renderPass = mRenderPass;
    pipelineInfo.subpass = 0;
    pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

    if (vkCreateGraphicsPipelines(mDevice, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &mGraphicsPipeline) != VK_SUCCESS) {
        throw std::runtime_error("failed to create graphics pipeline!");
    }

    vkDestroyShaderModule(mDevice, fragShaderModule, nullptr);
    vkDestroyShaderModule(mDevice, vertShaderModule, nullptr);
}

void ComputeShaderApplication::createComputePipeline()
{
    auto computeShaderCode = Utils::readFile("shaders/cscomp.spv");

    VkShaderModule computeShaderModule = createShaderModule(computeShaderCode);

    VkPipelineShaderStageCreateInfo computeShaderStageInfo{};
    computeShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    computeShaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    computeShaderStageInfo.module = computeShaderModule;
    computeShaderStageInfo.pName = "main";

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &mComputeDescriptorSetLayout;

    if(vkCreatePipelineLayout(mDevice,&pipelineLayoutInfo,nullptr,&mComputePipelineLayout) != VK_SUCCESS){
        throw std::runtime_error("failed to create compute pipeline layout!");
    }

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.stage = computeShaderStageInfo;
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = mComputePipelineLayout;

    if(vkCreateComputePipelines(mDevice,VK_NULL_HANDLE,1,&pipelineInfo,nullptr,&mComputePipeline) != VK_SUCCESS){
        throw std::runtime_error("failed to create compute pipeline!");
    }

    vkDestroyShaderModule(mDevice,computeShaderModule,nullptr);
}

void ComputeShaderApplication::createFramebuffers()
{
    mSwapChainFramebuffers.resize(mSwapChainImageViews.size());

    for (size_t i = 0; i < mSwapChainImageViews.size(); i++) {
        VkImageView attachments[] = {mSwapChainImageViews[i]};

        VkFramebufferCreateInfo framebufferInfo{};
        framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        framebufferInfo.renderPass = mRenderPass;
        framebufferInfo.attachmentCount = 1;
        framebufferInfo.pAttachments = attachments;
        framebufferInfo.width = mSwapChainExtent.width;
        framebufferInfo.height = mSwapChainExtent.height;
        framebufferInfo.layers = 1;

        if (vkCreateFramebuffer(mDevice, &framebufferInfo, nullptr, &mSwapChainFramebuffers[i]) != VK_SUCCESS) {
            throw std::runtime_error("failed to create framebuffer!");
        }
    }
}

void ComputeShaderApplication::createCommandPool()
{
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(mPhysicalDevice);

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsAndComputeFamily.value();

    if (vkCreateCommandPool(mDevice, &poolInfo, nullptr, &mCommandPool) != VK_SUCCESS) {
       throw std::runtime_error("failed to create graphics command pool!");
    }
}

void ComputeShaderApplication::createComputeDescriptorSets()
{
    std::vector<VkDescriptorSetLayout> layouts(max_frames_in_flight,mComputeDescriptorSetLayout);
    VkDescriptorSetAllocateInfo allocinfo{};
    allocinfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocinfo.descriptorPool = mDescriptorPool;
    allocinfo.descriptorSetCount = static_cast<uint32_t>(max_frames_in_flight);
    allocinfo.pSetLayouts = layouts.data();

    mComputeDescriptorSets.resize(max_frames_in_flight);
    if(vkAllocateDescriptorSets(mDevice,&allocinfo,mComputeDescriptorSets.data())!=VK_SUCCESS){
        throw std::runtime_error("failed to allocate descriptor sets!");
    }

    for(size_t i = 0; i < max_frames_in_flight; i++){
        VkDescriptorBufferInfo uniformBufferInfo{};
        uniformBufferInfo.buffer = mUniformBuffers[i];
        uniformBufferInfo.offset = 0;
        uniformBufferInfo.range = sizeof(float);

        std::array<VkWriteDescriptorSet,3> descriptorWrites{};
        descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[0].dstSet = mComputeDescriptorSets[i];
        descriptorWrites[0].dstBinding = 0;
        descriptorWrites[0].dstArrayElement = 0;
        descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptorWrites[0].descriptorCount = 1;
        descriptorWrites[0].pBufferInfo = &uniformBufferInfo;

        VkDescriptorBufferInfo storageBufferInfoLastFrame{};
        storageBufferInfoLastFrame.buffer = mShaderStorageBuffers[(i - 1)%max_frames_in_flight];
        storageBufferInfoLastFrame.offset = 0;
        storageBufferInfoLastFrame.range = sizeof(Particle) * PARTICLE_COUNT;

        descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[1].dstSet = mComputeDescriptorSets[i];
        descriptorWrites[1].dstBinding = 1;
        descriptorWrites[1].dstArrayElement = 0;
        descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[1].descriptorCount = 1;
        descriptorWrites[1].pBufferInfo = &storageBufferInfoLastFrame;

        VkDescriptorBufferInfo storageBufferInfoCurrentFrame{};
        storageBufferInfoCurrentFrame.buffer = mShaderStorageBuffers[i];
        storageBufferInfoCurrentFrame.offset = 0;
        storageBufferInfoCurrentFrame.range = sizeof(Particle) * PARTICLE_COUNT;

        descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptorWrites[2].dstSet = mComputeDescriptorSets[i];
        descriptorWrites[2].dstBinding = 2;
        descriptorWrites[2].dstArrayElement = 0;
        descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptorWrites[2].descriptorCount = 1;
        descriptorWrites[2].pBufferInfo = &storageBufferInfoCurrentFrame;

        vkUpdateDescriptorSets(mDevice,3,descriptorWrites.data(),0,nullptr);
    }
}

void ComputeShaderApplication::createShaderStorageBuffers()
{
    std::default_random_engine rndEngine((unsigned)time(nullptr));
    std::uniform_real_distribution<float> rndDist(0.0,1.0f);

    std::vector<Particle> particles(PARTICLE_COUNT);
    for(auto& particle : particles){
        float r = 0.25f * sqrt(rndDist(rndEngine));
        float theta = rndDist(rndEngine) * 2.0f * 3.14159265358979323846;
        float x = r * cos(theta) * mHeight / mWidth;
        float y = r * sin(theta);
        particle.position = glm::vec2(x,y);
        particle.velocity = glm::normalize(glm::vec2(x,y)) * 0.00025f;
        particle.color = glm::vec4(rndDist(rndEngine),rndDist(rndEngine),rndDist(rndEngine),1.0f);
    }

    VkDeviceSize bufferSize = sizeof(Particle) * PARTICLE_COUNT;

    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;
    createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);

    void* data;
    vkMapMemory(mDevice,stagingBufferMemory,0,bufferSize,0,&data);
    memcpy(data,particles.data(),(size_t)bufferSize);
    vkUnmapMemory(mDevice,stagingBufferMemory);

    mShaderStorageBuffers.resize(max_frames_in_flight);
    mShaderStorageBuffersMemory.resize(max_frames_in_flight);

    for(size_t i = 0; i < max_frames_in_flight;i++){
        createBuffer(bufferSize,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,mShaderStorageBuffers[i],mShaderStorageBuffersMemory[i]);
        
        copyBuffer(stagingBuffer,mShaderStorageBuffers[i],bufferSize);
    }

    vkDestroyBuffer(mDevice,stagingBuffer,nullptr);
    vkFreeMemory(mDevice,stagingBufferMemory,nullptr);
}

void ComputeShaderApplication::createUniformBuffers()
{
    VkDeviceSize bufferSize = sizeof(float);

    mUniformBuffers.resize(max_frames_in_flight);
    mUniformBuffersMemory.resize(max_frames_in_flight);
    mUniformBuffersMapped.resize(max_frames_in_flight);

    for (size_t i = 0; i < max_frames_in_flight; i++) {
        createBuffer(bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, mUniformBuffers[i], mUniformBuffersMemory[i]);

        vkMapMemory(mDevice, mUniformBuffersMemory[i], 0, bufferSize, 0, &mUniformBuffersMapped[i]);
    }
}

void ComputeShaderApplication::createDescriptorPool()
{
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    poolSizes[0].descriptorCount = static_cast<uint32_t>(max_frames_in_flight);
        
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = static_cast<uint32_t>(max_frames_in_flight) * 2;

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = static_cast<uint32_t>(max_frames_in_flight);

    if (vkCreateDescriptorPool(mDevice, &poolInfo, nullptr, &mDescriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("failed to create descriptor pool!");
    }
}

void ComputeShaderApplication::createCommandBuffers()
{
    mCommandBuffers.resize(max_frames_in_flight);

    VkCommandBufferAllocateInfo allocinfo{};
    allocinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocinfo.commandPool = mCommandPool;
    allocinfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocinfo.commandBufferCount = (uint32_t)mCommandBuffers.size();

    if(vkAllocateCommandBuffers(mDevice,&allocinfo,mCommandBuffers.data()) != VK_SUCCESS){
        throw std::runtime_error("Failed to allocate command buffers!");
    }
}

void ComputeShaderApplication::createComputeCommandBuffers()
{
    mComputeCommandBuffers.resize(max_frames_in_flight);

    VkCommandBufferAllocateInfo allocinfo{};
    allocinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocinfo.commandPool = mCommandPool;
    allocinfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocinfo.commandBufferCount = (uint32_t)mComputeCommandBuffers.size();

    if(vkAllocateCommandBuffers(mDevice,&allocinfo,mComputeCommandBuffers.data())!=VK_SUCCESS){
        throw std::runtime_error("Failed to allocate compute command buffers!");
    }
}

void ComputeShaderApplication::createSyncObjects()
{
    mImageAvailableSemaphores.resize(max_frames_in_flight);
    mRenderFinishedSemaphores.resize(max_frames_in_flight);
    mComputeFinishedSemaphores.resize(max_frames_in_flight);
    mInFlightFences.resize(max_frames_in_flight);
    mComputeInFlightFences.resize(max_frames_in_flight);

    VkSemaphoreCreateInfo semaphoreInfo{};
    semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for(size_t i = 0; i < max_frames_in_flight; i++){
        if( vkCreateSemaphore(mDevice,&semaphoreInfo,nullptr,&mImageAvailableSemaphores[i]) != VK_SUCCESS ||
            vkCreateSemaphore(mDevice,&semaphoreInfo,nullptr,&mRenderFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(mDevice,&fenceInfo,nullptr,&mInFlightFences[i]) != VK_SUCCESS){
            throw std::runtime_error("failed to create graphics synchronization objects for a frame!");
        }

        if( vkCreateSemaphore(mDevice,&semaphoreInfo,nullptr,&mComputeFinishedSemaphores[i]) != VK_SUCCESS ||
            vkCreateFence(mDevice,&fenceInfo,nullptr,&mComputeInFlightFences[i]) != VK_SUCCESS){
            throw std::runtime_error("Failed to create compute synchronization objects for a frame!");
        }
    }
}

bool ComputeShaderApplication::isDeviceSuitable(VkPhysicalDevice device)
{
    QueueFamilyIndices indices = findQueueFamilies(device);

    bool extensionsSupported = checkDeviceExtensionSupport(device);

    bool swapChainAdequate = false;
    if(extensionsSupported){
        SwapChainSupportDetails swapChainSupportDetails = querySwapChainSupport(device);
        swapChainAdequate = !swapChainSupportDetails.formats.empty() && !swapChainSupportDetails.presentModes.empty();
    }

    return indices.isComplete() && extensionsSupported && swapChainAdequate;
}

QueueFamilyIndices ComputeShaderApplication::findQueueFamilies(VkPhysicalDevice device)
{
    QueueFamilyIndices indices;

    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,nullptr);

    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,queueFamilies.data());

    int i = 0;
    for(const auto& queueFamily : queueFamilies){
        if((queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT) && (queueFamily.queueFlags & VK_QUEUE_COMPUTE_BIT)){
            indices.graphicsAndComputeFamily = i;
        }

        VkBool32 presentSupport = false;
        vkGetPhysicalDeviceSurfaceSupportKHR(device,i,mSurface,&presentSupport);

        if(presentSupport){
            indices.presentFamily = i;
        }
        if(indices.isComplete()) break;
        i++;
    }

    return indices;
}

bool ComputeShaderApplication::checkDeviceExtensionSupport(VkPhysicalDevice device)
{
    uint32_t extensionCount;
    vkEnumerateDeviceExtensionProperties(device,nullptr,&extensionCount,nullptr);

    std::vector<VkExtensionProperties> availableExtensions(extensionCount);
    vkEnumerateDeviceExtensionProperties(device,nullptr,&extensionCount,availableExtensions.data()); 

    std::set<std::string> requiredExtensions(DebugUtils::deviceExtensions.begin(),DebugUtils::deviceExtensions.end());

    for(const auto& extension : availableExtensions){
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

SwapChainSupportDetails ComputeShaderApplication::querySwapChainSupport(VkPhysicalDevice device)
{
    SwapChainSupportDetails details;

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device,mSurface,&details.capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device,mSurface,&formatCount,nullptr);

    if(formatCount != 0){
        details.formats.resize(formatCount);
        vkGetPhysicalDeviceSurfaceFormatsKHR(device,mSurface,&formatCount,details.formats.data());
    }

    uint32_t presentModeCount;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device,mSurface,&presentModeCount,nullptr);

    if(presentModeCount != 0){
        details.presentModes.resize(presentModeCount);
        vkGetPhysicalDeviceSurfacePresentModesKHR(device,mSurface,&presentModeCount,details.presentModes.data());
    }

    return details;
}

VkSurfaceFormatKHR ComputeShaderApplication::chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats)
{
    for(const auto& availableFormat : availableFormats){
        if(availableFormat.format == VK_FORMAT_R8G8B8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
            return availableFormat;
        }
    }
    return availableFormats[0];
}

VkPresentModeKHR ComputeShaderApplication::chooseSwapPresentMode(const std::vector<VkPresentModeKHR> &avalilablePresentModes)
{
    for(const auto& avalilablePresentMode : avalilablePresentModes){
        if(avalilablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
            return avalilablePresentMode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

VkExtent2D ComputeShaderApplication::chooseSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities)
{
    if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()){
        return capabilities.currentExtent;
    } else {
        int width ,height;
        glfwGetFramebufferSize(mWindow,&width,&height);

        VkExtent2D actualExtent = {static_cast<uint32_t>(width),static_cast<uint32_t>(height)};

        actualExtent.width = std::clamp(actualExtent.width,capabilities.minImageExtent.width,capabilities.maxImageExtent.width);
        actualExtent.height= std::clamp(actualExtent.height,capabilities.minImageExtent.height,capabilities.maxImageExtent.height);
    }

    return VkExtent2D();
}

VkShaderModule ComputeShaderApplication::createShaderModule(const std::vector<char> &code)
{
    VkShaderModuleCreateInfo createinfo{};
    createinfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createinfo.codeSize = code.size();
    createinfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if(vkCreateShaderModule(mDevice,&createinfo,nullptr,&shaderModule) != VK_SUCCESS){
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

void ComputeShaderApplication::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer &buffer, VkDeviceMemory &bufferMemory)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if(vkCreateBuffer(mDevice,&bufferInfo,nullptr,&buffer) != VK_SUCCESS){
        throw std::runtime_error("Failed to create buffer!");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(mDevice,buffer,&memRequirements);

    VkMemoryAllocateInfo allocinfo{};
    allocinfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocinfo.allocationSize = memRequirements.size;
    allocinfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,properties);

    if(vkAllocateMemory(mDevice,&allocinfo,nullptr,&bufferMemory) != VK_SUCCESS){
        throw std::runtime_error("failed to allocate buffer memory!");
    }

    vkBindBufferMemory(mDevice,buffer,bufferMemory,0);
}

void ComputeShaderApplication::copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandPool = mCommandPool;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(mDevice, &allocInfo, &commandBuffer);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    VkBufferCopy copyRegion{};
    copyRegion.size = size;
    vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);

    vkEndCommandBuffer(commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;

    vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(mGraphicsQueue);

    vkFreeCommandBuffers(mDevice, mCommandPool, 1, &commandBuffer);
}

uint32_t ComputeShaderApplication::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type!");
}

void ComputeShaderApplication::recordComputeCommandBuffer(VkCommandBuffer commandBuffer){
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if(vkBeginCommandBuffer(commandBuffer,&beginInfo) != VK_SUCCESS){
        throw std::runtime_error("Failed to begin recording compute command buffer!");
    }
    vkCmdBindPipeline(commandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,mComputePipeline);
    vkCmdBindDescriptorSets(commandBuffer,VK_PIPELINE_BIND_POINT_COMPUTE,mComputePipelineLayout,0,1,&mComputeDescriptorSets[mCurrentFrame],0,nullptr);
    vkCmdDispatch(commandBuffer,PARTICLE_COUNT / 256, 1,1);

    if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS){
        throw std::runtime_error("Failed to record compute command buffer!");
    }
}
void ComputeShaderApplication::recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
        throw std::runtime_error("failed to begin recording command buffer!");
    }

    VkRenderPassBeginInfo renderPassInfo{};
    renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    renderPassInfo.renderPass = mRenderPass;
    renderPassInfo.framebuffer = mSwapChainFramebuffers[imageIndex];
    renderPassInfo.renderArea.offset = {0, 0};
    renderPassInfo.renderArea.extent = mSwapChainExtent;

    VkClearValue clearColor = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
    renderPassInfo.clearValueCount = 1;
    renderPassInfo.pClearValues = &clearColor;

    vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, mGraphicsPipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = (float) mSwapChainExtent.width;
    viewport.height = (float) mSwapChainExtent.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = mSwapChainExtent;
    vkCmdSetScissor(commandBuffer, 0, 1, &scissor);            

    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &mShaderStorageBuffers[mCurrentFrame], offsets);

    vkCmdDraw(commandBuffer, PARTICLE_COUNT, 1, 0, 0);

    vkCmdEndRenderPass(commandBuffer);

    if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
        throw std::runtime_error("failed to record command buffer!");
    }
}

void ComputeShaderApplication::drawFrame() {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    // Compute submission        
    vkWaitForFences(mDevice, 1, &mComputeInFlightFences[mCurrentFrame], VK_TRUE, UINT64_MAX);

    updateUniformBuffer(mCurrentFrame);

    vkResetFences(mDevice, 1, &mComputeInFlightFences[mCurrentFrame]);

    vkResetCommandBuffer(mComputeCommandBuffers[mCurrentFrame],  0);
    recordComputeCommandBuffer(mComputeCommandBuffers[mCurrentFrame]);

    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &mComputeCommandBuffers[mCurrentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &mComputeFinishedSemaphores[mCurrentFrame];

    if (vkQueueSubmit(mComputeQueue, 1, &submitInfo, mComputeInFlightFences[mCurrentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit compute command buffer!");
    };

    // Graphics submission
    vkWaitForFences(mDevice, 1, &mInFlightFences[mCurrentFrame], VK_TRUE, UINT64_MAX);

    uint32_t imageIndex;
    VkResult result = vkAcquireNextImageKHR(mDevice, mSwapChain, UINT64_MAX, mImageAvailableSemaphores[mCurrentFrame], VK_NULL_HANDLE, &imageIndex);

    if (result == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapChain();
        return;
    } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("failed to acquire swap chain image!");
    }

    vkResetFences(mDevice, 1, &mInFlightFences[mCurrentFrame]);

    vkResetCommandBuffer(mCommandBuffers[mCurrentFrame], /*VkCommandBufferResetFlagBits*/ 0);
    recordCommandBuffer(mCommandBuffers[mCurrentFrame], imageIndex);

    VkSemaphore waitSemaphores[] = { mComputeFinishedSemaphores[mCurrentFrame], mImageAvailableSemaphores[mCurrentFrame] };
    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_VERTEX_INPUT_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    submitInfo = {};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

    submitInfo.waitSemaphoreCount = 2;
    submitInfo.pWaitSemaphores = waitSemaphores;
    submitInfo.pWaitDstStageMask = waitStages;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &mCommandBuffers[mCurrentFrame];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &mRenderFinishedSemaphores[mCurrentFrame];

    if (vkQueueSubmit(mGraphicsQueue, 1, &submitInfo, mInFlightFences[mCurrentFrame]) != VK_SUCCESS) {
        throw std::runtime_error("failed to submit draw command buffer!");
    }

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &mRenderFinishedSemaphores[mCurrentFrame];

    VkSwapchainKHR swapChains[] = {mSwapChain};
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = swapChains;

    presentInfo.pImageIndices = &imageIndex;

    result = vkQueuePresentKHR(mPresentQueue, &presentInfo);

    if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || bFrameBufferResized) {
        bFrameBufferResized = false;
        recreateSwapChain();
    } else if (result != VK_SUCCESS) {
        throw std::runtime_error("failed to present swap chain image!");
    }

    mCurrentFrame = (mCurrentFrame + 1) % max_frames_in_flight;
}