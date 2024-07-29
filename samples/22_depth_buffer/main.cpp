#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <fstream>

#include <array>
#include <chrono>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};

        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription,3> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription,3> attributeDescriptions{};
        
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex,pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex,color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex,texCoord);

        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f,-0.5f, 0.0f},{1.0f,0.0f,0.0f},{1.0f, 0.0f}},
    {{ 0.5f,-0.5f, 0.0f},{0.0f,1.0f,0.0f},{0.0f, 0.0f}},
    {{ 0.5f, 0.5f, 0.0f},{0.0f,0.0f,1.0f},{0.0f, 1.0f}},
    {{-0.5f, 0.5f, 0.0f},{1.0f,1.0f,1.0f},{1.0f, 1.0f}},

    {{-0.5f,-0.5f,-0.5f},{1.0f,0.0f,0.0f},{1.0f,0.0f}},
    {{ 0.5f,-0.5f,-0.5f},{0.0f,1.0f,0.0f},{0.0f,0.0f}},
    {{ 0.5f, 0.5f,-0.5f},{0.0f,0.0f,1.0f},{0.0f,1.0f}},
    {{-0.5f, 0.5f,-0.5f},{1.0f,1.0f,1.0f},{1.0f,1.0f}}
};

const std::vector<uint16_t> indices = {
    0,1,2,2,3,0,
    4,5,6,6,7,4
};

struct UniformBufferObject{
    glm::mat4 model;
    glm::mat4 view;
    glm::mat4 proj;
};

// Validation layers are optional components that hook into Vulkan function calls to apply additional operations
// Vulakn does not come with any validation layers. LunarG Vulkan SDK provides it.
// the LunarG validation layers are only available on PCs with the Vulkan SDK installed.
// instance layers would only check calls related to global Vulkan objects like instances
// device specific layers would only check calls related to a specific GPU.(have now been deprecated)
const std::vector<const char*> constGlobalValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> constGlobalDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete(){
        return graphicsFamily.has_value() && presentFamily.has_value();
    }
};

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance
                        ,const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo
                        ,VkAllocationCallbacks* pAllocator
                        ,VkDebugUtilsMessengerEXT* pDebugMessenger){
    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance,"vkCreateDebugUtilsMessengerEXT");
    if(func != nullptr){
        return func(instance,pCreateInfo,pAllocator,pDebugMessenger);
    }else{
        return VK_ERROR_EXTENSION_NOT_PRESENT;
    }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance
                    ,VkDebugUtilsMessengerEXT debugMessenger
                    ,const VkAllocationCallbacks* pAllocator){
    auto func = (PFN_vkDestroyDebugUtilsMessengerEXT) vkGetInstanceProcAddr(instance,"vkDestroyDebugUtilsMessengerEXT");
    if(func != nullptr){
        func(instance,debugMessenger,pAllocator);
    }
}


class HelloTriangle{
public:
    HelloTriangle(){}
    HelloTriangle(int w,int h) : mWidth(w),mHeight(h){}
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow* mWindow;
    int mWidth = 800, mHeight = 800;
    uint32_t mCurrentFrame = 0;
    VkInstance mInstance;
    VkDebugUtilsMessengerEXT mDebugMessenger;
    
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice;
    VkSurfaceKHR mSurface;
    VkSwapchainKHR mSwapChain;
    VkFormat mSwapChainImageFormat;
    VkExtent2D mSwapChainExtent;

    VkPipelineLayout mPipelineLayout;
    VkRenderPass mRenderPass;
    VkPipeline mGraphicsPipeline;

    VkCommandPool mCommandPool;

    std::vector<VkImage> mSwapChainImages;
    std::vector<VkImageView> mSwapChainImageViews;
    std::vector<VkFramebuffer> mSwapChainFrameBuffers;

    VkQueue mGraphicsQueue;
    VkQueue mPresentQueue;

    VkCommandBuffer mCommandBuffer;
    VkSemaphore mImageAvailableSemaphore;
    VkSemaphore mRenderFinishedSemaphore;
    VkFence mInFlightFence;

    bool bFramebufferResized = false;

    VkBuffer mVertexBuffer;
    VkDeviceMemory mVertexBufferMemory;
    VkBuffer mIndexBuffer;
    VkDeviceMemory mIndexBufferMemory;
    std::vector<VkBuffer> mUniformBuffers;
    std::vector<VkDeviceMemory> mUniformBuffersMemory;
    std::vector<void*> mUniformBuffersMapped;

    VkDescriptorSetLayout mDescriptorSetLayout;

    std::vector<VkCommandBuffer> mCommandBuffers;
    std::vector<VkSemaphore> mImageAvailableSemaphores;
    std::vector<VkSemaphore> mRenderFinishedSemaphores;
    std::vector<VkFence> mInFlightFences;

    VkImage mTextureImage;
    VkDeviceMemory mTextureImageMemory;
   
    VkImageView mTextureImageView;

    VkSampler mTextureSampler;

    VkDescriptorPool mDescriptorPool;
    std::vector<VkDescriptorSet> mDescriptorSets;

    VkImage mDepthImage;
    VkDeviceMemory mDepthImageMemory;
    VkImageView mDepthImageView;
  
    void mainLoop(){
        while(!glfwWindowShouldClose(mWindow)){
            glfwPollEvents();
            drawFrame();
        }

        vkDeviceWaitIdle(mDevice);
    }


    void cleanup(){
        cleanupSwapChain();

        vkDestroySampler(mDevice,mTextureSampler,nullptr);
        vkDestroyImageView(mDevice,mTextureImageView,nullptr);

        vkDestroyImage(mDevice,mTextureImage,nullptr);
        vkFreeMemory(mDevice,mTextureImageMemory,nullptr);

        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            vkDestroyBuffer(mDevice,mUniformBuffers[i],nullptr);
            vkFreeMemory(mDevice,mUniformBuffersMemory[i],nullptr);
        }

        vkDestroyDescriptorPool(mDevice,mDescriptorPool,nullptr);
        vkDestroyDescriptorSetLayout(mDevice,mDescriptorSetLayout,nullptr);

        vkDestroyBuffer(mDevice,mVertexBuffer,nullptr);
        vkFreeMemory(mDevice,mVertexBufferMemory,nullptr);

        vkDestroyBuffer(mDevice,mIndexBuffer,nullptr);
        vkFreeMemory(mDevice,mIndexBufferMemory,nullptr);

        vkDestroyPipeline(mDevice,mGraphicsPipeline,nullptr);
        vkDestroyPipelineLayout(mDevice,mPipelineLayout,nullptr);
        vkDestroyRenderPass(mDevice,mRenderPass,nullptr);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            vkDestroySemaphore(mDevice,mImageAvailableSemaphores[i],nullptr);
            vkDestroySemaphore(mDevice,mRenderFinishedSemaphores[i],nullptr);
            vkDestroyFence(mDevice,mInFlightFences[i],nullptr);
        }

        vkDestroyCommandPool(mDevice,mCommandPool,nullptr);

        vkDestroyDevice(mDevice, nullptr);
        if(enableValidationLayers){
            DestroyDebugUtilsMessengerEXT(mInstance,mDebugMessenger,nullptr);
        }
        vkDestroySurfaceKHR(mInstance,mSurface,nullptr);
        vkDestroyInstance(mInstance,nullptr);
        glfwDestroyWindow(mWindow);
        glfwTerminate();
    } 
    void initWindow(){
        glfwInit();
        //specifi window api, no use any opengl api
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);

        mWindow = glfwCreateWindow(mWidth,mHeight,"Vulkan",nullptr,nullptr);

        glfwSetWindowUserPointer(mWindow,this);
        glfwSetFramebufferSizeCallback(mWindow,framebufferResizeCallback);
    }

    static void framebufferResizeCallback(GLFWwindow* window,int widht,int height){
        auto app = reinterpret_cast<HelloTriangle*>(glfwGetWindowUserPointer(window));
        app->bFramebufferResized = true;
    }

    void initVulkan(){
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();    
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createFrameBuffers();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void createDepthResources(){
        VkFormat depthFormat = findDepthFormat();

        createImage(mSwapChainExtent.width,mSwapChainExtent.height,depthFormat,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                    ,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,mDepthImage,mDepthImageMemory);
        mDepthImageView = createImageView(mDepthImage,depthFormat,VK_IMAGE_ASPECT_DEPTH_BIT);
    }

    VkFormat findSupportedFormat(const std::vector<VkFormat> &candidates,VkImageTiling tiling,VkFormatFeatureFlags features){
        for(VkFormat format : candidates){
            VkFormatProperties props;
            vkGetPhysicalDeviceFormatProperties(mPhysicalDevice,format,&props);

            if(tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features){
                return format;
            } else if(tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features){
                return format;
            }
        }
        throw std::runtime_error("Failed to find support format!");
    }

    bool hasStencilComponent(VkFormat format){
        return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
    }

    VkFormat findDepthFormat(){
        return findSupportedFormat({VK_FORMAT_D32_SFLOAT,VK_FORMAT_D32_SFLOAT_S8_UINT,VK_FORMAT_D24_UNORM_S8_UINT},
                                    VK_IMAGE_TILING_OPTIMAL,VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
    }

    void createTextureSampler(){
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(mPhysicalDevice,&properties);

        VkSamplerCreateInfo samplerinfo{};
        samplerinfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
        samplerinfo.magFilter = VK_FILTER_LINEAR;
        samplerinfo.minFilter = VK_FILTER_LINEAR;
        samplerinfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerinfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerinfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
        samplerinfo.anisotropyEnable = VK_TRUE;
        samplerinfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
        samplerinfo.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
        samplerinfo.unnormalizedCoordinates = VK_FALSE;
        samplerinfo.compareEnable = VK_FALSE;
        samplerinfo.compareOp = VK_COMPARE_OP_ALWAYS;
        samplerinfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;

        if(vkCreateSampler(mDevice,&samplerinfo,nullptr,&mTextureSampler) != VK_SUCCESS){
            throw std::runtime_error("Failed to create texture sampler!");
        }
    }
    void createTextureImageView(){
        mTextureImageView = createImageView(mTextureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createImageViews(){
        mSwapChainImageViews.resize(mSwapChainImages.size());

        for(uint32_t i = 0; i < mSwapChainImages.size(); i++){
            mSwapChainImageViews[i] = createImageView(mSwapChainImages[i],mSwapChainImageFormat,VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    VkImageView createImageView(VkImage image,VkFormat format,VkImageAspectFlags aspectFlags){
        VkImageViewCreateInfo viewinfo{};
        viewinfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewinfo.image = image;
        viewinfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewinfo.format = format;
        viewinfo.subresourceRange.aspectMask = aspectFlags;
        viewinfo.subresourceRange.baseMipLevel = 0;
        viewinfo.subresourceRange.levelCount = 1;
        viewinfo.subresourceRange.baseArrayLayer = 0;
        viewinfo.subresourceRange.layerCount = 1;

        VkImageView imageView;
        if(vkCreateImageView(mDevice,&viewinfo,nullptr,&imageView) != VK_SUCCESS){
            throw std::runtime_error("Failed to create texture image view!");
        }
        return imageView;
    }
    void createTextureImage(){
        int texWidth,texHeight,texChannels;
        stbi_uc* pixels = stbi_load("assets/textures/texture.jpg",&texWidth,&texHeight,&texChannels,STBI_rgb_alpha);
        VkDeviceSize imageSize = texWidth * texHeight * 4;

        if(!pixels){
            throw std::runtime_error("Failed to load texture image!");
        }

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(imageSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        stagingBuffer,stagingBufferMemory);

        void* data;
        vkMapMemory(mDevice,stagingBufferMemory,0,imageSize,0,&data);
        memcpy(data,pixels,static_cast<size_t>(imageSize));
        vkUnmapMemory(mDevice,stagingBufferMemory);
        
        stbi_image_free(pixels);

        createImage(texWidth,texHeight,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_TILING_OPTIMAL,VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,mTextureImage,mTextureImageMemory);

        transitionImageLayout(mTextureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        copyBufferToImage(stagingBuffer,mTextureImage,static_cast<uint32_t>(texWidth),static_cast<uint32_t>(texHeight));
        transitionImageLayout(mTextureImage,VK_FORMAT_R8G8B8A8_SRGB,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        
        vkDestroyBuffer(mDevice,stagingBuffer,nullptr);
        vkFreeMemory(mDevice,stagingBufferMemory,nullptr);
    }

    void createImage(uint32_t width,uint32_t height,VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, 
                                        VkMemoryPropertyFlags properties,VkImage& image,VkDeviceMemory& imageMemory){
        VkImageCreateInfo imageinfo{};
        imageinfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
        imageinfo.imageType = VK_IMAGE_TYPE_2D;
        imageinfo.extent.width = width;
        imageinfo.extent.height = height;
        imageinfo.extent.depth = 1;
        imageinfo.mipLevels = 1;
        imageinfo.arrayLayers = 1;
        imageinfo.format = format;
        imageinfo.tiling = tiling;
        imageinfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        imageinfo.usage = usage;
        imageinfo.samples = VK_SAMPLE_COUNT_1_BIT;
        imageinfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if(vkCreateImage(mDevice,&imageinfo,nullptr,&image) != VK_SUCCESS){
            throw std::runtime_error("Failed to create image!");
        }

        VkMemoryRequirements memRequirements;
        vkGetImageMemoryRequirements(mDevice,image,&memRequirements);

        VkMemoryAllocateInfo allocinfo{};
        allocinfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocinfo.allocationSize = memRequirements.size;
        allocinfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,properties);

        if(vkAllocateMemory(mDevice,&allocinfo,nullptr,&imageMemory) != VK_SUCCESS){
            throw std::runtime_error("Failed to allocate image memory!");
        }

        vkBindImageMemory(mDevice,image,imageMemory,0);
    }

    void transitionImageLayout(VkImage image,VkFormat format,VkImageLayout oldLayout,VkImageLayout newLayout){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkImageMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkPipelineStageFlags sourcestage;
        VkPipelineStageFlags destinationStage;

        if(oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL){
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

            sourcestage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destinationStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL){
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

            sourcestage = VK_PIPELINE_STAGE_TRANSFER_BIT;
            destinationStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vkCmdPipelineBarrier(commandBuffer,sourcestage,destinationStage,0,0,nullptr,0,nullptr,1,&barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(VkBuffer buffer,VkImage image,uint32_t width,uint32_t height){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0,0,0};
        region.imageExtent = {width,height,1};

        vkCmdCopyBufferToImage(commandBuffer,buffer,image,VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,1,&region);

        endSingleTimeCommands(commandBuffer);
    }

    VkCommandBuffer beginSingleTimeCommands(){
        VkCommandBufferAllocateInfo allocinfo{};
        allocinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocinfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocinfo.commandPool = mCommandPool;
        allocinfo.commandBufferCount = 1;

        VkCommandBuffer commandBuffer;
        vkAllocateCommandBuffers(mDevice,&allocinfo,&commandBuffer);

        VkCommandBufferBeginInfo begininfo{};
        begininfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begininfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer,&begininfo);

        return commandBuffer;
    }

    void endSingleTimeCommands(VkCommandBuffer commandBuffer){
        vkEndCommandBuffer(commandBuffer);
        
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        vkQueueSubmit(mGraphicsQueue,1,&submitInfo,VK_NULL_HANDLE);
        vkQueueWaitIdle(mGraphicsQueue);

        vkFreeCommandBuffers(mDevice,mCommandPool,1,&commandBuffer);
    }

    void createDescriptorSets(){
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,mDescriptorSetLayout);

        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = mDescriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        mDescriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if(vkAllocateDescriptorSets(mDevice,&allocInfo,mDescriptorSets.data()) != VK_SUCCESS){
            throw std::runtime_error("Failed to allocate descriptor sets!");
        }

        for(size_t i = 0;i < MAX_FRAMES_IN_FLIGHT;i++){
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = mUniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = mTextureImageView;
            imageInfo.sampler = mTextureSampler;

            std::array<VkWriteDescriptorSet,2> descriptorWrites{};
            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = mDescriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = mDescriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;


            vkUpdateDescriptorSets(mDevice,static_cast<uint32_t>(descriptorWrites.size()),descriptorWrites.data(),0,nullptr);
        }
    }

    void createDescriptorPool(){
        std::array<VkDescriptorPoolSize,2> poolSizes{};
        poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        poolSizes[0].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        poolSizes[1].descriptorCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        
        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        poolInfo.maxSets = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);

        if(vkCreateDescriptorPool(mDevice,&poolInfo,nullptr,&mDescriptorPool) != VK_SUCCESS){
            throw std::runtime_error("Failed to create descriptor pool!");
        }
    }
    void createUniformBuffers(){
        VkDeviceSize bufferSize = sizeof(UniformBufferObject);

        mUniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        mUniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        mUniformBuffersMapped.resize(MAX_FRAMES_IN_FLIGHT);
        
        for(size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++){
            createBuffer(bufferSize,VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                mUniformBuffers[i],mUniformBuffersMemory[i]);
            vkMapMemory(mDevice,mUniformBuffersMemory[i],0,bufferSize,0,&mUniformBuffersMapped[i]);
        }
    }

    void createDescriptorSetLayout(){
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.pImmutableSamplers = nullptr;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding,2> bindings = {uboLayoutBinding,samplerLayoutBinding};
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        if(vkCreateDescriptorSetLayout(mDevice,&layoutInfo,nullptr,&mDescriptorSetLayout) != VK_SUCCESS){
            throw std::runtime_error("Failed to create descriptor set layout!");
        }
    }

    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkMemoryPropertyFlags properties, VkBuffer& buffer, 
            VkDeviceMemory &bufferMemory){
        
        VkBufferCreateInfo bufferinfo{};
        bufferinfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferinfo.size = size;
        bufferinfo.usage = usage;
        bufferinfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if(vkCreateBuffer(mDevice,&bufferinfo,nullptr,&buffer) != VK_SUCCESS){
            throw std::runtime_error("Failed to create buffer");
        }

        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(mDevice,buffer,&memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits,properties);

        if(vkAllocateMemory(mDevice,&allocInfo,nullptr,&bufferMemory) != VK_SUCCESS){
            throw std::runtime_error("Failed to allocate buffer memory!");
        }

        vkBindBufferMemory(mDevice,buffer,bufferMemory,0);
    }

    void createIndexBuffer(){
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer , stagingBufferMemory);
        
        void *data;
        vkMapMemory(mDevice,stagingBufferMemory,0,bufferSize,0,&data);
        memcpy(data,indices.data(),(size_t)bufferSize);
        vkUnmapMemory(mDevice,stagingBufferMemory);

        createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, mIndexBuffer,mIndexBufferMemory);
        copyBuffer(stagingBuffer,mIndexBuffer,bufferSize);
        vkDestroyBuffer(mDevice,stagingBuffer,nullptr);
        vkFreeMemory(mDevice,stagingBufferMemory,nullptr);
    }
    void createVertexBuffer(){
        VkDeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 stagingBuffer,stagingBufferMemory);

        void* data;
        vkMapMemory(mDevice,stagingBufferMemory,0,bufferSize,0,&data);
        memcpy(data,vertices.data(),(size_t)bufferSize);
        vkUnmapMemory(mDevice,stagingBufferMemory);

        createBuffer(bufferSize,VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT
                    ,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,mVertexBuffer,mVertexBufferMemory);
        
        copyBuffer(stagingBuffer,mVertexBuffer,bufferSize);

        vkDestroyBuffer(mDevice,stagingBuffer,nullptr);
        vkFreeMemory(mDevice,stagingBufferMemory,nullptr);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size){
        VkCommandBuffer commandBuffer = beginSingleTimeCommands();

        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer,srcBuffer,dstBuffer,1,&copyRegion);
        endSingleTimeCommands(commandBuffer);
    }

    void createVertexBuffer_OLD(){
                                 
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = sizeof(vertices[0]) * vertices.size();
        bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        if(vkCreateBuffer(mDevice,&bufferInfo,nullptr,&mVertexBuffer) != VK_SUCCESS){
            throw std::runtime_error("Failed to create vertex buffer!");
        }

        VkMemoryRequirements memoryRequirements;
        vkGetBufferMemoryRequirements(mDevice,mVertexBuffer,&memoryRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memoryRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits
                                                ,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                                |VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

        if(vkAllocateMemory(mDevice,&allocInfo,nullptr,&mVertexBufferMemory) != VK_SUCCESS){
            throw std::runtime_error("Failed to allocate vertex buffer memory!");
        }

        vkBindBufferMemory(mDevice,mVertexBuffer,mVertexBufferMemory,0);
        void* data;
        vkMapMemory(mDevice,mVertexBufferMemory,0,bufferInfo.size,0,&data);
        memcpy(data,vertices.data(),(size_t)bufferInfo.size);
        vkUnmapMemory(mDevice,mVertexBufferMemory);
    }
    
    uint32_t findMemoryType(uint32_t typeFilter,VkMemoryPropertyFlags properties){
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(mPhysicalDevice,&memProperties);

        for(uint32_t i = 0; i < memProperties.memoryTypeCount; i++){
            if((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties){
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

    void updateUniformBuffer(uint32_t currentImage){
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float,std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f),time*glm::radians(90.0f),glm::vec3(0.0f,0.0f,1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f,2.f,2.f),glm::vec3(.0f,.0f,.0f),glm::vec3(.0f,.0f,1.f));
        ubo.proj = glm::perspective(glm::radians(45.f),mSwapChainExtent.width/(float)mSwapChainExtent.height,0.1f,10.0f);
        ubo.proj[1][1] *= -1;
        memcpy(mUniformBuffersMapped[currentImage],&ubo,sizeof(ubo));
    }
    void drawFrame(){
        vkWaitForFences(mDevice,1,&mInFlightFences[mCurrentFrame],VK_TRUE,UINT64_MAX);

        uint32_t imageIndex;
        VkResult rst = vkAcquireNextImageKHR(mDevice,mSwapChain,UINT64_MAX,mImageAvailableSemaphores[mCurrentFrame],VK_NULL_HANDLE,&imageIndex);

        if(rst == VK_ERROR_OUT_OF_DATE_KHR){
            recreateSwapChain();
            return;
        } else if( rst != VK_SUCCESS && rst != VK_SUBOPTIMAL_KHR){
            throw std::runtime_error("Failed to acquire swap chain image!");
        }

        updateUniformBuffer(mCurrentFrame);

        vkResetFences(mDevice,1,&mInFlightFences[mCurrentFrame]);

        vkResetCommandBuffer(mCommandBuffers[mCurrentFrame],0);
        recordCommandBuffer(mCommandBuffers[mCurrentFrame],imageIndex);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {mImageAvailableSemaphores[mCurrentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;

        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &mCommandBuffers[mCurrentFrame];

        VkSemaphore signalSemaphores[] = {mRenderFinishedSemaphores[mCurrentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        if(vkQueueSubmit(mGraphicsQueue,1,&submitInfo,mInFlightFences[mCurrentFrame]) != VK_SUCCESS){
            throw std::runtime_error("Failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {mSwapChain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        rst = vkQueuePresentKHR(mPresentQueue,&presentInfo);

        if(rst == VK_ERROR_OUT_OF_DATE_KHR || rst == VK_SUBOPTIMAL_KHR || bFramebufferResized){
            bFramebufferResized = false;
            recreateSwapChain();
        } else if (rst != VK_SUCCESS){
            throw std::runtime_error("Failed to present swap chain image!");
        }

        mCurrentFrame = (mCurrentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }
    
    void createSyncObjects(){
        mImageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        mRenderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        mInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            if( vkCreateSemaphore(mDevice,&semaphoreInfo,nullptr,&mImageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(mDevice,&semaphoreInfo,nullptr,&mRenderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(mDevice,&fenceInfo,nullptr,&mInFlightFences[i]) != VK_SUCCESS){
                throw std::runtime_error("Failed to create semaphores!");
            }
        }
    }
    
    void createCommandBuffers(){
        mCommandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocinfo{};
        allocinfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocinfo.commandPool = mCommandPool;
        allocinfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocinfo.commandBufferCount = (uint32_t) mCommandBuffers.size();

        if(vkAllocateCommandBuffers(mDevice,&allocinfo,mCommandBuffers.data()) != VK_SUCCESS){
            throw std::runtime_error("Failed to allocate command buffers!");
        }
    }

    void recordCommandBuffer(VkCommandBuffer commandBuffer,uint32_t imageIndex){
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        if(vkBeginCommandBuffer(commandBuffer,&beginInfo) != VK_SUCCESS){
            throw std::runtime_error("failed to begin recording ocmmand buffer!");
        }

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = mRenderPass;
        renderPassInfo.framebuffer = mSwapChainFrameBuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0,0};
        renderPassInfo.renderArea.extent = mSwapChainExtent;

        std::array<VkClearValue,2> clearValues{};
        clearValues[0].color = {{0.0f,0.0f,0.0f,1.0f}};
        clearValues[1].depthStencil = {1.0f,0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer,&renderPassInfo,VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,mGraphicsPipeline);
 
        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = static_cast<float>(mSwapChainExtent.width);
        viewport.height = static_cast<float>(mSwapChainExtent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(commandBuffer,0,1,&viewport);

        VkRect2D scissor{};
        scissor.offset = {0,0};
        scissor.extent = mSwapChainExtent;
        vkCmdSetScissor(commandBuffer,0,1,&scissor);

        VkBuffer vertexBuffers[] = {mVertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffer,0,1,vertexBuffers,offsets);
        vkCmdBindIndexBuffer(commandBuffer,mIndexBuffer,0,VK_INDEX_TYPE_UINT16);
        
        vkCmdBindDescriptorSets(commandBuffer,VK_PIPELINE_BIND_POINT_GRAPHICS,mPipelineLayout,0,1,&mDescriptorSets[mCurrentFrame],0,nullptr);
        
        vkCmdDrawIndexed(commandBuffer,static_cast<uint32_t>(indices.size()),1,0,0,0);

        vkCmdEndRenderPass(commandBuffer);

        if(vkEndCommandBuffer(commandBuffer) != VK_SUCCESS){
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void createCommandPool(){
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(mPhysicalDevice);

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = queueFamilyIndices.graphicsFamily.value();

        if(vkCreateCommandPool(mDevice,&poolInfo,nullptr,&mCommandPool) != VK_SUCCESS){
            throw std::runtime_error("Failed to create command pool!");
        }
    }

    void createFrameBuffers(){
        mSwapChainFrameBuffers.resize(mSwapChainImageViews.size());

        for(size_t i = 0; i < mSwapChainImageViews.size(); i++){
            std::array<VkImageView,2> attachments = {mSwapChainImageViews[i],mDepthImageView};

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = mRenderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = mSwapChainExtent.width;
            framebufferInfo.height = mSwapChainExtent.height;
            framebufferInfo.layers = 1;

            if(vkCreateFramebuffer(mDevice,&framebufferInfo,nullptr,&mSwapChainFrameBuffers[i]) != VK_SUCCESS){
                throw std::runtime_error("Failed to create framebuffer!");
            }
        }
    }

    void createRenderPass(){
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = mSwapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = findDepthFormat();
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 0;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;

        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        std::array<VkAttachmentDescription,2> attachments = {colorAttachment,depthAttachment};
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = 1;
        renderPassInfo.pSubpasses = &subpass;
        renderPassInfo.dependencyCount = 1;
        renderPassInfo.pDependencies = &dependency;


        if(vkCreateRenderPass(mDevice,&renderPassInfo,nullptr,&mRenderPass) != VK_SUCCESS){
            throw std::runtime_error("failed to create render pass!");
        }
    }

    VkShaderModule createShaderModule(const std::vector<char>& code){
        VkShaderModuleCreateInfo createInfo{};

        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if(vkCreateShaderModule(mDevice,&createInfo,nullptr,&shaderModule) !=VK_SUCCESS){
            throw std::runtime_error("Failed to create shader module!");
        }
        return shaderModule;
    }

    void createGraphicsPipeline(){
        auto vertShaderCode = readFile("shaders/22vert.spv");
        auto fragShaderCode = readFile("shaders/22frag.spv");

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

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
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
        // rasterizer.depthBiasConstantFactor = 0.0f;
        // rasterizer.depthBiasClamp = 0.0f;
        // rasterizer.depthBiasSlopeFactor = 0.0f;

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f;
        multisampling.pSampleMask = nullptr;
        multisampling.alphaToCoverageEnable = VK_FALSE;
        multisampling.alphaToOneEnable = VK_FALSE;
        
        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
                                              VK_COLOR_COMPONENT_G_BIT |
                                              VK_COLOR_COMPONENT_B_BIT |
                                              VK_COLOR_COMPONENT_A_BIT;

        colorBlendAttachment.blendEnable = VK_FALSE;
        
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD;
        
        colorBlendAttachment.srcAlphaBlendFactor= VK_BLEND_FACTOR_ONE;
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD;

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

        std::vector<VkDynamicState> dynamicStates = {
            VK_DYNAMIC_STATE_VIEWPORT,
            VK_DYNAMIC_STATE_SCISSOR
        };
        VkPipelineDynamicStateCreateInfo dynamicState{};
        dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
        dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
        dynamicState.pDynamicStates = dynamicStates.data();

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &mDescriptorSetLayout;

        if(vkCreatePipelineLayout(mDevice,&pipelineLayoutInfo,nullptr,&mPipelineLayout) != VK_SUCCESS){
            throw std::runtime_error("Failed to create pipeline layout!");
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
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = &dynamicState;
        pipelineInfo.layout = mPipelineLayout;
        pipelineInfo.renderPass = mRenderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE;

        if(vkCreateGraphicsPipelines(mDevice,VK_NULL_HANDLE,1,&pipelineInfo,nullptr,&mGraphicsPipeline) != VK_SUCCESS){
            throw std::runtime_error("Failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(mDevice,fragShaderModule,nullptr);
        vkDestroyShaderModule(mDevice,vertShaderModule,nullptr);
    }

    void createSwapChain(){
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(mPhysicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount;
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
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(),indices.presentFamily.value()};

        if(indices.graphicsFamily != indices.presentFamily){
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

        // createinfo.oldSwapchain = VK_NULL_HANDLE;

        if(vkCreateSwapchainKHR(mDevice,&createinfo,nullptr,&mSwapChain) != VK_SUCCESS){
            throw std::runtime_error("failed to create swap chain");
        }

        vkGetSwapchainImagesKHR(mDevice,mSwapChain,&imageCount,nullptr);
        mSwapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(mDevice,mSwapChain,&imageCount,mSwapChainImages.data());

        mSwapChainImageFormat = surfaceFormat.format;
        mSwapChainExtent = extent;
    }

    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createinfo){
        createinfo = {};
        createinfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createinfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT 
                                    |VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                    |VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createinfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                |VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                |VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createinfo.pfnUserCallback = debugCallback;
        createinfo.pUserData = nullptr;
    }

    void setupDebugMessenger(){
        if(!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo;
        populateDebugMessengerCreateInfo(createInfo);

        if(CreateDebugUtilsMessengerEXT(mInstance,&createInfo,nullptr,&mDebugMessenger) != VK_SUCCESS){
            throw std::runtime_error("failed to set up debug messenger");
        } 
    }

    void createSurface(){
        if(glfwCreateWindowSurface(mInstance,mWindow,nullptr,&mSurface) != VK_SUCCESS){
            throw std::runtime_error("Failed to create window surface!");
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device){
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

    void printInstanceExtension(){
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr,&extensionCount,nullptr);

        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr,&extensionCount,extensions.data());

        std::cout << "available extensions:\n";
        for(const auto &e : extensions){
            std::cout << '\t' << e.extensionName << '\n';
        }
    }
    void createInstance(){
        if(enableValidationLayers && !checkValidationLayerSupport()){
            throw std::runtime_error("validation layers requested , but not available!");
        }

        // a struct with some infomation about our application
        VkApplicationInfo appinfo{};
        appinfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appinfo.pApplicationName = "Hello triangle";
        appinfo.applicationVersion = VK_MAKE_VERSION(1,3,0);
        appinfo.pEngineName = "No Engine";
        appinfo.engineVersion = VK_MAKE_VERSION(1,3,0);
        appinfo.apiVersion = VK_API_VERSION_1_3;

        // instance create info
        VkInstanceCreateInfo createinfo{};
        createinfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createinfo.pApplicationInfo = &appinfo;


        // get glfw instance extension
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // set extension info into create info structure
        auto extensions = getRequiredExtensions();
        createinfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createinfo.ppEnabledExtensionNames = extensions.data();
        
        
        VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};

        if(enableValidationLayers){
            createinfo.enabledLayerCount = static_cast<uint32_t>(constGlobalValidationLayers.size());
            createinfo.ppEnabledLayerNames = constGlobalValidationLayers.data();

            populateDebugMessengerCreateInfo(debugCreateInfo);
            createinfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT*)&debugCreateInfo;
        }else{
            createinfo.enabledLayerCount = 0;
            createinfo.pNext = nullptr;
        }

        if(vkCreateInstance(&createinfo,nullptr,&mInstance) != VK_SUCCESS){
            throw std::runtime_error("Failed to create instance");
        }
    }

    bool checkValidationLayerSupport(){
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount,nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount,availableLayers.data());

        for(const char* layerName : constGlobalValidationLayers){
            bool layerFound = false;

            for(const auto& layerProperties : availableLayers){
                if(strcmp(layerName , layerProperties.layerName) == 0){
                    layerFound = true;
                    break;
                }
            }

            if(!layerFound){
                return false;
            }
        }

        return true;
    }

    static std::vector<char> readFile(const std::string& filename){
        std::ifstream file(filename,std::ios::ate | std::ios::binary);

        if(!file.is_open()){
            throw std::runtime_error("Failed to open file: " + filename);
        }

        size_t filesize = (size_t)file.tellg();
        std::vector<char> buffer(filesize);
        file.seekg(0);
        file.read(buffer.data(),filesize);

        file.close();
        return buffer;
    }
    
    std::vector<const char*> getRequiredExtensions(){
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        std::vector<const char*> extensions(glfwExtensions,glfwExtensions + glfwExtensionCount);

        if(enableValidationLayers){
            extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        }
        return extensions;
    }

    void pickPhysicalDevice(){
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(mInstance,&deviceCount,nullptr);
        if(deviceCount == 0){
            throw std::runtime_error("Failed to find GPUs with Vulkan support");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(mInstance,&deviceCount,devices.data());

        for(const auto& device: devices){
            if(isDeviceSuitable(device)){
                mPhysicalDevice = device;
                break;
            }
        }

        if(mPhysicalDevice == VK_NULL_HANDLE){
            throw std::runtime_error("Failed to find a suitable GPU!");
        }
    }

    void createLogicalDevice(){
        QueueFamilyIndices indices = findQueueFamilies(mPhysicalDevice);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(),indices.presentFamily.value()};

        float queuePriority = 1.0;
        for(uint32_t queueFamily : uniqueQueueFamilies){
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        //Physical device
        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        //logical device
        VkDeviceCreateInfo createinfo{};
        createinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createinfo.pQueueCreateInfos = queueCreateInfos.data();
        createinfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createinfo.pEnabledFeatures = &deviceFeatures;
        createinfo.enabledExtensionCount = static_cast<uint32_t>(constGlobalDeviceExtensions.size());
        createinfo.ppEnabledExtensionNames = constGlobalDeviceExtensions.data();

        if(enableValidationLayers){
            createinfo.enabledLayerCount = static_cast<uint32_t>(constGlobalValidationLayers.size());
            createinfo.ppEnabledLayerNames = constGlobalValidationLayers.data();
        }else{
            createinfo.enabledLayerCount = 0;
        }

        if(vkCreateDevice(mPhysicalDevice,&createinfo,nullptr,&mDevice) != VK_SUCCESS){
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(mDevice,indices.graphicsFamily.value(),0,&mGraphicsQueue);
        vkGetDeviceQueue(mDevice,indices.presentFamily.value(),0,&mPresentQueue);
    }

    bool isDeviceSuitable(VkPhysicalDevice device){
        QueueFamilyIndices indices = findQueueFamilies(device);
        bool extensionSupported = checkDeviceExtensionSupport(device);

        bool swapChainAdequate = false;
        if(extensionSupported){
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device,&supportedFeatures);

        return indices.isComplete() && extensionSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats){

        for(const auto& availableFormat : availableFormats ){
            if(availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR){
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) {
        for(const auto& availablePresentMode : availablePresentModes){ 
            if(availablePresentMode == VK_PRESENT_MODE_MAILBOX_KHR){
                return availablePresentMode;
            }
        }
        return VK_PRESENT_MODE_FIFO_KHR;
    }
    
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities){
        if(capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()){
            return capabilities.currentExtent;
        }else{
            int width,height;
            glfwGetFramebufferSize(mWindow,&width,&height);

            VkExtent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };

            actualExtent.width = std::clamp(actualExtent.width,capabilities.minImageExtent.width,capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height,capabilities.minImageExtent.height,capabilities.maxImageExtent.height);

            return actualExtent;
        }
    }

    bool checkDeviceExtensionSupport(VkPhysicalDevice device){
        uint32_t extensionCount;

        vkEnumerateDeviceExtensionProperties(device,nullptr,&extensionCount,nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device,nullptr,&extensionCount,availableExtensions.data());

        std::set<std::string> requiredExtensions(constGlobalDeviceExtensions.begin(),constGlobalDeviceExtensions.end());

        for(const auto& extension : availableExtensions){
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device){
        QueueFamilyIndices indices;

        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,nullptr);
        
        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(device,&queueFamilyCount,queueFamilies.data());

        int i = 0;
        for(const auto& queueFamily : queueFamilies){
            if(queueFamily.queueFlags & VK_QUEUE_GRAPHICS_BIT){
                indices.graphicsFamily = i;
            }

            VkBool32 presentSupoort = false;
            vkGetPhysicalDeviceSurfaceSupportKHR(device,i,mSurface,&presentSupoort);

            if(presentSupoort){
                indices.presentFamily = i;
            }
            
            if(indices.isComplete()){
                break;
            }
            i++;
        }

        return indices;
    } 

    
    void recreateSwapChain(){
        int width = 0,height = 0;
        glfwGetFramebufferSize(mWindow,&width,&height);

        while(0 == width || 0 == height){
            glfwGetFramebufferSize(mWindow,&width,&height);
            glfwWaitEvents();
        }

        vkDeviceWaitIdle(mDevice);

        cleanupSwapChain();

        createSwapChain();
        createImageViews();
        createDepthResources();
        createFrameBuffers();
    } 

    void cleanupSwapChain(){
        vkDestroyImageView(mDevice,mDepthImageView,nullptr);
        vkDestroyImage(mDevice,mDepthImage,nullptr);
        vkFreeMemory(mDevice,mDepthImageMemory,nullptr);

        for (size_t i = 0; i < mSwapChainFrameBuffers.size(); i++){
            vkDestroyFramebuffer(mDevice,mSwapChainFrameBuffers[i],nullptr);
        }

        for(size_t i = 0; i < mSwapChainImageViews.size(); i++){
            vkDestroyImageView(mDevice,mSwapChainImageViews[i],nullptr);
        }

        vkDestroySwapchainKHR(mDevice,mSwapChain,nullptr);
    }

    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverty,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData) {
            std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;

            return VK_FALSE;
    }
};

int main(){
    HelloTriangle app(800,600);
    try{
        app.run();
    }catch (const std::exception& e){
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}