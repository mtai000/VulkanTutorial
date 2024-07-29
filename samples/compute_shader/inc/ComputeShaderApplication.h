#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "DebugUtils.h"
#include "Utils.h"
#include <optional>
#include <string>
#include <set>
#include <algorithm>
#include <array>
#include <random>

struct Particle{
    glm::vec2 position;
    glm::vec2 velocity;
    glm::vec4 color;

    static VkVertexInputBindingDescription getBindingDescription(){
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Particle);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription,2> getAttributeDescriptions(){
        std::array<VkVertexInputAttributeDescription,2> attributeDescriptions{};

        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Particle,position);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Particle,color);

        return attributeDescriptions;
    }
};

struct QueueFamilyIndices{
    std::optional<uint32_t> graphicsAndComputeFamily;
    std::optional<uint32_t> presentFamily;

    bool isComplete(){
        return graphicsAndComputeFamily.has_value() && presentFamily.has_value();
    }
};

struct SwapChainSupportDetails{
    VkSurfaceCapabilitiesKHR capabilities;
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

class ComputeShaderApplication{
public:
    ComputeShaderApplication(){}
    ComputeShaderApplication(uint32_t w,uint32_t h):mWidth(w),mHeight(h) {}
    ~ComputeShaderApplication(){}
    void run(){
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    GLFWwindow* mWindow;
    uint32_t mWidth = 800,mHeight = 600;

    VkInstance mInstance;
    VkSurfaceKHR mSurface;

    VkDebugUtilsMessengerEXT mDebugMessenger;

    // device
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice;

    // swapChain
    VkSwapchainKHR mSwapChain;
    std::vector<VkImage> mSwapChainImages;
    VkFormat mSwapChainImageFormat;
    VkExtent2D mSwapChainExtent;
    std::vector<VkImageView> mSwapChainImageViews;
    std::vector<VkFramebuffer> mSwapChainFramebuffers;

    // RenderPass
    VkRenderPass mRenderPass;
    VkPipelineLayout mPipelineLayout;
    VkPipeline mGraphicsPipeline;

    // Pipeline
    VkDescriptorSetLayout mComputeDescriptorSetLayout;
    VkPipelineLayout mComputePipelineLayout;
    VkPipeline mComputePipeline;

    // Command Pool
    VkCommandPool mCommandPool;

    // shader storage buffer
    std::vector<VkBuffer> mShaderStorageBuffers;
    std::vector<VkDeviceMemory> mShaderStorageBuffersMemory;

    // framebuffers
    std::vector<VkBuffer> mUniformBuffers;
    std::vector<VkDeviceMemory> mUniformBuffersMemory;
    std::vector<void*> mUniformBuffersMapped;

    // descriptor
    VkDescriptorPool mDescriptorPool;
    std::vector<VkDescriptorSet> mComputeDescriptorSets;

    // command buffers
    std::vector<VkCommandBuffer> mCommandBuffers;
    std::vector<VkCommandBuffer> mComputeCommandBuffers;

    // Semaphore
    std::vector<VkSemaphore> mImageAvailableSemaphores;
    std::vector<VkSemaphore> mRenderFinishedSemaphores;
    std::vector<VkSemaphore> mComputeFinishedSemaphores;

    // Queue
    VkQueue mGraphicsQueue;
    VkQueue mComputeQueue;
    VkQueue mPresentQueue;

    // Fence
    std::vector<VkFence> mInFlightFences;
    std::vector<VkFence> mComputeInFlightFences;

    uint32_t mCurrentFrame = 0;
    float mLastFrameTime = 0.0f;
    double mLastTime = 0.0;

    bool bFrameBufferResized = false;

    void initWindow();
    void initVulkan();
    void mainLoop();
    void cleanup();
    void cleanupSwapChain();
    void recreateSwapChain();

    void createInstance();
    void setupDebugMessenger();
    
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();

    void createSwapChain();
    void createImageViews();
    void createRenderPass();

    void createComputeDescriptorSetLayout();
    void createGraphicsPipeline();
    void createComputePipeline();

    void createFramebuffers();
    void createCommandPool();

    void createShaderStorageBuffers();
    void createUniformBuffers();
    void createDescriptorPool();

    void createComputeDescriptorSets();
    void createCommandBuffers();
    void createComputeCommandBuffers();
    void createSyncObjects();

    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device); 
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& avalilablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createBuffer(VkDeviceSize size,VkBufferUsageFlags usage,VkMemoryPropertyFlags properties,VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);

    void drawFrame();
    void updateUniformBuffer(uint32_t currentImage){
        float deltaTime = mLastFrameTime * 2.0f;
        memcpy(mUniformBuffersMapped[currentImage],&deltaTime,sizeof(float));
    }
    void recordComputeCommandBuffer(VkCommandBuffer commandBuffer);
    void recordCommandBuffer(VkCommandBuffer commandBuffer, uint32_t imageIndex);

    static void framebufferResizeCallback(GLFWwindow* window,int width,int height){
        auto app = reinterpret_cast<ComputeShaderApplication*>(glfwGetWindowUserPointer(window));
        app->bFrameBufferResized = true;
    }

};