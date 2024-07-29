#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

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
    VkInstance mInstance;
    VkDebugUtilsMessengerEXT mDebugMessenger;
    
    VkPhysicalDevice mPhysicalDevice = VK_NULL_HANDLE;
    VkDevice mDevice;
    VkSurfaceKHR mSurface;
    VkSwapchainKHR mSwapChain;
    VkFormat mSwapChainImageFormat;
    VkExtent2D mSwapChainExtent;

    std::vector<VkImage> mSwapChainImages;
    std::vector<VkImageView> mSwapChainImageViews;

    VkQueue mGraphicsQueue;
    VkQueue mPresentQueue;

    void initWindow(){
        glfwInit();
        //specifi window api, no use any opengl api
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        // create a not resizable window.
        glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);

        mWindow = glfwCreateWindow(mWidth,mHeight,"Vulkan",nullptr,nullptr);

    }

    void initVulkan(){
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();    
        createSwapChain();
        createImageViews();
        createGraphicsPipeline();
    }

    void createImageViews(){
        mSwapChainImageViews.resize(mSwapChainImages.size());
        for(size_t i = 0; i < mSwapChainImages.size();i++){
            VkImageViewCreateInfo createInfo{};
            createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
            createInfo.image = mSwapChainImages[i];
            createInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
            createInfo.format = mSwapChainImageFormat;
            createInfo.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
            createInfo.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

            createInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            createInfo.subresourceRange.baseMipLevel = 0;
            createInfo.subresourceRange.levelCount = 1;
            createInfo.subresourceRange.baseArrayLayer = 0;
            createInfo.subresourceRange.layerCount = 1;

            if(vkCreateImageView(mDevice,&createInfo,nullptr,&mSwapChainImageViews[i]) != VK_SUCCESS){
                throw std::runtime_error("Failed to create image views!");
            }
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
        auto vertShaderCode = readFile("shaders/08vert.spv");
        auto fragShaderCode = readFile("shaders/08frag.spv");

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

        createinfo.oldSwapchain = VK_NULL_HANDLE;

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

        return indices.isComplete() && extensionSupported && swapChainAdequate;
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

    void mainLoop(){
        while(!glfwWindowShouldClose(mWindow)){
            glfwPollEvents();
        }
    }
    void cleanup(){
        for(auto imageView : mSwapChainImageViews){
            vkDestroyImageView(mDevice,imageView,nullptr);
        }
        vkDestroySwapchainKHR(mDevice,mSwapChain,nullptr);
        vkDestroyDevice(mDevice, nullptr);
        if(enableValidationLayers){
            DestroyDebugUtilsMessengerEXT(mInstance,mDebugMessenger,nullptr);
        }
        vkDestroySurfaceKHR(mInstance,mSurface,nullptr);
        vkDestroyInstance(mInstance,nullptr);
        glfwDestroyWindow(mWindow);
        glfwTerminate();
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
    HelloTriangle app;
    try{
        app.run();
    }catch (const std::exception& e){
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}