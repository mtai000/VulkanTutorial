#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>

// Validation layers are optional components that hook into Vulkan function calls to apply additional operations
// Vulakn does not come with any validation layers. LunarG Vulkan SDK provides it.
// the LunarG validation layers are only available on PCs with the Vulkan SDK installed.
// instance layers would only check calls related to global Vulkan objects like instances
// device specific layers would only check calls related to a specific GPU.(have now been deprecated)
const std::vector<const char*> constGlobalValidationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

struct QueueFamilyIndices {
    std::optional<uint32_t> graphicsFamily;

    bool isComplete(){
        return graphicsFamily.has_value();
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
    VkQueue mGraphicsQueue;

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
        pickPhysicalDevice();
        createLogicalDevice();    
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

        VkDeviceQueueCreateInfo queueCreateInfo{};
        queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queueCreateInfo.queueFamilyIndex = indices.graphicsFamily.value();
        queueCreateInfo.queueCount = 1;

        float queuePriority = 1.0;
        queueCreateInfo.pQueuePriorities = &queuePriority;

        //Physical device
        VkPhysicalDeviceFeatures deviceFeatures{};
        //logical device
        VkDeviceCreateInfo createinfo{};
        createinfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createinfo.pQueueCreateInfos = &queueCreateInfo;
        createinfo.queueCreateInfoCount = 1;
        createinfo.pEnabledFeatures = &deviceFeatures;
        createinfo.enabledExtensionCount = 0;

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
    }

    bool isDeviceSuitable(VkPhysicalDevice device){
        QueueFamilyIndices indices = findQueueFamilies(device);

        return indices.isComplete();
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
        vkDestroyDevice(mDevice, nullptr);
        if(enableValidationLayers){
            DestroyDebugUtilsMessengerEXT(mInstance,mDebugMessenger,nullptr);
        }
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