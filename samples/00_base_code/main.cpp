#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>

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
    GLFWwindow* window;
    int mWidth = 800, mHeight = 800;

    void initWindow(){
        glfwInit();
        //specifi window api, no use any opengl api
        glfwWindowHint(GLFW_CLIENT_API,GLFW_NO_API);
        // create a not resizable window.
        glfwWindowHint(GLFW_RESIZABLE,GLFW_FALSE);

        window = glfwCreateWindow(mWidth,mHeight,"Vulkan",nullptr,nullptr);

    }

    void initVulkan(){

    }
    void mainLoop(){
        while(!glfwWindowShouldClose(window)){
            glfwPollEvents();
        }
    }
    void cleanup(){
        glfwDestroyWindow(window);
        glfwTerminate();
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