import os
import glob
import shutil

def findAllShaderFiles(dir):
    return glob.glob(dir + "\\*\\shaders\\*.frag") + glob.glob(dir + "\\*\\shaders\\*.vert") + glob.glob(dir + "\\*\\shaders\\*.comp")

cur_dir = os.path.dirname(os.path.abspath(__file__))
#print(cur_dir)
files = findAllShaderFiles(os.path.join(cur_dir,"samples"))
cmd = os.environ.get("VULKAN_SDK") + "\\bin\\glslc.exe {} -o {}"

for f in files:
    #print(cmd.format(f,f.replace(os.path.basename(f),os.path.basename(f).split('.')[-1]+'.spv')))
    output = os.path.basename(f).split('.')
    outputFile = f.replace(os.path.basename(f),output[0] + output[1] + '.spv')

    copySpvTarget = os.path.abspath(".\\shaders\\" + os.path.basename(outputFile))
    os.system(cmd.format(f, copySpvTarget))
    #shutil.copy(outputFile,copySpvTarget)
    print(copySpvTarget)
