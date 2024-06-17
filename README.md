sd3 自用懒人训练脚本
原项目在这里https://github.com/bghira/SimpleTuner

目前原项目还不支持windows,最好使用wsl或者linux，但也许未来会支持，我也写了一部分，测试不通过，感兴趣的可以自行尝试

配置过于繁琐，所以想写个自动化配置的东西，又不想写前端，只好借用一下comfyui，

使用方法：将文件夹放入comfyui的custom_nodes文件夹里

1.首先使用apt或者其他包管理软件在终端安装xterm：sudo apt install xterm 

2.下载 env文件夹，将python所在的路径填上，例如/media/xxxxx/env/bin

下载地址：链接: https://pan.baidu.com/s/1-PwTmYo85cL7s94kORJYiQ?pwd=x288 提取码: x288 

3.其他的也填上，就可以自动运行了 

modelname 填写你的sd3下载路径，例如/media/xxxxx/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/b1148b4028b9ec56ebd36444c193d56aeff7ab56或者使用stabilityai/stable-diffusion-3-medium-diffusers(不推荐)

image path填写你的图片文件夹，要包含txt标注文件

base dir任意填写你要保存的项目文件夹

######################################施工线#############################################

The original project is here https://github.com/bghira/SimpleTuner
Currently, the original project does not support Windows, it is best to use WSL or Linux, but it may be supported in the future, so I have also written a part of it, the tests do not pass, those interested can try it themselves.
The configuration is too cumbersome, so I want to write something for automated configuration, but I don't want to write a frontend, so I have to borrow comfyui.

Usage: Put the folder into the comfyui's custom_nodes folder.

1. First, use apt or another package manager in the terminal to install xterm: sudo apt install xterm

2. Download the env folder, fill in the path where Python is located, for example, /media/xxxxx/env/bin

Download link: https://drive.google.com/file/d/1pbFDgoGmI8-55eX3NcAMhCzG856TfRys/view?usp=sharing

3. Fill in the other fields, and it can run automatically.
For modelname, fill in your sd3 download path, for example, /media/xxxxx/models--stabilityai--stable-diffusion-3-medium-diffusers/snapshots/b1148b4028b9ec56ebd36444c193d56aeff7ab56 or use stabilityai/stable-diffusion-3-medium-diffusers (not recommended).
For image path, fill in your image folder, which should include a txt annotation file.
For base dir, fill in any project folder you want to save to.

![image](https://github.com/pzzmyc/comfyui-sd3-simple-simpletuner/assets/43562427/13df99cf-abc2-4488-91c3-4a3ea688ba47)
