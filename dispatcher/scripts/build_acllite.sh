# 设置环境变量，其中DDK_PATH中/usr/local请替换为实际CANN包的安装路径
export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub

# 安装，编译过程中会将库文件安装到/lib目录下，所以会有sudo命令，需要输入密码
bash $1/build_so.sh