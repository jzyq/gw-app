# 模型调度管理平台

## 构建

### 构建前：

1. 项目实际运行环境位 ascend 平台，确保有一个实体机可以用来构建项目。
2. 项目使用docker部署，构建项目前确保docker已经正确安装。
3. webapp/postprocess/notifier镜像构建于 `python:3.11-alpine` 镜像之上，可以提前拉取镜像加速，使用命令 `docker pull python:3.11-alpine` 拉取镜像
4. 构建dispatcher镜像依赖于[hw ascend-infer-310b dev](https://www.hiascend.com/developer/ascendhub/detail/fa3de8a80fd04830b3c396a4cb6f69b5) 开发镜像，这个镜像docker hub上找不到需要提前拉取到构建环境中。

    1. 注意拉取hw镜像需要注册hw账号。
    2. 拉取镜像时 ascend hub 有登录指引，按步骤进行即可。构建镜像不需要host安装依赖。
    3. 在ascend infer主页，点击`镜像版本`选项卡，进入版本列表，我们使用 `24.0.RC1-dev-arm`

5. 构建dispatcher还需要miniconda，如果没有miniconda的安装文件需要提前下载一个。

    1. miniconda下载页面 https://docs.anaconda.com/miniconda/install/
    2. 需要下载 linux aarch64 版本的
    3. 不知道怎么下载的直接使用命令 `curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh`

6. 克隆或复制项目文件到ascend上。

<br>

**拉取镜像慢的问题：**

如果 docker 拉取镜像慢，可以试着配置国内的 hub 镜像，配置方法可以参看这个[文档](https://gist.github.com/y0ngb1n/7e8f16af3242c7815e7ca2f0833d3ea6)，下面简述步骤：

1. 编辑 host (这里指的是ascend) 的 `/etc/docker/daemon.json` 文件，如果不存在则创建
2. 添加内容到文件内。注意如果已经有顶层大括号了就不要再添加大括号了，`register-mirrors` 写在顶层大括号内即可
    ```
    {
        "registry-mirrors": [
            "https://docker.unsee.tech",
            "https://hub.geekry.cn"
        ]
    }
    ```
3. mirrors地址参看列表，这里推荐unsee。
4. 运行指令 `sudo systemctl daemon-reload` 重载服务配置
5. 运行指令 `sudo systemctl restart docker` 重启docker服务
6. 重启 `docker` 后执行命令 `docker info`，如果看到 `Register Mirrors` 条目则说明镜像源配置好了。


## 构建步骤

1. cd 切换工作目录到项目目录中，假设项目文件放在 `/home/user/gw-app` 目录下，则 `cd /home/user/gw-app`
2. 将之前准本好的miniconda安装文件 `Miniconda3-latest-Linux-aarch64.sh` 拷贝到这个目录下。
3. 确保docker已经正确运行，可以执行指令 `docker info` 检查docker是否运行。
4. 执行指令 `docker compose build` 开始构建，然后等待构建完成即可。
5. 构建完成后执行 `docker images` 应该可以看到下面4个镜像。
    ```
    gw/dispatcher  latest <hash> 1 seconds ago 13.2GB
    gw/webapp      latest <hash> 1 seconds ago 94MB
    gw/notifier    latest <hash> 1 seconds ago 91.8MB
    gw/postprocess latest <hash> 1 secnods ago 88.8MB
    ```
    这样就构建好了


**通常安装docker的同时会自动安装 docker-compose插件，但是有些平台（如arch）不会自动安装，这时需要手动的安装好docker-compose插件，具体的安装方法参照docker文档**


### 构建后

如果需要导出镜像到其它环境，可以按照以下步骤进行：

1. 在构建环境，执行以下命令
    ```
    // -o 参数后跟导出文件名，可以自定义
    // 导出 gw/webapp gw/dispatcher gw/postprocess gw/notifier

    docker save -o gw.tar gw/webapp gw/dispatcher gw/postprocess gw/notifier
    ```

2. 拷贝这个文件到需要部署的环境中，然后执行以下命令
    ```
    docker load -i gw.tar
    ```
    镜像就可以正常导入了

3. 可以运行 `docker images` 检查镜像是否导入成功


## 运行

### 运行准备

1. 运行前，需要在ascend上正确的安装驱动和CANN固件，安装指引 https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC2alpha001/softwareinst/instg/instg_0001.html
2. 在host上创建用户 HwHiAiUser， 并且确保此用户的uid和gid都是1000，按以下步骤确认：
    1. 执行命令 `id` 检查 `uid` 和 `gid`
    2. 如果 `uid` 不是 `1000`, 执行指令 `usermod -u 1000 HwHiAiUser`
    3. 如果 `gid` 不是 `1000`, 执行指令 `groupmod -g 1000 HwHiAiUser`
3. 其它细节和启动参数参看此文档 https://gitee.com/ascend/ascend-docker-image/tree/master/ascend-infer-310b， 这里我们使用docker compose 启动
4. ***重要： 接下来准备模型和compose文件，如果已经拷贝或克隆了整个项目到运行环境中，可以略过。***
    1. 创建一个目录，比如 `/gwapp`
    2. 将项目根目录下的 `docker-compose.yaml` 文件和 `gwproc` 文件夹拷贝到这个目录中
    3. 修改 `docker-compose.yaml` 文件，在 `services:dispatcher:volumes` 下找到项目 `<path to gwproc>:/app/gwproc:rw`, 修改 `<path to gwproc>` 位刚刚拷贝过来的 `gwproc` 文件夹的路径，在此例中是 `/gwapp/gwproc`，所以修改后的项目位 `/gwapp/gwproc:/app/gwproc:rw`，确保服务能够找到业务模型。
    4. `cd` 切换工作目录到此目录下
    5. 运行指令启动服务栈。
        1. 前台启动指令 `docker compose up`
        2. 启动到后台，指令为 `docker compose up -d`
    

一切正确的话会启动 `redis` `webapp` `dispatcher` `postprocess` `notifier` 五个服务。前台启动时日志会直接通过标准输出流打印到终端。