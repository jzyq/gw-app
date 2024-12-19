# ACLLite

## 介绍
ACLLite库是对CANN提供的ACL接口进行的高阶封装，简化用户调用流程，为用户提供一组简易的公共接口。当前主要针对边缘场景设计。

## 软件架构
<table>
<tr><td width="25%"><b>命名空间</b></td><td width="25%"><b>模块</b></td><td width="50%"><b>说明</b></td></tr>
<tr><td rowspan="10">acllite</td>
<td>common</td>  <td>资源管理及公共函数模块</td>  </tr>
<tr><td>DVPPLite</td>  <td>DVPP高阶封装模块</td>  </tr>
<tr><td>OMExecute</td>  <td>离线模型执行高阶封装模块</td>  </tr>
<tr><td>Media</td>  <td>媒体功能高阶封装模块</td>  </tr>
</tr>
</table>


## 安装教程

- **版本配套表**

   | 配套                                                         | 版本    | 环境准备指导                                                 |
   | :------------------------------------------------------------: | :-------: | :------------------------------------------------------------: |
   | CANN                                                         | 8.0RC2 |[CANN软件包安装准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0011.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)|
   | Python                                                       | 3.7.5   |[Python安装准备](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/80RC3alpha003/softwareinst/instg/instg_0064.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit#ZH-CN_TOPIC_0000002017916412?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)|
   | 硬件设备型号                                                   | 310B系列；310P系列；910B系列  | -                                                            |

- **安装依赖：**

    安装ffmpeg

       (1) 通过如下命令查询OS版本

       ```linux
       lsb_release -a
       ```

       (2) 根据查询结果选择安装方式

       - apt安装

         Ubuntu 22.04及以上版本的用户建议使用此方式安装

         ```linux
         apt-get install ffmpeg libavcodec-dev libswscale-dev libavdevice-dev
         ```
       - yum安装

         OpenEuler 22.03及以上的用户建议使用此方式安装
         ```linux
         yum install ffmpeg ffmpeg-devel

         # 将yum安装的ffmpeg头文件软链到系统能默认识别的路径
         ln -s /usr/include/ffmpeg/* /usr/include/
         ```
       - 源码编译安装

         其他用户建议使用此方式安装

         ```linux
         wget https://ffmpeg.org/releases/ffmpeg-4.2.9.tar.gz
         tar -zxvf ffmpeg-4.2.9.tar.gz
         cd ffmpeg-4.2.9
         ./configure --disable-static --enable-shared --disable-doc --enable-ffplay --enable-ffprobe --enable-avdevice --disable-debug --enable-demuxers --enable-parsers --enable-protocols --enable-small --enable-avresample
         make -j8
         sudo make install
         
         # 为保证程序能识别动态库，请在/etc/ld.so.conf.d下添加ffmpeg.conf配置
         cd /etc/ld.so.conf.d
         # 创建ffmpeg.conf
         vim ffmpeg.conf
         # 添加内容：/usr/local/lib
         # 保存：wq
         # 生效配置文件：
         ldconfig
         # 设置ffmpeg安装路径环境变量，请替换为ffmpeg的实际安装路径
         export FFMPEG_PATH=/usr/local/lib
         ```

- **安装步骤：**   
  
    ```linux
    # 拉取ACLLite仓库，并进入目录
    git clone https://gitee.com/ascend/ACLLite.git
    cd ACLLite
    
    # 设置环境变量，其中DDK_PATH中/usr/local请替换为实际CANN包的安装路径
    export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub
    
    # 安装，编译过程中会将库文件安装到/lib目录下，所以会有sudo命令，需要输入密码
    bash build_so.sh
    ```

## API说明

- [common模块API说明](Doc/common.md)
- [DVPPLite模块API说明](Doc/dvpplite.md)
- [OMExcute模块API说明](Doc/omexcute.md)
- [Media模块API说明](Doc/media.md)

#### 参与贡献

1.  Fork 本仓库
2.  提交代码
3.  新建 Pull Request


#### 修订记录

| 日期  | 更新事项  |
|---|---|
| 2022/1/25  | 首次发布  |

