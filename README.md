# SPA Snake：双螺旋机器人控制与训练

本地网页工作台，面向两节螺旋机器人（M3 / M4）：USB 电机控制、SPA-CAM 蓝牙图像、全局相机测速、数据采集、预测模型预训练和残差 PPO。

## 快速启动

推荐 Windows、Python 3.12。克隆仓库后，在仓库根目录运行：

```powershell
py -3.12 -m venv spa_control/.venv
& .\spa_control\.venv\Scripts\python.exe -m pip install -r .\spa_control\requirements_camera.txt -r .\spa_control\requirements_ml.txt
& .\spa_control\.venv\Scripts\python.exe -m spa_control.web_server --port 8765
```

打开 <http://127.0.0.1:8765/>。以后可双击 `spa_control/start_web.bat` 启动。服务只监听本机，启动不会自动连接设备或运行电机。

需要图像表征与训练时，先下载官方 MobileNetV3-Small 权重到本机缓存：

```powershell
& .\spa_control\.venv\Scripts\python.exe -c "from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights; mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)"
```

网页不会自行下载模型。运行记录、照片、标定和训练检查点在本机生成，不随仓库提供。

## 使用顺序

1. 连接 C 板 USB，点击“连接 C 板并停止”，核对 M3 / M4 反馈。先排除故障原因，必要时点击“清除故障”；成功后仍保持停机。
2. 连接全局 USB 相机，按已知轨道距离标定并框选机器人头部，确认实测速度有效。训练还需连接已配对的 SPA-CAM 蓝牙传出串口并收到新鲜图像。
3. 设置前后电机、方向、标称转速、目标距离和最长时间，保存设置。单机测试可帮助辨认 M3 / M4。
4. 先进行基线运行，再在至少两组独立场地准备下分别采集预训练数据。
5. 训练并冻结预测模型，然后开始真机强化学习。每轮结束停机后更新 PPO；下一轮由操作者启动。

详细操作、奖励和故障行为见 [网页工作台说明](spa_control/README_web.md)。旧桌面程序入口为 `python -m spa_control.app`，其四电机流程与当前双电机网页不同，见 [旧版说明](spa_control/README.md)。

## 当前硬件与算法边界

- RoboMaster C 板 STM32F407 + GM6020，M3 / M4；配套固件见 [电机固件](firmware/stm32f407_motor_pair_usb_cdc/README.md)。
- ESP32-CAM / OV2640，通过 Bluetooth Classic SPP 传输 JPEG，见 [相机固件](firmware/esp32_cam_bluetooth/README.md)。串口编号按本机实际设备选择。
- 当前实物确认导程 **30 mm/转**、传动比 **1:1**：100 rpm 对应理想前进速度 0.05 m/s。真实对地速度来自全局相机；电机反馈是转子相对定子的速度。
- 指令限幅 ±500 rpm，残差限幅 ±5 rpm，非激活电机为零。STOP、过流保护、通信看门狗独立于策略；固件启动为零输出。机械固定和方向需要按实物核对。
- 全局相机仅提供训练监督、奖励与评价，不作为 actor 输入。历史轨迹只用于监督预训练，PPO 使用当前策略新采集的真实轨迹。
- 本仓库用于 **two-section SPA pilot study**。未随仓库发布实机数据、训练好的策略或论文实验结论；时序标定与真实控制收益需要独立验证。

## 目录

```text
spa_control/                            网页、桌面工具、控制与算法、软件测试
firmware/stm32f407_motor_pair_usb_cdc/    C 板 USB CDC 双电机固件
firmware/esp32_cam_bluetooth/            SPA-CAM 固件
vendor/light/                           STM32 构建所需依赖
THIRD_PARTY_NOTICES.md                   第三方依赖来源与许可证
```

## 软件验证

以下测试使用模拟设备与本地临时服务，不要求连接机器人：

```powershell
& .\spa_control\.venv\Scripts\python.exe -m unittest discover -s spa_control/tests
```

固件编译方法和依赖见对应 README。编译不等于烧录；上传仓库不会改变正在运行的设备。
