# GM6020 M3/M4 USB 配对固件 v4-full

2026-10-30。适用 RoboMaster C 板 STM32F407、12 MHz HSE，CAN1 PD0/PD1、
1 Mbps，反馈 M3=0x207、M4=0x208，电压控制报文 0x1FF。
本目录由 v3-2mff 固件修改而来。
应用户要求取消额外诊断输出限制，恢复完整电压指令范围和项目速度范围。
现场烧录清单、原始 Flash 备份和实测日志仅保存在原开发电脑，不随源码仓库发布。

## 用途与限制

默认 PAIR 模式，固件启动映射为 M3 负、M4 正。
电机背靠背安装时编码器正方向可能不同；上述符号必须通过实际外筒运动确认。

**C 板放在机器人外部，板载 IMU 不能测量中轴滚动。此固件没有中轴防滚闭环。**
遥测固定报告 `roll=UNAVAILABLE`。真正防滚需要中轴上固定的 IMU/姿态传感器，
并验证坐标方向、时间戳和失联停机，再标定反馈控制。
电机转速反馈是转子相对定子转速，不是外筒相对地面转速。

## 限制与保护

- 仅 M3/M4 可输出；M1/M2 恒为零。默认必须同时发送符合映射的两路目标。
- 目标范围恢复为 ±500 rpm；输出范围恢复为协议完整范围 ±25000（0x1FF 电压指令原始值，非电流）。
- 不再使用额外的 50 rpm / 2500 诊断限幅。输出仍由速度控制器计算，开机、STOP 和未激活通道输出为零。
- 前馈 60 电压指令单位/rpm，内部目标斜坡 400 rpm/s；PI Kp=10、Ki=10/s。
- 原始反馈电流绝对值达到 7000、温度达到 80°C 时锁存故障并将全部目标/输出归零。
- 运动所需反馈达到 100 ms 未更新，或指令接收后达到 300 ms 未刷新，停机锁存。
- 独立 5 ms TIM5 检查、硬件看门狗、USB/CAN 故障停机均保留。
- USB 会话变化、STOP 清除旧队列；成功 MODE/MAP 清除队列和未完成行。
- CAN ABOM 允许总线恢复，但不自动恢复目标或清除软件故障。
- 首次故障保存触发电机、原始电流、时间；STOP 不抹除，成功 CLEAR 才清除。
  非电机故障的时间使用最近控制回调时间，电机故障使用对应反馈接收时间。

## USB 命令

CDC 115200、8N1、DTR 开启，每条命令以换行结尾。正数无需加号。

```text
CTRL STOP
CTRL STATUS
CTRL DIAG
CTRL CLEAR
CTRL PAIR 20
CTRL PAIR 0
CTRL SPEED M3=-20 M4=20
CTRL MODE BENCH
CTRL SPEED M3=20
CTRL MODE PAIR
CTRL MAP M3=-1 M4=1
```

`CTRL PAIR q` 将 q 映射成 M3=dir3*q、M4=dir4*q，BENCH 下也操作两路。
PAIR 下非零 SPEED 必须同时给出两路，且 `M3*dir3 == M4*dir4`。
BENCH 允许单路诊断；省略通道立即归零。所有零目标都执行停机。
M1/M2 在范围内的目标被忽略并保持零；M5/M6 仅接受零。
非法命令、超范围或不匹配配对均锁存 BAD_COMMAND 并归零。

MODE/MAP 仅在所有目标/输出已为零、两路反馈均新鲜且 |speed|≤2 rpm 时成功。
成功配置后须等 ACK 再发新指令，旧排队指令会丢弃。模式/映射不持久化，重启恢复默认。
CLEAR 先停机，健康检查通过后才清故障，不恢复旧目标。
关闭串口/DTR 会锁存 USB_CLOSED；不要在反复重连后自动清故障并启动。

遥测为 `RM_C MOTOR_USB v4-full`，包含 mode、dir3/dir4、limit、output_limit、
current_trip、roll、trip_mask/trip_current/trip_ms 及各路角度/相对转速/原始电流。
`torque` 仅是兼容字段名，值为原始电流，不是 Nm。`tx_ok` 才表示真实发送完成。

## 编译与验证

依赖仓库中的 `vendor/light`、Ninja 和 Arm GNU Toolchain（验证版本 14.2.Rel1）。
将 `ninja`、`arm-none-eabi-gcc`、`arm-none-eabi-objcopy` 加入 PATH 后，在项目根目录运行：

```powershell
& .\firmware\stm32f407_motor_pair_usb_cdc\build.ps1
```

也可显式指定工具链的 `bin` 目录：

```powershell
& .\firmware\stm32f407_motor_pair_usb_cdc\build.ps1 -ToolchainBin 'C:\path\to\arm-gnu-toolchain\bin'
```

在其他系统上可从固件目录运行 `mkdir -p build` 后执行 `ninja`，所需工具同样须在 PATH 中。
构建脚本只编译，不连接或烧录设备。第三方依赖及其许可证见仓库根目录 `THIRD_PARTY_NOTICES.md`。

输出 `build/motor_usb_cdc.hex`。主机核心测试使用 `tests/test_pair_control.c`，
覆盖映射、BENCH/MAP 门控、限幅、超时、首次故障和停机；复制来的旧四电机测试不适用此配置。
可通过网页工作台连接后读取 STOP/STATUS/DIAG 反馈；连接不自动启动运动。

DFU 必须指定板子序列号，写入后校验并独立按 HEX 载荷回读比对。
HEX 段间空隙使用擦除值 FF，不能直接用默认 BIN 的 00 填充比较。
烧录后恢复 BOOT0 正常启动位置并按 RST，启动默认零输出。
