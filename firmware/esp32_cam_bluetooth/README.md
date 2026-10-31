# ESP32-CAM Bluetooth live view

Target: classic ESP32 / ESP-32S AI Thinker ESP32-CAM with OV2640.
This firmware uses Bluetooth Classic SPP, not BLE. The camera is independent
of robot motor control. It has no motor commands.

## Flash

Use a USB-UART with 3.3 V logic:

| USB-UART / supply | Camera |
| --- | --- |
| TX | U0R |
| RX | U0T |
| GND | GND |
| Stable 5 V supply | 5V |
| Temporary boot jumper | IO0 to GND |

Do not connect two 5 V sources together. After connecting the boot jumper,
press RST before upload. Remove the jumper and press RST after upload.

From the repository root:

```powershell
& .\spa_control\.venv\Scripts\python.exe -m platformio run --project-dir firmware/esp32_cam_bluetooth
& .\spa_control\.venv\Scripts\python.exe -m platformio run --project-dir firmware/esp32_cam_bluetooth --target upload --upload-port COM12
```

The COM number can change. Verify the current CH340 port first.
Build dependencies: `pip install platformio esptool`.

## Preview

The device advertises as `SPA-CAM`. Pair it in Windows Bluetooth settings.
Use the outgoing SPP COM port for SPA-CAM, not the CH340 USB download port
and not a COM port belonging to another Bluetooth device.

Open the `Head Camera` (Chinese: head-camera) tab in `spa_control.app`, refresh
ports, select the correct port and connect. A standalone viewer is also available:

```powershell
& .\spa_control\.venv\Scripts\python.exe -m spa_control.bluetooth_camera
```

USB preview works with the same firmware through CH340 at 921600 baud after
the boot jumper is removed. Bluetooth's COM baud setting does not determine
radio throughput. Flash upload remains at 115200 baud. This is QVGA JPEG
with a 50 ms interval after each received frame and a 30 Hz preview refresh;
actual frame rate depends on scene, image size, and link throughput. PSRAM
boards use two frame buffers and latest-frame capture to reduce stale images.
Update both firmware and host together: the original firmware used 115200
baud for USB viewing and is not compatible with the new default viewer baud.

## Wire Protocol

Host requests one frame with ASCII `F`. Only one request is outstanding.
Firmware replies with a 32-byte little-endian header `<8sIIQII`:

`SPACAM01`, boot ID, frame sequence, capture timestamp in microseconds,
JPEG length, standard CRC32 of JPEG. The JPEG payload follows.
`L`/`l` enable/disable the onboard light. `?` reports camera initialization.
On startup USB serial prints camera success/failure and Bluetooth status.

The receiver bounds JPEG length to 128 KiB, rejects corrupt frames, and
retains only the latest decoded image. Serial reception and image decoding
run outside Tk. It times out if no complete frame arrives for 8 seconds.

Recording saves JPEGs and `frames.jsonl` under `spa_control/runs/head_camera`.
Device capture time and host monotonic receive time are separate clock domains.
Robot snapshots are associated with reception, not synchronized exposure.
`clock_sync=not_calibrated` is intentional: clock mapping and robot acquisition
timestamps must be added before claiming frame/state synchronization.
