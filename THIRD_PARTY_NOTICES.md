# Third-party components

Third-party code retains its original copyright notices and licenses. This
document does not relicense those components or the project's own code.

## Vendored STM32 dependencies

`vendor/light` contains the headers and source subset used by the STM32F407
firmware with Arm GNU Toolchain. The subset was derived from the successful
build's `ninja -t deps` output and contains 50 source/header files plus the
licenses and their provenance. Other STM32 devices or compilers may require
additional upstream headers. The component versions below were checked against matching upstream
representative source files (ignoring CRLF/LF differences). No vendor source
was changed while preparing this repository.

| Component | Upstream version | Local directory | License |
| --- | --- | --- | --- |
| ST STM32F4 HAL | v1.8.5 | `vendor/light/hal` | BSD-3-Clause; see `LICENSE.md` |
| ST STM32F4 CMSIS Device | v2.6.11 | `vendor/light/cmsis_device` | Apache-2.0; see `LICENSE.md` |
| ST USB Device middleware | v2.11.4 | `vendor/light/usb` | ST SLA0044 Rev5; see `LICENSE.md` |
| Arm CMSIS Core | CMSIS_5 5.9.0 | `vendor/light/cube/Drivers/CMSIS` | Apache-2.0; see `LICENSE.txt` |

The ST USB Device license limits use to STMicroelectronics devices and has
additional redistribution conditions. Its full license is included without
substituting another project's license.

License source URLs and SHA-256 checksums are recorded in
[`vendor/light/license_sources.json`](vendor/light/license_sources.json).
The source files used to identify the versions were respectively
`Src/stm32f4xx_hal.c`, `Include/stm32f4xx.h`, `Core/Src/usbd_core.c`, and
`CMSIS/Core/Include/core_cm4.h` in the upstream repositories linked there.

## Installed dependencies

Python libraries are installed through `spa_control/requirements*.txt` and
retain their own licenses. The optional MobileNetV3-Small pretrained weights
are obtained from the official torchvision source; they are not included.

The ESP32 camera firmware uses PlatformIO's pinned `espressif32@6.12.0`
platform with the Arduino framework. PlatformIO downloads those dependencies;
its framework, toolchain, and build caches are not part of this source release.
