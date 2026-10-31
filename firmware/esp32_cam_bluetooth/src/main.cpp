#include <Arduino.h>
#include <BluetoothSerial.h>
#include "esp_camera.h"
#include "esp_timer.h"

BluetoothSerial bluetooth;
bool cameraReady = false;
uint32_t bootId;
uint32_t sequence = 0;

// Matches Python's <8sIIQII. The timestamp comes from camera capture, not send time.
struct __attribute__((packed)) FrameHeader {
  char magic[8];
  uint32_t boot;
  uint32_t sequence;
  uint64_t captureUs;
  uint32_t length;
  uint32_t crc;
};
static_assert(sizeof(FrameHeader) == 32, "Frame header size");

uint32_t crc32(const uint8_t *data, size_t length) {
  uint32_t crc = 0xffffffff;
  while (length--) {
    crc ^= *data++;
    for (int bit = 0; bit < 8; ++bit)
      crc = (crc >> 1) ^ (0xedb88320 & (0u - (crc & 1)));
  }
  return ~crc;
}

void sendFrame(Stream &stream) {
  if (!cameraReady) return;
  camera_fb_t *frame = esp_camera_fb_get();
  if (!frame) return;
  if (frame->format == PIXFORMAT_JPEG && frame->len <= 128 * 1024) {
    FrameHeader header;
    memcpy(header.magic, "SPACAM01", 8);
    header.boot = bootId;
    header.sequence = ++sequence;
    header.captureUs = uint64_t(frame->timestamp.tv_sec) * 1000000ULL + frame->timestamp.tv_usec;
    header.length = frame->len;
    header.crc = crc32(frame->buf, frame->len);
    stream.write(reinterpret_cast<const uint8_t *>(&header), sizeof(header));
    for (size_t offset = 0; offset < frame->len;) {
      size_t sent = stream.write(frame->buf + offset, min(size_t(512), frame->len - offset));
      if (!sent) break;
      offset += sent;
      delay(0);
    }
  }
  esp_camera_fb_return(frame);
}

void setup() {
  Serial.begin(921600);
  pinMode(4, OUTPUT);
  digitalWrite(4, LOW);
  bootId = esp_random();
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  // AI Thinker ESP32-CAM / OV2640 pinout.
  config.pin_d0 = 5; config.pin_d1 = 18; config.pin_d2 = 19; config.pin_d3 = 21;
  config.pin_d4 = 36; config.pin_d5 = 39; config.pin_d6 = 34; config.pin_d7 = 35;
  config.pin_xclk = 0; config.pin_pclk = 22; config.pin_vsync = 25; config.pin_href = 23;
  config.pin_sccb_sda = 26; config.pin_sccb_scl = 27;
  config.pin_pwdn = 32; config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_QVGA;
  config.jpeg_quality = 20;
  const bool hasPsram = psramFound();
  config.fb_count = hasPsram ? 2 : 1;
  config.fb_location = hasPsram ? CAMERA_FB_IN_PSRAM : CAMERA_FB_IN_DRAM;
  config.grab_mode = hasPsram ? CAMERA_GRAB_LATEST : CAMERA_GRAB_WHEN_EMPTY;
  esp_err_t error = esp_camera_init(&config);
  cameraReady = error == ESP_OK;
  Serial.printf("SPA_CAM camera=%s error=0x%x psram=%d boot=%u\n",
                cameraReady ? "OK" : "FAIL", error, psramFound(), bootId);
  if (!bluetooth.begin("SPA-CAM")) Serial.println("SPA_CAM bluetooth=FAIL");
  else Serial.println("SPA_CAM bluetooth=OK name=SPA-CAM");
}

void handle(Stream &stream) {
  if (!stream.available()) return;
  int command = stream.read();
  if (command == 'F') sendFrame(stream);
  else if (command == 'L') digitalWrite(4, HIGH);
  else if (command == 'l') digitalWrite(4, LOW);
  else if (command == '?') stream.printf("SPA_CAM camera=%s boot=%u\n", cameraReady ? "OK" : "FAIL", bootId);
}

void loop() {
  handle(Serial);
  if (bluetooth.hasClient()) handle(bluetooth);
  delay(1);
}
