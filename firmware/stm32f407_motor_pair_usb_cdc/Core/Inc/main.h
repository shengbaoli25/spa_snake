#ifndef MOTOR_MAIN_H
#define MOTOR_MAIN_H
#include <stdbool.h>
#include "stm32f4xx_hal.h"
#include "usbd_cdc.h"
#include "usbd_cdc_if.h"
#include "usbd_desc.h"
#include "motor_control.h"
extern USBD_HandleTypeDef USBD_Device;
void Error_Handler(void);
void motor_usb_link(bool open);
void motor_usb_receive(const uint8_t *data, uint32_t length);
void motor_emergency_fault(void);
uint32_t motor_lock(void);
void motor_unlock(uint32_t key);
#endif
