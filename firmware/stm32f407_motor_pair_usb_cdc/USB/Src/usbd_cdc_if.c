#include <stdbool.h>
#include <string.h>
#include "main.h"

static uint8_t rx_buffer[64];
static uint8_t tx_buffer[1536];
static volatile bool port_open;
static uint8_t line_coding[7] = {0x00,0xC2,0x01,0x00,0,0,8};

static int8_t cdc_init(void)
{
    CDC_ResetLink();
    USBD_CDC_SetTxBuffer(&USBD_Device, tx_buffer, 0);
    USBD_CDC_SetRxBuffer(&USBD_Device, rx_buffer);
    return USBD_OK;
}
static int8_t cdc_deinit(void)
{
    CDC_ResetLink();
    return USBD_OK;
}
static int8_t cdc_control(uint8_t cmd, uint8_t *buf, uint16_t length)
{
    if (cmd == CDC_SET_LINE_CODING && length == 7) memcpy(line_coding,buf,7);
    else if (cmd == CDC_GET_LINE_CODING) memcpy(buf,line_coding,7);
    else if (cmd == CDC_SET_CONTROL_LINE_STATE) {
        /* ST passes the setup request itself for zero-length controls. */
        USBD_SetupReqTypedef *req = (USBD_SetupReqTypedef *)buf;
        port_open = (req->wValue & 1U) != 0;
        CDC_UpdateLink();
    }
    return USBD_OK;
}
static int8_t cdc_receive(uint8_t *buf,uint32_t *length)
{
    motor_usb_receive(buf,*length);
    USBD_CDC_SetRxBuffer(&USBD_Device,rx_buffer);
    USBD_CDC_ReceivePacket(&USBD_Device);
    return USBD_OK;
}
static int8_t cdc_tx_complete(uint8_t *buf,uint32_t *length,uint8_t ep)
{ (void)buf; (void)length; (void)ep; return USBD_OK; }
USBD_CDC_ItfTypeDef USBD_CDC_fops = {
    cdc_init,cdc_deinit,cdc_control,cdc_receive,cdc_tx_complete
};
bool CDC_IsOpen(void)
{
    return port_open && USBD_Device.dev_state == USBD_STATE_CONFIGURED;
}
void CDC_ResetLink(void)
{
    port_open = false;
    motor_usb_link(false);
}
void CDC_UpdateLink(void)
{
    /* This updates transport availability only. Control fault/STOP state is
       retained, and motor_usb_link discards any old queued commands. */
    motor_usb_link(CDC_IsOpen());
}
uint8_t CDC_Write(const uint8_t *data,uint16_t length)
{
    if (length > sizeof(tx_buffer)) return USBD_FAIL;
    uint32_t key = motor_lock();
    USBD_CDC_HandleTypeDef *cdc = USBD_Device.pClassData;
    uint8_t result = USBD_BUSY;
    if (CDC_IsOpen() && cdc != NULL && cdc->TxState == 0) {
        memcpy(tx_buffer,data,length);
        USBD_CDC_SetTxBuffer(&USBD_Device,tx_buffer,length);
        result=USBD_CDC_TransmitPacket(&USBD_Device);
    }
    motor_unlock(key);
    return result;
}
