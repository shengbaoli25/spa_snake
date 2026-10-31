#ifndef USBD_CDC_IF_H
#define USBD_CDC_IF_H

#include "usbd_cdc.h"
#include <stdbool.h>

extern USBD_CDC_ItfTypeDef USBD_CDC_fops;
uint8_t CDC_Write(const uint8_t *data, uint16_t length);
bool CDC_IsOpen(void);
/* Reset/disconnect forget DTR; suspend preserves it until resume. */
void CDC_ResetLink(void);
void CDC_UpdateLink(void);

#endif
