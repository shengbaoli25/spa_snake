#include "main.h"
#include <stdio.h>
#include <string.h>

/* RoboMaster C: 12 MHz HSE; CAN1 PD0/PD1, 1 Mbps; GM6020 group 1. */
#define CAN_CONTROL_ID 0x1FFU
/* Match the original flash backup: BRP=3, BS1=10 tq, BS2=3 tq, SJW=1 tq. */
#define CAN_BIT_TIMING (2U|(9U<<16)|(2U<<20))
#define RX_QUEUE_SIZE 4U
#define LINE_SIZE 160U

USBD_HandleTypeDef USBD_Device;
static volatile bool can_ready;
static volatile uint32_t can_rx_count, can_tx_count, can_tx_fail, can_other;
static volatile uint32_t can_tx_ok, can_tx_error, can_tx_arbitration, can_tx_busy;
static volatile uint32_t can_tx_abort, last_tx_ok_ms, last_motor_rx_ms;
static volatile bool tx_ok_seen, motor_rx_seen, tx_inflight, tx_aborting;
static volatile bool first_error_seen;
static volatile uint32_t first_error_ms, first_error_esr, first_error_tsr, first_error_rx;
static volatile uint32_t main_alive_ms;
static volatile bool rx_fault;
static volatile uint8_t rx_head,rx_tail;
typedef struct { char line[LINE_SIZE]; uint32_t received_ms; } rx_command_t;
static rx_command_t rx_queue[RX_QUEUE_SIZE];
static volatile uint32_t command_generation;
static volatile bool stop_reply;
static char partial[LINE_SIZE];
static uint16_t partial_len;
static bool discard_line;

uint32_t motor_lock(void)
{ uint32_t key=__get_PRIMASK(); __disable_irq(); return key; }
void motor_unlock(uint32_t key) { __set_PRIMASK(key); }

static void clock_config(void)
{
    RCC_OscInitTypeDef osc={0};
    RCC_ClkInitTypeDef clk={0};
    __HAL_RCC_PWR_CLK_ENABLE();
    __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
    osc.OscillatorType=RCC_OSCILLATORTYPE_HSE;
    osc.HSEState=RCC_HSE_ON;
    osc.PLL.PLLState=RCC_PLL_ON;
    osc.PLL.PLLSource=RCC_PLLSOURCE_HSE;
    osc.PLL.PLLM=12; osc.PLL.PLLN=336;
    osc.PLL.PLLP=RCC_PLLP_DIV2; osc.PLL.PLLQ=7;
    if(HAL_RCC_OscConfig(&osc)!=HAL_OK) Error_Handler();
    clk.ClockType=RCC_CLOCKTYPE_SYSCLK|RCC_CLOCKTYPE_HCLK|
                  RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
    clk.SYSCLKSource=RCC_SYSCLKSOURCE_PLLCLK;
    clk.AHBCLKDivider=RCC_SYSCLK_DIV1;
    clk.APB1CLKDivider=RCC_HCLK_DIV4;
    clk.APB2CLKDivider=RCC_HCLK_DIV2;
    if(HAL_RCC_ClockConfig(&clk,FLASH_LATENCY_5)!=HAL_OK) Error_Handler();
}

static bool can_init(void)
{
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_CAN1_CLK_ENABLE();
    __HAL_RCC_CAN1_FORCE_RESET();
    __HAL_RCC_CAN1_RELEASE_RESET();
    GPIO_InitTypeDef gpio={0};
    gpio.Pin=GPIO_PIN_0|GPIO_PIN_1; gpio.Mode=GPIO_MODE_AF_PP;
    gpio.Pull=GPIO_NOPULL; gpio.Speed=GPIO_SPEED_FREQ_VERY_HIGH;
    gpio.Alternate=GPIO_AF9_CAN1; HAL_GPIO_Init(GPIOD,&gpio);
    /* Hardware may recover the bus; the control fault remains latched. */
    CAN1->MCR=CAN_MCR_INRQ|CAN_MCR_NART|CAN_MCR_ABOM;
    uint32_t start=HAL_GetTick();
    while(!(CAN1->MSR&CAN_MSR_INAK)) if(HAL_GetTick()-start>100) return false;
    /* APB1 42 MHz / (3 * (1+10+3)) = 1 Mbps, sample point 78.6%. */
    CAN1->BTR=CAN_BIT_TIMING;
    CAN1->FMR=CAN_FMR_FINIT|(14U<<8);
    CAN1->FA1R=0; CAN1->FM1R=0; CAN1->FS1R=1; CAN1->FFA1R=0;
    /* Accept all frames; count every reception, then validate motor IDs in ISR. */
    CAN1->sFilterRegister[0].FR1=0;
    CAN1->sFilterRegister[0].FR2=0;
    CAN1->FA1R=1; CAN1->FMR&=~CAN_FMR_FINIT;
    CAN1->MCR&=~CAN_MCR_INRQ;
    start=HAL_GetTick();
    while(CAN1->MSR&CAN_MSR_INAK) if(HAL_GetTick()-start>100) return false;
    CAN1->IER=CAN_IER_FMPIE0|CAN_IER_FOVIE0;
    HAL_NVIC_SetPriority(CAN1_RX0_IRQn,1,0);
    HAL_NVIC_EnableIRQ(CAN1_RX0_IRQn);
    can_ready=true;
    return true;
}

static void can_record_error(uint32_t now, uint32_t esr, uint32_t tsr)
{
    if(!first_error_seen) {
        first_error_ms=now; first_error_esr=esr;
        first_error_tsr=tsr; first_error_rx=can_rx_count;
        first_error_seen=true;
    }
}

/* Read TX completion before clearing RQCP. Submission is not success. */
static void can_reap(uint32_t now)
{
    uint32_t tsr=CAN1->TSR,esr=CAN1->ESR;
    uint32_t lec=(esr&CAN_ESR_LEC)>>4;
    if((lec>0U && lec<7U) || (esr&(CAN_ESR_EPVF|CAN_ESR_BOFF)))
        can_record_error(now,esr,tsr);
    if(!(tsr&CAN_TSR_RQCP0)) return;
    if(tx_inflight) {
        if(tsr&CAN_TSR_TXOK0) { ++can_tx_ok; tx_ok_seen=true; last_tx_ok_ms=now; }
        if(tsr&CAN_TSR_TERR0) {
            ++can_tx_error; ++can_tx_fail;
            can_record_error(now,esr,tsr); control_trip(CONTROL_FAULT_CAN);
        }
        if(tsr&CAN_TSR_ALST0) ++can_tx_arbitration;
        if(tx_aborting) ++can_tx_abort;
    }
    tx_inflight=false; tx_aborting=false;
    CAN1->TSR=CAN_TSR_RQCP0;
}

/* Only mailbox 0 is used. Never queue multiple generations of motor output. */
static bool can_send(void)
{
    if(!can_ready || (CAN1->ESR&CAN_ESR_BOFF)) return false;
    if(!(CAN1->TSR&CAN_TSR_TME0)) {
        ++can_tx_busy;
        can_record_error(HAL_GetTick(),CAN1->ESR,CAN1->TSR);
        tx_aborting=true;
        CAN1->TSR=CAN_TSR_ABRQ0;
        return false;
    }
    can_reap(HAL_GetTick());
    /* Reap can latch a fault. Read outputs only after completion handling. */
    int16_t output[4]; control_get_outputs(output);
    CAN1->TSR=CAN_TSR_RQCP0;
    uint8_t data[8];
    for(unsigned i=0;i<4;i++) {
        data[2*i]=(uint16_t)output[i]>>8;
        data[2*i+1]=(uint8_t)output[i];
    }
    CAN1->sTxMailBox[0].TDTR=8;
    CAN1->sTxMailBox[0].TDLR=(uint32_t)data[0]|((uint32_t)data[1]<<8)|
        ((uint32_t)data[2]<<16)|((uint32_t)data[3]<<24);
    CAN1->sTxMailBox[0].TDHR=(uint32_t)data[4]|((uint32_t)data[5]<<8)|
        ((uint32_t)data[6]<<16)|((uint32_t)data[7]<<24);
    tx_inflight=true;
    CAN1->sTxMailBox[0].TIR=(CAN_CONTROL_ID<<21)|CAN_TI0R_TXRQ;
    ++can_tx_count;
    return true;
}

void CAN1_RX0_IRQHandler(void)
{
    uint32_t key=motor_lock();
    if(CAN1->RF0R&CAN_RF0R_FOVR0) {
        can_record_error(HAL_GetTick(),CAN1->ESR,CAN1->TSR);
        CAN1->RF0R=CAN_RF0R_FOVR0;
        control_trip(CONTROL_FAULT_CAN);
    }
    while(CAN1->RF0R&CAN_RF0R_FMP0) {
        uint32_t rir=CAN1->sFIFOMailBox[0].RIR;
        uint32_t dlc=CAN1->sFIFOMailBox[0].RDTR&15;
        uint32_t lo=CAN1->sFIFOMailBox[0].RDLR;
        uint32_t hi=CAN1->sFIFOMailBox[0].RDHR;
        CAN1->RF0R=CAN_RF0R_RFOM0;
        ++can_rx_count;
        uint32_t id=rir>>21;
        if(!(rir&(CAN_RI0R_IDE|CAN_RI0R_RTR)) && dlc==8 && id>=0x205 && id<=0x208) {
            last_motor_rx_ms=HAL_GetTick(); motor_rx_seen=true;
            uint16_t angle=(uint16_t)(((lo&255)<<8)|((lo>>8)&255));
            int16_t speed=(int16_t)((((lo>>16)&255)<<8)|(lo>>24));
            int16_t current=(int16_t)(((hi&255)<<8)|((hi>>8)&255));
            control_on_feedback((uint8_t)(id-0x205),angle,speed,current,
                                (uint8_t)(hi>>16),HAL_GetTick());
        } else ++can_other;
    }
    motor_unlock(key);
}

void motor_usb_link(bool open)
{
    uint32_t key=motor_lock();
    /* Never arm host commands while the CAN controller failed to initialize. */
    control_set_link(open && can_ready);
    /* A new USB session must not replay old queued/partial commands. */
    rx_head=rx_tail=0; partial_len=0; discard_line=false;
    ++command_generation; stop_reply=false; rx_fault=false;
    motor_unlock(key);
}

void motor_usb_receive(const uint8_t *data,uint32_t length)
{
    uint32_t key=motor_lock();
    for(uint32_t i=0;i<length;i++) {
        uint8_t b=data[i];
        if(b=='\r') continue;
        if(b=='\n') {
            if(discard_line) { discard_line=false; partial_len=0; continue; }
            if(!partial_len) continue;
            partial[partial_len]=0;
            if(!strcmp(partial,"CTRL STOP")) {
                char unused[32];
                control_command(partial,HAL_GetTick(),unused,sizeof(unused));
                rx_head=rx_tail=0;
                ++command_generation; stop_reply=true;
            } else {
                uint8_t next=(rx_head+1)%RX_QUEUE_SIZE;
                if(next==rx_tail) {
                    control_trip(CONTROL_FAULT_BAD_COMMAND); rx_fault=true;
                    rx_head=rx_tail=0; ++command_generation;
                } else {
                    memcpy(rx_queue[rx_head].line,partial,partial_len+1);
                    rx_queue[rx_head].received_ms=HAL_GetTick(); rx_head=next;
                }
            }
            partial_len=0;
        } else if(b<32 || b>126 || partial_len>=LINE_SIZE-1) {
            control_trip(CONTROL_FAULT_BAD_COMMAND);
            rx_head=rx_tail=0; ++command_generation;
            rx_fault=true; discard_line=true; partial_len=0;
        } else if(!discard_line) partial[partial_len++]=(char)b;
    }
    motor_unlock(key);
}

/* TIM5 owns the output loop; USB printing and main cannot delay its checks. */
void TIM5_IRQHandler(void)
{
    if(!(TIM5->SR&TIM_SR_UIF)) return;
    TIM5->SR=0;
    uint32_t now=HAL_GetTick();
    uint32_t key=motor_lock();
    if(now-main_alive_ms>100U) control_trip(CONTROL_FAULT_FATAL);
    if(can_ready) can_reap(now);
    uint32_t esr=CAN1->ESR;
    if(can_ready && (esr&(CAN_ESR_BOFF|CAN_ESR_EPVF))) {
        can_record_error(now,esr,CAN1->TSR); control_trip(CONTROL_FAULT_CAN);
        if(tx_inflight && !(CAN1->TSR&CAN_TSR_TME0)) {
            tx_aborting=true; CAN1->TSR=CAN_TSR_ABRQ0;
        }
    }
    bool fresh=motor_rx_seen && (uint32_t)(now-last_motor_rx_ms)<CONTROL_FEEDBACK_TIMEOUT_MS;
    control_set_can_health(can_ready && !(esr&(CAN_ESR_BOFF|CAN_ESR_EPVF)) &&
        fresh && tx_ok_seen && (uint32_t)(now-last_tx_ok_ms)<CONTROL_FEEDBACK_TIMEOUT_MS);
    control_tick(now);
    /* Wait for actual motor traffic before zero-output handshaking. This
       avoids driving an unpowered/disconnected bus into bus-off at boot. */
    bool send_window=motor_rx_seen &&
        (uint32_t)(now-last_motor_rx_ms)<2U*CONTROL_FEEDBACK_TIMEOUT_MS;
    if(can_ready && send_window && !(esr&CAN_ESR_BOFF) && !can_send()) {
        ++can_tx_fail; control_trip(CONTROL_FAULT_CAN);
    }
    motor_unlock(key);
}

static void timer_init(void)
{
    __HAL_RCC_TIM5_CLK_ENABLE();
    TIM5->PSC=83; TIM5->ARR=4999; TIM5->EGR=TIM_EGR_UG;
    TIM5->SR=0; TIM5->DIER=TIM_DIER_UIE;
    HAL_NVIC_SetPriority(TIM5_IRQn,0,0); HAL_NVIC_EnableIRQ(TIM5_IRQn);
    TIM5->CR1=TIM_CR1_CEN;
}
static void led_init(void)
{
    /* C board user RGB LED: PH10 is the active-high blue channel. */
    __HAL_RCC_GPIOH_CLK_ENABLE();
    HAL_GPIO_WritePin(GPIOH,GPIO_PIN_10,GPIO_PIN_RESET);
    GPIO_InitTypeDef gpio={0};
    gpio.Pin=GPIO_PIN_10; gpio.Mode=GPIO_MODE_OUTPUT_PP;
    gpio.Pull=GPIO_NOPULL; gpio.Speed=GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOH,&gpio);
    HAL_GPIO_WritePin(GPIOH,GPIO_PIN_10,GPIO_PIN_SET);
}
static void watchdog_init(void)
{
    /* PR/RLR updates are synchronized to LSI; waiting before its clock starts
       can stall boot forever. Start LSI and the watchdog before updating. */
    RCC->CSR|=RCC_CSR_LSION;
    uint32_t start=HAL_GetTick();
    while(!(RCC->CSR&RCC_CSR_LSIRDY))
        if(HAL_GetTick()-start>100U) Error_Handler();
    IWDG->KR=0xCCCC;
    IWDG->KR=0x5555; IWDG->PR=4; IWDG->RLR=249;
    start=HAL_GetTick();
    while(IWDG->SR)
        if(HAL_GetTick()-start>100U) Error_Handler();
    IWDG->KR=0xAAAA;
}

void motor_emergency_fault(void)
{
    __disable_irq();
    control_trip(CONTROL_FAULT_FATAL);
    if(can_ready) { CAN1->TSR=CAN_TSR_ABRQ0; (void)can_send(); }
    /* IWDG resets to zero-output boot; do not refresh from a fault handler. */
    while(1) {}
}
void Error_Handler(void) { motor_emergency_fault(); }
void NMI_Handler(void) { motor_emergency_fault(); }
void HardFault_Handler(void) { motor_emergency_fault(); }
void MemManage_Handler(void) { motor_emergency_fault(); }
void BusFault_Handler(void) { motor_emergency_fault(); }
void UsageFault_Handler(void) { motor_emergency_fault(); }
void SysTick_Handler(void) { HAL_IncTick(); }
extern PCD_HandleTypeDef hpcd;
void OTG_FS_IRQHandler(void) { HAL_PCD_IRQHandler(&hpcd); }

static void report(void)
{
    control_snapshot_t s;
    uint32_t key=motor_lock(); control_snapshot(&s);
    uint32_t rx=can_rx_count,submitted=can_tx_count,ok=can_tx_ok,fail=can_tx_fail;
    uint32_t errors=can_tx_error,arbitration=can_tx_arbitration,busy=can_tx_busy;
    uint32_t aborted=can_tx_abort,other=can_other,esr=CAN1->ESR,btr=CAN1->BTR;
    uint32_t now=HAL_GetTick();
    motor_unlock(key);
    /* Four motor records plus full-width counters and v4 diagnostics. */
    char line[1536];
    int n=snprintf(line,sizeof(line),
        "RM_C MOTOR_USB v4-full can_ready=%u can_healthy=%u active=0x%02X rx=%lu tx_submit=%lu tx_ok=%lu tx_err=%lu arb=%lu busy=%lu abort=%lu fail=%lu other=%lu ESR=0x%08lX BTR=0x%08lX fault=%s reason=%s lease_ms=%u limit=%u output_limit=%u current_trip=%u mode=%s dir3=%d dir4=%d roll=UNAVAILABLE trip_mask=0x%02X trip_current=%d trip_ms=%lu",
        can_ready,s.can_healthy,(unsigned)CONTROL_ACTIVE_MASK,(unsigned long)rx,(unsigned long)submitted,(unsigned long)ok,
        (unsigned long)errors,(unsigned long)arbitration,(unsigned long)busy,
        (unsigned long)aborted,(unsigned long)fail,(unsigned long)other,
        (unsigned long)esr,(unsigned long)btr,
        s.fault_latched?control_reason_string(s.reason):"NONE",control_reason_string(s.reason),
        (unsigned)CONTROL_COMMAND_TIMEOUT_MS,(unsigned)CONTROL_SPEED_LIMIT,
        (unsigned)CONTROL_CURRENT_LIMIT,(unsigned)CONTROL_OVERCURRENT_RAW,
        s.pair_mode?"PAIR":"BENCH",(int)s.pair_dir3,(int)s.pair_dir4,
        (unsigned)s.fault_motor_mask,(int)s.fault_current_raw,(unsigned long)s.fault_ms);
    for(unsigned i=0;i<4 && n>0 && n<(int)sizeof(line)-2;i++) {
        n+=snprintf(line+n,sizeof(line)-(size_t)n,
            " | M%u id=0x%03X angle=%u spd=%d torque=%d setI=0 setV=%d target=%d temp=%u seen=%u rx=%lu age_ms=%lu",
            i+1,0x205+i,s.feedback[i].angle,s.feedback[i].speed,s.feedback[i].current,
            s.output[i],s.target[i],s.feedback[i].temperature,s.feedback[i].seen,
            (unsigned long)s.feedback[i].rx_count,
            (unsigned long)(s.feedback[i].seen? now-s.feedback[i].last_ms:999999U));
    }
    if(n>0 && n<(int)sizeof(line)-2) { line[n++]='\r'; line[n++]='\n'; (void)CDC_Write((uint8_t*)line,(uint16_t)n); }
}

/* Capture registers under the lock, format with interrupts enabled. */
typedef struct {
    uint32_t btr,mcr,msr,esr,tsr,rf0r,mode,af,input,cfgr,apb1;
    uint32_t first_seen,first_ms,first_esr,first_tsr,first_rx;
} can_diag_t;
static void diag_snapshot(can_diag_t *d)
{
    d->btr=CAN1->BTR; d->mcr=CAN1->MCR; d->msr=CAN1->MSR;
    d->esr=CAN1->ESR; d->tsr=CAN1->TSR; d->rf0r=CAN1->RF0R;
    d->mode=GPIOD->MODER; d->af=GPIOD->AFR[0]; d->input=GPIOD->IDR;
    d->cfgr=RCC->CFGR; d->apb1=RCC->APB1ENR;
    d->first_seen=first_error_seen; d->first_ms=first_error_ms;
    d->first_esr=first_error_esr; d->first_tsr=first_error_tsr; d->first_rx=first_error_rx;
}
static void diag_format(const can_diag_t *d,char *reply,size_t length)
{
    (void)snprintf(reply,length,
        "CTRL DIAG BTR=0x%08lX MCR=0x%08lX MSR=0x%08lX ESR=0x%08lX TSR=0x%08lX RF0R=0x%08lX PD_MODE=0x%08lX PD_AF=0x%08lX PD_IN=0x%08lX RCC_CFGR=0x%08lX RCC_APB1=0x%08lX first_seen=%lu first_ms=%lu first_ESR=0x%08lX first_TSR=0x%08lX first_rx=%lu\r\n",
        (unsigned long)d->btr,(unsigned long)d->mcr,(unsigned long)d->msr,
        (unsigned long)d->esr,(unsigned long)d->tsr,(unsigned long)d->rf0r,
        (unsigned long)d->mode,(unsigned long)d->af,(unsigned long)d->input,
        (unsigned long)d->cfgr,(unsigned long)d->apb1,(unsigned long)d->first_seen,
        (unsigned long)d->first_ms,(unsigned long)d->first_esr,
        (unsigned long)d->first_tsr,(unsigned long)d->first_rx);
}

int main(void)
{
    HAL_Init(); clock_config(); control_init(); led_init();
    /* Start CDC before CAN so a missing CAN clock/transceiver is diagnosable. */
    if(USBD_Init(&USBD_Device,&VCP_Desc,0)!=USBD_OK ||
       USBD_RegisterClass(&USBD_Device,USBD_CDC_CLASS)!=USBD_OK ||
       USBD_CDC_RegisterInterface(&USBD_Device,&USBD_CDC_fops)!=USBD_OK ||
       USBD_Start(&USBD_Device)!=USBD_OK) Error_Handler();
    if(!can_init()) control_trip(CONTROL_FAULT_CAN);
    CDC_UpdateLink();
    main_alive_ms=HAL_GetTick();
    timer_init(); watchdog_init();
    uint32_t report_at=0,led_at=HAL_GetTick();
    char pending_reply[768]={0};
    uint32_t pending_generation=command_generation;
    while(1) {
        main_alive_ms=HAL_GetTick(); IWDG->KR=0xAAAA;
        if(HAL_GetTick()-led_at>=500U) {
            led_at=HAL_GetTick(); HAL_GPIO_TogglePin(GPIOH,GPIO_PIN_10);
        }
        bool diag=false;
        can_diag_t d;
        uint32_t key=motor_lock();
        if(pending_generation!=command_generation) pending_reply[0]=0;
        pending_generation=command_generation;
        if(stop_reply) {
            strcpy(pending_reply,"CTRL OK STOP\r\n"); stop_reply=false;
        }
        if(!pending_reply[0] && rx_head!=rx_tail) {
            const rx_command_t *cmd=&rx_queue[rx_tail];
            if(!strcmp(cmd->line,"CTRL DIAG")) { diag_snapshot(&d); diag=true; }
            else (void)control_command_at(cmd->line,cmd->received_ms,HAL_GetTick(),
                                          pending_reply,sizeof(pending_reply));
            rx_tail=(rx_tail+1)%RX_QUEUE_SIZE;
            if(!strncmp(pending_reply,"CTRL OK MODE ",13) ||
               !strncmp(pending_reply,"CTRL OK MAP ",12)) {
                /* A configuration acknowledgement is a queue barrier. Commands
                 * received before it must never run under the new mapping. */
                rx_head=rx_tail=0;
                if(partial_len) discard_line=true;
                partial_len=0;
                ++command_generation;
                pending_generation=command_generation;
            }
        }
        if(rx_fault && !pending_reply[0]) {
            strcpy(pending_reply,"CTRL ERR RX_OVERFLOW_OR_INVALID\r\n"); rx_fault=false;
        }
        motor_unlock(key);
        if(diag) diag_format(&d,pending_reply,sizeof(pending_reply));
        /* STOP/session changes can interrupt formatting, so recheck before TX. */
        key=motor_lock();
        if(pending_generation!=command_generation || stop_reply) pending_reply[0]=0;
        if(pending_reply[0] && CDC_Write((uint8_t*)pending_reply,(uint16_t)strlen(pending_reply))==USBD_OK)
            pending_reply[0]=0;
        motor_unlock(key);
        if(HAL_GetTick()-report_at>=100U) { report_at=HAL_GetTick(); if(!pending_reply[0]) report(); }
    }
}
