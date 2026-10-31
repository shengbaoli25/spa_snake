#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H

#include <stddef.h>
#include <stdint.h>

#define CONTROL_MOTOR_COUNT 4u
/* This paired full-range build drives M3 + M4 only. */
#ifndef CONTROL_ACTIVE_MASK
#define CONTROL_ACTIVE_MASK 0x0Cu
#endif
/* Feed-forward: voltage-command units per commanded rpm, measured on the
 * bench (100 rpm needed ~5-7k units through the integral alone). Kp alone
 * cannot overcome stiction, so the loop used to crawl up then overshoot. */
#define CONTROL_FF_UNITS_PER_RPM 60
/* Soft start: the internal setpoint ramps at this rate instead of stepping,
 * so the feed-forward voltage cannot demand a large inrush current. 300 rpm
 * takes 0.75 s at 400 rpm/s. */
#define CONTROL_TARGET_SLEW_RPM_PER_S 400
/* Restore the project-wide command range; no extra diagnostic speed cap. */
#define CONTROL_SPEED_LIMIT 500
/* Full GM6020 voltage-command range, not measured current or an instruction
 * to output full power. PI/feed-forward still calculate the requested output. */
#define CONTROL_CURRENT_LIMIT 25000
#define CONTROL_MODE_BENCH 0u
#define CONTROL_MODE_PAIR 1u
#define CONTROL_PAIR_MASK 0x0Cu
#define CONTROL_OVERCURRENT_RAW 7000
#define CONTROL_TEMPERATURE_LIMIT 80u
#define CONTROL_FEEDBACK_TIMEOUT_MS 100u
#define CONTROL_COMMAND_TIMEOUT_MS 300u
#define CONTROL_COMMAND_MAX_LENGTH 192u

typedef enum {
    CONTROL_FAULT_NONE = 0,
    CONTROL_FAULT_STOP,
    CONTROL_FAULT_USB,
    CONTROL_FAULT_COMMAND_TIMEOUT,
    CONTROL_FAULT_FEEDBACK_STALE,
    CONTROL_FAULT_OVERCURRENT,
    CONTROL_FAULT_OVERTEMP,
    CONTROL_FAULT_BAD_COMMAND,
    CONTROL_FAULT_CAN,
    CONTROL_FAULT_FATAL
} control_reason_t;

typedef struct {
    uint16_t angle;
    int16_t speed;
    int16_t current;
    uint8_t temperature;
    uint8_t seen;
    uint32_t rx_count;
    uint32_t last_ms;
} control_feedback_t;

typedef struct {
    control_feedback_t feedback[CONTROL_MOTOR_COUNT];
    int16_t target[CONTROL_MOTOR_COUNT];
    int16_t output[CONTROL_MOTOR_COUNT];
    uint32_t last_command_ms;
    uint32_t tick_count;
    control_reason_t reason;
    uint8_t fault_latched;
    uint8_t link_open;
    uint8_t can_healthy;
    uint8_t active_mask;
    uint8_t fault_required_mask;
    uint8_t pair_mode;
    int8_t pair_dir3;
    int8_t pair_dir4;
    /* First genuine fault sample; STOP and later faults never overwrite it.
     * For non-motor faults, mask/current are zero. */
    uint8_t fault_motor_mask;
    int16_t fault_current_raw;
    uint32_t fault_ms;
} control_snapshot_t;

/* Pure C singleton. The HAL caller MUST serialize every API with IRQ critical
 * sections, including snapshots, commands and feedback. tick runs every 5 ms.
 * Motor indexes are zero based. No API sends CAN or blocks on USB. */
void control_init(void);
void control_set_link(uint8_t open);
/* Restored health never clears a latched fault or restores old targets. */
void control_set_can_health(uint8_t healthy);
void control_trip(control_reason_t reason);
void control_on_feedback(uint8_t index, uint16_t angle, int16_t speed,
                         int16_t current, uint8_t temperature, uint32_t now);
int control_command(const char *line, uint32_t now, char *reply, size_t reply_len);
/* received_ms is captured when the complete line arrives, not when dequeued. */
int control_command_at(const char *line, uint32_t received_ms, uint32_t now,
                       char *reply, size_t reply_len);
void control_tick(uint32_t now);
void control_get_outputs(int16_t output[CONTROL_MOTOR_COUNT]);
void control_snapshot(control_snapshot_t *snapshot);
const char *control_reason_string(control_reason_t reason);

#endif
