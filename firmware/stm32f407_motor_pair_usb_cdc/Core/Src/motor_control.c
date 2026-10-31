#include "motor_control.h"

#include <stdio.h>
#include <string.h>

static control_snapshot_t state;
/* Integral units are 1/20 raw voltage command; delta per 5 ms is error / 20.
 * This implements Ki * dt = 0.05 without floating point in an interrupt. */
static int32_t integral20[CONTROL_MOTOR_COUNT];
/* Ramped setpoint actually used by the loop (soft start). */
static int16_t ramped[CONTROL_MOTOR_COUNT];
/* Timestamp for callbacks whose legacy API has no now argument. The main
 * loop calls control_tick every 5 ms; motor faults use their exact RX time. */
static uint32_t last_observed_ms;

#if CONTROL_ACTIVE_MASK != CONTROL_PAIR_MASK
#error "Paired diagnostic controller requires M3/M4 active mask 0x0C"
#endif

const char *control_reason_string(control_reason_t reason)
{
    switch (reason) {
    case CONTROL_FAULT_NONE: return "NONE";
    case CONTROL_FAULT_STOP: return "STOP";
    case CONTROL_FAULT_USB: return "USB_CLOSED";
    case CONTROL_FAULT_COMMAND_TIMEOUT: return "COMMAND_TIMEOUT";
    case CONTROL_FAULT_FEEDBACK_STALE: return "FEEDBACK_STALE";
    case CONTROL_FAULT_OVERCURRENT: return "OVERCURRENT";
    case CONTROL_FAULT_OVERTEMP: return "OVERTEMP";
    case CONTROL_FAULT_BAD_COMMAND: return "BAD_COMMAND";
    case CONTROL_FAULT_CAN: return "CAN_ERROR";
    case CONTROL_FAULT_FATAL: return "FATAL";
    default: return "UNKNOWN";
    }
}

static void zero_outputs(void)
{
    unsigned i;
    state.active_mask = 0u;
    for (i = 0u; i < CONTROL_MOTOR_COUNT; ++i) {
        state.target[i] = 0;
        state.output[i] = 0;
        integral20[i] = 0;
        ramped[i] = 0;
    }
}

static void trip_sample(control_reason_t reason, uint8_t required,
                        uint8_t motor_mask, int16_t current_raw, uint32_t now)
{
    state.fault_required_mask |= (uint8_t)(required | state.active_mask);
    /* Keep the first fault as the cause. Repeated STOP never unlocks a fault. */
    if (!state.fault_latched) {
        state.reason = reason;
        state.fault_motor_mask = motor_mask;
        state.fault_current_raw = current_raw;
        state.fault_ms = now;
    }
    state.fault_latched = 1u;
    zero_outputs();
}

static void trip_required(control_reason_t reason, uint8_t required)
{
    trip_sample(reason, required, 0u, 0, last_observed_ms);
}

void control_init(void)
{
    memset(&state, 0, sizeof(state));
    memset(integral20, 0, sizeof(integral20));
    memset(ramped, 0, sizeof(ramped));
    last_observed_ms = 0u;
    state.pair_mode = CONTROL_MODE_PAIR;
    /* Colleague's command mapping. Physical orientation must be observed;
     * this is not a measured chassis roll cancellation calibration. */
    state.pair_dir3 = -1;
    state.pair_dir4 = 1;
}

void control_trip(control_reason_t reason)
{
    if (reason == CONTROL_FAULT_NONE || reason == CONTROL_FAULT_STOP) {
        reason = CONTROL_FAULT_FATAL;
    }
    trip_required(reason, 0u);
}

void control_set_link(uint8_t open)
{
    open = (uint8_t)(open != 0u);
    if (!open && (state.link_open || state.active_mask)) {
        trip_required(CONTROL_FAULT_USB, 0u);
    }
    state.link_open = open;
    if (!open) {
        zero_outputs();
    }
}

void control_set_can_health(uint8_t healthy)
{
    state.can_healthy = (uint8_t)(healthy != 0u);
    if (!state.can_healthy && state.active_mask) {
        trip_required(CONTROL_FAULT_CAN, 0u);
    }
}

static control_reason_t hazard(uint8_t required, uint32_t now,
                               uint8_t *motor_mask, int16_t *current_raw)
{
    unsigned i;
    *motor_mask = 0u;
    *current_raw = 0;
    required &= (uint8_t)CONTROL_ACTIVE_MASK;
    for (i = 0u; i < CONTROL_MOTOR_COUNT; ++i) {
        const control_feedback_t *f = &state.feedback[i];
        if (f->seen) {
            const int32_t current = f->current;
            if (current >= CONTROL_OVERCURRENT_RAW ||
                current <= -CONTROL_OVERCURRENT_RAW) {
                *motor_mask = (uint8_t)(1u << i);
                *current_raw = f->current;
                return CONTROL_FAULT_OVERCURRENT;
            }
            if (f->temperature >= CONTROL_TEMPERATURE_LIMIT) {
                *motor_mask = (uint8_t)(1u << i);
                *current_raw = f->current;
                return CONTROL_FAULT_OVERTEMP;
            }
        }
        if ((required & (1u << i)) &&
            (!f->seen || (uint32_t)(now - f->last_ms) >= CONTROL_FEEDBACK_TIMEOUT_MS)) {
            *motor_mask = (uint8_t)(1u << i);
            return CONTROL_FAULT_FEEDBACK_STALE;
        }
    }
    return CONTROL_FAULT_NONE;
}

void control_on_feedback(uint8_t index, uint16_t angle, int16_t speed,
                         int16_t current, uint8_t temperature, uint32_t now)
{
    control_feedback_t *f;
    if (index >= CONTROL_MOTOR_COUNT) {
        return;
    }
    last_observed_ms = now;
    f = &state.feedback[index];
    f->angle = angle;
    f->speed = speed;
    f->current = current;
    f->temperature = temperature;
    f->seen = 1u;
    f->last_ms = now;
    ++f->rx_count;
    if ((int32_t)current >= CONTROL_OVERCURRENT_RAW ||
        (int32_t)current <= -CONTROL_OVERCURRENT_RAW) {
        trip_sample(CONTROL_FAULT_OVERCURRENT, (uint8_t)(1u << index),
                    (uint8_t)(1u << index), current, now);
    } else if (temperature >= CONTROL_TEMPERATURE_LIMIT) {
        trip_sample(CONTROL_FAULT_OVERTEMP, (uint8_t)(1u << index),
                    (uint8_t)(1u << index), current, now);
    }
}

static int whitespace(char c)
{
    return c == ' ' || c == '\t';
}

static const char *skip_space(const char *p)
{
    while (whitespace(*p)) {
        ++p;
    }
    return p;
}

static int token(const char **p, const char *word)
{
    const size_t n = strlen(word);
    if (strncmp(*p, word, n) != 0 || ((*p)[n] != '\0' && !whitespace((*p)[n]))) {
        return 0;
    }
    *p = skip_space(*p + n);
    return 1;
}

static int parse_speeds(const char *p, int16_t target[CONTROL_MOTOR_COUNT], uint8_t *mask)
{
    uint8_t seen = 0u;
    unsigned i;
    *mask = 0u;
    while (*p) {
        unsigned channel;
        uint32_t magnitude = 0u;
        int sign = 1;
        int value;
        if (*p++ != 'M' || *p < '1' || *p > '6') {
            return 0;
        }
        channel = (unsigned)(*p++ - '1');
        if (*p++ != '=' || (seen & (1u << channel))) {
            return 0;
        }
        if (*p == '+' || *p == '-') {
            if (*p++ == '-') {
                sign = -1;
            }
        }
        if (*p < '0' || *p > '9') {
            return 0;
        }
        do {
            const unsigned digit = (unsigned)(*p++ - '0');
            /* Guard against both overflow and values above the command limit. */
            if (magnitude > (CONTROL_SPEED_LIMIT - (int)digit) / 10u) {
                return 0;
            }
            magnitude = magnitude * 10u + digit;
        } while (*p >= '0' && *p <= '9');
        if (*p != '\0' && !whitespace(*p)) {
            return 0;
        }
        value = sign * (int)magnitude;
        if (channel >= CONTROL_MOTOR_COUNT) {
            if (value != 0) {
                return 0;
            }
        } else {
            target[channel] = (int16_t)value;
            if (value != 0) {
                *mask |= (uint8_t)(1u << channel);
            }
        }
        seen |= (uint8_t)(1u << channel);
        p = skip_space(p);
    }
    for (i = 0u; i < CONTROL_MOTOR_COUNT; ++i) {
        if (!(CONTROL_ACTIVE_MASK & (1u << i))) {
            /* Physically absent channel: never drive it, ignore any value. */
            target[i] = 0;
            *mask &= (uint8_t)~(1u << i);
        }
    }
    /* Missing channels are explicitly zero, never carried over from an old
     * command. PAIR validates both nonzero channels below. All-zero is STOP. */
    return seen != 0u;
}

static int parse_pair(const char *p, int16_t target[CONTROL_MOTOR_COUNT], uint8_t *mask)
{
    unsigned magnitude = 0u;
    int sign = 1;
    if (*p == '+' || *p == '-') {
        if (*p++ == '-') { sign = -1; }
    }
    if (*p < '0' || *p > '9') { return 0; }
    do {
        unsigned digit = (unsigned)(*p++ - '0');
        if (magnitude > (CONTROL_SPEED_LIMIT - digit) / 10u) { return 0; }
        magnitude = magnitude * 10u + digit;
    } while (*p >= '0' && *p <= '9');
    if (*skip_space(p)) { return 0; }
    target[2] = (int16_t)(sign * (int)magnitude * state.pair_dir3);
    target[3] = (int16_t)(sign * (int)magnitude * state.pair_dir4);
    *mask = magnitude ? CONTROL_PAIR_MASK : 0u;
    return 1;
}

static int parse_map(const char *p, int8_t *dir3, int8_t *dir4)
{
    unsigned seen = 0u;
    while (*p) {
        unsigned channel;
        int sign = 1;
        if (*p++ != 'M' || (*p != '3' && *p != '4')) { return 0; }
        channel = (unsigned)(*p++ - '3');
        if (*p++ != '=' || (seen & (1u << channel))) { return 0; }
        if (*p == '+' || *p == '-') {
            if (*p++ == '-') { sign = -1; }
        }
        if (*p++ != '1' || (*p && !whitespace(*p))) { return 0; }
        if (channel == 0u) { *dir3 = (int8_t)sign; }
        else { *dir4 = (int8_t)sign; }
        seen |= 1u << channel;
        p = skip_space(p);
    }
    return seen == 3u;
}

static int configuration_stopped(uint32_t now)
{
    unsigned i;
    if (state.active_mask) { return 0; }
    for (i = 0u; i < CONTROL_MOTOR_COUNT; ++i) {
        if (state.target[i] || state.output[i] || ramped[i] || integral20[i]) { return 0; }
    }
    for (i = 2u; i < 4u; ++i) {
        const control_feedback_t *f = &state.feedback[i];
        if (!f->seen || (uint32_t)(now - f->last_ms) >= CONTROL_FEEDBACK_TIMEOUT_MS ||
            f->speed < -2 || f->speed > 2) { return 0; }
    }
    return 1;
}

static void reply_error(char *reply, size_t reply_len)
{
    if (reply && reply_len) {
        (void)snprintf(reply, reply_len, "CTRL ERR %s\r\n", control_reason_string(state.reason));
    }
}

int control_command_at(const char *line, uint32_t received_ms, uint32_t now,
                       char *reply, size_t reply_len)
{
    char buffer[CONTROL_COMMAND_MAX_LENGTH + 1u];
    size_t n = 0u;
    const char *p;
    int16_t target[CONTROL_MOTOR_COUNT] = {0, 0, 0, 0};
    uint8_t mask = 0u;
    uint8_t hazard_motor_mask;
    int16_t hazard_current;
    control_reason_t reason;
    unsigned i;
    last_observed_ms = now;
    if (reply && reply_len) {
        reply[0] = '\0';
    }
    if (!line) {
        goto bad_command;
    }
    while (n <= CONTROL_COMMAND_MAX_LENGTH && line[n]) {
        ++n;
    }
    if (n > CONTROL_COMMAND_MAX_LENGTH) {
        goto bad_command;
    }
    memcpy(buffer, line, n + 1u);
    /* Accept an optional line terminator, but never another embedded line. */
    if (n && buffer[n - 1u] == '\n') {
        buffer[--n] = '\0';
    }
    if (n && buffer[n - 1u] == '\r') {
        buffer[--n] = '\0';
    }
    p = skip_space(buffer);
    if (!token(&p, "CTRL")) {
        goto bad_command;
    }
    if (token(&p, "STOP") && !*p) {
        zero_outputs();
        if (!state.fault_latched) {
            state.reason = CONTROL_FAULT_STOP;
        }
        if (reply && reply_len) {
            (void)snprintf(reply, reply_len, "CTRL OK STOP\r\n");
        }
        return 1;
    }
    p = skip_space(buffer);
    (void)token(&p, "CTRL");
    if (token(&p, "STATUS") && !*p) {
        if (reply && reply_len) {
            (void)snprintf(reply, reply_len,
                "CTRL STATUS link=%u fault=%u reason=%s active=0x%02X M1=%d M2=%d M3=%d M4=%d mode=%s dir3=%d dir4=%d speed_limit=%d output_limit=%d fault_motor=0x%02X fault_current=%d fault_ms=%lu\r\n",
                state.link_open, state.fault_latched, control_reason_string(state.reason),
                (unsigned)CONTROL_ACTIVE_MASK,
                state.target[0], state.target[1], state.target[2], state.target[3],
                state.pair_mode == CONTROL_MODE_PAIR ? "PAIR" : "BENCH",
                state.pair_dir3, state.pair_dir4, CONTROL_SPEED_LIMIT, CONTROL_CURRENT_LIMIT,
                state.fault_motor_mask, state.fault_current_raw, (unsigned long)state.fault_ms);
        }
        return 1;
    }
    p = skip_space(buffer);
    (void)token(&p, "CTRL");
    if (token(&p, "CLEAR") && !*p) {
        zero_outputs();
        reason = hazard(state.fault_required_mask, now, &hazard_motor_mask, &hazard_current);
        if (!state.link_open || !state.can_healthy || reason != CONTROL_FAULT_NONE) {
            if (reason != CONTROL_FAULT_NONE) {
                trip_sample(reason, state.fault_required_mask, hazard_motor_mask, hazard_current, now);
            } else if (!state.link_open) {
                trip_required(CONTROL_FAULT_USB, 0u);
            } else {
                trip_required(CONTROL_FAULT_CAN, 0u);
            }
            reply_error(reply, reply_len);
            return 0;
        }
        state.fault_latched = 0u;
        state.fault_required_mask = 0u;
        state.reason = CONTROL_FAULT_NONE;
        state.fault_motor_mask = 0u;
        state.fault_current_raw = 0;
        state.fault_ms = 0u;
        if (reply && reply_len) {
            (void)snprintf(reply, reply_len, "CTRL OK CLEAR\r\n");
        }
        return 1;
    }
    p = skip_space(buffer);
    (void)token(&p, "CTRL");
    if (token(&p, "MODE")) {
        uint8_t mode;
        if (token(&p, "PAIR")) {
            if (*p) { goto bad_command; }
            mode = CONTROL_MODE_PAIR;
        }
        else if (token(&p, "BENCH")) {
            if (*p) { goto bad_command; }
            mode = CONTROL_MODE_BENCH;
        }
        else { goto bad_command; }
        if (!configuration_stopped(now)) { goto bad_command; }
        zero_outputs();
        state.pair_mode = mode;
        if (reply && reply_len) {
            (void)snprintf(reply, reply_len, "CTRL OK MODE %s\r\n",
                          mode == CONTROL_MODE_PAIR ? "PAIR" : "BENCH");
        }
        return 1;
    }
    p = skip_space(buffer);
    (void)token(&p, "CTRL");
    if (token(&p, "MAP")) {
        int8_t dir3 = 0, dir4 = 0;
        if (!parse_map(p, &dir3, &dir4) || !configuration_stopped(now)) { goto bad_command; }
        zero_outputs();
        state.pair_dir3 = dir3;
        state.pair_dir4 = dir4;
        if (reply && reply_len) {
            (void)snprintf(reply, reply_len, "CTRL OK MAP M3=%d M4=%d\r\n", dir3, dir4);
        }
        return 1;
    }
    p = skip_space(buffer);
    (void)token(&p, "CTRL");
    if (token(&p, "PAIR")) {
        if (!parse_pair(p, target, &mask)) { goto bad_command; }
    } else if (!token(&p, "SPEED") || !parse_speeds(p, target, &mask)) {
        goto bad_command;
    }
    if (mask && state.pair_mode == CONTROL_MODE_PAIR &&
        (mask != CONTROL_PAIR_MASK ||
         target[2] * state.pair_dir3 != target[3] * state.pair_dir4)) {
        goto bad_command;
    }
    if (!mask) {
        zero_outputs();
        state.last_command_ms = now;
        if (!state.fault_latched) {
            state.reason = CONTROL_FAULT_STOP;
        }
    } else {
        if ((uint32_t)(now - received_ms) >= CONTROL_COMMAND_TIMEOUT_MS) {
            trip_required(CONTROL_FAULT_COMMAND_TIMEOUT, mask);
            reply_error(reply, reply_len);
            return 0;
        }
        if (state.fault_latched) {
            zero_outputs();
            reply_error(reply, reply_len);
            return 0;
        }
        if (!state.link_open) {
            trip_required(CONTROL_FAULT_USB, mask);
            reply_error(reply, reply_len);
            return 0;
        }
        if (!state.can_healthy) {
            trip_required(CONTROL_FAULT_CAN, mask);
            reply_error(reply, reply_len);
            return 0;
        }
        reason = hazard(mask, now, &hazard_motor_mask, &hazard_current);
        if (reason != CONTROL_FAULT_NONE) {
            trip_sample(reason, mask, hazard_motor_mask, hazard_current, now);
            reply_error(reply, reply_len);
            return 0;
        }
        for (i = 0u; i < CONTROL_MOTOR_COUNT; ++i) {
            /* A newly inactive or sign-reversed channel loses old integral. */
            if (!target[i] || (int32_t)target[i] * state.target[i] <= 0) {
                integral20[i] = 0;
                state.output[i] = 0;
                ramped[i] = 0;
            }
            state.target[i] = target[i];
        }
        state.active_mask = mask;
        state.last_command_ms = received_ms;
        state.reason = CONTROL_FAULT_NONE;
    }
    if (reply && reply_len) {
        (void)snprintf(reply, reply_len, "CTRL OK M1=%d M2=%d M3=%d M4=%d\r\n",
            target[0], target[1], target[2], target[3]);
    }
    return 1;

bad_command:
    trip_required(CONTROL_FAULT_BAD_COMMAND, 0u);
    reply_error(reply, reply_len);
    return 0;
}

int control_command(const char *line, uint32_t now, char *reply, size_t reply_len)
{
    return control_command_at(line, now, now, reply, reply_len);
}

void control_tick(uint32_t now)
{
    unsigned i;
    control_reason_t reason;
    uint8_t hazard_motor_mask;
    int16_t hazard_current;
    last_observed_ms = now;
    ++state.tick_count;
    if (state.fault_latched || !state.link_open || !state.active_mask) {
        zero_outputs();
        return;
    }
    if (!state.can_healthy) {
        trip_required(CONTROL_FAULT_CAN, 0u);
        return;
    }
    if ((uint32_t)(now - state.last_command_ms) >= CONTROL_COMMAND_TIMEOUT_MS) {
        trip_required(CONTROL_FAULT_COMMAND_TIMEOUT, 0u);
        return;
    }
    reason = hazard(state.active_mask, now, &hazard_motor_mask, &hazard_current);
    if (reason != CONTROL_FAULT_NONE) {
        trip_sample(reason, 0u, hazard_motor_mask, hazard_current, now);
        return;
    }
    for (i = 0u; i < CONTROL_MOTOR_COUNT; ++i) {
        int32_t error, proportional, feedforward, candidate_i, candidate_output, output;
        int32_t setpoint, step;
        if (!state.target[i] || !(CONTROL_ACTIVE_MASK & (1u << i))) {
            state.output[i] = 0;
            integral20[i] = 0;
            ramped[i] = 0;
            continue;
        }
        setpoint = (int32_t)state.target[i];
        step = CONTROL_TARGET_SLEW_RPM_PER_S / 200; /* 5 ms tick */
        if (step < 1) { step = 1; }
        if ((int32_t)ramped[i] * setpoint < 0) { ramped[i] = 0; }
        if ((int32_t)ramped[i] < setpoint) {
            int32_t next = (int32_t)ramped[i] + step;
            ramped[i] = (int16_t)(next > setpoint ? setpoint : next);
        } else if ((int32_t)ramped[i] > setpoint) {
            int32_t next = (int32_t)ramped[i] - step;
            ramped[i] = (int16_t)(next < setpoint ? setpoint : next);
        }
        error = (int32_t)ramped[i] - state.feedback[i].speed;
        feedforward = (int32_t)ramped[i] * CONTROL_FF_UNITS_PER_RPM;
        proportional = error * 10;
        candidate_i = integral20[i] + error;
        if (candidate_i > CONTROL_CURRENT_LIMIT * 20) {
            candidate_i = CONTROL_CURRENT_LIMIT * 20;
        } else if (candidate_i < -CONTROL_CURRENT_LIMIT * 20) {
            candidate_i = -CONTROL_CURRENT_LIMIT * 20;
        }
        candidate_output = feedforward + proportional + candidate_i / 20;
        /* Conditional integration: permit integral to unwind saturation. */
        if (!((candidate_output > CONTROL_CURRENT_LIMIT && error > 0) ||
              (candidate_output < -CONTROL_CURRENT_LIMIT && error < 0))) {
            integral20[i] = candidate_i;
        }
        output = feedforward + proportional + integral20[i] / 20;
        if (output > CONTROL_CURRENT_LIMIT) {
            output = CONTROL_CURRENT_LIMIT;
        } else if (output < -CONTROL_CURRENT_LIMIT) {
            output = -CONTROL_CURRENT_LIMIT;
        }
        state.output[i] = (int16_t)output;
    }
}

void control_get_outputs(int16_t output[CONTROL_MOTOR_COUNT])
{
    if (output) {
        memcpy(output, state.output, sizeof(state.output));
    }
}

void control_snapshot(control_snapshot_t *snapshot)
{
    if (snapshot) {
        *snapshot = state;
    }
}
