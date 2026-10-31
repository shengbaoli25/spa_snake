#include "motor_control.h"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static unsigned checks;
static char reply[384];
#define CHECK(x) do { ++checks; assert(x); } while (0)

static control_snapshot_t snap(void)
{
    control_snapshot_t s;
    control_snapshot(&s);
    return s;
}

static int cmd(const char *line, uint32_t now)
{
    return control_command(line, now, reply, sizeof(reply));
}

static void fresh(uint32_t now)
{
    control_on_feedback(2u, 100u, 0, 0, 25u, now);
    control_on_feedback(3u, 200u, 0, 0, 25u, now);
}

static void ready(uint32_t now)
{
    control_init();
    control_set_link(1u);
    control_set_can_health(1u);
    fresh(now);
}

static void zero(void)
{
    unsigned i;
    int16_t outputs[4];
    control_get_outputs(outputs);
    for (i = 0u; i < 4u; ++i) {
        CHECK(outputs[i] == 0);
        CHECK(snap().target[i] == 0);
    }
    CHECK(snap().active_mask == 0u);
}

static void fault(control_reason_t reason)
{
    CHECK(snap().fault_latched);
    CHECK(snap().reason == reason);
    zero();
}

static void defaults_and_pair(void)
{
    ready(100u);
    CHECK(snap().pair_mode == CONTROL_MODE_PAIR);
    CHECK(snap().pair_dir3 == -1 && snap().pair_dir4 == 1);
    CHECK(CONTROL_SPEED_LIMIT == 500 && CONTROL_CURRENT_LIMIT == 25000);
    CHECK(cmd("CTRL PAIR 20", 100u));
    CHECK(snap().target[2] == -20 && snap().target[3] == 20);
    control_tick(105u);
    CHECK(snap().output[2] < 0 && snap().output[3] > 0);
    CHECK(snap().output[0] == 0 && snap().output[1] == 0);
    CHECK(cmd("CTRL STATUS", 105u));
    CHECK(strstr(reply, "mode=PAIR") != NULL);
    CHECK(strstr(reply, "speed_limit=500 ") != NULL);
    CHECK(strstr(reply, "output_limit=25000 ") != NULL);
    CHECK(cmd("CTRL PAIR -20", 106u));
    control_tick(110u);
    CHECK(snap().output[2] > 0 && snap().output[3] < 0);
    CHECK(cmd("CTRL SPEED M3=0", 110u));
    zero();
    CHECK(cmd("CTRL PAIR +20\r\n", 110u));
    CHECK(cmd("CTRL PAIR 0", 110u));
    zero();
}

static void pair_rejections(void)
{
    static const char *bad[] = {
        "CTRL SPEED M3=20 M4=20", "CTRL SPEED M3=-20 M4=19",
        "CTRL SPEED M3=-20", "CTRL SPEED M4=20", "CTRL SPEED M3=0 M4=20",
        "CTRL SPEED M3=-20 M4=0", "CTRL PAIR 501", "CTRL PAIR -501",
        "CTRL PAIR 4294967296", "CTRL PAIR 1 2", "CTRL PAIR +",
        "CTRL SPEED M3=-501 M4=501", "CTRL SPEED M3=501 M4=-501",
        "CTRL SPEED M3=-20 M3=-20 M4=20",
        "CTRL PAIR 20\nCTRL STOP", "CTRL PAIR", "CTRL SPEED"
    };
    unsigned i;
    for (i = 0u; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        ready(10u);
        CHECK(cmd("CTRL PAIR 20", 10u));
        control_tick(10u);
        CHECK(!cmd(bad[i], 11u));
        fault(CONTROL_FAULT_BAD_COMMAND);
        CHECK(cmd("CTRL PAIR 0", 12u));
        fault(CONTROL_FAULT_BAD_COMMAND);
    }
}

static void full_range_boundaries(void)
{
    static const struct {
        const char *command;
        int16_t m3;
        int16_t m4;
    } accepted[] = {
        { "CTRL PAIR 500", -500, 500 },
        { "CTRL PAIR -500", 500, -500 },
        { "CTRL SPEED M3=-500 M4=500", -500, 500 },
        { "CTRL SPEED M3=500 M4=-500", 500, -500 }
    };
    unsigned i;
    for (i = 0u; i < sizeof(accepted) / sizeof(accepted[0]); ++i) {
        ready(100u);
        CHECK(cmd(accepted[i].command, 100u));
        CHECK(snap().target[2] == accepted[i].m3);
        CHECK(snap().target[3] == accepted[i].m4);
        CHECK(snap().target[0] == 0 && snap().target[1] == 0);
        control_tick(105u);
        CHECK(!snap().fault_latched);
        CHECK(cmd("CTRL STOP", 106u));
        zero();
    }
    ready(100u);
    CHECK(cmd("CTRL MODE BENCH", 100u));
    CHECK(cmd("CTRL SPEED M3=500", 100u));
    CHECK(snap().target[2] == 500 && snap().target[3] == 0);
    CHECK(cmd("CTRL SPEED M4=-500", 101u));
    CHECK(snap().target[2] == 0 && snap().target[3] == -500);
    CHECK(!cmd("CTRL SPEED M3=501", 102u));
    fault(CONTROL_FAULT_BAD_COMMAND);
    CHECK(cmd("CTRL CLEAR", 102u));
    CHECK(!cmd("CTRL SPEED M4=-501", 103u));
    fault(CONTROL_FAULT_BAD_COMMAND);
}

static void modes_and_maps(void)
{
    static const char *bad[] = {
        "CTRL MAP M3=0 M4=1", "CTRL MAP M3=-1", "CTRL MAP M4=1",
        "CTRL MAP M3=2 M4=1", "CTRL MAP M3=-1 M3=1 M4=1",
        "CTRL MAP M3=-01 M4=1", "CTRL MAP M3=-1 M4=1 M1=1",
        "CTRL MODE PAIR BENCH", "CTRL MODE BENCH PAIR", "CTRL MODE",
        "CTRL MAP M3=-1 M4=1junk", "CTRL MAP M3=-1 M4="
    };
    unsigned i;
    ready(100u);
    CHECK(cmd("CTRL MODE BENCH", 100u));
    CHECK(snap().pair_mode == CONTROL_MODE_BENCH);
    CHECK(cmd("CTRL SPEED M3=20", 100u));
    control_tick(105u);
    CHECK(snap().output[2] > 0 && snap().output[3] == 0);
    CHECK(!cmd("CTRL MODE PAIR", 105u));
    fault(CONTROL_FAULT_BAD_COMMAND);
    CHECK(snap().pair_mode == CONTROL_MODE_BENCH);
    CHECK(cmd("CTRL CLEAR", 105u));
    CHECK(cmd("CTRL MAP M4=-1 M3=+1", 105u));
    CHECK(snap().pair_dir3 == 1 && snap().pair_dir4 == -1);
    CHECK(cmd("CTRL MODE PAIR", 105u));
    CHECK(cmd("CTRL PAIR 20", 105u));
    CHECK(snap().target[2] == 20 && snap().target[3] == -20);
    CHECK(cmd("CTRL STOP", 106u));
    CHECK(cmd("CTRL MAP M3=1 M4=1", 106u));
    CHECK(cmd("CTRL PAIR -10", 106u));
    CHECK(snap().target[2] == -10 && snap().target[3] == -10);
    for (i = 0u; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        ready(100u);
        CHECK(!cmd(bad[i], 100u));
        fault(CONTROL_FAULT_BAD_COMMAND);
        CHECK(snap().pair_mode == CONTROL_MODE_PAIR);
        CHECK(snap().pair_dir3 == -1 && snap().pair_dir4 == 1);
    }
    ready(100u);
    control_on_feedback(3u, 200u, 3, 0, 25u, 100u);
    CHECK(!cmd("CTRL MODE BENCH", 100u));
    ready(100u);
    control_on_feedback(2u, 100u, -3, 0, 25u, 100u);
    CHECK(!cmd("CTRL MAP M3=1 M4=-1", 100u));
    ready(100u);
    CHECK(!cmd("CTRL MODE BENCH", 200u));
    ready(100u);
    control_on_feedback(2u, 100u, -2, 0, 25u, 100u);
    control_on_feedback(3u, 200u, 2, 0, 25u, 100u);
    CHECK(cmd("CTRL MODE BENCH", 100u));
    zero();
}

static void saturation_and_inactive(void)
{
    uint32_t t;
    unsigned i;
    int sign;
    for (sign = -1; sign <= 1; sign += 2) {
        ready(0u);
        for (t = 0u; t <= 2000u; t += 5u) {
            /* Simulate opposing measured speed while remaining immediately
             * below the independent raw-current trip. This exercises true
             * positive/negative output saturation, not just numeric bounds. */
            control_on_feedback(2u, 100u, (int16_t)(sign * 800),
                                (int16_t)(sign * 6999), 25u, t);
            control_on_feedback(3u, 200u, (int16_t)(-sign * 800),
                                (int16_t)(-sign * 6999), 25u, t);
            if (t % 100u == 0u) {
                CHECK(cmd(sign > 0 ? "CTRL PAIR 500" : "CTRL PAIR -500", t));
            }
            control_tick(t);
            for (i = 0u; i < 4u; ++i) {
                CHECK(snap().output[i] >= -25000 && snap().output[i] <= 25000);
            }
            CHECK(snap().target[0] == 0 && snap().target[1] == 0);
            CHECK(snap().output[0] == 0 && snap().output[1] == 0);
            CHECK(!snap().fault_latched);
        }
        CHECK(snap().output[2] == -sign * 25000 && snap().output[3] == sign * 25000);
        /* Full-scale voltage allowance must not weaken the 7000 raw-current
         * trip: the exact boundary immediately stops BOTH outputs. */
        control_on_feedback(3u, 200u, 0, (int16_t)(sign * 7000), 25u, 2001u);
        fault(CONTROL_FAULT_OVERCURRENT);
        CHECK(snap().fault_motor_mask == 8u && snap().fault_current_raw == sign * 7000);
    }
    ready(100u);
    CHECK(cmd("CTRL SPEED M1=50 M2=-50 M3=-20 M4=20 M5=0 M6=0", 100u));
    control_tick(105u);
    CHECK(snap().target[0] == 0 && snap().target[1] == 0);
    CHECK(snap().output[0] == 0 && snap().output[1] == 0);
}

static void protection_and_first_fault(void)
{
    ready(0u);
    CHECK(cmd("CTRL PAIR 20", 0u));
    control_on_feedback(2u, 0u, 0, 0, 25u, 99u);
    control_tick(100u);
    fault(CONTROL_FAULT_FEEDBACK_STALE);
    CHECK(snap().fault_motor_mask == 8u);
    ready(100u);
    CHECK(cmd("CTRL PAIR 20", 100u));
    control_on_feedback(3u, 100u, 0, -7100, 25u, 107u);
    fault(CONTROL_FAULT_OVERCURRENT);
    CHECK(snap().fault_motor_mask == 8u && snap().fault_current_raw == -7100);
    CHECK(snap().fault_ms == 107u);
    control_on_feedback(2u, 100u, 0, 7400, 25u, 108u);
    CHECK(cmd("CTRL STOP", 109u));
    control_set_link(0u);
    CHECK(snap().fault_motor_mask == 8u && snap().fault_current_raw == -7100);
    CHECK(snap().fault_ms == 107u && snap().reason == CONTROL_FAULT_OVERCURRENT);
    CHECK(!cmd("CTRL CLEAR", 110u));
    control_set_link(1u);
    fresh(111u);
    CHECK(cmd("CTRL CLEAR", 111u));
    CHECK(snap().fault_motor_mask == 0u && snap().fault_current_raw == 0 && snap().fault_ms == 0u);
    zero();
    CHECK(cmd("CTRL PAIR 20", 111u));
    control_on_feedback(2u, 100u, 0, 13, 80u, 113u);
    fault(CONTROL_FAULT_OVERTEMP);
    CHECK(snap().fault_motor_mask == 4u && snap().fault_current_raw == 13 && snap().fault_ms == 113u);
    ready(100u);
    CHECK(cmd("CTRL PAIR 20", 100u));
    control_set_link(0u);
    fault(CONTROL_FAULT_USB);
    control_set_link(1u);
    CHECK(!cmd("CTRL PAIR 20", 101u));
    fault(CONTROL_FAULT_USB);
    CHECK(cmd("CTRL CLEAR", 101u));
    zero();
    CHECK(cmd("CTRL PAIR 20", 101u));
    control_set_can_health(0u);
    fault(CONTROL_FAULT_CAN);
    control_set_can_health(1u);
    CHECK(!cmd("CTRL PAIR 20", 102u));
    fault(CONTROL_FAULT_CAN);
    CHECK(cmd("CTRL CLEAR", 102u));
    zero();
    CHECK(cmd("CTRL PAIR 20", 102u));
    CHECK(cmd("CTRL STOP", 102u));
    CHECK(!snap().fault_latched && snap().reason == CONTROL_FAULT_STOP);
    zero();
    control_tick(500u);
    zero();
}

static void leases_and_parser_bounds(void)
{
    char oversized[CONTROL_COMMAND_MAX_LENGTH + 3u];
    ready(1000u);
    CHECK(!control_command_at("CTRL PAIR 20", 700u, 1000u, reply, sizeof(reply)));
    fault(CONTROL_FAULT_COMMAND_TIMEOUT);
    ready(1000u);
    CHECK(control_command_at("CTRL PAIR 20", 701u, 1000u, reply, sizeof(reply)));
    control_tick(1001u);
    fault(CONTROL_FAULT_COMMAND_TIMEOUT);
    ready(UINT32_MAX - 10u);
    CHECK(cmd("CTRL PAIR 20", UINT32_MAX - 10u));
    fresh(10u);
    control_tick(10u);
    CHECK(!snap().fault_latched);
    fresh(289u);
    control_tick(289u);
    fault(CONTROL_FAULT_COMMAND_TIMEOUT);
    ready(100u);
    memset(oversized, 'A', sizeof(oversized));
    oversized[sizeof(oversized) - 1u] = '\0';
    CHECK(!cmd(oversized, 100u));
    fault(CONTROL_FAULT_BAD_COMMAND);
    ready(100u);
    CHECK(!cmd(NULL, 100u));
    fault(CONTROL_FAULT_BAD_COMMAND);
    CHECK(control_command("CTRL STOP", 100u, NULL, 0u));
}

int main(void)
{
    defaults_and_pair();
    pair_rejections();
    full_range_boundaries();
    modes_and_maps();
    saturation_and_inactive();
    protection_and_first_fault();
    leases_and_parser_bounds();
    printf("pair control: %u checks passed\n", checks);
    return 0;
}
