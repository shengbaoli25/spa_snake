#include "motor_control.h"

/* Some optimized host compiler modes define NDEBUG. These tests must still
 * evaluate every command and assertion in that configuration. */
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

static char reply[256];
static unsigned checks;

#define CHECK(condition) do { ++checks; assert(condition); } while (0)

static control_snapshot_t snapshot(void)
{
    control_snapshot_t s;
    control_snapshot(&s);
    return s;
}

static void zero(void)
{
    unsigned i;
    int16_t out[4] = {1, 1, 1, 1};
    control_get_outputs(out);
    for (i = 0; i < 4; ++i) {
        CHECK(out[i] == 0);
        CHECK(snapshot().target[i] == 0);
    }
}

static int command(const char *line, uint32_t now)
{
    const int accepted = control_command(line, now, reply, sizeof(reply));
    CHECK(reply[sizeof(reply) - 1] == '\0' || strlen(reply) < sizeof(reply));
    return accepted;
}

static int command_at(const char *line, uint32_t received_ms, uint32_t now)
{
    const int accepted = control_command_at(line, received_ms, now, reply, sizeof(reply));
    CHECK(reply[sizeof(reply) - 1] == '\0' || strlen(reply) < sizeof(reply));
    return accepted;
}

static void feedback_all(uint32_t now)
{
    unsigned i;
    for (i = 0; i < 4; ++i) {
        control_on_feedback((uint8_t)i, (uint16_t)(1000 + i), 0, 0, 25, now);
    }
}

static void ready(uint32_t now)
{
    control_init();
    control_set_link(1);
    control_set_can_health(1);
    feedback_all(now);
}

static void start(uint32_t now)
{
    CHECK(command("CTRL SPEED M1=50 M2=-50 M3=50 M4=-50 M5=0 M6=0", now));
    control_tick(now);
    CHECK(snapshot().output[0] > 0);
    CHECK(snapshot().output[1] < 0);
}

static void expect_fault(control_reason_t reason)
{
    (void)reason;
    CHECK(snapshot().fault_latched);
    CHECK(snapshot().reason == reason);
    zero();
}

static void test_startup_and_link(void)
{
    control_init();
    control_tick(500);
    zero();
    CHECK(!snapshot().fault_latched);
    CHECK(!snapshot().link_open);
    feedback_all(500);
    CHECK(command("CTRL STOP", 500));
    CHECK(!command("CTRL SPEED M1=1 M2=0 M3=0 M4=0", 500));
    expect_fault(CONTROL_FAULT_USB);
    CHECK(!command("CTRL CLEAR", 500));
    control_set_link(1);
    control_set_can_health(1);
    CHECK(command("CTRL CLEAR", 500));
    CHECK(!snapshot().fault_latched);
    CHECK(command("CTRL SPEED M1=1 M2=0 M3=0 M4=0", 500));
    control_tick(505);
    CHECK(snapshot().output[0] > 0);
    CHECK(snapshot().output[1] == 0);
    control_set_link(0);
    expect_fault(CONTROL_FAULT_USB);
    control_set_link(1);
    CHECK(!command("CTRL SPEED M1=1 M2=0 M3=0 M4=0", 505));
    CHECK(command("CTRL STOP", 505));
    CHECK(snapshot().fault_latched);
    CHECK(command("CTRL CLEAR", 505));
    zero();
}

static void test_speed_pi_and_zero(void)
{
    unsigned i;
    ready(0);
    CHECK(command("  CTRL\tSPEED M4=-500 M2=+500 M1=500 M3=-500\r\n", 0));
    control_tick(5);
    CHECK(snapshot().output[0] == 2000);
    CHECK(snapshot().output[3] == -2000);
    CHECK(command("CTRL STOP", 5));
    zero();
    CHECK(command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 5));
    for (i = 0; i < 10; ++i) {
        control_tick(5 + i * 5);
    }
    CHECK(snapshot().output[0] == 105);
    CHECK(command("CTRL STOP", 50));
    CHECK(command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 50));
    control_tick(55);
    CHECK(snapshot().output[0] == 100);
    CHECK(command("CTRL SPEED M1=0 M2=10 M3=0 M4=0", 55));
    control_tick(60);
    CHECK(snapshot().output[0] == 0);
    CHECK(snapshot().output[1] == 100);
    CHECK(command("CTRL SPEED M1=0 M2=0 M3=0 M4=0", 60));
    zero();
    CHECK(!snapshot().fault_latched);
    CHECK(command("CTRL STATUS", 60));
    CHECK(strstr(reply, "CTRL STATUS") != NULL);
}

static void test_malformed(void)
{
    const char *bad[] = {
        "", "CTRL", "CTRL SPEED", "CTRL SPEED M1=1 M2=2 M3=3",
        "CTRL SPEED M1=1 M2=2 M3=3 M4=4 M1=2",
        "CTRL SPEED M1=501 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=600 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=-501 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=9999999999999999999999999999999 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=4294967296 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0 M5=1",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0 M6=-1",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0 M7=0",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0 M5=0 M5=0",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0 rubbish",
        "CTRL SPEED M1=1M2=0 M3=0 M4=0",
        "CTRL SPEED M1=1.5 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=+ M2=0 M3=0 M4=0",
        "CTRL SPEED M1=--1 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0\nCTRL STOP",
        "CTRL CURRENT M1=1 M2=0 M3=0 M4=0",
        "CTRL STOP NOW", "CTRL CLEAR NOW", "CTRL STATUS EXTRA",
        "CTRL SPEED M=1", "CTRL SPEED M1", "CTRL SPEED M1=",
        "CTRL SPEED M01=1 M2=0 M3=0 M4=0",
        "CTRL SPEED M1=1 M2=0 M3=0 M4=0\r\r\n"
    };
    unsigned i;
    char longline[CONTROL_COMMAND_MAX_LENGTH + 3];
    for (i = 0; i < sizeof(bad) / sizeof(bad[0]); ++i) {
        ready(0);
        start(0);
        CHECK(!command(bad[i], 1));
        expect_fault(CONTROL_FAULT_BAD_COMMAND);
        CHECK(!command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 2));
        CHECK(command("CTRL STOP", 2));
        CHECK(snapshot().fault_latched);
        CHECK(command("CTRL CLEAR", 2));
    }
    memset(longline, 'X', sizeof(longline));
    longline[sizeof(longline) - 1] = '\0';
    ready(0);
    CHECK(!command(longline, 0));
    expect_fault(CONTROL_FAULT_BAD_COMMAND);
    ready(0);
    CHECK(!control_command(NULL, 0, NULL, 0));
    expect_fault(CONTROL_FAULT_BAD_COMMAND);
}

static void test_command_deadline(void)
{
    ready(1000);
    start(1000);
    feedback_all(1299);
    control_tick(1299);
    CHECK(!snapshot().fault_latched);
    CHECK(command("CTRL STATUS", 1299));
    control_tick(1300);
    expect_fault(CONTROL_FAULT_COMMAND_TIMEOUT);
    feedback_all(1300);
    CHECK(command("CTRL CLEAR", 1300));
    start(1300);
    feedback_all(1599);
    CHECK(command("CTRL SPEED M1=50 M2=-50 M3=50 M4=-50", 1599));
    control_tick(1600);
    CHECK(!snapshot().fault_latched);
    CHECK(snapshot().output[0] > 0);
}

static void test_can_health_gate_and_recovery(void)
{
    control_init();
    control_set_link(1);
    feedback_all(0);
    CHECK(!snapshot().can_healthy);
    control_set_can_health(0);
    CHECK(!snapshot().fault_latched); /* Waiting at rest is not itself a fault. */
    CHECK(!command("CTRL CLEAR", 0));
    expect_fault(CONTROL_FAULT_CAN);
    control_set_can_health(1);
    CHECK(snapshot().can_healthy);
    CHECK(snapshot().fault_latched);
    control_tick(5);
    zero();
    CHECK(!command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 5));
    CHECK(command("CTRL CLEAR", 5));
    zero();
    start(5);
    control_set_can_health(0);
    expect_fault(CONTROL_FAULT_CAN); /* Health loss stops immediately, before tick. */
    CHECK(snapshot().fault_required_mask == 0x0fu);
    CHECK(!command("CTRL CLEAR", 5));
    control_set_can_health(1);
    control_tick(10);
    expect_fault(CONTROL_FAULT_CAN);
    CHECK(command("CTRL CLEAR", 10));
    control_tick(10);
    zero(); /* Even explicit recovery needs a new SPEED command. */

    control_init();
    control_set_link(1);
    feedback_all(0);
    CHECK(!command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 0));
    expect_fault(CONTROL_FAULT_CAN);
    CHECK(snapshot().fault_required_mask == 1u);

    ready(0);
    start(0);
    control_on_feedback(0, 1, 0, CONTROL_OVERCURRENT_RAW, 25, 1);
    control_set_can_health(0);
    expect_fault(CONTROL_FAULT_OVERCURRENT); /* CAN cannot overwrite first fault. */
}

static void test_received_command_deadline(void)
{
    ready(1299);
    CHECK(command_at("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 1000, 1299));
    CHECK(snapshot().last_command_ms == 1000);
    control_tick(1299);
    CHECK(snapshot().output[0] > 0);
    control_tick(1300); /* Processing delay must not extend the original lease. */
    expect_fault(CONTROL_FAULT_COMMAND_TIMEOUT);

    ready(1000);
    start(1000);
    feedback_all(1300);
    CHECK(!command_at("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 1000, 1300));
    expect_fault(CONTROL_FAULT_COMMAND_TIMEOUT); /* Exactly 300 ms is too old. */

    ready(149);
    CHECK(command_at("CTRL SPEED M1=10 M2=0 M3=0 M4=0", UINT32_MAX - 149u, 149));
    CHECK(snapshot().last_command_ms == UINT32_MAX - 149u);
    control_tick(149);
    CHECK(snapshot().output[0] > 0);
    control_tick(150);
    expect_fault(CONTROL_FAULT_COMMAND_TIMEOUT);

    ready(150);
    CHECK(!command_at("CTRL SPEED M1=10 M2=0 M3=0 M4=0", UINT32_MAX - 149u, 150));
    expect_fault(CONTROL_FAULT_COMMAND_TIMEOUT);

    ready(1000);
    start(1000);
    CHECK(command_at("CTRL SPEED M1=0 M2=0 M3=0 M4=0", 0, 1000));
    zero(); /* An old stop command still stops a currently active motor. */
    CHECK(!snapshot().fault_latched);
    control_set_can_health(0);
    CHECK(command_at("CTRL SPEED M1=0 M2=0 M3=0 M4=0", 0, 1000));
    CHECK(command_at("CTRL STOP", 0, 1000));
    zero();
    CHECK(!snapshot().fault_latched);
    CHECK(!command("CTRL CLEAR", 1000));
    CHECK(command_at("CTRL SPEED M1=0 M2=0 M3=0 M4=0", 0, 1000));
    CHECK(command_at("CTRL STOP", 0, 1000));
    expect_fault(CONTROL_FAULT_CAN); /* Stop never unlocks a latched failure. */
}

static void test_feedback_deadline_and_required_mask(void)
{
    control_init();
    control_set_link(1);
    control_set_can_health(1);
    control_on_feedback(0, 10, 0, 0, 25, 1000);
    CHECK(command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 1000));
    control_tick(1099);
    CHECK(!snapshot().fault_latched);
    control_tick(1100);
    expect_fault(CONTROL_FAULT_FEEDBACK_STALE);
    CHECK(!command("CTRL CLEAR", 1100));
    control_on_feedback(0, 11, 0, 0, 25, 1100);
    CHECK(command("CTRL CLEAR", 1100));
    CHECK(!command("CTRL SPEED M1=0 M2=10 M3=0 M4=0", 1100));
    expect_fault(CONTROL_FAULT_FEEDBACK_STALE);
    CHECK(!command("CTRL CLEAR", 1100));
    control_on_feedback(1, 12, 0, 0, 25, 1100);
    CHECK(command("CTRL CLEAR", 1100));
    zero();
}

static void test_current_temperature_and_clear(void)
{
    ready(0);
    start(0);
    control_on_feedback(2, 10, 0, 6999, 79, 1);
    CHECK(!snapshot().fault_latched);
    control_on_feedback(2, 10, 0, 7000, 79, 2);
    expect_fault(CONTROL_FAULT_OVERCURRENT);
    CHECK(!command("CTRL CLEAR", 2));
    feedback_all(2);
    CHECK(command("CTRL CLEAR", 2));
    CHECK(!snapshot().fault_latched);
    CHECK(command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 2));
    control_tick(3);
    /* Even a nonactive motor causes immediate stop when current is unsafe. */
    control_on_feedback(3, 10, 0, -7000, 25, 3);
    expect_fault(CONTROL_FAULT_OVERCURRENT);
    feedback_all(3);
    CHECK(command("CTRL CLEAR", 3));
    CHECK(command("CTRL SPEED M1=10 M2=0 M3=0 M4=0", 3));
    control_on_feedback(3, 10, 0, 0, 80, 4);
    expect_fault(CONTROL_FAULT_OVERTEMP);
    CHECK(!command("CTRL CLEAR", 4));
    control_on_feedback(3, 10, 0, 0, 79, 4);
    CHECK(command("CTRL CLEAR", 4));
    control_on_feedback(0, 10, 0, INT16_MIN, 25, 5);
    expect_fault(CONTROL_FAULT_OVERCURRENT);
}

static void test_faults_and_time_wrap(void)
{
    control_reason_t reasons[] = {CONTROL_FAULT_CAN, CONTROL_FAULT_FATAL, CONTROL_FAULT_USB};
    unsigned i;
    for (i = 0; i < sizeof(reasons) / sizeof(reasons[0]); ++i) {
        ready(0);
        start(0);
        control_trip(reasons[i]);
        expect_fault(reasons[i]);
        CHECK(!command("CTRL SPEED M1=1 M2=0 M3=0 M4=0", 0));
        CHECK(command("CTRL CLEAR", 0));
    }
    ready(UINT32_MAX - 49u);
    start(UINT32_MAX - 49u);
    control_tick(49u); /* elapsed 99 ms across wrap */
    CHECK(!snapshot().fault_latched);
    control_tick(50u); /* elapsed 100 ms */
    expect_fault(CONTROL_FAULT_FEEDBACK_STALE);
    ready(UINT32_MAX - 149u);
    start(UINT32_MAX - 149u);
    feedback_all(149u);
    control_tick(149u); /* elapsed 299 ms */
    CHECK(!snapshot().fault_latched);
    control_tick(150u);
    expect_fault(CONTROL_FAULT_COMMAND_TIMEOUT);
}

static void test_saturation_and_no_stored_braking(void)
{
    unsigned i;
    ready(0);
    CHECK(command("CTRL SPEED M1=500 M2=-500 M3=0 M4=0", 0));
    for (i = 0; i < 1000; ++i) {
        uint32_t t = i * 5u;
        feedback_all(t);
        if (i % 20 == 0) {
            CHECK(command("CTRL SPEED M1=500 M2=-500 M3=0 M4=0", t));
        }
        control_tick(t);
        CHECK(snapshot().output[0] == 2000);
        CHECK(snapshot().output[1] == -2000);
    }
    control_on_feedback(0, 1, 500, 0, 25, 5000);
    control_on_feedback(1, 1, -500, 0, 25, 5000);
    control_tick(5000);
    CHECK(snapshot().output[0] == 0);
    CHECK(snapshot().output[1] == 0);
    CHECK(command("CTRL SPEED M1=0 M2=0 M3=0 M4=0", 5000));
    control_on_feedback(0, 1, 300, 0, 25, 5001);
    control_tick(5005);
    zero();
    CHECK(!snapshot().fault_latched);
}

int main(void)
{
    test_startup_and_link();
    test_speed_pi_and_zero();
    test_malformed();
    test_command_deadline();
    test_can_health_gate_and_recovery();
    test_received_command_deadline();
    test_feedback_deadline_and_required_mask();
    test_current_temperature_and_clear();
    test_faults_and_time_wrap();
    test_saturation_and_no_stored_braking();
    printf("motor control: %u checks passed\n", checks);
    return 0;
}
