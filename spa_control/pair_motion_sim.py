"""Pure simulator for a prospective paired-drive distance-stop state machine.

No serial access, timers or web endpoints. Not a physical safety system. Values
are simulated inputs, and commands are *strings returned to the caller* only.
Real integration requires a single serial owner, independent MCU watchdog,
validated calibration/tracking, emergency stop and physical tests.
"""
from __future__ import annotations

import math


class PairMotionSimulation:
    BASELINE_S = 1.2
    FRAME_TIMEOUT_S = .25
    CAMERA_TIMEOUT_S = .5
    ACK_TIMEOUT_S = .2
    RENEW_S = .1
    STOP_TIMEOUT_S = .7

    def __init__(self):
        self.phase = 'idle'
        self.reason = None
        self.stop_verified = False
        self._last_time = None
        self._pending_ack = None
        self._stop_ack = False
        self._stop_sent_at = None
        self._baseline_at = None
        self._start_at = None
        self._last_send = None
        self._camera_revision = None
        self._start_position = None
        self._last_counts = None
        self._last_frame_id = None
        self._baseline_samples = 0
        self._cancel_requested = False
        self._cfg = None

    def arm(self, *, rpm, distance_m, direction, expected_dirs, max_duration_s, direction_confirmed):
        if self.phase != 'idle':
            raise ValueError('A simulation is already in progress or finished')
        if type(rpm) is not int or rpm == 0 or abs(rpm) > 500:
            raise ValueError('rpm must be a nonzero integer within ±500')
        if isinstance(distance_m, bool) or not isinstance(distance_m, (float, int)) or not math.isfinite(distance_m) or distance_m <= 0:
            raise ValueError('distance must be positive and finite')
        if type(direction) is not int or direction not in (-1, 1) or direction_confirmed is not True:
            raise ValueError('direction must be explicitly confirmed')
        if not isinstance(expected_dirs, (tuple, list)) or len(expected_dirs) != 2 or any(type(d) is not int or d not in (-1, 1) for d in expected_dirs):
            raise ValueError('two confirmed motor directions are required')
        if isinstance(max_duration_s, bool) or not isinstance(max_duration_s, (float, int)) or not math.isfinite(max_duration_s) or not 1 <= max_duration_s <= 120:
            raise ValueError('simulation timeout must be within 1–120 s')
        self._cfg = dict(rpm=rpm, distance=float(distance_m), direction=direction,
                         dirs=tuple(expected_dirs), timeout=float(max_duration_s))
        self.phase = 'request_stop'

    def emergency_stop(self):
        if self.phase not in ('idle', 'complete', 'aborted'):
            self._cancel_requested = True

    def receive(self, line):
        """Ingest one board reply; unrelated/stale replies are not acknowledgments."""
        if line.startswith('CTRL ERR') and self.phase not in ('idle', 'complete', 'aborted'):
            self._cancel_requested = True
            self.reason = 'board_error'
        elif line == 'CTRL OK STOP' and self.phase in ('wait_initial_stop', 'stopping'):
            self._stop_ack = True
        elif self.phase == 'running' and self._pending_ack is not None:
            d3, d4 = self._cfg['dirs']
            q = self._cfg['rpm']
            if line == f'CTRL OK M1=0 M2=0 M3={d3*q} M4={d4*q}':
                self._pending_ack = None

    @staticmethod
    def _finite(value):
        return type(value) in (int, float) and math.isfinite(value)

    def _check(self, now, motor, camera, baseline=False):
        if not isinstance(motor, dict) or not isinstance(camera, dict):
            return 'telemetry_missing'
        if not motor.get('connected') or not self._finite(motor.get('frame_at')) or not 0 <= now-motor['frame_at'] <= self.FRAME_TIMEOUT_S:
            return 'motor_stale'
        if motor.get('fault') != 'NONE' or motor.get('can_ready') is not True or motor.get('can_healthy') is not True:
            return 'motor_fault'
        directions = motor.get('directions')
        if motor.get('mode') != 'PAIR' or not isinstance(directions, (tuple, list)) or tuple(directions) != self._cfg['dirs']:
            return 'configuration_changed'
        motors = motor.get('motors')
        if not isinstance(motors, (tuple, list)) or len(motors) != 2:
            return 'motor_feedback_missing'
        for m in motors:
            if not isinstance(m, dict) or not m.get('fresh') or not all(self._finite(m.get(k)) for k in ('speed', 'current', 'temperature', 'target', 'output', 'rx')):
                return 'motor_feedback_missing'
            if abs(m['current']) >= 7000 or m['temperature'] >= 80:
                return 'motor_limit'
            if baseline and (abs(m['speed']) > 2 or abs(m['current']) >= 1000 or m['temperature'] >= 50 or m['target'] != 0 or m['output'] != 0):
                return 'not_stopped'
        if not camera.get('valid') or not camera.get('calibrated') or not self._finite(camera.get('sample_at')) or not 0 <= now-camera['sample_at'] <= self.CAMERA_TIMEOUT_S or not self._finite(camera.get('position_m')):
            return 'camera_stale'
        if type(camera.get('revision')) is not int or camera.get('sample_revision') != camera.get('revision'):
            return 'camera_revision_missing'
        if self._camera_revision is not None and self._camera_revision != camera['revision']:
            return 'camera_revision_changed'
        return None

    def _begin_stop(self, now, reason):
        self.phase = 'stopping'
        self.reason = reason
        self._pending_ack = None
        self._stop_ack = False
        self._stop_sent_at = now
        return ['CTRL STOP']

    def _stopped(self, now, motor):
        if not isinstance(motor, dict) or not motor.get('connected') or not self._finite(motor.get('frame_at')) or not 0 <= now - motor['frame_at'] <= self.FRAME_TIMEOUT_S:
            return False
        motors = motor.get('motors')
        if not isinstance(motors, (tuple, list)) or len(motors) != 2:
            return False
        return all(isinstance(m, dict) and m.get('fresh') and
                   all(self._finite(m.get(k)) for k in ('speed', 'target', 'output')) and
                   abs(m['speed']) <= 2 and m['target'] == 0 and m['output'] == 0 for m in motors)

    def tick(self, now, motor, camera):
        """Advance one simulated tick and return suggested command strings."""
        if not self._finite(now) or (self._last_time is not None and now <= self._last_time):
            raise ValueError('time must be strictly increasing and finite')
        self._last_time = now
        if self.phase in ('idle', 'complete', 'aborted'):
            return []
        if self._cancel_requested and self.phase != 'stopping':
            return self._begin_stop(now, self.reason or 'emergency_stop')
        if self.phase == 'request_stop':
            self.phase = 'wait_initial_stop'
            self._stop_sent_at = now
            self._stop_ack = False
            return ['CTRL STOP']
        if self.phase == 'stopping':
            if self._stop_ack and self._stopped(now, motor):
                self.stop_verified = True
                self.phase = 'complete' if self.reason == 'distance_reached' else 'aborted'
            elif now-self._stop_sent_at > self.STOP_TIMEOUT_S:
                self.phase = 'aborted'
                self.stop_verified = False
            return []
        if self.phase == 'wait_initial_stop':
            if now-self._stop_sent_at > self.STOP_TIMEOUT_S:
                return self._begin_stop(now, 'initial_stop_timeout')
            if not self._stop_ack:
                return []
            self.phase = 'baseline'
        problem = self._check(now, motor, camera, baseline=self.phase == 'baseline')
        if problem:
            return self._begin_stop(now, problem)
        if self.phase == 'baseline':
            if self._camera_revision is None:
                self._camera_revision = camera['revision']
            frame_id = motor.get('frame_id')
            counts = tuple(m['rx'] for m in motor['motors'])
            if frame_id != self._last_frame_id:
                if self._last_counts is not None and any(b <= a for a,b in zip(self._last_counts,counts)):
                    return self._begin_stop(now, 'motor_feedback_not_advancing')
                self._last_counts, self._last_frame_id = counts, frame_id
                self._baseline_samples += 1
            if self._baseline_at is None:
                self._baseline_at = now
            if now-self._baseline_at < self.BASELINE_S:
                return []
            if self._last_counts is None or frame_id is None or self._baseline_samples < 8:
                return self._begin_stop(now, 'motor_feedback_not_advancing')
            self.phase = 'running'
            self._start_at, self._start_position = now, camera['position_m']
        if self.phase == 'running':
            if now-self._start_at >= self._cfg['timeout']:
                return self._begin_stop(now, 'duration_limit')
            travel_direction = self._cfg['direction'] * (1 if self._cfg['rpm'] > 0 else -1)
            travel = travel_direction * (camera['position_m']-self._start_position)
            if travel >= self._cfg['distance']:
                return self._begin_stop(now, 'distance_reached')
            if self._pending_ack is not None and now-self._pending_ack > self.ACK_TIMEOUT_S:
                return self._begin_stop(now, 'motion_ack_timeout')
            if self._pending_ack is None and (self._last_send is None or now-self._last_send >= self.RENEW_S):
                self._pending_ack = self._last_send = now
                return [f"CTRL PAIR {self._cfg['rpm']}"]
        return []
