import copy
import unittest

from spa_control.pair_motion_sim import PairMotionSimulation


def sample(now, frame=1):
    m = dict(fresh=True, speed=0, current=0, temperature=25, target=0, output=0, rx=frame)
    motor = dict(connected=True, frame_at=now, frame_id=frame, fault='NONE', can_ready=True,
                 can_healthy=True, mode='PAIR', directions=(-1, -1), motors=[copy.copy(m), copy.copy(m)])
    camera = dict(valid=True, calibrated=True, sample_at=now, revision=3, sample_revision=3, position_m=0.0)
    return motor, camera


class PairSimulationTest(unittest.TestCase):
    def arm(self):
        s = PairMotionSimulation()
        s.arm(rpm=100, distance_m=.2, direction=1, expected_dirs=(-1,-1), max_duration_s=5, direction_confirmed=True)
        return s

    def running(self):
        s=self.arm(); m,c=sample(0)
        self.assertEqual(s.tick(0,m,c), ['CTRL STOP'])
        s.receive('CTRL OK STOP')
        for i in range(1,15):
            t=i/10; m,c=sample(t,i)
            commands=s.tick(t,m,c)
            if commands:
                break
        self.assertEqual(s.phase,'running')
        self.assertEqual(commands,['CTRL PAIR 100'])
        return s

    def test_validation(self):
        for change in [dict(rpm=501),dict(rpm=0),dict(distance_m=-1),dict(direction_confirmed=False),dict(max_duration_s=float('nan'))]:
            p=dict(rpm=100,distance_m=.2,direction=1,expected_dirs=(-1,-1),max_duration_s=5,direction_confirmed=True);p.update(change)
            with self.assertRaises(ValueError): PairMotionSimulation().arm(**p)

    def test_ack_and_renew(self):
        s=self.running(); s.receive('CTRL OK M1=0 M2=0 M3=-100 M4=-100');m,c=sample(1.51,16)
        self.assertEqual(s.tick(1.51,m,c),['CTRL PAIR 100'])

    def test_ack_timeout_and_no_resume(self):
        s=self.running();m,c=sample(1.61,17)
        self.assertEqual(s.tick(1.61,m,c),['CTRL STOP'])
        self.assertEqual(s.reason,'motion_ack_timeout')
        s.receive('CTRL OK M1=0 M2=0 M3=-100 M4=-100')
        self.assertEqual(s.tick(1.7,*sample(1.7,18)),[])

    def test_emergency_wins_over_renewal(self):
        s=self.running();s.receive('CTRL OK M1=0 M2=0 M3=-100 M4=-100');s.emergency_stop()
        self.assertEqual(s.tick(1.51,*sample(1.51,16)),['CTRL STOP'])
        self.assertEqual(s.reason,'emergency_stop')
        s.receive('CTRL OK STOP'); self.assertEqual(s.tick(1.6,*sample(1.6,17)),[])
        self.assertEqual(s.phase,'aborted');self.assertTrue(s.stop_verified)
        self.assertEqual(s.tick(1.8,*sample(1.8,18)),[])

    def test_camera_loss_and_revision_change_stop(self):
        for kind in ('lost','revision','sample_revision'):
            s=self.running();m,c=sample(1.5,16)
            if kind=='lost': c['valid']=False
            elif kind=='revision': c['revision']=c['sample_revision']=4
            else: c['sample_revision']=2
            self.assertEqual(s.tick(1.5,m,c),['CTRL STOP'])
            self.assertIn(s.reason,('camera_stale','camera_revision_changed','camera_revision_missing'))

    def test_motor_faults_stop(self):
        for kind in ('disconnect','fault','overcurrent','overtemp','mapping','stale'):
            s=self.running();m,c=sample(1.5,16)
            if kind=='disconnect': m['connected']=False
            elif kind=='fault': m['fault']='CAN_ERROR'
            elif kind=='overcurrent': m['motors'][0]['current']=7000
            elif kind=='overtemp': m['motors'][1]['temperature']=80
            elif kind=='mapping': m['directions']=(-1,1)
            else: m['frame_at']=1
            self.assertEqual(s.tick(1.5,m,c),['CTRL STOP'])

    def test_distance_stop_ack_without_camera(self):
        s=self.running();m,c=sample(1.5,16);c['position_m']=.21
        self.assertEqual(s.tick(1.5,m,c),['CTRL STOP']);self.assertEqual(s.reason,'distance_reached')
        s.receive('CTRL OK STOP');m,c=sample(1.6,17);c['valid']=False
        self.assertEqual(s.tick(1.6,m,c),[]); self.assertEqual(s.phase,'complete');self.assertTrue(s.stop_verified)

    def test_unacknowledged_stop_is_not_verified(self):
        s=self.running();s.emergency_stop();s.tick(1.5,*sample(1.5,16))
        s.tick(2.3,*sample(2.3,17)); self.assertEqual(s.phase,'aborted');self.assertFalse(s.stop_verified)

    def test_baseline_needs_advancing_feedback(self):
        s=self.arm();s.tick(0,*sample(0));s.receive('CTRL OK STOP')
        for i in range(1,15):
            t=i/10;m,c=sample(t,1);m['frame_id']=1
            actions=s.tick(t,m,c)
            if actions: break
        self.assertEqual(actions,['CTRL STOP']);self.assertEqual(s.reason,'motor_feedback_not_advancing')

    def test_late_emergency_never_sends_motion(self):
        s=self.arm();s.emergency_stop();self.assertEqual(s.tick(0,*sample(0)),['CTRL STOP'])
        self.assertEqual(s.tick(.1,*sample(.1)),[])

    def test_duration_limit_stops(self):
        s=self.running(); s._start_at=1.3
        s.receive('CTRL OK M1=0 M2=0 M3=-100 M4=-100')
        self.assertEqual(s.tick(6.31,*sample(6.31,30)),['CTRL STOP'])
        self.assertEqual(s.reason,'duration_limit')

    def test_board_rejection_stops(self):
        s=self.running();s.receive('CTRL ERR BAD_COMMAND')
        self.assertEqual(s.tick(1.5,*sample(1.5,16)),['CTRL STOP'])
        self.assertEqual(s.reason,'board_error')

if __name__ == '__main__': unittest.main()
