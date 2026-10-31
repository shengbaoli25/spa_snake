import unittest
import numpy as np

from spa_control.serial_preview import (ACTIONS, CategoricalPPO, DemoExperiment, Packet,
    RewardWeights, Transition, anchored_preview, causal_windows, failure_tail,
    measured_reward, measured_load_costs, project_candidates, response_features)


class SerialPreviewTests(unittest.TestCase):
    def test_causal_window_boundaries_no_future(self):
        packets = [Packet(t, t) for t in [0., .5, 1., 1.5, 1.5001, 2., 2.1]]
        windows = causal_windows(packets, 2., recent_s=.5, width_s=.5, count=3)
        self.assertEqual([[p.values for p in w] for w in windows], [[1.5], [1.], [.5]])

    def test_null_anchor_and_shared_attention(self):
        calls = []
        def f(z, recent, action):
            calls.append(z.copy())
            return np.array([z[0] + action[0]])
        ref = lambda a: np.array([7. + a[0]])
        z, alpha0, ps, alpha = anchored_preview(np.array([2.]), [np.array([99.])],
            [False], [float('nan')], 0., np.array([0.]), [np.array([1.])], ref, f)
        self.assertEqual(alpha0, 1.)
        self.assertEqual(z[0], 0.)
        self.assertEqual(ps[0][0], 8.)
        self.assertEqual(calls, [])
        z, alpha0, ps, alpha = anchored_preview(np.array([2.]), [np.array([4.])],
            [True], [0.], 0., np.array([0.]), [np.array([1.]), np.array([3.])], ref, f)
        self.assertAlmostEqual(alpha0, .5)
        self.assertAlmostEqual(z[0], 1.)
        self.assertAlmostEqual(ps[0][0], 9.)
        self.assertAlmostEqual(ps[1][0], 11.)

    def test_projection_limits_and_dedup(self):
        v, mask = project_candidates([300,300,500,500], [-1,1,1,1],
            [False,False,True,True], [True,True,True,True])
        np.testing.assert_array_equal(v[0], [-300,300,495,495])
        np.testing.assert_array_equal(v[1], [-300,300,500,500])
        np.testing.assert_array_equal(mask, [True,True,False])
        v, _ = project_candidates([300]*4, [-1,1,1,1], [0,0,1,1], [1,0,1,1])
        self.assertTrue(np.all(v[:,1] == 0))
        with self.assertRaises(ValueError):
            project_candidates([501]*4, [1]*4, [0,0,1,1], [1]*4)

    def test_ordered_response_quantiles_and_valid_current_grid(self):
        p = response_features([1., 0., 0., -5., 2., -4., 1., 0., 0., -1000.])
        for i in range(0, 9, 3):
            self.assertLessEqual(p[i], p[i+1])
            self.assertLessEqual(p[i+1], p[i+2])
        self.assertTrue(np.all(p[3:6] >= 0))
        self.assertTrue(np.all((p[6:9] >= 0) & (p[6:9] <= 1)))
        cost, occ, coverage = measured_load_costs([[50, 150], [10, 10], [200, 20]],
            [[1,1],[1,0],[1,1]], [100,100], .5)
        self.assertAlmostEqual(cost, (0 + 1 + 2.25 + 0) / 4)
        self.assertEqual(occ, 1.)
        self.assertAlmostEqual(coverage, 2/3)
        with self.assertRaises(ValueError):
            measured_load_costs([[1,1]], [[1,0]], [100,100], .5)

    def test_reward_missing_and_failure_tail(self):
        w = RewardWeights(.2,.1,.05,.04,2.,.5,.01)
        equal = measured_reward(.1,.1,0,0,[0,0],[0,0],False,w)
        self.assertEqual(equal,1.)
        above = measured_reward(.11,.1,0,0,[0,0],[0,0],False,w)
        below = measured_reward(.09,.1,0,0,[0,0],[0,0],False,w)
        self.assertAlmostEqual(above, below)
        with self.assertRaises(ValueError):
            measured_reward(None,.1,0,0,[0,0],[0,0],False,w)
        self.assertAlmostEqual(failure_tail(2,.9,w),-(.2+.1+.05+.04)*(.9+.81))

    def make_rollout(self, p):
        rng = np.random.default_rng(11)
        ts = []
        for i in range(16):
            obs = rng.normal(size=p.obs_dim)
            mask = np.array([True,True, i % 2 == 0])
            s = p.act(obs, mask)
            done = i == 15
            ts.append(Transition(obs,mask,s.action_index,s.log_prob,s.value,
                float(ACTIONS[s.action_index] == 5), 0. if done else .1, done, s.version,
                source='simulation'))
        return ts

    def test_ppo_updates_and_rejects_stale_or_historical(self):
        p = CategoricalPPO(6, seed=3)
        ts = self.make_rollout(p)
        before = {k: v.copy() for k,v in p.params.items()}
        with self.assertRaises(ValueError):
            p.update(ts)
        result = p.update(ts, allow_simulation=True)
        self.assertEqual(result['policy_version'], 1)
        self.assertTrue(any(np.any(before[k] != p.params[k]) for k in before))
        with self.assertRaises(ValueError):
            p.update(ts, allow_simulation=True)

    def test_ppo_never_mixes_simulated_and_real_rollouts(self):
        from dataclasses import replace
        p = CategoricalPPO(6, seed=2)
        ts = self.make_rollout(p)
        ts[0] = replace(ts[0], source='real_on_policy')
        with self.assertRaisesRegex(ValueError, 'cannot share'):
            p.update(ts, allow_simulation=True)

    def test_ppo_rejects_changed_support_and_probabilities(self):
        from dataclasses import replace
        p = CategoricalPPO(6, seed=4)
        ts = self.make_rollout(p)
        ts[0] = replace(ts[0], old_log_prob=ts[0].old_log_prob + .2)
        with self.assertRaisesRegex(ValueError, 'probability'):
            p.update(ts, allow_simulation=True)
        ts = self.make_rollout(p)
        sampled = ts[0].action_index
        altered = np.zeros(3,dtype=bool)
        altered[sampled] = True
        ts[0] = replace(ts[0], support=altered)
        with self.assertRaisesRegex(ValueError, 'probability'):
            p.update(ts, allow_simulation=True)

    def test_forced_step_has_no_action_label(self):
        from dataclasses import replace
        p = CategoricalPPO(6, seed=1)
        ts = self.make_rollout(p)
        ts[3] = replace(ts[3], action_index=None, old_log_prob=None, actor_weight=0.)
        p.update(ts, allow_simulation=True)
        p = CategoricalPPO(6, seed=1)
        ts = self.make_rollout(p)
        ts[3] = replace(ts[3], action_index=None, old_log_prob=None, actor_weight=1.)
        with self.assertRaises(ValueError):
            p.update(ts, allow_simulation=True)

    def test_demo_deterministic_and_no_physical_claim(self):
        a,b = DemoExperiment(14),DemoExperiment(14)
        for _ in range(12):
            self.assertEqual(a.step(),b.step())
        self.assertTrue(a.status()['synthetic'])
        self.assertFalse(a.status()['trained'])
        self.assertEqual(a.status()['step'],12)
        self.assertEqual(a.reset(2)['step'],0)

    def test_demo_trains_fresh_synthetic_rollouts_without_real_claim(self):
        demo = DemoExperiment(5)
        before = {k: value.copy() for k, value in demo.policy.params.items()}
        trained = demo.train_batch()
        self.assertEqual(trained['train_samples'], 240)
        self.assertEqual(trained['policy_version'], 1)
        self.assertEqual(trained['train_batches'], 1)
        self.assertTrue(trained['synthetic'])
        self.assertFalse(trained['trained'])
        self.assertTrue(any(np.any(before[k] != demo.policy.params[k]) for k in before))
        self.assertEqual(demo.train_batch()['policy_version'], 2)
        self.assertEqual(demo.reset(5)['policy_version'], 0)

    def test_demo_observation_includes_previous_action_and_remaining_time(self):
        demo = DemoExperiment(2)
        initial, _ = demo._observation()
        self.assertEqual(initial[2], 0.)
        self.assertEqual(initial[3], 1.)
        demo.previous = 5
        demo.steps = 24
        later, _ = demo._observation()
        self.assertEqual(later[2], 1.)
        self.assertEqual(later[3], .5)


if __name__ == '__main__':
    unittest.main()
