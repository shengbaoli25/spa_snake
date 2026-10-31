import unittest
import numpy as np
import torch

from spa_control.serial_preview import CategoricalPPO, Transition, measured_load_costs, measured_reward, RewardWeights
from spa_control.serial_preview_pipeline import DrivePacket, PipelineScales
from spa_control.two_motor_preview import TwoMotorConfig, TwoMotorModel, TwoMotorPipeline, project_two, run_two_motor_demo


class Vision:
    def encode_window(self, frames, cutoff_ns, max_frames=8):
        accepted = sorted((x for x in frames if x['receive_monotonic_ns'] <= cutoff_ns),
                          key=lambda x:x['receive_monotonic_ns'])[-max_frames:]
        return {'frames':accepted, 'future_frames_excluded':sum(x['receive_monotonic_ns']>cutoff_ns for x in frames)}


class TwoMotorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model=TwoMotorModel(TwoMotorConfig())
        self.policy=CategoricalPPO(147,seed=2)
        self.pipeline=TwoMotorPipeline(self.model,self.policy,Vision(),PipelineScales(7000,500,.25,.5))
        self.cut=10_000_000_000
        self.frames=[{'receive_monotonic_ns':self.cut-100_000_000,'valid':True,'features':[0.1]*576}]
        self.drives=[DrivePacket(c,self.cut-age*1_000_000,100+c,-20-c,-60,self.cut-age*1_000_000)
                     for age in range(4400,0,-100) for c in (2,3)]
        self.args=dict(frames=self.frames,drives=self.drives,cutoff_ns=self.cut,quality=[0,0,0,0,1],
                       nominal=[60,60],front_motor=3)

    def test_candidates_both_physical_mappings_and_bounds(self):
        a, features, mask=project_two([60,60],3)
        self.assertEqual(a.tolist(), [[0,0,-60,-55],[0,0,-60,-60],[0,0,-60,-65]])
        b, _, _=project_two([60,60],4)
        self.assertEqual(b.tolist(), [[0,0,-55,-60],[0,0,-60,-60],[0,0,-65,-60]])
        self.assertEqual(features.shape,(3,4))
        self.assertEqual(mask.tolist(),[True,True,True])
        a, _, mask=project_two([500,500],3)
        self.assertEqual(mask.tolist(),[True,True,False])
        self.assertLessEqual(np.max(np.abs(a)),500)
        with self.assertRaises(ValueError): project_two([501,60],3)
        with self.assertRaises(ValueError): project_two([60,60],2)

    def test_true_dimension_and_independent_motor_samples(self):
        batch,meta=self.pipeline.assemble(**self.args)
        self.assertEqual(tuple(batch['recent'].shape[-1:]),(12,))
        self.assertEqual(batch['front'].shape[-1],6)
        self.assertEqual(batch['candidates'].shape[-1],4)
        self.assertNotEqual(float(batch['recent'][0,-1,0]),float(batch['recent'][0,-1,1]))
        output=self.pipeline.infer(**self.args)
        self.assertEqual(len(output['observation']),147)
        self.assertEqual(len(output['history_available']),8)
        self.assertTrue(all(output['history_available']))
        self.assertFalse(output['physical_policy_allowed'])

    def test_future_packets_and_frames_never_affect_observation(self):
        a=self.pipeline.infer(**self.args)
        self.drives.append(DrivePacket(2,self.cut+1,6900,500,500,self.cut+1))
        self.frames.append({'receive_monotonic_ns':self.cut+1,'valid':True,'features':[999]*576})
        b=self.pipeline.infer(**self.args)
        np.testing.assert_allclose(a['observation'],b['observation'])
        self.assertEqual(b['timing']['future_drive_packets_excluded'],1)
        self.assertEqual(b['timing']['future_frames_excluded'],1)

    def test_null_anchor_without_history(self):
        self.args['drives']=[p for p in self.drives if p.receive_ns >= self.cut-200_000_000]
        batch,_=self.pipeline.assemble(**self.args)
        with torch.no_grad(): out=self.model(**batch)
        self.assertTrue(torch.equal(out['raw'],out['base_raw']))
        self.assertAlmostEqual(float(out['null_weight'][0,0]),1.)
        self.assertFalse(bool(out['valid_windows'].any()))

    def test_lag_window_boundaries(self):
        self.pipeline.scales=PipelineScales(7000,500,.051,.5)
        self.args['drives']=[DrivePacket(2,self.cut-1_000_000_000,10,1,-60,self.cut-1_000_000_000),
                             DrivePacket(2,self.cut-40_000_000,10,1,-60,self.cut-40_000_000),
                             DrivePacket(3,self.cut-40_000_000,10,1,-60,self.cut-40_000_000)]
        _,meta=self.pipeline.assemble(**self.args)
        self.assertEqual(meta['history_coverage'][0],0)
        self.assertAlmostEqual(meta['history_coverage'][1],.2)

    def test_missing_and_stale_feedback_fails(self):
        self.args['drives']=[p for p in self.drives if p.channel==2]
        with self.assertRaises(ValueError): self.pipeline.infer(**self.args)
        self.args['drives']=self.drives
        self.args['cutoff_ns']=self.cut+1_000_000_000
        with self.assertRaises(ValueError): self.pipeline.infer(**self.args)

    def test_measured_reward_requires_real_measurements(self):
        costs=measured_load_costs(np.array([[100.,200.],[300.,np.nan]]),
                                  np.array([[True,True],[True,False]]),[1000.,1000.],.1)
        self.assertEqual(costs[2],.5)
        w=RewardWeights(.1,.1,.1,.1,.1,.5,.1)
        self.assertTrue(np.isfinite(measured_reward(.5,.5,costs[0],costs[1],[0,5],[0,0],False,w)))
        with self.assertRaises(ValueError): measured_reward(None,.5,costs[0],costs[1],[0,5],[0,0],False,w)

    def test_ppo_rejects_historical_mixed_and_stale(self):
        obs=np.zeros(147); support=np.ones(3,dtype=bool)
        s=self.policy.act(obs,support)
        def t(source,version=0):
            return Transition(obs,support,s.action_index,s.log_prob,s.value,1.,0.,True,version,source=source)
        with self.assertRaises(ValueError): self.policy.update([t('historical'),t('historical')],allow_simulation=True)
        with self.assertRaises(ValueError): self.policy.update([t('simulation'),t('real_on_policy')],allow_simulation=True)
        self.policy.update([t('simulation'),t('simulation')],allow_simulation=True)
        with self.assertRaises(ValueError): self.policy.update([t('simulation'),t('simulation')],allow_simulation=True)

    def test_demo_is_explicit_synthetic(self):
        a=run_two_motor_demo(4)
        self.assertEqual(a['drive_source'],'synthetic')
        self.assertEqual(a['image_source'],'synthetic_feature_vector')
        self.assertFalse(a['mapping_confirmed'])
        self.assertFalse(a['model_trained_on_robot'])
        self.assertFalse(a['physical_policy_allowed'])
        self.assertEqual(a['timing']['front_motor'],4)

if __name__=='__main__': unittest.main()
