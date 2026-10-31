import unittest
import io
import numpy as np
import torch

from spa_control.serial_preview import CategoricalPPO, Transition
from spa_control.serial_preview_model import ModelConfig, SerialPreviewModel, supervised_step
from spa_control.serial_preview_pipeline import DrivePacket, PipelineScales, SerialPreviewPipeline


class FakeVision:
    def encode_window(self, frames, cutoff_ns, max_frames=8):
        selected = [f for f in frames if f['receive_monotonic_ns'] <= cutoff_ns]
        selected.sort(key=lambda f: f['receive_monotonic_ns'])
        return {'frames': [dict(f, valid=True, features=[float(f['jpeg'][0])] * 576)
                           for f in selected[-max_frames:]],
                'future_frames_excluded': len(frames)-len(selected)}


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model = SerialPreviewModel(ModelConfig(24,12,5))
        self.policy = CategoricalPPO(159, seed=1)
        self.pipeline = SerialPreviewPipeline(self.model, self.policy, FakeVision(),
                                             PipelineScales(7000,500,.25,.5))
        self.cut = 10_000_000_000
        self.drives = []
        for t in (9_250_000_000,9_450_000_000,9_800_000_000,9_980_000_000):
            for c in range(4):
                self.drives.append(DrivePacket(c,t,100+c,20+c,100,t))
        self.frames = [{'jpeg':b'\x01jpeg','receive_monotonic_ns':9_900_000_000}]
        self.args = dict(frames=self.frames, drives=self.drives, cutoff_ns=self.cut,
                         quality=[0.,0.,0.,0.,1.], nominal=[300]*4,
                         signs=[-1,1,-1,1], rear=[False,False,True,True], active=[True]*4)

    def test_real_pipeline_with_injected_vision(self):
        out = self.pipeline.infer(**self.args)
        self.assertTrue(out['untrained'])
        self.assertFalse(out['physical_policy_allowed'])
        self.assertEqual(len(out['observation']),159)
        self.assertEqual(len(out['prediction_features']),3)
        self.assertAlmostEqual(sum(out['candidate_probabilities']),1.)
        self.assertEqual(out['timing']['candidate_vectors'][0],[-300,300,-295,295])
        self.assertTrue(out['history_available'][0])
        self.assertEqual(out['timing']['frame_count'],1)

    def test_synthetic_jpeg_with_real_mobilenet_weights(self):
        # Entirely synthetic image and drive values; this is not camera/hardware validation.
        from PIL import Image
        from spa_control.serial_preview_vision import MobileNetWindowEncoder, VisionModelError
        stream = io.BytesIO()
        Image.new('RGB', (320, 240), (30, 100, 160)).save(stream, format='JPEG')
        self.args['frames'] = [{'jpeg':stream.getvalue(),'receive_monotonic_ns':9_900_000_000}]
        self.pipeline.visual_encoder = MobileNetWindowEncoder()
        try:
            result = self.pipeline.infer(**self.args)
        except VisionModelError as e:
            self.skipTest(f'本机视觉运行环境/官方权重不可用: {e}')
        self.assertEqual(result['timing']['frame_count'],1)
        self.assertEqual(len(result['observation']),159)
        self.assertTrue(np.isfinite(result['observation']).all())
        self.assertFalse(result['physical_policy_allowed'])

    def test_future_frames_and_motor_packets_cannot_change_inference(self):
        first = self.pipeline.infer(**self.args)
        self.frames.append({'jpeg':b'\xfffake','receive_monotonic_ns':self.cut+1})
        for c in range(4):
            self.drives.append(DrivePacket(c,self.cut+1,999999,999999,500,self.cut+1))
        second = self.pipeline.infer(**self.args)
        np.testing.assert_array_equal(first['observation'], second['observation'])
        self.assertEqual(second['timing']['future_frames_excluded'],1)
        self.assertEqual(second['timing']['future_drive_packets_excluded'],4)
        self.assertEqual(first['timing']['history_coverage'],second['timing']['history_coverage'])

    def test_grid_coverage_single_sample_is_one_of_five(self):
        self.pipeline.scales = PipelineScales(7000,500,.051,.5)
        sparse = []
        for c in (0,1):
            sparse.append(DrivePacket(c,9_450_000_000,100,10,100,9_450_000_000))
        for c in range(4):
            sparse.append(DrivePacket(c,9_980_000_000,100,10,100,9_980_000_000))
        self.args['drives'] = sparse
        batch, metadata = self.pipeline.assemble(**self.args)
        self.assertAlmostEqual(metadata['history_coverage'][0],.2)
        self.assertAlmostEqual(float(batch['front_coverage'][0,0]),.2,places=6)
        self.assertTrue(all(x == 0. for x in metadata['history_coverage'][1:]))

    def test_missing_or_stale_channel_rejected(self):
        self.args['drives'] = [p for p in self.drives if p.channel != 3]
        with self.assertRaisesRegex(ValueError,'four channels'):
            self.pipeline.infer(**self.args)
        self.args['drives'] = self.drives + [DrivePacket(3,self.cut-1,0,0,0,self.cut-1,valid=False)]
        with self.assertRaisesRegex(ValueError,'four channels'):
            self.pipeline.infer(**self.args)

    def test_command_cannot_arrive_from_future(self):
        self.args['drives'] = self.drives + [DrivePacket(0,self.cut-1,10,10,10,self.cut)]
        with self.assertRaisesRegex(ValueError,'command time'):
            self.pipeline.infer(**self.args)

    def test_requires_verified_structural_order_and_integral_channel(self):
        self.args['rear'] = [True, True, False, False]
        with self.assertRaisesRegex(ValueError, 'structural'):
            self.pipeline.infer(**self.args)
        self.args['rear'] = [False, False, True, True]
        self.args['active'] = [True, True, True, False]
        with self.assertRaisesRegex(ValueError, 'four drives'):
            self.pipeline.infer(**self.args)
        self.args['active'] = [True, True, True, True]
        self.args['drives'] = self.drives + [DrivePacket(1.0,self.cut-1,10,10,10,self.cut-1)]
        with self.assertRaisesRegex(ValueError, 'channel'):
            self.pipeline.infer(**self.args)

    def test_supervised_and_fresh_simulation_ppo_interfaces(self):
        batch, metadata = self.pipeline.assemble(**self.args)
        self.model.set_phase('reference')
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=.001)
        loss = supervised_step(self.model,optimizer,batch,torch.tensor([1]),
            torch.tensor([[.1,.2,.1,0.]]),torch.ones(1,4,dtype=torch.bool),[1.,1.,1.])
        self.assertTrue(np.isfinite(loss))
        self.model.set_phase('frozen')
        with torch.no_grad():
            obs = self.model(**batch)['augmentation'][0].numpy().astype(float)
        mask = np.asarray(metadata['support'])
        transitions = []
        for i in range(3):
            sample = self.policy.act(obs,mask)
            transitions.append(Transition(obs,mask,sample.action_index,sample.log_prob,
                sample.value,float(i),0.,i==2,sample.version,source='simulation'))
        result = self.policy.update(transitions,allow_simulation=True)
        self.assertEqual(result['policy_version'],1)


if __name__ == '__main__':
    unittest.main()
