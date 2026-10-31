import tempfile
import unittest
from pathlib import Path
import torch

from spa_control.serial_preview_model import (ModelConfig, SerialPreviewModel,
                                               supervised_loss, supervised_step)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.cfg = ModelConfig(recent_dim=8, front_dim=6, quality_dim=5)
        self.model = SerialPreviewModel(self.cfg)
        b, steps = 2, 3
        self.batch = dict(
            visual=torch.randn(b, 3, 576), visual_times=torch.tensor([[9.8, 9.9, 10.1]]*b),
            visual_valid=torch.ones(b,3,dtype=torch.bool),
            recent=torch.randn(b,steps,8), recent_times=torch.tensor([[9.7,9.9,10.1]]*b),
            recent_valid=torch.ones(b,steps,dtype=torch.bool),
            front=torch.randn(b,8,steps,6), front_times=torch.zeros(b,8,steps),
            front_valid=torch.zeros(b,8,steps,dtype=torch.bool),
            cutoff=torch.tensor([10.,10.]), quality=torch.randn(b,5),
            candidates=torch.tanh(torch.randn(b,3,8)), support=torch.ones(b,3,dtype=torch.bool))
        for k in range(8):
            upper = 9.5 - k*.5
            self.batch['front_times'][:,k,:] = torch.tensor([upper-.4,upper-.2,upper])
        self.batch['front_valid'][:,0,:] = True

    def test_null_is_exact_and_ordered_quantiles(self):
        self.batch['front_valid'].fill_(False)
        out = self.model(**self.batch)
        self.assertTrue(torch.equal(out['correction'], torch.zeros_like(out['correction'])))
        self.assertTrue(torch.equal(out['zbar'], torch.zeros_like(out['zbar'])))
        self.assertTrue(torch.equal(out['null_weight'], torch.ones_like(out['null_weight'])))
        pred = out['prediction']
        for start in [0,3,6]:
            self.assertTrue(bool((pred[...,start] <= pred[...,start+1]).all()))
            self.assertTrue(bool((pred[...,start+1] <= pred[...,start+2]).all()))
        self.assertEqual(out['augmentation'].shape, (2, 159))

    def test_future_and_wrong_window_are_never_used(self):
        self.batch['front_times'][:,0,2] = 10.2
        before = self.model(**self.batch)
        self.batch['visual'][:,2,:] = 999999
        self.batch['recent'][:,2,:] = 999999
        self.batch['front'][:,0,2,:] = 999999
        after = self.model(**self.batch)
        torch.testing.assert_close(before['augmentation'], after['augmentation'], rtol=0, atol=0)

    def test_explicit_coverage_is_padding_invariant(self):
        self.batch['front_coverage'] = torch.tensor([[.2,0,0,0,0,0,0,0]]*2)
        before = self.model(**self.batch)
        self.batch['front'] = torch.nn.functional.pad(self.batch['front'], (0,0,0,5))
        self.batch['front_times'] = torch.nn.functional.pad(self.batch['front_times'], (0,5))
        self.batch['front_valid'] = torch.nn.functional.pad(self.batch['front_valid'], (0,5), value=False)
        after = self.model(**self.batch)
        torch.testing.assert_close(before['augmentation'],after['augmentation'],rtol=0,atol=0)
        torch.testing.assert_close(after['front_coverage'], self.batch['front_coverage'])
        self.batch.pop('front_coverage')
        conservative = self.model(**self.batch)
        self.assertTrue(torch.equal(conservative['front_coverage'],torch.zeros(2,8)))
        self.batch['front_coverage'] = torch.full((2,8),1.2)
        with self.assertRaisesRegex(ValueError,'front_coverage'):
            self.model(**self.batch)

    def test_train_phases_and_frozen_parameters(self):
        targets = torch.tensor([[.1,.2,.4,0.],[.2,.3,.6,1.]])
        mask = torch.ones(2,4,dtype=torch.bool)
        actions = torch.tensor([0,2])
        self.model.set_phase('reference')
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=.002)
        ref_before = self.model.reference[0].weight.detach().clone()
        history_before = self.model.correction[0].weight.detach().clone()
        loss = supervised_step(self.model, optimizer, self.batch, actions, targets, mask, [1.,1.,1.])
        self.assertGreater(loss,0.)
        self.assertFalse(torch.equal(ref_before,self.model.reference[0].weight))
        self.assertTrue(torch.equal(history_before,self.model.correction[0].weight))
        self.model.set_phase('history')
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad), lr=.002)
        ref_before = self.model.reference[0].weight.detach().clone()
        history_before = self.model.correction[0].weight.detach().clone()
        supervised_step(self.model, optimizer, self.batch, actions, targets, mask, [1.,1.,1.])
        self.assertTrue(torch.equal(ref_before,self.model.reference[0].weight))
        self.assertFalse(torch.equal(history_before,self.model.correction[0].weight))

    def test_missing_labels_and_unsupported_sample(self):
        out = self.model(**self.batch)
        targets = torch.tensor([[float('nan'),.2,float('nan'),0.],[.2,.3,.6,1.]])
        mask = torch.tensor([[0,1,0,1],[1,1,1,1]],dtype=torch.bool)
        loss = supervised_loss(out['raw'],torch.tensor([1,2]),targets,mask,[1.,1.,1.])
        self.assertTrue(bool(torch.isfinite(loss)))
        with self.assertRaises(ValueError):
            supervised_loss(out['raw'],torch.tensor([1,2]),targets,torch.ones_like(mask),[1.,1.,1.])
        self.model.set_phase('reference')
        optimizer = torch.optim.AdamW((p for p in self.model.parameters() if p.requires_grad))
        self.batch['support'][0,1] = False
        with self.assertRaisesRegex(ValueError, 'support'):
            supervised_step(self.model,optimizer,self.batch,torch.tensor([1,2]),targets,mask,[1.,1.,1.])
        self.model.set_phase('history')
        with self.assertRaisesRegex(ValueError, 'optimizer'):
            supervised_step(self.model,optimizer,self.batch,torch.tensor([1,2]),targets,mask,[1.,1.,1.])

    def test_checkpoint(self):
        self.model.set_phase('frozen')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            self.model.export_checkpoint(path)
            loaded = SerialPreviewModel.load_checkpoint(path)
            torch.testing.assert_close(self.model(**self.batch)['raw'], loaded(**self.batch)['raw'])
            self.assertEqual(loaded.phase, 'frozen')
            self.assertFalse(any(p.requires_grad for p in loaded.parameters()))


if __name__ == '__main__':
    unittest.main()
