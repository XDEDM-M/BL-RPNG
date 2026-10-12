import unittest
from unittest.mock import patch

import torch

from blrpng.training.gray_scott_r3 import balanced_field_loss, field_errors, advance


class GrayScottR3Tests(unittest.TestCase):
    def test_equal_relative_errors_have_equal_weight(self):
        target = torch.ones(2, 2, 8, 8)
        target[:, 1] *= 0.02
        prediction = (target * 1.2).requires_grad_()
        self.assertAlmostEqual(float(balanced_field_loss(prediction, target).detach()), 0.2, places=6)
        self.assertTrue(torch.allclose(field_errors(prediction, target), torch.full((2, 2), .2)))

    def test_exact_and_empty_fields_have_finite_gradients(self):
        for value in (0.0, 1.0):
            target = torch.full((1, 2, 8, 8), value)
            pred = target.clone().requires_grad_()
            loss = balanced_field_loss(pred, target)
            loss.backward()
            self.assertEqual(float(loss.detach()), 0.0)
            self.assertTrue(torch.isfinite(pred.grad).all())

    def test_closed_loop_keeps_gradient_through_previous_steps(self):
        class Linear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.gain = torch.nn.Parameter(torch.tensor(2.0))
            def forward(self, x, lift, mask):
                return x[:, :2] * self.gain, {}
        model = Linear()
        state = torch.ones(1, 2, 4, 4)
        geo = (state * 0, state.bool(), state[:, :0], state)
        for _ in range(3):
            state = advance(model, state, geo)
        state.mean().backward()
        self.assertAlmostEqual(float(model.gain.grad), 12.0)

    def test_engine_balanced_loss_is_opt_in(self):
        from blrpng.training.engine import _train_batch
        class Toy(torch.nn.Module):
            def forward(self, x, lift, mask): return x, {}
        target = torch.ones(1, 2, 8, 8)
        target[:, 1] *= .02
        prediction = target.clone(); prediction[:, 1] *= 2
        batch = {'x': prediction, 'y': target, 'boundary_lift': target * 0,
                 'boundary_mask': torch.zeros_like(target, dtype=torch.bool)}
        balanced, _ = _train_batch(model=Toy(), batch=batch, objective='final_only',
                                   training_config={'final_loss': 'field_balanced_relative_l2'})
        legacy, _ = _train_batch(model=Toy(), batch=batch, objective='final_only', training_config={})
        self.assertAlmostEqual(float(balanced), .5, places=6)
        self.assertLess(float(legacy), .021)

    def test_v4_validation_metric_matches_field_loss(self):
        from blrpng.evaluation.rollout import projected_rollout_index
        class Identity(torch.nn.Module):
            def forward(self, x, lift, mask): return x[:, :2], {}
        state = torch.ones(2, 8, 8); state[1] *= .02
        target = state.clone(); target[1] *= 2
        payload = {'trajectory': torch.stack((state, target)), 'boundary_lift': state * 0,
                   'constraint_mask': torch.zeros_like(state, dtype=torch.bool),
                   'ids': {'initial_condition_id': 1}}
        with patch('blrpng.evaluation.rollout.read_index', return_value={'shards': [{'path': 'unused'}]}), \
             patch('blrpng.evaluation.rollout.resolve_in_project', return_value='unused'), \
             patch('blrpng.evaluation.rollout.torch.load', return_value=payload):
            balanced = projected_rollout_index(Identity(), 'unused', device=torch.device('cpu'), field_balanced=True)
            joint = projected_rollout_index(Identity(), 'unused', device=torch.device('cpu'))
        self.assertAlmostEqual(balanced['rollout_rel_l2'], .25, places=6)
        self.assertLess(joint['rollout_rel_l2'], .021)

    def test_endpoint_auxiliary_uses_physical_fields_without_mutating_newton_labels(self):
        from types import SimpleNamespace
        from blrpng.training.engine import _train_batch
        from blrpng.training.losses import correction_direction_magnitude_loss
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.gain = torch.nn.Parameter(torch.tensor(1.1))
                self.config = SimpleNamespace(out_channels=2)
            def forward(self,x,lift,mask):
                y = self.gain*x[:,:2]
                return y, {'layer1:hook0':y*.1}
            def path_active_parameters(self,layer): return [self.gain]
        x = torch.ones(1,5,8,8); y = torch.full((1,2,8,8),2.)
        newton = torch.full_like(y,.25)
        batch = dict(x=x,y=y,newton_correction=newton,valid_correction_mask=torch.ones(1,dtype=torch.bool),
                     boundary_lift=torch.zeros_like(y),boundary_mask=torch.zeros_like(y,dtype=torch.bool))
        config = dict(path_layer=1,path_log_magnitude_weight=.1,path_epsilon=1e-8,
                      path_gradient_ratio=.05,auxiliary_target='endpoint_increment')
        with patch('blrpng.training.engine.correction_direction_magnitude_loss',
                   wraps=correction_direction_magnitude_loss) as loss_call:
            model = Toy()
            loss,_ = _train_batch(model=model,batch=batch,objective='blrpng',training_config=config)
            self.assertTrue(torch.equal(loss_call.call_args.args[1],torch.ones_like(y)))
            loss.backward(); self.assertTrue(torch.isfinite(model.gain.grad))
        self.assertTrue(torch.equal(newton,torch.full_like(y,.25)))
