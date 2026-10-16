import unittest
import torch
from blrpng.training.rollout_supervision import supervised_rollout_loss
from blrpng.training.losses import per_sample_relative_l2


class ScalarOperator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(0.5))
        self.frames = []

    def forward(self, x, lift, mask):
        state = torch.where(mask, lift, x[:, :1] * self.factor)
        self.frames.append(state)
        return state, {}


class RolloutSupervisionTest(unittest.TestCase):
    def batch(self, boundary=False):
        x = torch.ones(1, 3, 3, 3)
        mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
        if boundary:
            mask[:, :, (0, -1), :] = True
            mask[:, :, :, (0, -1)] = True
        return {"x": x, "y": x[:, :1].clone(),
                "future_targets": torch.ones(1, 3, 1, 3, 3),
                "boundary_lift": torch.ones(1, 1, 3, 3), "boundary_mask": mask}

    def loss(self, prediction, target):
        return per_sample_relative_l2(prediction, target).mean()

    def test_h1_exact_value_and_gradient(self):
        m = ScalarOperator(); b = self.batch()
        p, _ = m(b["x"], b["boundary_lift"], b["boundary_mask"])
        old = self.loss(p, b["y"])
        new = supervised_rollout_loss(m, b, p, horizon=1, loss_function=self.loss)
        self.assertTrue(torch.equal(old, new))
        self.assertTrue(torch.equal(torch.autograd.grad(old, m.factor, retain_graph=True)[0],
                                    torch.autograd.grad(new, m.factor)[0]))
        self.assertEqual(len(m.frames), 1)

    def test_h3_recursive_gradient_not_detached(self):
        m = ScalarOperator(); b = self.batch()
        p, _ = m(b["x"], b["boundary_lift"], b["boundary_mask"])
        loss = supervised_rollout_loss(m, b, p, horizon=3, loss_function=self.loss)
        self.assertAlmostEqual(float(loss.detach()), (0.5 + 0.75 + 0.875) / 3, places=6)
        grad = torch.autograd.grad(loss, m.factor)[0]
        self.assertAlmostEqual(float(grad), -(1 + 1 + 0.75) / 3, places=6)
        self.assertGreater(abs(float(grad) + (1 + 0.5 + 0.25) / 3), 0.1)

    def test_boundary_every_recursive_step(self):
        m = ScalarOperator(); b = self.batch(boundary=True)
        p, _ = m(b["x"], b["boundary_lift"], b["boundary_mask"])
        supervised_rollout_loss(m, b, p, horizon=3, loss_function=self.loss)
        self.assertEqual(len(m.frames), 3)
        for frame in m.frames:
            self.assertTrue(torch.equal(frame[b["boundary_mask"]], b["boundary_lift"][b["boundary_mask"]]))

    def test_fail_closed_on_missing_or_misaligned_window(self):
        m = ScalarOperator(); b = self.batch(); p, _ = m(b["x"], b["boundary_lift"], b["boundary_mask"])
        b["future_targets"][:, 0] *= 2
        with self.assertRaises(ValueError):
            supervised_rollout_loss(m, b, p, horizon=3, loss_function=self.loss)
        b.pop("future_targets")
        with self.assertRaises(ValueError):
            supervised_rollout_loss(m, b, p, horizon=3, loss_function=self.loss)

    def test_real_fno_engine_h3_backpropagation(self):
        from blrpng.models.fno import FNOConfig, DirichletOperator
        from blrpng.training.engine import _train_batch
        torch.manual_seed(42)
        model = DirichletOperator(FNOConfig(in_channels=4, out_channels=1,
                                            modes1=1, modes2=1, width=4, layers=2))
        b = self.batch(boundary=True)
        loss, metrics = _train_batch(model=model, batch=b, objective="final_only",
                                    training_config={"rollout_horizon": 3})
        loss.backward()
        self.assertEqual(metrics["supervised_rollout_horizon"], 3)
        self.assertTrue(torch.isfinite(loss))
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
        self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)


if __name__ == "__main__":
    unittest.main()
