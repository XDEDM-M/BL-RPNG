from __future__ import annotations

import copy
import unittest

import torch

from blrpng.config import load_config
from blrpng.data.generate import generate_from_config
from blrpng.data.manifest import read_manifest, validate_split_isolation
from blrpng.data.schema import load_dataset, validate_coarse_fine_pairing
from blrpng.evaluation.checkpoint import load_checkpoint_model
from blrpng.evaluation.rollout import projected_rollout
from blrpng.models.fno import build_model
from blrpng.paths import resolve_in_project
from blrpng.training.engine import train
from blrpng.utils import set_seed, state_dict_sha256


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/smoke.yaml")
        if not resolve_in_project(cls.config["data"]["train_path"]).exists():
            generate_from_config(cls.config)
        # Regenerate old smoke artifacts when optional dual-stage fields were added.
        data = torch.load(
            resolve_in_project(cls.config["data"]["train_path"]),
            map_location="cpu",
            weights_only=False,
        )
        transition = data["y"] - data["x"][:, : data["y"].shape[1]]
        if (
            "newton_correction_p075" not in data
            or float(transition.abs().max()) == 0.0
        ):
            generate_from_config(cls.config)

    def test_manifest_and_pairing(self):
        manifest = read_manifest(self.config["data"]["manifest_path"])
        report = validate_split_isolation(manifest)
        self.assertTrue(report["unique_pair_ids"])
        coarse = load_dataset(self.config["data"]["validation_path"])
        fine = load_dataset(self.config["data"]["test_path"])
        self.assertTrue(validate_coarse_fine_pairing(coarse, fine)["ids_match"])

    def test_one_epoch_checkpoint_reload_and_fine_rollout(self):
        result = train(
            self.config,
            output_checkpoint="artifacts/checkpoints/test_integration.pt",
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, checkpoint = load_checkpoint_model(result["checkpoint"], device=device)
        fine = load_dataset(self.config["data"]["test_path"], modes2=8)
        metrics = projected_rollout(model, fine, device=device, horizon=2)
        coarse_train = load_dataset(self.config["data"]["train_path"], modes2=8)
        batched_metrics = projected_rollout(
            model, coarse_train, device=device, horizon=2, rollout_batch_size=2
        )
        serial_metrics = projected_rollout(
            model, coarse_train, device=device, horizon=2, rollout_batch_size=1
        )
        self.assertTrue(torch.isfinite(torch.tensor(metrics["rollout_rel_l2"])))
        # cuFFT convolution kernels can differ slightly across batch shapes even
        # with deterministic algorithms enabled.  This checks evaluation
        # equivalence at a tolerance far below the reported metric precision.
        self.assertLess(
            abs(
                batched_metrics["rollout_rel_l2"]
                - serial_metrics["rollout_rel_l2"]
            ),
            5e-5,
        )
        self.assertEqual(len(batched_metrics["per_step_rel_l2"]), 4)
        self.assertEqual(checkpoint["representation"], "lifted")
        self.assertIn("ema_weights", checkpoint)

    def test_deterministic_initialization(self):
        set_seed(123, True)
        first = build_model(self.config["model"]).state_dict()
        first_hash = state_dict_sha256(first)
        set_seed(123, True)
        second = build_model(self.config["model"]).state_dict()
        self.assertEqual(first_hash, state_dict_sha256(second))

    def test_deterministic_training_and_resume(self):
        first = train(
            self.config,
            output_checkpoint="artifacts/checkpoints/deterministic_a.pt",
        )
        second = train(
            self.config,
            output_checkpoint="artifacts/checkpoints/deterministic_b.pt",
        )
        first_checkpoint = torch.load(
            first["checkpoint"], map_location="cpu", weights_only=False
        )
        second_checkpoint = torch.load(
            second["checkpoint"], map_location="cpu", weights_only=False
        )
        self.assertEqual(
            state_dict_sha256(first_checkpoint["raw_weights"]),
            state_dict_sha256(second_checkpoint["raw_weights"]),
        )
        resumed_config = copy.deepcopy(self.config)
        resumed_config["training"]["epochs"] = 2
        resumed = train(
            resumed_config,
            resume_checkpoint="artifacts/checkpoints/deterministic_a_recovery.pt",
            output_checkpoint="artifacts/checkpoints/resumed.pt",
        )
        resumed_checkpoint = torch.load(
            resumed["checkpoint"], map_location="cpu", weights_only=False
        )
        self.assertEqual(len(resumed_checkpoint["history"]), 2)


if __name__ == "__main__":
    unittest.main()
