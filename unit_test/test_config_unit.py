# test_config_unit.py
import unittest
import os
from config.config import Config

class TestConfig(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.cfg = Config()
        cls.cfg.ensure_dirs()

    def test_directories_exist(self):
        dirs = [
            self.cfg.OUTPUT_DIR,
            self.cfg.PROCESSED_DATA_DIR,
            self.cfg.PROCESSED_STATIC_DIR,
            self.cfg.PROCESSED_DYNAMIC_DIR,
            self.cfg.EVALUATION_DIR,
            self.cfg.REPORTS_DIR,
            self.cfg.MODELS_DIR,
            self.cfg.MODELS_DIR_STATIC_TRAIN,
            self.cfg.LOG_DIR,
            self.cfg.DATA_DIR,
            self.cfg.RAW_DATA_DIR,
            self.cfg.MAPPING_DIR,
        ]
        for d in dirs:
            with self.subTest(directory=d):
                self.assertTrue(os.path.exists(d), f"Directory {d} should exist")

    def test_vae_config(self):
        vae_cfg = self.cfg.VAE_CONFIG
        self.assertIn("input_dim", vae_cfg)
        self.assertIn("latent_dim", vae_cfg)
        self.assertIsInstance(vae_cfg["epochs"], int)

    def test_ratios_sum(self):
        total = self.cfg.TRAIN_RATIO + self.cfg.VALID_RATIO + self.cfg.TEST_RATIO
        self.assertAlmostEqual(total, 1.0, msg="Train, validation, test ratios should sum to 1")

if __name__ == "__main__":
    unittest.main()
