import unittest

from blrpng.cli.gray_scott_r5_safe_router import choose_route


class GrayScottR5RouterTests(unittest.TestCase):
    def test_rejects_more_accurate_but_unstable_candidate(self):
        candidates = {
            'final': {'lr_validation_balanced': .20, 'free20_comm128_256': .03},
            'path': {'lr_validation_balanced': .10, 'free20_comm128_256': .20},
        }
        self.assertEqual(choose_route(candidates, threshold=.06), 'final')

    def test_uses_lr_error_when_both_are_stable(self):
        candidates = {
            'final': {'lr_validation_balanced': .20, 'free20_comm128_256': .03},
            'path': {'lr_validation_balanced': .10, 'free20_comm128_256': .04},
        }
        self.assertEqual(choose_route(candidates, threshold=.06), 'path')

    def test_falls_back_to_least_unstable_candidate(self):
        candidates = {
            'final': {'lr_validation_balanced': .20, 'free20_comm128_256': .10},
            'path': {'lr_validation_balanced': .10, 'free20_comm128_256': .20},
        }
        self.assertEqual(choose_route(candidates, threshold=.06), 'final')


if __name__ == '__main__':
    unittest.main()
