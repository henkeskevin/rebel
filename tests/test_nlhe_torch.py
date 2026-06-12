import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class PokerModelTest(unittest.TestCase):
    def test_structured_model_shapes_and_gradients(self):
        from cfvpy.nlhe.model import PokerModelConfig, PokerReBeLNet
        from cfvpy.nlhe.pbs import POLICY_INPUT_SIZE, VALUE_INPUT_SIZE

        model = PokerReBeLNet(PokerModelConfig.profile("smoke"))
        value = model(torch.zeros(2, VALUE_INPUT_SIZE), head="value")
        policy = model(torch.zeros(2, POLICY_INPUT_SIZE), head="policy")
        self.assertEqual(tuple(value.shape), (2, 1326))
        self.assertEqual(tuple(policy.shape), (2, 1326, 9))
        (value.mean() + policy.mean()).backward()
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in model.parameters()
                if parameter.requires_grad
            )
        )

    def test_h100_profile_is_substantial(self):
        from cfvpy.nlhe.model import PokerModelConfig, PokerReBeLNet

        model = PokerReBeLNet(PokerModelConfig.profile("h100"))
        self.assertGreater(model.parameter_count(), 10_000_000)


if __name__ == "__main__":
    unittest.main()
