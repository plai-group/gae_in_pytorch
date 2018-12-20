from pyro.distributions import Bernoulli
from torch.distributions.utils import broadcast_all
from torch.nn.functional import binary_cross_entropy_with_logits

# from pyro.distributions.torch_distribution import TorchDistribution

class WeightedBernoulli(Bernoulli):
    """Bernoulli distribution with a weighted cross entropy. Used for imbalanced data when you
    want to increase the penalizization of the positive class. """

    def __init__(self, *args, **kwargs):
        self.weight = kwargs.pop('weight', 1.0)
        super(WeightedBernoulli, self).__init__(*args, **kwargs)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        logits, value = broadcast_all(self.logits, value)
        return -binary_cross_entropy_with_logits(logits, value, reduction='none', pos_weight=self.weight)
