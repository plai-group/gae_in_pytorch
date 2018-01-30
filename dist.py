from pyro.distributions.bernoulli import Bernoulli
from pyro.distributions.random_primitive import RandomPrimitive
import torch
from torch.nn import CrossEntropyLoss

class WeightedBernoulli(Bernoulli):
    """Bernoulli distribution with a weighted cross entropy. Used for imbalanced data when you 
    want to increase the penalizization of the positive class. """
    def __init__(self, *args, **kwargs):
        self.weight = kwargs.pop('weight', 1.0)
        super(WeightedBernoulli, self).__init__(*args, **kwargs)

    def batch_log_pdf(self, x):
        """
        Ref: :py:meth:`pyro.distributions.distribution.Distribution.batch_log_pdf`
        """
        batch_log_pdf_shape = self.batch_shape(x) + (1,)
        max_val = (-self.logits).clamp(min=0)
        # ----------
        # The following two lines are the only difference between WeightedBernoulli and Bernoulli
        # Cf. derivation at:
        # https://www.tensorflow.org/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        weight = (1 + (self.weight - 1) * x)
        binary_cross_entropy = self.logits*(1 - x) + max_val + weight*((-max_val).exp() + (-self.logits - max_val).exp()).log()
        # ----------
        log_prob = -binary_cross_entropy
        # XXX this allows for the user to mask out certain parts of the score, for example
        # when the data is a ragged tensor. also useful for KL annealing. this entire logic
        # will likely be done in a better/cleaner way in the future
        if self.log_pdf_mask is not None:
            log_prob = log_prob * self.log_pdf_mask
        return torch.sum(log_prob, -1).contiguous().view(batch_log_pdf_shape)


# function aliases
weighted_bernoulli = RandomPrimitive(WeightedBernoulli)
