import torch
import torch.nn as nn

class StochasticDepth(nn.Module):
    """
    Implements Stochastic Depth as described in:
    "Deep Networks with Stochastic Depth" (https://arxiv.org/abs/1603.09382)
    
    During training, randomly drops entire residual blocks with probability p.
    During inference, scales the residual by (1-p).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1.0 - drop_prob
    
    def forward(self, x, residual):
        if not self.training or self.drop_prob == 0.0:
            # During inference, scale the residual
            return x + self.keep_prob * residual
        
        # During training, randomly drop the residual
        batch_size = x.shape[0]
        random_tensor = torch.rand(batch_size, 1, 1, 1, device=x.device) >= self.drop_prob
        random_tensor = random_tensor.float() / self.keep_prob  # Scale by keep_prob
        
        return x + random_tensor * residual 