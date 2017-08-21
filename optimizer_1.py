import torch
from torch.optim import Optimizer

required = object()

class Optimizer_1(Optimizer):
    """Implements Neural Optimizer Search's Optimizer_1 for PyTorch
    
    Proposed in 'Neural Optimizer Search with Reinforcement Learning' by 
    Irwan Bello, Barret Zoph, Vijay Vasudevan and Quoc Le
    
    http://proceedings.mlr.press/v70/bello17a/bello17a.pdf
    
    Arguments:
      params (iterable): iterable of parameters to optimize or dicts defining
         parameter groups
      lr (float, optional): learning rate (default: 1e-3)
      beta (float, optional): decay for the running exponential average
         (momentum) range (0,1) (default: 0.9)
      weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, beta=0.9, weight_decay=0):
        defaults = dict(lr=lr, beta=beta, weight_decay=weight_decay)
        super(Optimizer_1, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Optimizer_1, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    # Times step has been called. Used for bias correction.
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                
                exp_avg = state['exp_avg']
                beta = group['beta']
                
                state['step'] += 1
                
                # Apply weight decay (L2 penalty)
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                # Decay the momentum running average coefficient
                exp_avg.mul_(beta).add_(1 - beta, grad)
                
                # Correct bias from early elements of the running average
                avg_corr = exp_avg / (1 - beta ** state['step'])
                
                update = grad.mul(torch.exp(torch.sign(grad) * torch.sign(avg_corr)))
                
                p.data.add_(-group['lr'], update)
        return loss