import torch
import torch.optim as optim
import numpy as np

class SGLD(optim.Optimizer):
    # XXX: Copied from pybnn library, modified some data type issue
    """ 
    Stochastic Gradient Langevin Dynamics Sampler
    """

    def __init__(self, params, lr = 1e-4):

        """ Set up a SGLD Optimizer.

        Parameters
        ----------
        params : iterable
            Parameters serving as optimization variable.
        lr : float, optional
            Base learning rate for this optimizer.
            Must be tuned to the specific function being minimized.
            Default: `1e-2`.
        """
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None

        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for parameter in group["params"]:

                if parameter.grad is None:
                    continue

                state = self.state[parameter]
                lr    = group["lr"]
                # the average gradient over the batch, i.e N/n sum_i g_theta_i + g_prior
                gradient = parameter.grad.data
                #  State initialization
                if len(state) == 0:
                    state["iteration"] = 0

                sigma = torch.tensor(lr, device = parameter.device).sqrt()
                delta = (0.5 * lr * gradient + sigma * torch.normal(mean=torch.zeros_like(gradient), std=torch.ones_like(gradient)))

                parameter.data.add_(-delta)
                state["iteration"] += 1
                state["sigma"] = sigma

        return loss
