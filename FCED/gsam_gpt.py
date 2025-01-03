import torch


class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, alpha=0.1, adaptive=True, **kwargs):
        """
        Gradient-aware SAM (GSAM) Optimizer.
        Parameters:
        - params: Model parameters to optimize.
        - base_optimizer: The underlying optimizer (e.g., SGD, Adam).
        - rho: Radius of the perturbation ball.
        - alpha: Weight for gradient alignment regularization.
        - adaptive: Use parameter-wise scaling for adaptive SAM.
        """
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert alpha >= 0.0, f"Invalid alpha, should be non-negative: {alpha}"

        # Initialize defaults
        defaults = dict(rho=rho, alpha=alpha, adaptive=adaptive, **kwargs)
        super(GSAM, self).__init__(params, defaults)

        # Setup the base optimizer
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        First step: Perturb weights along the gradient direction.
        """
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: 
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (p.data.abs() if group["adaptive"] else 1.0) * p.grad * scale.to(p.device)
                p.add_(e_w)  # Perturb weights

        if zero_grad: 
            self.zero_grad()
            
    @torch.no_grad()
    def second_step(self, zero_grad=False, closure=None):
        """
        Perform the second optimization step. The closure is optional and not required if the loss is precomputed.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # Restore original parameters

        if closure is not None:
            closure()

        self.base_optimizer.step()  # Perform the optimization update

        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def step(self, closure=None):
        """
        A full GSAM optimization step.
        """
        assert closure is not None, "GSAM requires closure for sharpness-aware optimization."
        closure = torch.enable_grad()(closure)  # Compute gradients via closure

        # Save current gradients for alignment computation
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self.state[p]["prev_grad"] = p.grad.clone()

        self.first_step(zero_grad=True)  # Perturb weights
        self.second_step(closure, zero_grad=True)  # Apply second step

    def _grad_norm(self):
        """
        Compute the L2 norm of the gradients across all parameters.
        """
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
            torch.stack([
                ((p.data.abs() if group["adaptive"] else 1.0) * p.grad).norm(2).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            p=2
        )
        return norm

    def load_state_dict(self, state_dict):
        """
        Overload load_state_dict to sync param_groups with base optimizer.
        """
        super(GSAM, self).load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
