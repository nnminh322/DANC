import torch


class FSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(FSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        # Compute the norm of the gradients
        grad_norm = self._grad_norm()
        
        # Compute the total number of parameters (d)
        d = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Loop through parameter groups
        for group in self.param_groups:
            # Compute scaling factor: γ = sqrt(rho) / sqrt(d)
            rho = group["rho"]
            gamma = torch.sqrt(torch.tensor(rho))  # γ = sqrt(rho)
            scale = gamma / torch.sqrt(torch.tensor(d).float())  # scale factor: γ / sqrt(d)

            # Loop through parameters in the current group
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Store the original parameter value
                self.state[p]["old_p"] = p.data.clone()

                # Compute the perturbation e_w
                sign_grad = torch.sign(p.grad)  # Get the sign of the gradient
                e_w = sign_grad * torch.abs(p.grad) * scale.to(p)  # perturbation: γ * (|∇L(θ)|) * 1/√d

                # Apply the perturbation (in-place update is avoided)
                p.data.add_(e_w)  # Apply the perturbation in-place: p = p + e_w
                
        # Zero gradients if specified
        if zero_grad:
            self.zero_grad()


    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()
        

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups