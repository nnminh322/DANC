import torch


def projection(source_vector, target_vector):
    a = torch.sum(source_vector * target_vector)
    long = torch.sum(target_vector**2)
    return a / long * target_vector


class GSAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(GSAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    def update_rho(self, lr_max=2e-5, lr_min=0.0, rho_max=0.05, rho_min=0.05):
        # Lấy learning rate hiện tại từ base_optimizer
        lr_t = self.base_optimizer.param_groups[0]["lr"]

        # Kiểm tra và tính toán rho mới
        if not (lr_min <= lr_t <= lr_max):
            raise ValueError(f"lr_t={lr_t} should be within [{lr_min}, {lr_max}]")

        new_rho = rho_min + (rho_max - rho_min) * (lr_t - lr_min) / (lr_max - lr_min)

        # Cập nhật rho cho từng group
        for group in self.param_groups:
            group["rho"] = new_rho

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        GSAM First Step: Perform gradient ascent to calculate w_adv.
        """
        self.update_rho()
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                # Save current parameters
                self.state[p]["old_p"] = p.data.clone()
                self.state[p]["old_grad"] = p.grad
                # Calculate ascent step
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # Move to w_adv = w + e(w)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False, alpha=0.4):
        """
        GSAM Second Step: Perform the final GSAM update step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Restore old parameters w from w_adv
                p.data = self.state[p]["old_p"]

            # Calculate surrogate gradient ∇f_GSAM = ∇f_p - α∇f_perp
            for p in group["params"]:
                if p.grad is None:
                    continue
                old_grad = self.state[p]["old_grad"]
                grad_fp = projection(old_grad, p.grad)
                # grad_fp = self._get_parallel_component(p.grad)  # Parallel gradient ∇f_p
                grad_f_perp = p.grad - grad_fp  # Orthogonal gradient ∇f_perp
                surrogate_grad = grad_fp - alpha * grad_f_perp
                p.grad.copy_(surrogate_grad)

        # Perform the gradient descent step
        self.base_optimizer.step()
        self.update_rho()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert (
            closure is not None
        ), "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(
            closure
        )  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][
            0
        ].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack(
                [
                    (p.grad).norm(p=2).to(shared_device)
                    for group in self.param_groups
                    for p in group["params"]
                    if p.grad is not None
                ]
            ),
            p=2,
        )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
