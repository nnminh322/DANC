import torch
from collections import defaultdict

class ASAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.5, eta=0.01, **kwargs):
        defaults = dict(rho=rho, eta=eta, **kwargs)
        super(ASAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.state = defaultdict(dict)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        wgrads = []
        rho = self.defaults["rho"]
        eta = self.defaults["eta"]

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                if t_w is None:
                    t_w = torch.clone(p).detach()
                    self.state[p]["eps"] = t_w

                if "weight" in p.name:  # Áp dụng cho trọng số
                    t_w[...] = p[...]
                    t_w.abs_().add_(eta)
                    p.grad.mul_(t_w)
                wgrads.append(torch.norm(p.grad, p=2))

        wgrad_norm = torch.norm(torch.stack(wgrads), p=2) + 1.e-16

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                t_w = self.state[p].get("eps")
                if "weight" in p.name:  # Áp dụng lại trọng số
                    p.grad.mul_(t_w)
                eps = t_w
                eps[...] = p.grad[...]
                eps.mul_(rho / wgrad_norm)
                p.add_(eps)

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["eps"])
        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "ASAM requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # Closure cần thực hiện forward-backward

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

