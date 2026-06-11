"""TENT: test-time entropy minimization on unlabeled target embeddings."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import torch


class Tent:
    """Adapt a feature-wise affine map by minimizing target prediction entropy."""

    USES_TARGET: ClassVar[bool] = True

    def __init__(
        self,
        steps: int = 20,
        lr: float = 1e-2,
        batch_size: int = 4096,
        eps: float = 1e-6,
        seed: int = 0,
    ):
        self.steps = steps
        self.lr = lr
        self.batch_size = batch_size
        self.eps = eps
        self.seed = seed

    def fit(self, x, y=None, groups=None, x_paired=None):
        self.x_target_ = x_paired
        return self

    def transform(self, x):
        return x

    def adapt_test_features(self, clf, x):
        if len(x) == 0:
            return x
        scaler = clf.named_steps.get("standardscaler")
        logistic = clf.named_steps.get("logisticregression")
        if scaler is None or logistic is None:
            return x

        torch.manual_seed(self.seed)
        x_np = np.asarray(x, dtype=np.float32)
        device = torch.device("cpu")
        xt = torch.as_tensor(x_np, dtype=torch.float32, device=device)
        gamma = torch.ones(x_np.shape[1], dtype=torch.float32, device=device, requires_grad=True)
        beta = torch.zeros(x_np.shape[1], dtype=torch.float32, device=device, requires_grad=True)
        mean = torch.as_tensor(scaler.mean_, dtype=torch.float32, device=device)
        scale = torch.as_tensor(scaler.scale_, dtype=torch.float32, device=device).clamp_min(self.eps)
        coef = torch.as_tensor(logistic.coef_, dtype=torch.float32, device=device)
        intercept = torch.as_tensor(logistic.intercept_, dtype=torch.float32, device=device)
        opt = torch.optim.Adam([gamma, beta], lr=self.lr)

        for _ in range(self.steps):
            order = torch.randperm(len(xt), device=device)
            for start in range(0, len(xt), self.batch_size):
                xb = xt[order[start : start + self.batch_size]]
                logits = (((xb * gamma + beta) - mean) / scale) @ coef.T + intercept
                logits = torch.cat([-0.5 * logits, 0.5 * logits], dim=1) if logits.shape[1] == 1 else logits
                probs = torch.softmax(logits, dim=1).clamp_min(self.eps)
                loss = -(probs * probs.log()).sum(dim=1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

        adapted = x_np * gamma.detach().cpu().numpy() + beta.detach().cpu().numpy()
        return adapted.astype(np.float32, copy=False)


def variants(task_kind: str) -> dict[str, dict]:
    return {} if task_kind == "regression" else {"tent": {}}
