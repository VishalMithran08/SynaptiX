"""
Fast Exponential Moving Average.

The shadow weights live on the SAME device as the model.

This is intentional:
- the model is only ~17M parameters
- GPU memory cost is modest
- copying 17M parameters CPU<->GPU every optimizer step would severely
  reduce throughput

EMA tracks parameters, not arbitrary state_dict buffers. NAFNet does not
depend on BatchNorm running statistics, so this is appropriate for the
current architecture.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class EMA:

    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.999,
        warmup: int = 2000,
    ):
        if not 0.0 < decay < 1.0:
            raise ValueError(
                "EMA decay must be in (0,1)."
            )

        self.model = model
        self.decay = float(decay)
        self.warmup = int(warmup)
        self._step = 0

        self.shadow: Dict[str, torch.Tensor] = {
            name: param.detach().clone().float()
            for name, param
            in model.named_parameters()
        }

        self._backup = None

    def _effective_decay(self):
        if self.warmup <= 0:
            return self.decay

        return min(
            self.decay,
            (1.0 + self._step)
            / (
                self.warmup
                + self._step
            ),
        )

    @torch.no_grad()
    def update(self):

        d = self._effective_decay()

        for name, param in self.model.named_parameters():

            if not param.dtype.is_floating_point:
                continue

            self.shadow[name].mul_(d)
            self.shadow[name].add_(
                param.detach().float(),
                alpha=1.0 - d,
            )

        self._step += 1

    @torch.no_grad()
    def apply_shadow(self):

        if self._backup is not None:
            raise RuntimeError(
                "EMA shadow already applied."
            )

        self._backup = {
            name: param.detach().clone()
            for name, param
            in self.model.named_parameters()
        }

        for name, param in self.model.named_parameters():
            param.copy_(
                self.shadow[name].to(
                    device=param.device,
                    dtype=param.dtype,
                )
            )

    @torch.no_grad()
    def restore(self):

        if self._backup is None:
            raise RuntimeError(
                "EMA restore called without apply_shadow."
            )

        for name, param in self.model.named_parameters():
            param.copy_(
                self._backup[name]
            )

        self._backup = None

    def state_dict(self):
        return {
            "shadow": {
                k: v.detach().cpu().clone()
                for k, v in self.shadow.items()
            },
            "step": self._step,
            "decay": self.decay,
            "warmup": self.warmup,
        }

    def load_state_dict(self, state):

        shadow = state["shadow"]

        expected = set(
            self.shadow.keys()
        )

        received = set(
            shadow.keys()
        )

        if expected != received:
            missing = expected - received
            extra = received - expected

            raise RuntimeError(
                "EMA checkpoint mismatch. "
                f"Missing={missing}, Extra={extra}"
            )

        self.shadow = {
            name: tensor.to(
                device=next(
                    self.model.parameters()
                ).device,
                dtype=torch.float32,
            ).clone()
            for name, tensor in shadow.items()
        }

        self._step = int(
            state.get("step", 0)
        )

        self.decay = float(
            state.get(
                "decay",
                self.decay,
            )
        )

        self.warmup = int(
            state.get(
                "warmup",
                self.warmup,
            )
        )
