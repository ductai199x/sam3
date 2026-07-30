# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import math

# NOTE: nothing here scales base_lr by world size, and that is deliberate.
#
# Adding a sqrt(world_size)-style multiplier is tempting when changing GPU count, since batch_size
# is per-GPU and DDP averages gradients, so more ranks means a larger effective batch at an
# unchanged step size. Resist it. Raising base_lr does not shift a stability threshold here; it
# only widens the tail of an already heavy-tailed step distribution (with an effective batch of a
# few samples, consecutive per-step losses can differ by more than an order of magnitude), and the
# failures in this codebase are discrete cliffs rather than gradual divergence.
#
# Empirically: a 1.41x multiplier produced `Loss is nan` around epoch 21 while the *decayed* LR at
# that point was 4.6x BELOW the peak the same run had already survived in epoch 1 -- so LR
# magnitude was not the proximate cause. Fix the cliff (see the NaN guards in loss_fns.py,
# decoder.py, box_ops.py and matcher.py, and the GradScaler dtype gate in trainer.py) instead of
# tuning the step size around it.


class InverseSquareRootParamScheduler:
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        cooldown_steps: int,
        timescale: int,
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.cooldown_steps = cooldown_steps
        self.timescale = timescale

    def __call__(self, step: int, where: float):
        lr = self.base_lr

        if where > 0:
            total_steps = step / where
            progress = (step - self.warmup_steps) / float(
                total_steps - self.warmup_steps
            )
            progress = max(min(progress, 1), 0)
        else:
            progress = 0
            total_steps = 1

        shift = self.timescale - self.warmup_steps
        if self.warmup_steps < step:
            lr = lr / math.sqrt((step + shift) / self.timescale)

        if self.warmup_steps:
            lr = lr * min(1.0, step / self.warmup_steps)
        if self.cooldown_steps:
            lr = lr * min(1.0, (total_steps - step) / self.cooldown_steps)

        return lr
