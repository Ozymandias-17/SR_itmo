import math
from torch.optim.lr_scheduler import _LRScheduler

class CosineAnnealingRestartLR(_LRScheduler):
    def __init__(self, optimizer, periods, restart_weights=(1, ), eta_min=1e-7, last_epoch=-1):
        self.periods = periods
        self.restart_weights = restart_weights
        self.eta_min = eta_min
        assert len(self.periods) == len(self.restart_weights), 'periods and restart_weights should have the same length.'
        self.cumulative_periods = [sum(self.periods[:i + 1]) for i in range(len(self.periods))]
        super(CosineAnnealingRestartLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        idx = self.get_period_idx()
        if idx == 0:
            current_period = self.cumulative_periods[0]
            current_idx = self.last_epoch
        else:
            current_period = self.periods[idx]
            current_idx = self.last_epoch - self.cumulative_periods[idx - 1]

        return [
            self.eta_min + (base_lr * self.restart_weights[idx] - self.eta_min) *
            (1 + math.cos(math.pi * current_idx / current_period)) / 2
            for base_lr in self.base_lrs
        ]

    def get_period_idx(self):
        for i, cumulative_period in enumerate(self.cumulative_periods):
            if self.last_epoch < cumulative_period:
                return i
        return len(self.cumulative_periods) - 1