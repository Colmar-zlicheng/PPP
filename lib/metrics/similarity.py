import math
from typing import Dict

from lib.metrics.basic_metric import AverageMeter, Metric


class PSNR(Metric):
    def __init__(self, cfg=None, name="") -> None:
        super().__init__()
        self.avg_meter = AverageMeter()
        self.name = "PSNR"
        self.reset()

    def reset(self):
        self.avg_meter.reset()

    def feed(self, img0, img_1, **kwargs):
        mse = ((img0 - img_1) ** 2).mean()
        psnr = -10 * math.log10(mse)
        self.avg_meter.update(psnr, n=1)
        return psnr

    def get_measures(self, **kwargs) -> Dict[str, float]:
        return {self.name: self.avg_meter.avg}

    def get_result(self):
        return self.avg_meter.avg

    def __str__(self):
        return f"{self.name}: {self.avg_meter.avg:6.4f}"
