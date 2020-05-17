import torch
from torch import nn
import math
from hnms import _c as hnms_c


class MultiHNMS(nn.ModuleList):
    def __init__(self, num, alpha):
        all_hash_rect = []
        for i in range(num):
            curr_w0 = math.exp(1. * i / num * (-math.log(alpha)))
            curr_h0 = math.exp(1. * i / num * (-math.log(alpha)))
            bx = 1. * i / num
            by = 1. * i / num

            hr = HNMS(alpha=alpha,
                    w0=curr_w0,
                    h0=curr_h0,
                    bx=bx,
                    by=by)
            all_hash_rect.append(hr)
        super(MultiHNMS, self).__init__(all_hash_rect)

    def forward(self, rects, conf):
        for i, hr in enumerate(self):
            if i == 0:
                curr_keep = hr(rects, conf)
                keep = curr_keep
            else:
                curr_keep = hr(rects[keep], conf[keep])
                keep = keep[curr_keep]
        return keep

class HNMS(nn.Module):
    def __init__(self, alpha, w0=1., h0=1., bx=0.5, by=0.5):
        super().__init__()
        self.w0 = float(w0)
        self.h0 = float(h0)
        self.alpha = alpha
        self.bx = bx
        self.by = by

    def __call__(self, rects, conf):
        result = hnms_c.hnms(rects, conf,
                self.w0, self.h0,
                self.alpha,
                self.bx, self.by)
        return result

