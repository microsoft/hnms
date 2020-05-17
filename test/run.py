import torch
from hnms import MultiHNMS

hnms = MultiHNMS(num=1, alpha=0.7)

xywh = [[10, 20, 10, 20], [10, 20, 10, 20], [30, 6, 4, 5]]
conf = [0.9, 0.8, 0.9]
xywh = torch.tensor(xywh).float()
conf = torch.tensor(conf)
keep = hnms(xywh, conf)
print(keep)

