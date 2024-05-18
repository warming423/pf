import torch
import torchvision
import sys
print(sys.path)
from prof.profiler import profile
import prof.profiler


print(dir(profile))

model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()
with profile(
    activities=[
        prof.profiler.DeviceActivity.CPU,
        prof.profiler.DeviceActivity.CUDA,
    ],
    on_trace_ready=prof.profiler.tensorboard_trace_handler("./")
) as p:
    model(x)
print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))
