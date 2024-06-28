import torch
import torchvision
import Prof.torchprof as profiler


model = torchvision.models.alexnet(pretrained=False).cuda()
x = torch.rand([1, 3, 224, 224]).cuda()
with profiler.prof(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    on_trace_ready=profiler.export_files("./data"),
    record_shapes=True,
    profile_memory=True,
    with_stack=False
) as p:
    model(x)
    
print(p.key_averages().table(
    sort_by="self_cuda_time_total", row_limit=-1))
