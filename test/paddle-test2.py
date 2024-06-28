import paddle
import Prof.paddleprof as profiler

paddle.device.set_device("cpu")
model=paddle.vision.alexnet(pretrained=False)
x=paddle.rand([1, 3, 224, 224])
with profiler.prof(
    targets=[
        profiler.ProfilerDevice.CPU
    ],
    scheduler=(0,2),
    on_trace_ready=profiler.export_files("./data",file_type="excel"),
    timer_only=False,
) as p:
    for iter in range(10):
        model(x)
    
print(p.summary())        
        

