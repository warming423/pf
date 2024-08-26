import paddle
import Prof.paddleprof as profiler

# print(paddle.device.get_device())

model=paddle.vision.alexnet(pretrained=False)
x=paddle.rand([1, 3, 224, 224])
with profiler.prof(
    device=[
      profiler.ProfilerDevice.CPU,
      #profiler.ProfilerDevice.GPU
        
    ],
    schedule=(0,2),
    on_trace_ready=profiler.export_files("./data",file_type="json"),
    record_shapes=True,
    with_flops=True,
    profile_memory=False,
    timer_only=False,
) as p:
        model(x)
    
#print(p.summary())        
        

