from paddle.base.core import (
    ProfilerOptions,
    TracerEventType,
    _Profiler,
    disable_memory_recorder,
    disable_op_info_recorder,
    enable_memory_recorder,
    enable_op_info_recorder,
    _ProfilerResult
)
import inspect

class_name="_ProfilerResult"
cls = globals()[class_name]

# 获取类的属性和方法
attributes = [attr for attr in dir(cls) if not callable(getattr(cls, attr)) and not attr.startswith("__")]
methods = [method for method in dir(cls) if callable(getattr(cls, method)) and not method.startswith("__")]

# 打印属性和方法
print(f"Attributes of class {class_name}: {attributes}")
print(f"Methods of class {class_name}: {methods}")

# 获取方法的参数信息
for method_name in methods:
    method = getattr(cls, method_name)
    signature = inspect.signature(method)
    print(f"Parameters of method {method_name}: {signature}")