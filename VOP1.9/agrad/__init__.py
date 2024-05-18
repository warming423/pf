# Import all native method/classes
from stubs._autograd import (DeviceType, ProfilerActivity, ProfilerState, ProfilerConfig, ProfilerEvent,
                                _enable_profiler_legacy, _disable_profiler_legacy, _profiler_enabled,
                                _enable_record_function, _set_empty_test_observer, kineto_available,
                                _supported_kineto_activities, _add_metadata_json)
if kineto_available():
    print("kineto is available")
    from stubs._autograd import (_ProfilerResult, _KinetoEvent,
                                    _prepare_profiler, _enable_profiler, _disable_profiler)

from . import profiler
