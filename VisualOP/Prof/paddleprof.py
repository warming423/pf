import datetime
import importlib
import json,tempfile,gzip
import os,time
import socket
import pandas as pd
from enum import Enum
from typing import Any, Callable, Iterable, Optional, Union
from warnings import warn

import paddle
from paddle.base.core import (
    ProfilerOptions,
    TracerEventType,
    _Profiler,
    disable_memory_recorder,
    disable_op_info_recorder,
    enable_memory_recorder,
    enable_op_info_recorder,
)
from paddle.profiler import utils

from paddle_utils.show_statistic import (
    SortedKeys,
    StatisticData,
    _build_table,
    gen_layer_flops,
)
from paddle_utils.timer import benchmark
from paddle_utils.utils import RecordEvent, wrap_optimizers


class SummaryView(Enum):
    r"""
    - **SummaryView.DeviceView** : The device summary view.
    - **SummaryView.OverView** : The overview summary view.
    - **SummaryView.ModelView** : The model summary view.
    - **SummaryView.DistributedView** : The distributed summary view.
    - **SummaryView.KernelView** : The kernel summary view.
    - **SummaryView.OperatorView** : The operator summary view.
    - **SummaryView.MemoryView** : The memory summary view.
    - **SummaryView.MemoryManipulationView** : The meomory manipulation summary view.
    - **SummaryView.UDFView** : The user defined summary view.
    """
    DeviceView = 0
    OverView = 1
    ModelView = 2
    DistributedView = 3
    KernelView = 4
    OperatorView = 5
    MemoryView = 6
    MemoryManipulationView = 7
    UDFView = 8

class ProfilerDevice(Enum):
    r"""
    - **ProfilerTarget.CPU** : Profile events on CPU.
    - **ProfilerTarget.GPU** : Profile events on GPU.
    - **ProfilerTarget.XPU** : Profile events on XPU.
    """
    CPU = 0
    GPU = 1
    XPU = 2
    CUSTOM_DEVICE = 3


class ProfilerAction(Enum):
    r"""
    Profiler actions that can be taken at the specified intervals
    - **ProfilerState.CLOSED** : The profiler is closed, and no profiling data will be recorded.
    - **ProfilerState.READY** : The profiler is open, but the data will not be recorded. This state is used for reducing overhead influence when profiler starts.
    - **ProfilerState.RECORD** : The profiler is open, and the data will be recorded.
    - **ProfilerState.RECORD_AND_RETURN** : The profiler is open, and this state stands for the last batch of "RECORD" state in current profiling period. The collected data will be returned in this state.
    """
    CLOSED = 0
    READY = 1
    RECORD = 2
    RECORD_AND_SAVE = 3  


def schedule(*, closed: int, ready: int, record: int, repeat: int = 0, skip_first: int = 0) -> Callable:
    r"""
        (CLOSED)  (CLOSED)    (CLOSED)  (READY)    (RECORD,last SAVE)      (CLOSED)
        START -> skip_first -> closed -> ready    ->    record       ->      END
                                |                        |
                                |                        | (if has_repeated < repeat)
                                - - - - - - - - - - - -
        Note that repeat <= 0 means the cycle will continue until the profiler exits.

    Args:
        closed(int): The number of steps in state ProfilerAction.CLOSED.
        ready(int):  The number of steps in state ProfilerAction.READY.
        record(int): The number of steps in state ProfilerAction.RECORD, and the state in last step will be set as ProfilerAction.RECORD_AND_SAVE.
        repeat(int, optional): The number of cycles to repeat above state transform. Default value is 0, which means it will repeat this cycle until profiler exits.
        skip_first(int, optional): The number of first steps to drop, not participate in the state transform, and at ProfilerAction.CLOSED state. 

    Returns:
        A scheduler function, conforms to above profiler action setting. 
        The function will takes one parameter `step_num`, and returns corresponding ProfilerAction.

    Examples:
        profiling range [2, 5].Assume batch 0: closed, batch 1: ready, batch [2, 5] record.
        profiler.make_scheduler(closed=1, ready=1, record=4, repeat=1)
    """

    def getAction(step: int) -> ProfilerAction:
        assert step >= 0
        if step < skip_first:  # within skip_first, just skip
            return ProfilerAction.CLOSED
        step = step - skip_first
        num_steps = closed + ready + record
        if repeat > 0 and step // num_steps >= repeat:  # the period has repeated repeat times, return CLOSED state
            return ProfilerAction.CLOSED
        mod_step = step % num_steps
        if mod_step < closed:
            return ProfilerAction.CLOSED
        elif  mod_step < closed + ready:
            return ProfilerAction.READY
        else:
            if mod_step < num_steps - 1:
                return ProfilerAction.RECORD
            else:
                return ProfilerAction.RECORD_AND_SAVE

    assert (closed >= 0 and ready >= 0 and record > 0 \
            and repeat >= 0 and skip_first >= 0), "Invalid profiler schedule arguments"
    if ready == 0:
        warn("Profiler won't be using ready, this can skew profiler results")
    return getAction


def _default_schedule(_: int) -> ProfilerAction:
    r"""
    A default state scheduler, keep recording from the beginning of the profiler until ending.
    """
    return ProfilerAction.RECORD


def export_chrome_tracing(dir_name: str, worker_name: Optional[str] = None) -> Callable:
    r"""
    Return a callable, used for outputing tracing data to chrome tracing format file.

    Args:
        dir_name(str): Directory to save profiling data.
        worker_name(str, optional): Prefix of the file name saved, default is `[hostname]_[pid]`.

    Returns:
        A callable, which takes a Profiler object as parameter and calls its export method to save data to chrome tracing format file.

    Examples:
        profiler.prof(on_trace_ready=profiler.export_chrome_tracing('./log'))

    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(
                f"Can not create directory '{dir_name}' for saving profiling results."
            )

    def handle_fn(prof):
        nonlocal worker_name
        if not worker_name:
            worker_name = f"host_{socket.gethostname()}pid_{str(os.getpid())}"
        now = datetime.datetime.now()
        filename = '{}_time_{}.paddle_trace.json'.format(
            worker_name, now.strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        prof.export(os.path.join(dir_name, filename), "json")

    return handle_fn

def export_files(dir_name:str, file_name: Optional[str]=None, file_type: Optional[str]='json'):
    
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(f"Can not create directory '{dir_name}' for saving profiling data.")
    
    def handler_fn(prof) -> None: 
        nonlocal file_name
        if not file_name:
            worker_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        file_name = "{}.{}.pfdata".format(worker_name, int(time.time() * 1000))
        
        if file_type=='json':
            file_name=file_name+'.json'
        elif file_type=='gzip':
            file_name=file_name+'.gz'
        elif file_type== 'excel':
            file_name=file_name+'.xlsx'
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        prof.export(os.path.join(dir_name, file_name))
    return handler_fn

def export_protobuf(dir_name: str, worker_name: Optional[str] = None) -> Callable:
    r"""
    Return a callable, used for outputing tracing data to protobuf file.

    Args:
        dir_name(str): Directory to save profiling data.
        worker_name(str, optional): Prefix of the file name saved, default is `[hostname]_[pid]`.
        
    example:
        profiler.prof(on_trace_ready = profiler.export_protobuf('./log'))
            
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(
                f"Can not create directory '{dir_name}' for saving profiling results."
            )

    def handle_fn(prof):
        nonlocal worker_name
        if not worker_name:
            worker_name = f"host_{socket.gethostname()}pid_{str(os.getpid())}"
        now = datetime.datetime.now()
        filename = '{}_time_{}.paddle_trace.pb'.format(
            worker_name, now.strftime('%Y_%m_%d_%H_%M_%S_%f')
        )
        prof.export(os.path.join(dir_name, filename), "pb")

    return handle_fn


def _get_supported_device() -> Iterable[ProfilerDevice]:
    r"""
    Get the current supported profiler device in the system.
    """
    if _Profiler.is_cupti_supported():
        return [
            ProfilerDevice.CPU,
            ProfilerDevice.GPU,
            ProfilerDevice.CUSTOM_DEVICE,
        ]
    if _Profiler.is_cnpapi_supported():
        return [
            ProfilerDevice.CPU,
            ProfilerDevice.CUSTOM_DEVICE,
        ]
    if _Profiler.is_xpti_supported():
        return [
            ProfilerDevice.CPU,
            ProfilerDevice.XPU,
            ProfilerDevice.CUSTOM_DEVICE,
        ]
    return [ProfilerDevice.CPU, ProfilerDevice.CUSTOM_DEVICE]

class profile(_Profiler):
    pass

class prof:
    r"""
    Profiler context manager, user interface to manage profiling process to start, stop, export profiling data and print summary table.

    Args:
        targets (list, optional): specify target devices to profile, and all existing and supported devices will be chosen by default. 
        Currently supported values, CPU, GPU, XPU, CUSTOM_DEVICE.
        using example: p = profiler.prof(targets=[profiler.ProfilerTarget.CPU, profiler.ProfilerTarget.GPU])
        
        scheduler (Callable|tuple, optional): If it is a callable object, it takes a step number as parameter and return the corresponding working "ProfilerState".
        This callable object can be generated by "make_scheduler" function.If not provided (None), the default scheduler will keep tracing until the profiler exits.
        If it is a tuple, it has two values start_batch and end_batch, which means profiling range [start_batch, end_batch).
        using example: p = profiler.Profiler(scheduler = (2, 5)) or p = profiler.Profiler(scheduler = profiler.make_scheduler(closed=1, ready=1, record=3, repeat=3))
    
        on_trace_ready (Callable, optional): Callable object, serves as callback function, and takes the Profiler object as parameter, which provides a way to do post-processing.
            This callable object will be called when `scheduler` returns `ProfilerState.RECORD_AND_RETURN`. 
            The default value is :ref:`export_chrome_tracing <api_paddle_profiler_export_chrome_tracing>`.
        using example: p = profiler.Profiler(on_trace_ready=profiler.export_chrome_tracing("./data"))
        
        timer_only (bool, optional): If it is True, the cost of Dataloader and every step of the model will be count without profiling.
            Otherwise, the model will be timed and profiled. Default: False.
            
        record_shapes (bool, optional): If it is True, collect op's input shape information. Default: False.
        
        profile_memory (bool, optional): If it is True, collect tensor memory allocation and release information. Default: False.
        
        custom_device_types (list, optional): If targets contain profiler.ProfilerTarget.CUSTOM_DEVICE, custom_device_types select the custom device type for profiling. 
        The default value represents all custom devices will be selected.
        
        with_flops (bool, optional): If it is True, the flops of the op will be calculated. Default: False.

    Examples:
        1. Use profiler with context manager
            model=paddle.vision.alexnet(pretrained=False)
            x=paddle.rand([1, 3, 224, 224])
            with profiler.torchprof(activities=[profiler.ProfilerActivity.CPU,profiler.ProfilerActivity.CUDA],
                on_trace_ready=profiler.tensorboard_trace_handler("./data",worker_name="test1"),
                record_shapes=True,
                profile_memory=True,
                with_flops=False,
                with_stack=False
            ) as p:
                model(x)
            print(p.summary())        

        2. Use profiler without context manager
            p = profiler.Profiler()
            p.start()
            for iter in range(10):
                #train()
                p.step()
                p.stop()
            p.summary()
    """
    
    def __init__(
        self,
        *,
        targets: Optional[Iterable[ProfilerDevice]] = None,
        scheduler: Union[Callable[[int], ProfilerAction], tuple, None] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes: Optional[bool] = False,
        profile_memory: Optional[bool] = False,
        timer_only: Optional[bool] = False,
        custom_device_types: Optional[list] = [],
        with_flops: Optional[bool] = False,
    ):
        supported_targets = _get_supported_device()
        if targets:
            self.targets = set(targets)
            for target in targets:
                if target not in supported_targets:
                    self.targets.remove(target)
                    warn(
                        f"Profiling {target} is not supported now."
                    )
        else:
            self.targets = supported_targets
            
        profileoption = ProfilerOptions()
        if ProfilerDevice.CPU in self.targets:
            profileoption.trace_switch |= 1
        if ProfilerDevice.GPU in self.targets:
            profileoption.trace_switch |= 1 << 1
        if ProfilerDevice.XPU in self.targets:
            profileoption.trace_switch |= 1 << 2
        if ProfilerDevice.CUSTOM_DEVICE in self.targets:
            profileoption.trace_switch |= 1 << 3
            if not custom_device_types:
                custom_device_types = paddle.device.get_all_custom_device_type()
        wrap_optimizers()
        self.profiler = _Profiler.create(profileoption, custom_device_types)
        if callable(scheduler):
            self.scheduler = scheduler
        elif isinstance(scheduler, (tuple, list)):
            assert len(scheduler) == 2 and scheduler[1] > scheduler[0]
            start_batch, end_batch = scheduler
            start_batch = max(start_batch, 0)
            if start_batch >= 1:
                self.scheduler = schedule(
                    closed=max(start_batch - 1, 0),
                    ready=1,
                    record=(end_batch - start_batch),
                    repeat=1,
                )
            else:
                self.scheduler = schedule(
                    closed=0,
                    ready=0,
                    record=(end_batch - start_batch),
                    repeat=1,
                )
        else:
            self.scheduler = _default_schedule

        if on_trace_ready is None:
            self.on_trace_ready = export_chrome_tracing('./profiling_data/')
        else:
            self.on_trace_ready = on_trace_ready
        self.step_num = 0
        self.previous_state = ProfilerAction.CLOSED
        self.current_state = self.scheduler(self.step_num)
        self.record_event = None
        self.profiler_result = None
        self.timer_only = timer_only
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_flops = with_flops

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def start(self):
        r'''
        Start profiler and enter the first profiler step(0).
        State transformed from CLOSED to self.current_state and trigger corresponding action.
        '''
        # Timing only without profiling.
        benchmark().begin()
        if not self.timer_only:
            utils._is_profiler_used = True
        if self.timer_only:
            return
        if self.record_shapes or self.with_flops:
            enable_op_info_recorder()
        if self.profile_memory:
            enable_memory_recorder()
        # CLOSED -> self.current_state
        if self.current_state == ProfilerAction.READY:
            self.profiler.prepare()
        elif self.current_state == ProfilerAction.RECORD:
            self.profiler.prepare()
            self.profiler.start()
        elif self.current_state == ProfilerAction.RECORD_AND_SAVE:
            self.profiler.prepare()
            self.profiler.start()
        self.record_event = RecordEvent(
            name=f"ProfileStep#{self.step_num}",
            event_type=TracerEventType.ProfileStep,
        )
        self.record_event.begin()

    def stop(self):
        r'''
        Stop profiler and State transformed from self.current_state to CLOSED.
        Trigger corresponding action and post-process profiler result using self.on_trace_ready if result exists.
        
        '''
        benchmark().end()
        if self.timer_only:
            return
        if self.record_shapes or self.with_flops:
            disable_op_info_recorder()
        if self.profile_memory:
            disable_memory_recorder()
        # self.current_state -> CLOSED
        # In this situation, RECORD state is regarded as RECORD_AND_RETURN.
        if self.record_event:
            self.record_event.end()
            self.record_event = None
        if self.current_state == ProfilerAction.READY:
            warn(
                "Inproper Profiler state transform: READY->CLOSED, profiler will start and stop without saving data"
            )
            self.profiler.start()
            self.profiler.stop()
        if (
            self.current_state == ProfilerAction.RECORD
            or self.current_state == ProfilerAction.RECORD_AND_SAVE
        ):
            self.profiler_result = self.profiler.stop()
            if self.on_trace_ready:
                self.on_trace_ready(self)
        utils._is_profiler_used = False

    def step(self, num_samples: Optional[int] = None):
        r"""
        Signals the profiler that the next profiling step has started.
        Get the new ProfilerState and trigger corresponding action.

        Args:
            num_samples (int|None, optional): Specifies the batch size of every step of the model
                that is used to compute throughput when `timer_only` is True. Default: None.
        """
        benchmark().step(num_samples)
        if self.timer_only:
            return
        if self.record_event:
            self.record_event.end()
            self.record_event = None
        self.previous_state = self.current_state
        self.step_num += 1
        self.current_state = self.scheduler(self.step_num)
        self._trigger_action()
        self.record_event = RecordEvent(
            name=f"ProfileStep#{self.step_num}",
            event_type=TracerEventType.ProfileStep,
        )
        self.record_event.begin()        

    def _trigger_action(self):
        if self.previous_state == ProfilerAction.CLOSED:
            if self.current_state == ProfilerAction.READY:  # CLOSED -> READY
                self.profiler.prepare()
            if self.current_state == ProfilerAction.RECORD:  # CLOSED -> RECORD
                self.profiler.prepare()
                self.profiler.start()
            if (
                self.current_state == ProfilerAction.RECORD_AND_SAVE
            ):  # CLOSED -> RECORD_AND_RETURN
                self.profiler.prepare()
                self.profiler.start()

        elif self.previous_state == ProfilerAction.READY:
            if self.current_state == ProfilerAction.CLOSED:  # READY -> CLOSED
                warn(
                    "Improper schedule: READY->CLOSED, profiler will start and stop without saving data"
                )
                self.profiler.start()
                self.profiler.stop()
            if self.current_state == ProfilerAction.RECORD:  # READY -> RECORD
                self.profiler.start()
            if (
                self.current_state == ProfilerAction.RECORD_AND_SAVE
            ):  # READY -> RECORD_AND_RETURN
                self.profiler.start()

        elif self.previous_state == ProfilerAction.RECORD:
            if self.current_state == ProfilerAction.CLOSED:  # RECORD -> CLOSED
                warn(
                    "Improper schedule: RECORD->CLOSED, profiler will not saving data"
                )
                self.profiler.stop()

            if self.current_state == ProfilerAction.READY:  # RECORD -> READY
                warn(
                    "Improper schedule: RECORD->READY, profiler will stop and re-prepare"
                )
                self.profiler.stop()
                self.profiler.prepare()
            if (
                self.current_state == ProfilerAction.RECORD_AND_SAVE
            ):  # RECORD -> RECORD_AND_RETURN
                pass

        else:
            assert self.previous_state == ProfilerAction.RECORD_AND_SAVE
            if (
                self.current_state == ProfilerAction.CLOSED
            ):  # RECORD_AND_RETURN -> CLOSED
                self.profiler_result = self.profiler.stop()
            if (
                self.current_state == ProfilerAction.READY
            ):  # RECORD_AND_RETURN -> READY
                self.profiler_result = self.profiler.stop()
                self.profiler.prepare()
            if (
                self.current_state == ProfilerAction.RECORD
            ):  # RECORD_AND_RETURN -> RECORD
                self.profiler_result = self.profiler.stop()
                self.profiler.prepare()
                self.profiler.start()
            if (
                self.current_state == ProfilerAction.RECORD_AND_SAVE
            ):  # RECORD_AND_RETURN -> RECORD_AND_RETURN
                self.profiler_result = self.profiler.stop()
                self.profiler.prepare()
                self.profiler.start()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    # def export(self, path="", format="json"):
    #     r"""
    #     Exports the tracing data to file.

    #     Args:
    #         path(str): file path of the output.
    #         format(str, optional): output format, can be chosen from ['json', 'pb'], 'json' for chrome tracing and 'pb' for protobuf, default value is 'json'.
    #     Examples:
    #         prof.export(path="./profiler_data.json", format="json")
    #     """
    #     if self.profiler_result:
    #         self.profiler_result.save(path, format)
            
    def export(self, path:str):
        """
        Exports the collected trace in Chrome JSON format.
        """
        assert self.profiler_result
        
        if path.endswith('.gz'):
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
            fp.close()
            retvalue = self.profiler_result.save(fp.name,"json")
            with open(fp.name) as fin:
                with gzip.open(path, 'wt') as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
              
        if path.endswith('.xlsx'):
            fp=tempfile.NamedTemporaryFile('w+t',suffix='.json',delete=False)
            fp.close()
            retvalue = self.profiler_result.save(fp.name,"json")
            with open(fp.name, 'r', encoding='utf-8') as f:
                trace_data = json.load(f)
                trace_events = trace_data.get('traceEvents', [])
                normalized_data = []
                for event in trace_events:
                    normalized_data.append(pd.json_normalize(event))
                if normalized_data:
                    trace_df = pd.concat(normalized_data, ignore_index=True)
                    trace_df.to_excel(path, index=False)
                else:               
                    # If there are no trace events, create an empty DataFrame and save
                    pd.DataFrame().to_excel(path, index=False)
            return retvalue
        else:
            return self.profiler_result.save(path,"json")
        
    def summary(
        self,
        sorted_by=SortedKeys.CPUTotal,
        op_detail=True,
        thread_sep=False,
        time_unit='ms',
        views=None,
    ):
        r"""
        Print the Summary table. Currently support overview, model, distributed, operator, memory manipulation and userdefined summary.

        Args:
            sorted_by( :ref:`SortedKeys <api_paddle_profiler_SortedKeys>` , optional): how to rank the op table items, default value is SortedKeys.CPUTotal.
            op_detail(bool, optional): expand each operator detail information, default value is True.
            thread_sep(bool, optional): print op table each thread, default value is False.
            time_unit(str, optional): time unit for display, can be chosen form ['s', 'ms', 'us', 'ns'], default value is 'ms'.
            views(SummaryView|list[SummaryView], optional): summary tables to print, default to None means all views to be printed.
       
        examples:
        prof.summary(sorted_by=profiler.SortedKeys.CPUTotal, op_detail=True, thread_sep=False, time_unit='ms')
        """
        if isinstance(views, SummaryView):
            views = [views]

        if self.profiler_result:
            statistic_data = StatisticData(
                self.profiler_result.get_data(),
                self.profiler_result.get_extra_info(),
            )
            print(
                _build_table(
                    statistic_data,
                    sorted_by=sorted_by,
                    op_detail=op_detail,
                    thread_sep=thread_sep,
                    time_unit=time_unit,
                    views=views,
                )
            )

        if self.with_flops:
            self._print_flops()

    def _print_flops(self, repeat=1):
        if not self.with_flops:
            print('ERROR: with_flops disabled.')
            return

        print(" Flops Profiler Begin ".center(100, "-"))
        print(gen_layer_flops(self.profiler_result.get_data(), repeat))
        print("- Flops Profiler End -".center(100, "-"))

