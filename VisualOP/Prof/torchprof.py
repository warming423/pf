import gzip
import json
import os
import pandas as pd
import socket
import time
import tempfile
from enum import Enum
from typing import Any, Callable, Iterable, Optional
from warnings import warn

import torch
from torch_utils.profiler import profile, record_function
from torch.autograd import kineto_available, ProfilerActivity


class ProfilerDevice(ProfilerActivity):
    pass


class ProfilerAction(Enum):
    """
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


def schedule(*,
             closed: int,
             ready: int,
             record: int,
             repeat: int = 0,
             skip_first: int = 0) -> Callable:
    """
        (CLOSED)  (CLOSED)    (CLOSED)  (READY)    (RECORD,last SAVE)      (CLOSED)
        START -> skip_first -> closed -> ready    ->    record       ->      END
                                |                        |
                                |                        | (if has_repeated < repeat)
                                - - - - - - - - - - - -
        Note that repeat <= 0 means the cycle will continue until the profiler exits.

    Args:
        closed(int): The number of steps in state ProfilerState.CLOSED.
        ready(int):  The number of steps in state ProfilerState.READY.
        record(int): The number of steps in state ProfilerState.RECORD, and the state in last step will be set as ProfilerState.RECORD_AND_SAVE.
        repeat(int, optional): The number of cycles to repeat above state transform. Default value is 0, which means it will repeat this cycle until profiler exits.
        skip_first(int, optional): The number of first steps to drop, not participate in the state transform, and at ProfilerState.CLOSED state. 

    Returns:
        A scheduler function, conforms to above profiler action setting. 
        The function will takes one parameter `step_num`, and returns corresponding ProfilerAction.

    Examples:
        profiling range [2, 5].Assume batch 0: closed, batch 1: ready, batch [2, 5] record.
        profiler.make_scheduler(closed=1, ready=1, record=4, repeat=1)
    """

    def getAction(step: int) -> ProfilerAction:
        assert step >= 0
        if step < skip_first:  # less than skip_first, just skip
            return ProfilerAction.CLOSED
        step -= skip_first
        num_steps = closed + ready + record
        if repeat > 0 and step // num_steps >= repeat:
            return ProfilerAction.CLOSED
        mod_step = step % num_steps
        if mod_step < closed:
            return ProfilerAction.CLOSED
        elif mod_step < closed + ready:
            return ProfilerAction.READY
        else:
            if mod_step < num_steps - 1:
                return ProfilerAction.RECORD
            else:
                return ProfilerAction.RECORD_AND_SAVE

    assert (closed >= 0 and ready >= 0 and record > 0 and \
           repeat >= 0 and skip_first >= 0), "Invalid profiler schedule arguments"
    if ready == 0:
        warn("Profiler won't be using ready, this can skew profiler results")
    return getAction


def _default_schedule(_: int) -> ProfilerAction:
    """
    A default state scheduler, keep recording from the beginning of the profiler until ending.
    """
    return ProfilerAction.RECORD

def export_files(dir_name: str,
                 file_name: Optional[str] = None,
                 file_type: Optional[str] = 'json'):
    '''
    Args:
        dir_name(str): Directory to save profiling data.
        file_name(str, optional): name of file, default is `[hostname]_[pid]`.
        file_type(str,optional): type of file, default is json. The value can be: json, gzip, excel, protobuf.

    Returns:
        A callable, which takes a profiler object as parameter and calls its export method to save data to file.

    Examples:
        profiler.prof(on_trace_ready=profiler.export_file('./log',file_name="your_file_name",file_type="gzip"))
    '''
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name, exist_ok=True)
        except Exception:
            raise RuntimeError(f"Can not create directory '{dir_name}' for saving profiling data.")

    def handler_fn(prof) -> None:
        nonlocal file_name
        if not file_name:
            file_name = "{}_{}".format(socket.gethostname(), str(os.getpid()))
        last_name = "{}.{}.pt.trace".format(file_name, int(time.time() * 1000))

        if file_type == 'json':
            last_name = last_name + '.json'
        elif file_type == 'gzip':
            last_name = last_name + '.gz'
        elif file_type == 'excel':
            last_name = last_name + '.xlsx'
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        prof.export(os.path.join(dir_name, last_name))

    return handler_fn


def supported_activities():
    """
    Returns a set of supported profiler tracing activities.

    Note: profiler uses CUPTI library to trace on-device CUDA kernels.
    In case when CUDA is enabled but CUPTI is not available, passing
    ``ProfilerActivity.CUDA`` to profiler results in using the legacy CUDA
    profiling code (same as in the legacy ``torch.autograd.profiler``).
    This, in turn, results in including CUDA time in the profiler table output,
    but not in the JSON trace.
    """
    return torch.autograd._supported_kineto_activities()


class prof():
    """Profiler context manager.

    Args:
        activities (iterable): list of activity groups (CPU, CUDA) to use in profiling, supported values:
            ``torch.profiler.ProfilerActivity.CPU``, ``torch.profiler.ProfilerActivity.CUDA``.
            Default value: ProfilerActivity.CPU and (when available) ProfilerActivity.CUDA.
        schedule (callable): callable that takes step (int) as a single parameter and returns
            ``ProfilerAction`` value that specifies the profiler action to perform at each step.
        on_trace_ready (callable): callable that is called at each step when ``schedule``
            returns ``ProfilerAction.RECORD_AND_SAVE`` during the profiling.
        record_shapes (bool): save information about operator's input shapes.
        profile_memory (bool): track tensor memory allocation/deallocation.
        with_stack (bool): record source information (file and line number) for the ops.
        with_flops (bool): use formula to estimate the FLOPS of specific operators
            (matrix multiplication and 2D convolution).

    .. note::
        Use :func:`~torch.profiler.schedule` to generate the callable schedule.
        Non-default schedules are useful when profiling long training jobs
        and allow the user to obtain multiple traces at the different iterations
        of the training process.
        The default schedule simply records all the events continuously for the
        duration of the context manager.

    .. note::
        Use :func:`~torch.profiler.tensorboard_trace_handler` to generate result files for TensorBoard:

        ``on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)``

        After profiling, result files can be found in the specified directory. Use the command:

        ``tensorboard --logdir dir_name``

        to see the results in TensorBoard.
        For more information, see
        `PyTorch Profiler TensorBoard Plugin <https://github.com/pytorch/kineto/tree/master/tb_plugin>`__

    .. note::
        Enabling shape and stack tracing results in additional overhead.

    Examples:

    .. code-block:: python

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as p:
            code_to_profile()
        print(p.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    Using the profiler's ``schedule``, ``on_trace_ready`` and ``step`` functions:

    .. code-block:: python

        # Non-default profiler schedule allows user to turn profiler on and off
        # on different iterations of the training loop;
        # trace_handler is called every time a new trace becomes available
        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],

            # In this example with wait=1, warmup=1, active=2,
            # profiler will skip the first step/iteration,
            # start warming up on the second, record
            # the third and the forth iterations,
            # after which the trace will become available
            # and on_trace_ready (when set) is called;
            # the cycle repeats starting with the next step

            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=2),
            on_trace_ready=trace_handler
            # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
            # used when outputting for tensorboard
            ) as p:
                for iter in range(N):
                    code_iteration_to_profile(iter)
                    # send a signal to the profiler that the next iteration has started
                    p.step()
    """

    def __init__(
            self,
            *,
            activities: Optional[Iterable[ProfilerActivity]] = None,
            schedule: Optional[Callable[[int], ProfilerAction]] = None,
            on_trace_ready: Optional[Callable[..., Any]] = None,
            record_shapes: bool = False,
            profile_memory: bool = False,
            with_stack: bool = False,
            with_flops: bool = False,          
        ):
        if activities:
            self.activities = set(activities)
        else:
            self.activities = supported_activities()

        assert len(self.activities) > 0, "No valid profiler activities found"

        if schedule:
            self.schedule = schedule
            # add step markers into the trace and table view
            self.record_steps = True
        else:
            self.schedule = _default_schedule
            self.record_steps = False
        self.on_trace_ready = on_trace_ready
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.step_num = 0
        self.current_action = self.schedule(self.step_num)
        self.profiler: Optional[profile] = None
        self.step_rec_fn: Optional[record_function] = None

    def __enter__(self):
        self._enter_actions()
        if self.record_steps:
            self.step_rec_fn = record_function("ProfilerStep#" +
                                               str(self.step_num))
            self.step_rec_fn.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        self._exit_actions()

    def step(self):
        """
        Signals the profiler that the next profiling step has started.
        """
        if self.record_steps and self.step_rec_fn:
            self.step_rec_fn.__exit__(None, None, None)
        prev_action = self.current_action
        self.step_num += 1
        self.current_action = self.schedule(self.step_num)

        if self.current_action == ProfilerAction.CLOSED:
            if prev_action == ProfilerAction.CLOSED:
                pass
            elif prev_action == ProfilerAction.READY:
                warn("Incorrect schedule: WARMUP followed by NONE")
                self._start_trace()
                self._stop_trace()
            elif prev_action == ProfilerAction.RECORD:
                warn("Incorrect schedule: RECORD followed by NONE")
                self._stop_trace()
            else:
                assert prev_action == ProfilerAction.RECORD_AND_SAVE
                self._stop_trace()
                if self.on_trace_ready:
                    self.on_trace_ready(self)
        elif self.current_action == ProfilerAction.READY:
            if prev_action == ProfilerAction.CLOSED:
                self._start_warmup()
            elif prev_action == ProfilerAction.READY:
                pass
            elif prev_action == ProfilerAction.RECORD:
                warn("Incorrect schedule: RECORD followed by WARMUP")
                self._stop_trace()
            else:
                assert prev_action == ProfilerAction.RECORD_AND_SAVE
                self._stop_trace()
                if self.on_trace_ready:
                    self.on_trace_ready(self)
                self._start_warmup()
        elif self.current_action in \
                [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]:
            if prev_action == ProfilerAction.CLOSED:
                self._start_warmup()
                self._start_trace()
            elif prev_action == ProfilerAction.READY:
                self._start_trace()
            elif prev_action == ProfilerAction.RECORD:
                pass
            else:
                assert prev_action == ProfilerAction.RECORD_AND_SAVE
                self._stop_trace()
                if self.on_trace_ready:
                    self.on_trace_ready(self)
                self._start_warmup()
                self._start_trace()

        if self.record_steps:
            self.step_rec_fn = record_function("ProfilerStep#" +
                                               str(self.step_num))
            self.step_rec_fn.__enter__()

    def export(self, path: str):
        """
        Exports the collected trace in Chrome JSON format.
        """
        assert self.profiler
        if path.endswith('.gz'):
            fp = tempfile.NamedTemporaryFile('w+t', suffix='.json', delete=False)
            fp.close()
            retvalue = self.profiler.export_chrome_trace(fp.name)
            with open(fp.name) as fin:
                with gzip.open(path, 'wt') as fout:
                    fout.writelines(fin)
            os.remove(fp.name)
            return retvalue
        
        if path.endswith('.xlsx'):
            fp=tempfile.NamedTemporaryFile('w+t',suffix='.json',delete=False)
            fp.close()
            retvalue = self.profiler.export_chrome_trace(fp.name)
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
            return self.profiler.export_chrome_trace(path)

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        """Save stack traces in a file in a format suitable for visualization.

        Args:
            path (str): save stacks file to this location;
            metric (str): metric to use: "self_cpu_time_total" or "self_cuda_time_total"

        .. note::
            Example of using FlameGraph tool:

            - git clone https://github.com/brendangregg/FlameGraph
            - cd FlameGraph
            - ./flamegraph.pl --title "CPU time" --countname "us." profiler.stacks > perf_viz.svg
        """
        assert self.profiler
        return self.profiler.export_stacks(path, metric)

    def key_averages(self,
                     group_by_input_shape: bool = False,
                     group_by_stack_n: int = 0):
        """Averages events, grouping them by operator name and (optionally) input shapes and
        stack.

        .. note::
            To use shape/stack functionality make sure to set record_shapes/with_stack
            when creating profiler context manager.
        """
        assert self.profiler
        return self.profiler.key_averages(group_by_input_shape,
                                          group_by_stack_n)

    def events(self):
        """
        Returns the list of unaggregated profiler events,
        to be used in the trace callback or after the profiling is finished
        """
        assert self.profiler
        return self.profiler.function_events

    def add_metadata(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a string value
        into the trace file
        """
        wrapped_value = "\"" + value.replace('"', '\\"') + "\""
        torch.autograd._add_metadata_json(key, wrapped_value)

    def add_metadata_json(self, key: str, value: str):
        """
        Adds a user defined metadata with a string key and a valid json value
        into the trace file
        """
        torch.autograd._add_metadata_json(key, value)

    def _get_distributed_info(self):
        import torch.distributed as dist
        if not dist.is_available() or not dist.is_initialized():
            return None

        return {
            "backend": dist.get_backend(),
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size()
        }

    def _enter_actions(self):
        if self.current_action == ProfilerAction.READY:
            self._start_warmup()
        elif self.current_action in \
                [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]:
            self._start_warmup()
            self._start_trace()

    def _exit_actions(self):
        if self.current_action == ProfilerAction.READY:
            self._start_trace()
            self._stop_trace()
        elif self.current_action in \
                [ProfilerAction.RECORD, ProfilerAction.RECORD_AND_SAVE]:
            self._stop_trace()
            if self.on_trace_ready:
                self.on_trace_ready(self)

    def _start_warmup(self):
        self.profiler = profile(
            use_cuda=(ProfilerActivity.CUDA in self.activities),
            use_cpu=(ProfilerActivity.CPU in self.activities),
            record_shapes=self.record_shapes,
            with_flops=self.with_flops,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            use_kineto=True,
        )
        self.profiler._prepare_trace()

    def _start_trace(self):
        assert self.profiler is not None
        self.profiler._start_trace()

        if kineto_available():
            dist_info = self._get_distributed_info()
            if dist_info:
                self.add_metadata_json("distributedInfo",
                                       json.dumps(dist_info))

    def _stop_trace(self):
        assert self.profiler is not None
        self.profiler.__exit__(None, None, None)
