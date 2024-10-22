# define the startup method of child process (eg. fork or spawn)
import multiprocessing as mp
import os


def get_start_method():
    return os.getenv('TORCH_PROFILER_START_METHOD', 'spawn')


__all__ = [x for x in dir(mp.get_context(get_start_method())) if not x.startswith('_')]
# global methods and attributes
globals().update((name, getattr(mp.get_context(get_start_method()), name)) for name in __all__) 
