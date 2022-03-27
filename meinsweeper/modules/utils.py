# ----------------------------------------------------------------------------------------------------------------------
# Taken from https://github.com/XuehaiPan/nvitop/blob/main/nvitop/core/utils.py
KiB = 1 << 10
MiB = 1 << 20
GiB = 1 << 30
TiB = 1 << 40
PiB = 1 << 50
NA = -1

def bytes2human(x):  # pylint: disable=too-many-return-statements
    if x is None or x == NA:
        return NA

    if not isinstance(x, int):
        try:
            x = round(float(x))
        except ValueError:
            return NA

    if x < KiB:
        return f'{x}B'
    if x < MiB:
        return f'{round(x / KiB)}KiB'
    if x <= 20 * GiB:
        return f'{round(x / MiB)}MiB'
    if x < 100 * GiB:
        return '{:.2f}GiB'.format(round(x / GiB, 2))
    if x < 1000 * GiB:
        return '{:.1f}GiB'.format(round(x / GiB, 1))
    if x < 100 * TiB:
        return '{:.2f}TiB'.format(round(x / TiB, 2))
    if x < 1000 * TiB:
        return '{:.1f}TiB'.format(round(x / TiB, 1))
    if x < 100 * PiB:
        return '{:.2f}PiB'.format(round(x / PiB, 2))
    return '{:.1f}PiB'.format(round(x / PiB, 1))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
import itertools
# Taken from https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
import pynvml as pynvml
import psutil
def check_gpu_usage():
    pynvml.nvmlInit()
    print ("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        print ("Device", i, ":", pynvml.nvmlDeviceGetName(handle), [bytes2human(x) for x in [mem.total, mem.free, mem.used]])

        procs = [*pynvml.nvmlDeviceGetComputeRunningProcesses(handle), *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
        for p in procs:
            # print(p.pid, p.usedGpuMemory if isinstance(p.usedGpuMemory, int) else -1)
            process = psutil.Process(p.pid)
            print(process.name(), process.username())
    pynvml.nvmlShutdown()
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
import subprocess, os
# This function should be called after all imports,
# in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
def assign_free_gpus(threshold_vram_usage=1500, max_gpus=2):
    """Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable

    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 1500 (MiB).
                                              
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
    """
    # Get the list of GPUs via nvidia-smi
    smi_query_result = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU', shell=True)
    # Extract the usage information
    gpu_info = smi_query_result.decode('utf-8').split('\n')
    gpu_info = list(filter(lambda info: 'Used' in info, gpu_info))
    gpu_info = [int(x.split(':')[1].replace('MiB', '').strip()) for x in gpu_info] # Remove garbage
    gpu_info = gpu_info[:min(max_gpus, len(gpu_info))] # Limit to max_gpus
    # Assign free gpus to the current process
    gpus_to_use = ','.join([str(i) for i, x in enumerate(gpu_info) if x < threshold_vram_usage])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpus_to_use
    # print(f'Using GPUs {gpus_to_use}' if gpus_to_use else 'No free GPUs found')
# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# Random Stuff
from pathlib import Path
from contextlib import contextmanager
# Taken from https://stackoverflow.com/questions/41742317/how-can-i-change-directory-with-python-pathlib
@contextmanager
def set_directory(path: Path):
    # Sets the cwd within the contex
    origin = Path().absolute()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(origin)

from datetime import datetime
from time import time
def get_time_diff(start_time):
    # It took me almost 30 minutes to get this working
    # Kill me 
    # Expects starttime to be a datetime time
    if isinstance(start_time, float):
        start_time = datetime.fromtimestamp(start_time)
    dt = datetime.fromtimestamp(time()) - start_time
    hours, remainder = divmod(dt.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    f = ''
    if hours > 0:
        f += f'{hours} Hours, '
    if minutes > 0:
        f += f'{minutes} Minutes '
    if  (hours + minutes) > 0:
        f += 'and '
    f += f'{seconds} Seconds'
    return f 
# ----------------------------------------------------------------------------------------------------------------------