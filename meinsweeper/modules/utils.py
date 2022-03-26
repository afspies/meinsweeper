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