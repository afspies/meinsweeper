import pynvml as pynvml
import psutil

#? pynvml is much faster than nvidia-smi, but an additional dependency
#? Maybe should process this via https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
def check_gpu_usage(process_exceptions=['Xorg'], user_exceptions=[''], min_memory=6, base_on_memory=True, base_on_process=True):
    # Process exceptions -> ignore these procs
    # User exceptions -> avoid GPUs if proc running from these users
    pynvml.nvmlInit()
    deviceCount = pynvml.nvmlDeviceGetCount()
    free_gpus = []
    for i in range(deviceCount):

        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free_memory = mem.free/(1024**3)
        if base_on_memory and free_memory < min_memory:
            continue

        free = True 
        if base_on_process:
            procs = [*pynvml.nvmlDeviceGetComputeRunningProcesses(handle), *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
            for p in procs:
                try:
                    process = psutil.Process(p.pid)
                except psutil.NoSuchProcess:
                    continue

                if process.name not in process_exceptions and process.username() in user_exceptions:
                    free = False
                    break
        if free:
            free_gpus.append(str(i))

    print(f"[[GPU INFO]] [{','.join(free_gpus)}] Free")
    pynvml.nvmlShutdown()


check_gpu_usage()


# # ----------------------------------------------------------------------------------------------------------------------
# import pynvml as pynvml
# import psutil
# def check_gpu_usage():
#     pynvml.nvmlInit()
#     print ("Driver Version:", pynvml.nvmlSystemGetDriverVersion())
#     deviceCount = pynvml.nvmlDeviceGetCount()
#     for i in range(deviceCount):
#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
#         mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         print ("Device", i, ":", pynvml.nvmlDeviceGetName(handle), [bytes2human(x) for x in [mem.total, mem.free, mem.used]])

#         procs = [*pynvml.nvmlDeviceGetComputeRunningProcesses(handle), *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
#         for p in procs:
#             # print(p.pid, p.usedGpuMemory if isinstance(p.usedGpuMemory, int) else -1)
#             process = psutil.Process(p.pid)
#             print(process.name(), process.username())
#     pynvml.nvmlShutdown()
# ----------------------------------------------------------------------------------------------------------------------