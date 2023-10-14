# import pynvml as pynvml
# import psutil

#? pynvml is much faster than nvidia-smi, but an additional dependency
#? Maybe should process this via https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
import os
import subprocess
import time

# process_exceptions=['Xorg'], user_exceptions=[''], min_memory=6, base_on_memory=True, base_on_process=True
def check_gpu_usage(threshold_vram_usage=4000, max_gpus=2, wait=False, sleep_time=10):
    """
    Assigns free gpus to the current process via the CUDA_AVAILABLE_DEVICES env variable
    This function should be called after all imports,
    in case you are setting CUDA_AVAILABLE_DEVICES elsewhere
    Borrowed and fixed from https://gist.github.com/afspies/7e211b83ca5a8902849b05ded9a10696
    Args:
        threshold_vram_usage (int, optional): A GPU is considered free if the vram usage is below the threshold
                                              Defaults to 4000 (MiB).
        max_gpus (int, optional): Max GPUs is the maximum number of gpus to assign.
                                  Defaults to 2.
        wait (bool, optional): Whether to wait until a GPU is free. Default False.
        sleep_time (int, optional): Sleep time (in seconds) to wait before checking GPUs, if wait=True. Default 10.
    """

    def _check():
        # Get the list of GPUs via nvidia-smi
        smi_query_result = subprocess.check_output(
            "nvidia-smi -q -d Memory | grep -A4 GPU", shell=True
        )
        # Extract the usage information
        gpu_info = smi_query_result.decode("utf-8").split("\n")
        gpu_info = list(filter(lambda info: "Used" in info, gpu_info))
        gpu_info = [
            int(x.split(":")[1].replace("MiB", "").strip()) for x in gpu_info
        ]  # Remove garbage
        # Keep gpus under threshold only
        free_gpus = [
            str(i) for i, mem in enumerate(gpu_info) if mem < threshold_vram_usage
        ]
        free_gpus = free_gpus[: min(max_gpus, len(free_gpus))]
        gpus_to_use = ",".join(free_gpus)
        return gpus_to_use

    # while True:
        # gpus_to_use = _check()
        # if gpus_to_use or not wait:
            # break
        # print(f"No free GPUs found, retrying in {sleep_time}s")
        # time.sleep(sleep_time)

    gpus_to_use = _check()
    print(f"[[GPU INFO]] [{gpus_to_use}] Free")

# def check_gpu_usage(process_exceptions=['Xorg'], user_exceptions=[''], min_memory=6, base_on_memory=True, base_on_process=True):
#     # Process exceptions -> ignore these procs
#     # User exceptions -> avoid GPUs if proc running from these users
#     pynvml.nvmlInit()
#     deviceCount = pynvml.nvmlDeviceGetCount()
#     free_gpus = []
#     for i in range(deviceCount):

#         handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        
#         mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
#         free_memory = mem.free/(1024**3)
#         if base_on_memory and free_memory < min_memory:
#             continue

#         free = True 
#         if base_on_process:
#             procs = [*pynvml.nvmlDeviceGetComputeRunningProcesses(handle), *pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)]
#             for p in procs:
#                 try:
#                     process = psutil.Process(p.pid)
#                 except psutil.NoSuchProcess:
#                     continue

#                 if process.name not in process_exceptions and process.username() in user_exceptions:
#                     free = False
#                     break
#         if free:
#             free_gpus.append(str(i))

#     print(f"[[GPU INFO]] [{','.join(free_gpus)}] Free")
#     pynvml.nvmlShutdown()


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