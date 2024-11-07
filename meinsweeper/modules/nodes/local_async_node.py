import asyncio
import os
import subprocess
from pathlib import Path
import tempfile
import shlex
import threading
import sys
import traceback

from meinsweeper.modules.helpers.debug_logging import init_node_logger, DEBUG
from meinsweeper.modules.helpers.utils import timeout_iterator
from .abstract import ComputeNode

# Use environment variables with default values
MINIMUM_VRAM = int(os.environ.get('MINIMUM_VRAM', 10)) * 1024  # Convert GB to MB
USAGE_CRITERION = float(os.environ.get('USAGE_CRITERION', 0.1))

class LocalAsyncNode(ComputeNode):
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, node_name, log_q, available_gpus, timeout=1200):
        with cls._lock:
            if node_name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[node_name] = instance
                instance.initialized = False
            return cls._instances[node_name]

    def __init__(self, node_name, log_q, available_gpus, timeout=1200):
        if not hasattr(self, 'initialized') or not self.initialized:
            self.log_q = log_q
            self.name = node_name
            self.available_gpus = available_gpus
            self.RUN_TIMEOUT = timeout
            self.node_logger = init_node_logger(self.name)
            self.temp_dirs = {}
            self.log_lock = threading.Lock()
            self.free_gpus = []
            self.initialized = True
            self.log_info("INIT", f"LocalAsyncNode initialized with name: {self.name}, available GPUs: {self.available_gpus}")
        elif self.name != node_name or self.available_gpus != available_gpus:
            # If the instance exists but with different parameters, update it
            self.name = node_name
            self.available_gpus = available_gpus
            self.RUN_TIMEOUT = timeout
            self.node_logger = init_node_logger(self.name)
            self.log_info("REINIT", f"LocalAsyncNode reinitialized with name: {self.name}, available GPUs: {self.available_gpus}")

    def log_info(self, label, message):
        with self.log_lock:
            self.node_logger.info(f"{self.name} | {label} | {message}")

    def log_error(self, label, message):
        with self.log_lock:
            self.node_logger.error(f"{self.name} | {label} | {message}")

    def log_warning(self, label, message):
        with self.log_lock:
            self.node_logger.warning(f"{self.name} | {label} | {message}")

    def log_debug(self, label, message):
        with self.log_lock:
            self.node_logger.debug(f"{self.name} | {label} | {message}")

    async def open_connection(self):
        self.log_info("INIT", f"Initializing LocalAsyncNode {self.name}")
        self.log_debug("INIT", f"Available GPUs for {self.name}: {self.available_gpus}")
        self.free_gpus = await self.check_gpu_free()
        self.log_debug("INIT", f"Free GPUs for {self.name}: {self.free_gpus}")
        return bool(self.free_gpus)

    async def check_gpu_free(self):
        self.log_info("GPU_CHECK", "Checking available GPUs")
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, check=True)
            gpu_info = result.stdout.strip().split('\n')
            self.log_debug("GPU_CHECK", f"GPU info: {gpu_info}")
            free_gpus = []
            for gpu in gpu_info:
                index, total_memory, free_memory, gpu_util = map(int, gpu.split(','))
                if (str(index) in self.available_gpus and
                    free_memory >= MINIMUM_VRAM and
                    gpu_util <= USAGE_CRITERION * 100):  # Convert USAGE_CRITERION to percentage
                    free_gpus.append(str(index))
            self.log_debug("GPU_CHECK", f"Free GPUs after filtering: {free_gpus}")
            return free_gpus
        except subprocess.CalledProcessError as e:
            self.log_warning("GPU_CHECK", f"Failed to get GPU information: {e}")
            return []

    async def run(self, command, label):
        self.log_info(label, f"Attempting to run job on {self.name}. Available GPUs: {self.available_gpus}, Free GPUs: {self.free_gpus}")
        if not self.free_gpus:
            self.log_warning(label, f"No free GPUs available on {self.name}")
            return False

        gpu_to_use = self.free_gpus.pop(0)
        self.log_info(label, f"Selected GPU {gpu_to_use} for job on {self.name}")
        temp_dir = tempfile.mkdtemp(prefix=f"meinsweeper_job_{label}_")
        self.temp_dirs[label] = temp_dir

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_to_use

        self.log_info(label, f"Running command on {self.name} GPU {gpu_to_use} (CUDA_VISIBLE_DEVICES={gpu_to_use})")
        await self.log_q.put((({'status': 'running'}, 'running'), self.name, label))

        try:
            self.log_debug(label, f"Creating subprocess with command: {command}")
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=Path.cwd()
            )

            async def read_stream(stream, is_stderr=False):
                while True:
                    try:
                        line = await stream.readline()
                        if not line:
                            self.log_debug(label, "Stream reached EOF")
                            break
                        line = line.decode().strip()
                        if line:  # Only process non-empty lines
                            if is_stderr:
                                self.log_error(label, f"{self.name} stderr: {line}")
                                await self.log_q.put((f"ERROR: {line}", self.name, label))
                            else:
                                self.log_info(label, f"{self.name} stdout: {line}")
                                await self.log_q.put((line, self.name, label))
                    except Exception as e:
                        self.log_error(label, f"Error reading from {'stderr' if is_stderr else 'stdout'}: {e}\nTraceback:\n{traceback.format_exc()}")

            # Read both stdout and stderr
            await asyncio.gather(
                read_stream(process.stdout),
                read_stream(process.stderr, is_stderr=True)
            )

            return_code = await process.wait()
            self.log_debug(label, f"Process exit code: {return_code}")
            
            if return_code != 0:
                self.log_warning(label, f"Job failed with return code {return_code} on {self.name}")
                await self.log_q.put(({"status": "failed", "return_code": return_code}, self.name, label))
                return False

        except asyncio.CancelledError:
            self.log_error(label, f"Job was cancelled on {self.name}")
            await self.log_q.put(({"status": "cancelled"}, self.name, label))
            return False

        except Exception as e:
            self.log_error(label, f"Error running job on {self.name}: {str(e)}\nTraceback:\n{traceback.format_exc()}")
            await self.log_q.put(({"status": "failed", "error": str(e), "traceback": traceback.format_exc()}, self.name, label))
            return False

        finally:
            self.free_gpus.append(gpu_to_use)
            self.log_info(label, f"Job completed on {self.name}. GPU {gpu_to_use} returned to free_gpus. Current free_gpus: {self.free_gpus}")

        self.log_info(label, f"Job completed successfully on {self.name}")
        await self.log_q.put(({"status": "completed"}, self.name, label))
        return True

    @staticmethod
    def parse_log_line(line):
        out = None
        if '[[LOG_ACCURACY TRAIN]]' in line:
            out = {}
            line = line.split('[[LOG_ACCURACY TRAIN]]')[1]
            line = line.split(';')
            for section in line:
                if 'Elapsed' in section:
                    continue
                elif "Losses" in section:
                    loss_terms = section.split('Losses:')[1].split(',')
                    for loss_term in loss_terms:
                        loss_name, loss_value = map(lambda x: x.strip(), loss_term.split(':'))
                        out[f'loss_total'] = float(loss_value)
                elif 'Step' in section:
                    out['completed'] = int(section.split('Step:')[1])
        elif '[[LOG_ACCURACY TEST]]' in line:
            line = line.split(':')[1]
            out = {'test_acc': float(line.strip())}
        elif 'error' in line or 'RuntimeError' in line or 'failed' in line or 'Killed' in line:
            out = 'FAILED'
        return out

    def cleanup(self):
        for temp_dir in self.temp_dirs.values():
            try:
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(temp_dir)
            except Exception as e:
                self.node_logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

    def __del__(self):
        self.cleanup()