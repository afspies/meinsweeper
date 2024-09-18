import socket
from pathlib import Path
import asyncio
import asyncssh
import threading
import os

from meinsweeper.modules.helpers.debug_logging import init_node_logger
from meinsweeper.modules.helpers.utils import timeout_iterator
from .abstract import ComputeNode

# Use environment variables with default values
MINIMUM_VRAM = int(os.environ.get('MINIMUM_VRAM', 10)) * 1024  # Convert GB to MB
USAGE_CRITERION = float(os.environ.get('USAGE_CRITERION', 0.1))

class SSHNode(ComputeNode):
    _instances = {}
    _lock = threading.Lock()

    def __new__(cls, node_name, log_q, address, username, password=None, key_path=None, timeout=5):
        with cls._lock:
            if node_name not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[node_name] = instance
                instance.initialized = False
            return cls._instances[node_name]

    def __init__(self, node_name, log_q, address, username, password=None, key_path=None, timeout=5):
        if not hasattr(self, 'initialized') or not self.initialized:
            assert password or key_path, "Either password or key_path must be provided."
            self.log_q = log_q
            self.name = node_name
            self.connection_info = {
                'address': address,
                'username': username,
                'password': password,
                'key_path': key_path,
                'timeout': timeout
            }
            self.RUN_TIMEOUT = timeout
            self.node_logger = init_node_logger(self.name)
            self.log_lock = threading.Lock()
            self.initialized = True
            self.log_info("INIT", f"SSHNode initialized with name: {self.name}, address: {address}")
        elif self.name != node_name:
            # If the instance exists but with a different name, update it
            self.name = node_name
            self.node_logger = init_node_logger(self.name)
            self.log_info("REINIT", f"SSHNode reinitialized with name: {self.name}, address: {address}")

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
        address, username, key_path, timeout = self.connection_info['address'], self.connection_info[
            'username'], self.connection_info['key_path'], self.connection_info['timeout']
        try:
            self.log_info("INIT", f"Connecting SSH to {address}")
            self.conn = await asyncssh.connect(
                address,
                username=username,
                known_hosts=None,
                login_timeout=timeout,
                encoding='utf-8',
                client_keys=[key_path] if key_path else None,
                term_type='bash'
            )
            self.log_info("INIT", f"Connecting SCP to {address}")
            self.scp_conn = await asyncssh.connect(
                address,
                username=username,
                known_hosts=None,
                login_timeout=timeout,
                encoding='utf-8',
                client_keys=[key_path] if key_path else None
            )

        except asyncssh.PermissionDenied as auth_err:
            self.log_warning("INIT", f"Authentication failed for {address}: {auth_err}")
            return False
        except asyncssh.DisconnectError as ssh_err:
            self.log_warning("INIT", f"Disconnected from {address}: {ssh_err}")
            return False
        except TimeoutError:
            self.log_warning("INIT", f"Timeout connecting to {address}")
            return False
        except socket.gaierror as target_err:
            self.log_warning("INIT", f"Could not connect to {address}: {target_err}")
            return False
        except OSError as err:
            self.log_warning("INIT", f"OS error when connecting to {address}: {err}")
            return False

        # Check if GPU is free
        self.free_gpus = await self.check_gpu_free()
        return bool(self.free_gpus)

    async def check_gpu_free(self):
        self.log_info("GPU_CHECK", f"Checking GPU on {self.connection_info['address']}")
        try:
            await asyncssh.scp(Path(__file__).parent.parent / 'check_gpu.py', (self.scp_conn, '/tmp/check_gpu.py'))
        except asyncssh.sftp.SFTPFailure:
            self.log_warning("GPU_CHECK", f"Could not copy check_gpu.py to {self.connection_info['address']}")
            return []

        result = await self.conn.run('python /tmp/check_gpu.py')
        gpu_info = result.stdout.strip()
        if 'No Module named' in gpu_info:
            self.log_warning("GPU_CHECK", f"Missing pynvml module on {self.connection_info['address']} - cannot check GPU")
            return []

        free_gpus = []
        for line in gpu_info.split('\n'):
            if line.startswith('GPU'):
                parts = line.split(',')
                if len(parts) == 4:
                    index, total_memory, free_memory, gpu_util = parts
                    index = index.split()[1]
                    free_memory = int(free_memory.split()[0])
                    gpu_util = float(gpu_util.split()[0])
                    if (free_memory >= MINIMUM_VRAM and
                        gpu_util <= USAGE_CRITERION * 100):
                        free_gpus.append(index)

        self.log_debug("GPU_CHECK", f"Free GPUs after filtering: {free_gpus}")
        return free_gpus

    async def run(self, command, label):
        if not self.free_gpus:
            self.log_warning(label, f"No free GPUs available on {self.connection_info['address']}")
            return False

        gpu_to_use = self.free_gpus.pop(0)
        self.log_info(label, f"Selected GPU {gpu_to_use} for job on {self.connection_info['address']}")

        env = f"CUDA_VISIBLE_DEVICES={gpu_to_use}"
        full_command = f"{env} {command}"

        self.log_info(label, f"Running command on {self.connection_info['address']} GPU {gpu_to_use}")
        await self.log_q.put((({'status': 'running'}, 'running'), self.connection_info['address'], label))

        try:
            async with self.conn.create_process(full_command) as proc: 
                async for line in timeout_iterator(proc.stdout, self.RUN_TIMEOUT, "TIMEOUT"):
                    if line == "TIMEOUT":
                        self.log_warning(label, f"Timeout on {self.connection_info['address']}")
                        await self.log_q.put(({"status": "failed"}, self.connection_info["address"], label))
                        return False

                    parsed_line = self.parse_log_line(line)
                    if parsed_line == "FAILED":
                        self.log_warning(label, f"Failed (caught via parsed line) on {self.connection_info['address']}")
                        await self.log_q.put(({"status": "failed"}, self.connection_info["address"], label))
                        return False
                    self.log_info(label, f"stdout: {line}")
                    await self.log_q.put((parsed_line, self.connection_info["address"], label))

                async for err_line in proc.stderr:
                    if err_line != "":
                        self.log_error(label, f"stderr: {err_line}")
                        await self.log_q.put(({"status": "failed"}, self.connection_info["address"], label))
                        return False

        except asyncssh.misc.ConnectionLost as err:
            self.log_warning(label, f"Connection lost on {self.connection_info['address']}: {err}")
            await self.log_q.put(({"status": "failed"}, self.connection_info["address"], label))
            return False
        except Exception as err:
            self.log_error(label, f'Unexpected error on {self.connection_info["address"]}: {err}')
            await self.log_q.put(({"status": "failed"}, self.connection_info["address"], label))
            return False
        finally:
            self.free_gpus.append(gpu_to_use)
            self.log_info(label, f"Job completed on {self.connection_info['address']}. GPU {gpu_to_use} returned to free_gpus. Current free_gpus: {self.free_gpus}")

        self.log_info(label, f"Job completed successfully on {self.connection_info['address']}")
        await self.log_q.put(({"status": "completed"}, self.connection_info["address"], label))
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

    def __str__(self) -> str:
        return f'SSH Node {self.name} ({self.connection_info["address"]}) with user {self.connection_info["username"]}'