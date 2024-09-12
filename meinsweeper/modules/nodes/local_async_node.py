import asyncio
import os
import subprocess
from pathlib import Path
import tempfile
import sys
from io import StringIO

from meinsweeper.modules.helpers.debug_logging import init_node_logger, DEBUG
from meinsweeper.modules.helpers.utils import timeout_iterator
from .abstract import ComputeNode

class CaptureIO(StringIO):
    def __init__(self, node, label, log_q):
        super().__init__()
        self.node = node
        self.label = label
        self.log_q = log_q

    def write(self, s):
        super().write(s)
        if s.strip():
            self.node.node_logger.debug(f"Job {self.label} output: {s.strip()}")
            asyncio.create_task(self.log_q.put((s.strip(), self.node.name, self.label)))

class LocalAsyncNode(ComputeNode):
    def __init__(self, log_q, available_gpus, timeout=1200):
        self.log_q = log_q
        self.name = 'local-async'
        self.available_gpus = available_gpus
        self.RUN_TIMEOUT = timeout
        self.node_logger = init_node_logger(self.name)
        self.temp_dirs = {}

    async def open_connection(self):
        self.node_logger.info("Initializing LocalAsyncNode")
        self.free_gpus = await self.check_gpu_free()
        self.node_logger.debug(f"Free GPUs: {self.free_gpus}")
        return True if self.free_gpus else False

    async def check_gpu_free(self):
        self.node_logger.info("Checking available GPUs")
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], 
                                    capture_output=True, text=True, check=True)
            gpu_info = result.stdout.strip().split('\n')
            self.node_logger.debug(f"GPU info: {gpu_info}")
            free_gpus = [gpu.split(',')[0] for gpu in gpu_info if int(gpu.split(',')[1]) > 1000]  # Consider GPUs with >1GB free memory
            self.node_logger.debug(f"Free GPUs before filtering: {free_gpus}")
            return [gpu for gpu in free_gpus if gpu in self.available_gpus]
        except subprocess.CalledProcessError as e:
            self.node_logger.warning(f"Failed to get GPU information: {e}")
            return None

    async def run(self, command, label):
        if not self.free_gpus:
            self.node_logger.warning("No free GPUs available")
            return False

        gpu_to_use = self.free_gpus.pop(0)
        temp_dir = tempfile.mkdtemp(prefix=f"meinsweeper_job_{label}_")
        self.temp_dirs[label] = temp_dir

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_to_use

        self.node_logger.info(f"Running command on local GPU {gpu_to_use}")
        await self.log_q.put((({'status': 'running'}, 'running'), self.name, label))

        # Write the command to a temporary Python file
        script_path = Path(temp_dir) / f"{label}_script.py"
        with open(script_path, 'w') as f:
            f.write(command)

        try:
            process = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,  # Redirect stderr to stdout
                env=env,
                cwd=temp_dir
            )

            async def read_stream(stream):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    line = line.decode().strip()
                    self.node_logger.info(f"Job {label} output: {line}")
                    
                    # Explicitly log MSLogger outputs
                    if '[[LOG_ACCURACY' in line:
                        self.node_logger.info(f"MSLogger output: {line}")
                    
                    await self.log_q.put((line, self.name, label))

            # Read only stdout (which now includes stderr)
            await read_stream(process.stdout)

            return_code = await process.wait()
            if return_code != 0:
                self.node_logger.warning(f"Job {label} failed with return code {return_code}")
                await self.log_q.put(({"status": "failed"}, self.name, label))
                return False

        except Exception as e:
            self.node_logger.error(f"Error running job {label}: {str(e)}")
            await self.log_q.put(({"status": "failed"}, self.name, label))
            return False

        finally:
            self.free_gpus.append(gpu_to_use)

        self.node_logger.info(f"Job {label} completed successfully")
        return True

    @staticmethod
    def parse_log_line(line):
        # We'll keep this method for internal use, but we're not using its output for the log queue anymore
        # print line to stdout in spite of rich table
        # logging.debug(line)
        # logging.debug("PARSING LINE")

        out = None
        #! FIX this can fail if the log line is part of an error message - we will try to parse
        #! but the f-strings won't have been evaluated (meaning we can't convert {step} or {loss} to float)
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
        elif '[[LOG_ACCURACY TEST]]' in line:  #TODO something is broken here
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