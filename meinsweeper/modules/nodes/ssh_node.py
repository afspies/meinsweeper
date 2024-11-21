import socket
from pathlib import Path
import asyncio
import asyncssh
import threading
import os
import traceback
from datetime import datetime
import time

from meinsweeper.modules.helpers.debug_logging import init_node_logger
from meinsweeper.modules.helpers.utils import timeout_iterator, debug_print
from meinsweeper.modules.helpers.retry import RetryStrategy
from .abstract import ComputeNode

# Use environment variables with default values
MINIMUM_VRAM = int(os.environ.get('MINIMUM_VRAM', 10)) * 1024  # Convert GB to MB
USAGE_CRITERION = float(os.environ.get('USAGE_CRITERION', 0.1))

class RetryableError(Exception):
    """Custom exception for retryable SSH operations"""
    pass

class SSHNode(ComputeNode):
    _instances = {}
    _connections = {}
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
            self.conn = None
            self.scp_conn = None
            self.process = None
            self.free_gpus = []
            self.last_heartbeat = time.time()
            self.HEARTBEAT_TIMEOUT = 600  # 10 minutes
            self.RUN_TIMEOUT = 30  # 30 seconds for individual operations
            self.connection_lock = asyncio.Lock()  # Lock for connection operations
            self.max_buffer_size = 1024 * 1024  # 1MB max buffer size
            self.read_chunk_size = 4096  # Read in 4KB chunks
            self.node_logger = init_node_logger(self.name)
            self.log_lock = threading.Lock()
            self.initialized = True
            debug_print(f"SSHNode initialized with name: {self.name}, address: {address}")
            self.warning_timestamps = {}
            self.WARNING_INTERVAL = 5

    async def open_connection(self):
        """Open a new SSH connection with proper isolation"""
        async with self.connection_lock:
            try:
                # Close existing connections if any
                await self._close_connections()
                
                # Create new connection with specific buffer sizes and timeouts
                self.conn = await asyncio.wait_for(
                    asyncssh.connect(
                        self.connection_info['address'],
                        username=self.connection_info['username'],
                        known_hosts=None,
                        login_timeout=self.connection_info['timeout'],
                        encoding='utf-8',
                        client_keys=[self.connection_info['key_path']] if self.connection_info['key_path'] else None,
                        keepalive_interval=30,
                        keepalive_count_max=4,
                        window=self.max_buffer_size,
                        max_pktsize=32768
                    ),
                    timeout=self.connection_info['timeout']
                )
                
                # Verify connection is working
                test_result = await asyncio.wait_for(
                    self.conn.run('echo "test"'),
                    timeout=5.0
                )
                if test_result.exit_status != 0:
                    raise ConnectionError("Connection test failed")
                
                self.last_heartbeat = time.time()
                return True
                
            except Exception as e:
                self.log_error("CONN", f"Failed to open connection: {str(e)}\n{traceback.format_exc()}")
                # Ensure connections are cleaned up on failure
                await self._close_connections()
                return False

    async def _close_connections(self):
        """Safely close all connections"""
        try:
            if self.conn:
                self.conn.close()
                await self.conn.wait_closed()
            if self.scp_conn:
                self.scp_conn.close()
                await self.scp_conn.wait_closed()
        except Exception as e:
            self.log_warning("CONN", f"Error closing connections: {str(e)}")
        finally:
            self.conn = None
            self.scp_conn = None

    def log_info(self, label, message):
        with self.log_lock:
            self.node_logger.info(f"{self.name} | {label} | {message}")

    def log_error(self, label, message):
        with self.log_lock:
            self.node_logger.error(f"{self.name} | {label} | {message}")

    def log_warning(self, label, message):
        with self.log_lock:
            # Create a key for this specific warning
            warning_key = f"{label}_{message}"
            current_time = time.time()
            
            # Check if we should log this warning
            last_time = self.warning_timestamps.get(warning_key, 0)
            if current_time - last_time >= self.WARNING_INTERVAL:
                self.node_logger.warning(f"{self.name} | {label} | {message}")
                self.warning_timestamps[warning_key] = current_time

    def log_debug(self, label, message):
        with self.log_lock:
            self.node_logger.debug(f"{self.name} | {label} | {message}")

    async def check_gpu_free(self):
        """Check which GPUs are free on the remote node"""
        address = self.connection_info['address']
        self.log_info("GPU_CHECK", f"Starting GPU check on {address}")
        
        retry_strategy = RetryStrategy(
            max_retries=3,
            initial_delay=1.0,
            max_delay=30.0,
            backoff_factor=2.0,
            logger=self.node_logger
        )

        try:
            # First ensure we have a healthy connection
            if not await self._check_connection_health():
                self.log_info("GPU_CHECK", "Connection unhealthy, attempting to reconnect")
                if not await self.open_connection():
                    raise ConnectionError(f"Failed to establish connection to {address}")

            # Now execute with retry strategy
            success, result, last_error = await retry_strategy.execute(self._check_gpu_free_internal)
            
            if not success:
                self.log_error("GPU_CHECK", f"GPU check failed after retries: {last_error}")
                return []
            
            # Ensure we return a list of integers
            if result is None:
                return []
            
            if not isinstance(result, list):
                self.log_error("GPU_CHECK", f"Unexpected result type: {type(result)}")
                return []
            
            return list(result)  # Ensure we return a mutable list
            
        except Exception as e:
            self.log_error("GPU_CHECK", f"Unexpected error during GPU check on {address}: {str(e)}\n{traceback.format_exc()}")
            return []

    async def _check_gpu_free_internal(self):
        """Internal implementation of GPU check with proper connection handling"""
        try:
            # Verify connection before starting
            if not await self._check_connection_health():
                raise ConnectionError("Connection lost before GPU check")

            script_path = await self._copy_gpu_check_script()
            gpu_info = await self._run_gpu_check(script_path)
            gpus = self._parse_gpu_check_output(gpu_info)
            self.log_debug("GPU_CHECK", f"Internal GPU check result type: {type(gpus)}, value: {gpus}")
            
            # Clean up the temporary script
            try:
                await self.conn.run(f'rm -f {script_path}')
            except Exception as e:
                self.log_warning("GPU_CHECK", f"Failed to clean up temporary script {script_path}: {str(e)}")
            
            # Ensure we return a list of integers
            if isinstance(gpus, (list, tuple)):
                try:
                    # Convert string GPU indices to integers
                    gpu_list = [int(str(gpu).strip()) for gpu in gpus if str(gpu).strip()]
                    self.log_debug("GPU_CHECK", f"Parsed GPU indices: {gpu_list}")
                    return gpu_list
                except (ValueError, TypeError, AttributeError) as e:
                    self.log_error("GPU_CHECK", f"Error parsing GPU indices: {str(e)}")
                    return []
            elif isinstance(gpus, bool):
                self.log_error("GPU_CHECK", "Received boolean instead of GPU list")
                return []
            else:
                self.log_error("GPU_CHECK", f"Unexpected GPU list type: {type(gpus)}")
                return []

        except asyncssh.misc.ChannelOpenError:
            self.log_warning("GPU_CHECK", "SSH channel closed, attempting to reconnect")
            if await self.open_connection():
                raise RetryableError("Reconnected, retrying GPU check")
            raise
            
        except Exception as e:
            if isinstance(e, (asyncssh.misc.ChannelOpenError, ConnectionError)):
                raise RetryableError(str(e))
            self.log_error("GPU_CHECK", f"Unexpected error in GPU check: {str(e)}\n{traceback.format_exc()}")
            return []

    async def _copy_gpu_check_script(self):
        """Helper to copy the GPU check script with node-specific name"""
        try:
            if not await self._check_connection_health():
                raise ConnectionError("Connection unhealthy before copying GPU check script")

            script_path = Path(__file__).parent.parent / 'check_gpu.py'
            remote_script = f'/tmp/check_gpu_{self.name}_{os.getpid()}.py'
            self.log_debug("GPU_CHECK", f"Copying {script_path} to remote {remote_script}")
            
            # Use the scp_conn from a fresh connection if needed
            if not hasattr(self, 'scp_conn') or self.scp_conn is None:
                self.log_debug("GPU_CHECK", "Creating new SCP connection")
                self.scp_conn = await asyncio.wait_for(
                    asyncssh.connect(
                        self.connection_info['address'],
                        username=self.connection_info['username'],
                        known_hosts=None,
                        login_timeout=self.connection_info['timeout'],
                        encoding='utf-8',
                        client_keys=[self.connection_info['key_path']] if self.connection_info['key_path'] else None
                    ),
                    timeout=self.connection_info['timeout']
                )

            await asyncssh.scp(script_path, (self.scp_conn, remote_script))
            self.log_debug("GPU_CHECK", "Successfully copied check script")
            return remote_script
            
        except Exception as e:
            self.log_error("GPU_CHECK", f"Error copying GPU check script: {str(e)}")
            raise

    async def _run_gpu_check(self, script_path):
        """Helper to run the GPU check script"""
        try:
            self.log_debug("GPU_CHECK", f"Running {script_path}")
            result = await asyncio.wait_for(
                self.conn.run(f'python {script_path}'),
                timeout=self.RUN_TIMEOUT
            )
            self.log_debug("GPU_CHECK", f"check_gpu.py output: {result.stdout.strip()}")
            return result.stdout.strip()
        except asyncio.TimeoutError:
            self.log_error("GPU_CHECK", f"Timeout running GPU check after {self.RUN_TIMEOUT}s")
            raise
        except Exception as e:
            self.log_error("GPU_CHECK", f"Error running GPU check: {str(e)}\n{traceback.format_exc()}")
            raise

    def _parse_gpu_check_output(self, gpu_info: str) -> list[str]:
        """Helper to parse GPU check output"""
        if not isinstance(gpu_info, str):
            self.log_error("GPU_CHECK", f"Invalid GPU info type: {type(gpu_info)}")
            return []

        if 'No Module named' in gpu_info:
            self.log_warning("GPU_CHECK", f"Missing pynvml module on {self.connection_info['address']} - cannot check GPU")
            return []

        free_gpus = []
        for line in gpu_info.split('\n'):
            if '[[GPU INFO]]' in line:
                # Extract GPU indices from the format "[[GPU INFO]] [0,1,2] Free"
                line = line.replace('[[GPU INFO]]', '').strip()
                # Extract content between square brackets
                if '[' in line and ']' in line:
                    gpu_list = line[line.find('[')+1:line.find(']')]
                    if gpu_list:  # Only process if there are GPUs listed
                        if ',' in gpu_list:
                            free_gpus = [gpu.strip() for gpu in gpu_list.split(',') if gpu.strip()]
                        else:
                            gpu = gpu_list.strip()
                            if gpu:  # Only add non-empty values
                                free_gpus = [gpu]
                        self.log_debug("GPU_CHECK", f"Found free GPUs: {free_gpus}")

        if not free_gpus:
            self.log_warning("GPU_CHECK", f"No free GPUs found on {self.connection_info['address']} - got {gpu_info}")

        return free_gpus

    async def _handle_output_buffer(self, stdout_buffer, label):
        """Handle buffered output lines and flush them to logs."""
        for buffered_line in stdout_buffer:
            self.log_info(label, f"stdout: {buffered_line}")
            parsed_line = self.parse_log_line(buffered_line)
            if parsed_line == "FAILED":
                self.log_warning(label, f"Failed (caught via parsed line)")
                await self.log_q.put(({"status": "failed"}, self.connection_info["address"], label))
                return False
            await self.log_q.put((parsed_line, self.connection_info["address"], label))
        return True

    async def _process_stderr(self, stderr, label):
        """Process stderr output and handle any errors."""
        if not stderr:
            return True
            
        stderr_lines = stderr.strip().split('\n')
        for err_line in stderr_lines:
            if err_line:
                self.log_error(label, f"stderr: {err_line}")
                if "StepAlreadyExistsError" in err_line:
                    await self.log_q.put(({"status": "failed", "error": "StepAlreadyExistsError", "message": err_line}, 
                                        self.connection_info["address"], label))
                    return False
        
        await self.log_q.put(({"status": "failed", "stderr": stderr}, self.connection_info["address"], label))
        return False

    async def _get_process_info(self, proc):
        """Get detailed information about a running process"""
        try:
            if (not hasattr(proc, 'pid') or 
                proc.pid is None or 
                hasattr(proc, 'returncode') and proc.returncode is not None):
                return None

            # Run ps command to get process info
            result = await self.conn.run(
                f'ps -p {proc.pid} -o pid,ppid,rss,vsize,pcpu,pmem,state,time,etime,command --no-headers'
            )
            
            if result.exit_status != 0:
                return None

            info = result.stdout.strip()
            if not info:
                return None

            # Parse ps output into a dictionary
            fields = ['pid', 'ppid', 'rss', 'vsize', 'cpu', 'mem', 'state', 'time', 'elapsed', 'command']
            values = info.split(None, len(fields)-1)
            return dict(zip(fields, values))

        except Exception as e:
            self.log_debug("PROC_INFO", f"Error getting process info: {str(e)}")  # Changed to debug level
            return None

    async def _handle_process_output(self, proc, label):
        """Handle process output streaming and buffering with improved buffer management"""
        stdout_buffer = []
        buffer_size = 0
        last_flush = time.time()
        last_output = time.time()
        last_heartbeat_log = time.time()
        FLUSH_INTERVAL = 1.0
        SILENCE_TIMEOUT = 300.0
        HEARTBEAT_LOG_INTERVAL = 300.0
        MAX_BUFFER_SIZE = self.max_buffer_size
        process_completed = False
        
        # Store recent output for completion detection
        if not hasattr(proc, '_recent_output'):
            proc._recent_output = []
        RECENT_OUTPUT_MAX = 20  # Keep last 20 lines

        async def flush_buffer():
            nonlocal stdout_buffer, buffer_size
            if stdout_buffer:
                if not await self._handle_output_buffer(stdout_buffer, label):
                    return False
                stdout_buffer = []
                buffer_size = 0
            return True

        try:
            while True:
                current_time = time.time()
                
                # Check for prolonged silence
                if current_time - last_output > SILENCE_TIMEOUT:
                    if current_time - last_heartbeat_log > HEARTBEAT_LOG_INTERVAL:
                        self.log_warning(label, f"No output received for {int(current_time - last_output)}s")
                        last_heartbeat_log = current_time
                        
                        if not await self._check_connection_health():
                            await flush_buffer()
                            raise ConnectionError("Connection lost during prolonged silence")

                try:
                    line = await asyncio.wait_for(proc.stdout.readline(), timeout=30.0)
                except asyncio.TimeoutError:
                    await flush_buffer()
                    if not await self._check_connection_health():
                        raise ConnectionError("Connection lost during read timeout")
                    continue

                if not line:
                    self.log_debug(label, "Process stdout closed")
                    process_completed = True
                    break

                self.last_heartbeat = current_time
                last_output = current_time

                line = line.strip()
                if line:
                    # Store recent output for completion detection
                    proc._recent_output.append(line)
                    if len(proc._recent_output) > RECENT_OUTPUT_MAX:
                        proc._recent_output.pop(0)

                    line_size = len(line.encode('utf-8'))
                    if buffer_size + line_size > MAX_BUFFER_SIZE:
                        await flush_buffer()
                    
                    stdout_buffer.append(line)
                    buffer_size += line_size

                    if (current_time - last_flush >= FLUSH_INTERVAL or 
                        buffer_size >= MAX_BUFFER_SIZE):
                        if not await flush_buffer():
                            return False
                        last_flush = current_time

                        async with self.connection_lock:
                            if not await self._check_connection_health():
                                raise ConnectionError("SSH connection lost after log flush")

        except Exception as e:
            self.log_error(label, f"Error in process output handling: {str(e)}\n{traceback.format_exc()}")
            await flush_buffer()
            raise
        finally:
            if process_completed:
                # Send explicit completion status with final state
                await self.log_q.put(({
                    "status": "completed",
                    "completed": 100,  # Ensure 100% completion
                    "visible": False,  # Hide from display
                    "final": True     # Mark as final update
                }, self.connection_info["address"], label))
            return await flush_buffer()

    async def _check_connection_health(self):
        """Enhanced connection health check with detailed diagnostics"""
        try:
            if not hasattr(self, 'conn') or self.conn is None:
                self.log_warning("CONN_CHECK", "No connection object exists")
                return False

            # Try a simple echo command
            result = await asyncio.wait_for(
                self.conn.run('echo "connection_test"'),
                timeout=5.0
            )
            
            if result.exit_status != 0:
                self.log_warning("CONN_CHECK", f"Health check command failed with status {result.exit_status}")
                return False
                
            return True
            
        except asyncio.TimeoutError:
            self.log_warning("CONN_CHECK", "Connection health check timed out")
            return False
        except Exception as e:
            self.log_warning("CONN_CHECK", f"Connection check failed: {str(e)}")
            return False

    async def _terminate_process(self, proc):
        """Enhanced process termination with cleanup"""
        try:
            self.log_warning("PROC_TERM", f"Terminating process {proc}")
            if hasattr(proc, 'terminate'):
                await proc.terminate()
            if hasattr(proc, 'kill'):
                await proc.kill()
            self.log_info("PROC_TERM", "Process terminated")
        except Exception as e:
            self.log_error("PROC_TERM", f"Error terminating process: {str(e)}")

    async def _check_completion_indicators(self, proc, label):
        """Check for various completion indicators in process output"""
        recent_output = getattr(proc, '_recent_output', [])
        
        # Common completion indicators (all lowercase for case-insensitive comparison)
        completion_indicators = [
            "cleaning up temporary",
            "completed successfully",
            "training completed",
            "completed [",  # Common format: "Completed [74.95s]"
            "wandb: synced",  # WandB completion message
            "wandb: \\ finished",
            "finished successfully"
        ]
        
        # Error indicators that shouldn't be treated as completion (all lowercase)
        error_indicators = [
            "error:",
            "exception:",
            "failed:",
            "killed",
            "terminated",
            "core dumped"
        ]
        
        # Check for completion indicators
        found_completion = False
        for line in recent_output:
            line_lower = line.lower()
            
            # Skip if line contains error indicators
            if any(err in line_lower for err in error_indicators):
                continue
            
            if any(indicator in line_lower for indicator in completion_indicators):
                found_completion = True
                self.log_info(label, f"Found completion indicator: {line}")
                break
        
        # Additional checks for specific patterns (also case-insensitive)
        if not found_completion:
            # Check for wandb run completion pattern
            wandb_pattern = any("wandb: synced" in line.lower() and "artifact file(s)" in line.lower() 
                              for line in recent_output)
            if wandb_pattern:
                found_completion = True
                self.log_info(label, "Found WandB completion pattern")
        
        return found_completion

    async def run(self, command, label):
        """Run a command on the remote node with GPU allocation."""
        max_connection_retries = 3
        connection_retry_count = 0
        
        while connection_retry_count < max_connection_retries:
            try:
                # First validate SSH connection
                if not hasattr(self, 'conn') or not self.conn:
                    self.log_error(label, "No SSH connection available, attempting to reconnect")
                    if not await self.open_connection():
                        connection_retry_count += 1
                        await asyncio.sleep(5 * (connection_retry_count + 1))
                        continue

                # Validate connection health
                if not await self._check_connection_health():
                    self.log_warning(label, f"SSH connection unhealthy (attempt {connection_retry_count + 1}/{max_connection_retries})")
                    if not await self.open_connection():
                        connection_retry_count += 1
                        await asyncio.sleep(5 * (connection_retry_count + 1))
                        continue

                if not self.free_gpus:
                    # Double check GPU availability before reporting none available
                    self.log_debug(label, "No free GPUs, rechecking availability")
                    self.free_gpus = await self.check_gpu_free()  # This now returns a list of integers
                    if not self.free_gpus:
                        self.log_warning(label, f"No free GPUs available on {self.connection_info['address']}")
                        return False

                try:
                    gpu_to_use = self.free_gpus.pop(0)  # This will now be an integer
                    self.log_info(label, f"Selected GPU {gpu_to_use} for job on {self.connection_info['address']}")
                except (IndexError, AttributeError) as e:
                    self.log_error(label, f"Error selecting GPU: {str(e)} - free_gpus type: {type(self.free_gpus)}")
                    return False

                env = f"CUDA_VISIBLE_DEVICES={gpu_to_use}"
                full_command = f"{env} {command}"

                self.log_info(label, f"Running command on {self.connection_info['address']} GPU {gpu_to_use}")
                self.log_debug(label, f"Full command: {full_command}")

                await self.log_q.put((({'status': 'running'}, 'running'), self.connection_info['address'], label))

                try:
                    self.log_debug(label, f"Creating process for command: {full_command}")
                    async with self.conn.create_process(full_command, term_type='xterm') as proc:
                        self.log_debug(label, "Process created successfully")
                        self.process = proc
                        monitor_task = asyncio.create_task(self._monitor_process(proc, label))

                        try:
                            if not await self._handle_process_output(proc, label):
                                self.log_error(label, "Process output handling failed")
                                return False

                            stderr = await proc.stderr.read()
                            if not await self._process_stderr(stderr, label):
                                self.log_error(label, "Stderr processing failed")
                                return False

                            exit_status = await proc.wait()
                            
                            # Enhanced completion detection
                            if exit_status != 0:
                                # Check for completion indicators
                                if await self._check_completion_indicators(proc, label):
                                    self.log_info(label, "Process completed successfully despite non-zero exit status")
                                    return True
                                    
                                # If no completion indicators found, log the error and fail
                                self.log_error(label, f"Process exited with non-zero status: {exit_status}")
                                self.log_debug(label, f"Last output lines: {getattr(proc, '_recent_output', [])[-5:]}")
                                await self.log_q.put(({"status": "failed", "exit_status": exit_status}, 
                                                    self.connection_info["address"], label))
                                return False

                            return True

                        except asyncio.TimeoutError:
                            self.log_error(label, "Process timed out")
                            await self._terminate_process(proc)
                            return False

                        finally:
                            if monitor_task and not monitor_task.done():
                                monitor_task.cancel()
                                try:
                                    await monitor_task
                                except asyncio.CancelledError:
                                    pass

                except (asyncssh.misc.ChannelOpenError, ConnectionError) as e:
                    self.log_error(label, f"SSH error: {str(e)}")
                    connection_retry_count += 1
                    if connection_retry_count < max_connection_retries:
                        await asyncio.sleep(5 * (connection_retry_count + 1))
                        continue
                    return False

                finally:
                    # Always return the GPU to the pool
                    if gpu_to_use not in self.free_gpus:
                        self.log_info(label, f"Returning GPU {gpu_to_use} to pool")
                        self.free_gpus.append(gpu_to_use)

            except Exception as e:
                self.log_error(label, f"Unexpected error in run method: {str(e)}\n{traceback.format_exc()}")
                return False

        return False  # All retries exhausted

    async def _monitor_process(self, proc, label):
        """Monitor process resources and health"""
        try:
            while True:
                try:
                    # Check process existence and resource usage
                    if proc.returncode is not None:
                        self.log_warning(label, f"Process terminated with return code {proc.returncode}")
                        break
                    
                    # Check SSH connection health
                    if not await self._check_connection_health():
                        self.log_error(label, "SSH connection appears to be dead")
                        break
                    
                    # Update heartbeat since we confirmed connection is alive
                    self.last_heartbeat = time.time()
                    
                    # Log process stats periodically
                    mem_info = await self._get_memory_usage(proc)
                    if mem_info:
                        self.log_debug(label, f"Process stats: {mem_info}")
                    
                    await asyncio.sleep(30)  # Check every 30 seconds instead of 60
                    
                except Exception as e:
                    self.log_error(label, f"Error monitoring process: {e}")
                    break
                    
        except asyncio.CancelledError:
            pass

    async def _get_memory_usage(self, proc):
        """Get memory usage of the remote process"""
        try:
            result = await self.conn.run(f'ps -p {proc.pid} -o pid,ppid,rss,vsize,pcpu,pmem,comm,state')
            return result.stdout
        except:
            return None

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

    @classmethod
    def clear_instances(cls):
        """Clear all stored instances and connections"""
        with cls._lock:
            debug_print(f"Clearing SSHNode instances. Current instances: {list(cls._instances.keys())}")
            # Close all connections
            for conn, scp_conn in cls._connections.values():
                conn.close()
                scp_conn.close()
            cls._connections.clear()
            cls._instances.clear()
            debug_print("SSHNode instances cleared")

    async def has_available_gpus(self):
        """Check if the node has any available GPUs"""
        try:
            free_gpus = await self.check_gpu_free()
            return len(free_gpus) > 0
        except Exception as e:
            self.log_error("GPU_CHECK", f"Error checking GPU availability: {str(e)}")
            return False