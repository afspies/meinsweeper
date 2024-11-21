from abc import ABC, abstractmethod
import socket
import asyncio
import os
import asyncssh
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
import time
import traceback

from meinsweeper.modules.helpers.debug_logging import (
    init_node_logger, 
    global_logger, 
    clear_log_files, 
    cleanup_sweep, 
    cleanup_inactive_sweeps
)
from .nodes import *
import textwrap
from .nodes.local_async_node import LocalAsyncNode
from meinsweeper.modules.helpers.utils import debug_print

# Use environment variables with default values
MINIMUM_VRAM = int(os.environ.get('MINIMUM_VRAM', 8))  # in GigaBytes
USAGE_CRITERION = float(os.environ.get('USAGE_CRITERION', 0.8))  # percentage (float) or -1 => no processes other than xorg
MAX_PROCESSES = int(os.environ.get('MAX_PROCESSES', -1))  # -1 => no limit, Otherwise number of processes = min(#nodes, #tbc_runs)
RUN_TIMEOUT = int(os.environ.get('RUN_TIMEOUT', 2100))  # in seconds
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))  # removed from PQ after this - should readd after some interval
RETRY_INTERVAL = int(os.environ.get('MEINSWEEPER_RETRY_INTERVAL', 450))  # Default to 7.5 minutes, but allow override

class RunManager(object):
    """ The RunManager spawns processes on the target nodes from the pool of available ones
    """
    def __init__(self, targets: dict, task_q: asyncio.Queue, log_q: asyncio.Queue) -> None:
        # Clear any existing SSH node instances
        from .nodes.ssh_node import SSHNode
        SSHNode.clear_instances()
        
        # Clean up any inactive sweeps before starting a new one
        cleanup_inactive_sweeps()
        
        debug_print("\n=== MeinSweeper Run Manager Initialization ===")
        debug_print(f"Initializing with targets: {list(targets.keys())}")
        
        self.targets = targets
        self.max_proc = min(len(targets), MAX_PROCESSES) if MAX_PROCESSES != -1 else len(targets)
        self.log_q = log_q
        self.task_q = task_q
        self.running_proc = 0
        self.tasks = []
        
        # Create a separate initialization log
        init_logger = init_node_logger('run_manager_init')
        init_logger.info(f"Initialized with targets: {list(self.targets.keys())}")
        
        self.logger = init_node_logger(f'run_manager_{os.getpid()}')
        self.unavailable_targets = {}
        self.retry_event = asyncio.Event()
        self.stop_event = asyncio.Event()

        # Clear log files when RunManager is instantiated
        clear_log_files()
        debug_print(f"Max concurrent processes: {self.max_proc}")
        debug_print(f"Retry interval: {RETRY_INTERVAL} seconds")
        debug_print("==========================================\n")

    async def start_run(self):
        try:
            self.target_q = asyncio.PriorityQueue()
            for name, target in self.targets.items():
                self.logger.info(f"Putting target {name} into queue")
                await self.target_q.put(Target(name, target, retries=MAX_RETRIES))

            self.retry_task = asyncio.create_task(self.retry_unavailable_targets())
            self.tasks = [asyncio.create_task(self.spawn_worker(worker_id=i)) for i in range(self.max_proc)]
            
            await self.task_q.join()  # Wait for all jobs to complete
        finally:
            self.stop_event.set()  # Signal all tasks to stop
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.retry_task.cancel()  # Cancel the retry task after all workers have finished
            cleanup_sweep()  # Clean up this sweep's resources

    async def retry_unavailable_targets(self):
        self.logger.info("Starting retry_unavailable_targets task")
        while not self.stop_event.is_set():
            current_time = time.time()
            targets_to_retry = []
            
            # Add debug logging for unavailable targets
            if self.unavailable_targets:
                self.logger.info(f"Current unavailable targets: {list(self.unavailable_targets.keys())}")
            
            for name, (timestamp, _) in list(self.unavailable_targets.items()):
                time_diff = current_time - timestamp
                if time_diff >= RETRY_INTERVAL:
                    targets_to_retry.append(name)
            
            if targets_to_retry:
                self.logger.info(f"Retrying targets: {targets_to_retry}")
                for name in targets_to_retry:
                    _, target = self.unavailable_targets[name]
                    new_target = Target(name, target, retries=MAX_RETRIES)
                    await self.target_q.put(new_target)
                    del self.unavailable_targets[name]
                self.retry_event.set()
            
            await asyncio.sleep(RETRY_INTERVAL)

    async def spawn_worker(self, worker_id):
        """Worker coroutine to process tasks"""
        self.logger.info(f"Starting worker {worker_id}")
        
        while True:
            try:
                task = await self.task_q.get()
                if task is None:  # Shutdown signal
                    self.logger.info(f"Worker {worker_id} received shutdown signal")
                    break

                command, task_id = task
                self.logger.info(f"Worker {worker_id}: Starting task {task_id}")
                
                start_time = time.time()
                last_activity = start_time

                try:
                    node = None
                    retry_count = 0
                    max_retries = 3
                    
                    while retry_count < max_retries:
                        try:
                            # Get a node and track it
                            target = await self.target_q.get()
                            node = self.create_node(target)
                            
                            if not node:
                                self.logger.error(f"Worker {worker_id}: No nodes available for task {task_id}")
                                await asyncio.sleep(10)
                                retry_count += 1
                                continue

                            # Check if node has available GPUs
                            if not await node.has_available_gpus():
                                self.logger.warning(f"Worker {worker_id}: Node {node.name} has no available GPUs")
                                # Move node to unavailable targets and continue
                                self.unavailable_targets[target.name] = (time.time(), target.details)
                                self.target_q.task_done()
                                # Don't put the node back in the queue
                                continue

                            self.logger.info(f"Worker {worker_id}: Got node {node.name} for task {task_id}")
                            
                            monitor_task = asyncio.create_task(
                                self._monitor_task_activity(worker_id, task_id, node, start_time)
                            )

                            success = await node.run(command, task_id)
                            last_activity = time.time()

                            if success:
                                self.logger.info(f"Worker {worker_id}: Task {task_id} completed successfully on {node.name}")
                                # Only put the node back if it was successful
                                await self.target_q.put(target)
                                break
                            else:
                                # Check if failure was due to GPU availability
                                if not await node.has_available_gpus():
                                    self.logger.warning(f"Worker {worker_id}: Node {node.name} has no available GPUs after failed run")
                                    self.unavailable_targets[target.name] = (time.time(), target.details)
                                    self.target_q.task_done()
                                    # Don't put the node back in the queue
                                    continue
                                
                                self.logger.error(f"Worker {worker_id}: Task {task_id} failed on {node.name}, retrying")
                                # Only put the node back if failure wasn't due to GPU availability
                                await self.target_q.put(target)
                                retry_count += 1
                                
                        except asyncio.CancelledError:
                            self.logger.warning(f"Worker {worker_id}: Task {task_id} cancelled")
                            raise
                        except Exception as e:
                            self.logger.error(f"Worker {worker_id}: Error running task {task_id} on {node.name if node else 'unknown'}: {str(e)}\n{traceback.format_exc()}")
                            retry_count += 1
                        finally:
                            # Clean up monitoring task
                            if 'monitor_task' in locals() and not monitor_task.done():
                                monitor_task.cancel()
                                try:
                                    await monitor_task
                                except asyncio.CancelledError:
                                    pass
                            
                    # Wait before retry
                    if retry_count < max_retries:
                        await asyncio.sleep(10 * (retry_count + 1))  # Exponential backoff

                    if retry_count >= max_retries:
                        self.logger.error(f"Worker {worker_id}: Task {task_id} failed after {max_retries} retries")
                        # Put the task back in the queue for retry
                        await self.task_q.put((command, task_id))
                        
                except Exception as e:
                    self.logger.error(f"Worker {worker_id}: Unhandled error in task {task_id}: {str(e)}\n{traceback.format_exc()}")
                finally:
                    self.task_q.task_done()
                    
            except asyncio.CancelledError:
                self.logger.info(f"Worker {worker_id} cancelled")
                break
            except Exception as e:
                self.logger.error(f"Worker {worker_id}: Unhandled error: {str(e)}\n{traceback.format_exc()}")
                continue

    async def _monitor_task_activity(self, worker_id, task_id, node, start_time):
        """Monitor task activity and log warnings for inactive tasks"""
        ACTIVITY_CHECK_INTERVAL = 60  # Check every minute
        ACTIVITY_TIMEOUT = 300  # Warning after 5 minutes of inactivity
        last_warning = 0  # Track when we last issued a warning
        
        try:
            while True:
                await asyncio.sleep(ACTIVITY_CHECK_INTERVAL)
                current_time = time.time()
                elapsed = current_time - start_time
                
                # Check node connection health periodically
                connection_healthy = await node._check_connection_health()
                if not connection_healthy:
                    self.logger.warning(f"Worker {worker_id}: Node {node.name} connection unhealthy for task {task_id} after {int(elapsed)}s")
                
                # Only check process info if process exists and is running
                if (hasattr(node, 'process') and 
                    node.process and 
                    hasattr(node.process, 'returncode') and 
                    node.process.returncode is None):  # Process is still running
                    
                    try:
                        if hasattr(node, '_get_process_info'):
                            process_info = await node._get_process_info(node.process)
                            if process_info:
                                self.logger.debug(f"Worker {worker_id}: Task {task_id} process info: {process_info}")
                            elif current_time - last_warning > ACTIVITY_TIMEOUT:
                                self.logger.warning(f"Worker {worker_id}: Unable to get process info for task {task_id}")
                                last_warning = current_time
                    except Exception as e:
                        if current_time - last_warning > ACTIVITY_TIMEOUT:
                            self.logger.warning(f"Worker {worker_id}: Error checking process for task {task_id}: {str(e)}")
                            last_warning = current_time
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Worker {worker_id}: Error in task monitor for {task_id}: {str(e)}")

    def create_node(self, target):
        if target.details['type'] == 'ssh':
            return SSHNode(
                node_name=target.name,
                log_q=self.log_q,
                address=target.details['params']['address'],
                username=target.details['params']['username'],
                password=target.details['params'].get('password'),
                key_path=target.details['params'].get('key_path'),
                timeout=RUN_TIMEOUT
            )
        elif target.details['type'] == 'local_async':
            return LocalAsyncNode(target.name, self.log_q, available_gpus=target.details['params']['gpus'], timeout=RUN_TIMEOUT)
        else:
            self.logger.warning(f"Unknown target type: {target.details['type']}")
            return None

    def handle_failed_target(self, target):
        if target.not_failed_us():
            target.retry()
            self.unavailable_targets[target.name] = (time.time(), target.details)
            self.logger.info(f"Target {target.name} will be retried later. Retries left: {target.retries}")
        else:
            self.logger.warning(f"Target {target.name} has failed too many times and will not be retried.")
        self.target_q.task_done()

class Target(object):
    def __init__(self, name, target, retries=MAX_RETRIES) -> None:
        self.name = name
        self.details = target
        self.retries = retries

    def retry(self):
        self.retries -= 1

    def not_failed_us(self):
        return self.retries > 0

    def __lt__(self, other):
        return self.retries > other.retries

    def __str__(self):
        return f"Target({self.name})"
