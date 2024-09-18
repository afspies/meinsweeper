from abc import ABC, abstractmethod
import socket
import asyncio
import os
import asyncssh
from pathlib import Path
import logging
import xml.etree.ElementTree as ET
import time

from meinsweeper.modules.helpers.debug_logging import init_node_logger, global_logger, clear_log_files
from .nodes import *
import textwrap
from .nodes.local_async_node import LocalAsyncNode

# Use environment variables with default values
MINIMUM_VRAM = int(os.environ.get('MINIMUM_VRAM', 8))  # in GigaBytes
USAGE_CRITERION = float(os.environ.get('USAGE_CRITERION', 0.8))  # percentage (float) or -1 => no processes other than xorg
MAX_PROCESSES = int(os.environ.get('MAX_PROCESSES', -1))  # -1 => no limit, Otherwise number of processes = min(#nodes, #tbc_runs)
RUN_TIMEOUT = int(os.environ.get('RUN_TIMEOUT', 1200))  # in seconds
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))  # removed from PQ after this - should readd after some interval
RETRY_INTERVAL = int(os.environ.get('MEINSWEEPER_RETRY_INTERVAL', 450))  # Default to 7.5 minutes, but allow override

class RunManager(object):
    """ The RunManager spawns processes on the target nodes from the pool of available ones
    """
    def __init__(self, targets: dict, task_q: asyncio.Queue, log_q: asyncio.Queue) -> None:
        self.targets = targets
        self.max_proc = min(len(targets), MAX_PROCESSES) if MAX_PROCESSES != -1 else len(targets)
        self.log_q = log_q
        self.task_q = task_q
        self.running_proc = 0
        self.tasks = []
        self.logger = init_node_logger('run_manager')
        self.unavailable_targets = {}
        self.retry_event = asyncio.Event()
        self.stop_event = asyncio.Event()

        # Clear log files when RunManager is instantiated
        clear_log_files()
        self.logger.info(f"Starting new sweep. Log files have been cleared. Max concurrent processes: {self.max_proc}")
        self.logger.info(f"Retry interval set to {RETRY_INTERVAL} seconds")

    async def start_run(self):
        self.target_q = asyncio.PriorityQueue()
        for name, target in self.targets.items():
            await self.target_q.put(Target(name, target, retries=MAX_RETRIES))

        self.retry_task = asyncio.create_task(self.retry_unavailable_targets())
        self.tasks = [asyncio.create_task(self.spawn_worker()) for _ in range(self.max_proc)]
        
        try:
            await self.task_q.join()  # Wait for all jobs to complete
        finally:
            self.stop_event.set()  # Signal all tasks to stop
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.retry_task.cancel()  # Cancel the retry task after all workers have finished

    async def retry_unavailable_targets(self):
        self.logger.info("Starting retry_unavailable_targets task")
        while not self.stop_event.is_set():
            current_time = time.time()
            targets_to_retry = []
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
            
            if self.unavailable_targets:
                self.logger.info(f"Unavailable targets: {list(self.unavailable_targets.keys())}. Next check in {RETRY_INTERVAL} seconds.")
            
            await asyncio.sleep(RETRY_INTERVAL)

    async def spawn_worker(self) -> None:
        while not self.stop_event.is_set():
            if self.target_q.empty() and self.unavailable_targets:
                await self.retry_event.wait()
                self.retry_event.clear()
                continue

            if self.target_q.empty() and self.task_q.empty():
                self.logger.info("No more tasks or targets. Worker exiting.")
                return

            try:
                target = await asyncio.wait_for(self.target_q.get(), timeout=5)
            except asyncio.TimeoutError:
                continue

            self.logger.info(f"Connecting to {target.name}")
            node = self.create_node(target)
            if node is None:
                self.logger.warning(f"Failed to create node for {target.name}")
                self.handle_failed_target(target)
                continue

            connected = await node.open_connection()

            if not connected:
                self.logger.info(f"Failed to connect to {target.name}, will retry later")
                self.handle_failed_target(target)
                continue

            self.logger.info(f"Successfully connected to {target.name}")

            while not self.task_q.empty():
                cfg, label = await self.task_q.get()
                cmd = cfg
                self.logger.info(f"Assigning task {label} to {target.name}")
                
                success = await node.run(cmd, label)
                if success:
                    self.logger.info(f"Task {label} completed successfully on {target.name}")
                else:
                    self.logger.info(f"Task {label} failed on {target.name}, retrying")
                    await self.task_q.put((cfg, label))
                self.task_q.task_done()

            if not self.task_q.empty():
                await self.target_q.put(target)
            self.target_q.task_done()

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

