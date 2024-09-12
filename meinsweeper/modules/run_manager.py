from abc import ABC, abstractmethod
import socket
import asyncio
import os
import asyncssh
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

from meinsweeper.modules.helpers.debug_logging import global_logger, clear_log_files
from .nodes import *
import textwrap
from .nodes.local_async_node import LocalAsyncNode

# Use environment variables with default values
MINIMUM_VRAM = int(os.environ.get('MINIMUM_VRAM', 8))  # in GigaBytes
USAGE_CRITERION = float(os.environ.get('USAGE_CRITERION', 0.8))  # percentage (float) or -1 => no processes other than xorg
MAX_PROCESSES = int(os.environ.get('MAX_PROCESSES', -1))  # -1 => no limit, Otherwise number of processes = min(#nodes, #tbc_runs)
RUN_TIMEOUT = int(os.environ.get('RUN_TIMEOUT', 1200))  # in seconds
MAX_RETRIES = int(os.environ.get('MAX_RETRIES', 3))  #! removed from PQ after this - should readd after some interval

class RunManager(object):
    """ The RunManager spawns processes on the target nodes from the pool of available ones
    """
    def __init__(self, targets: dict, task_q: asyncio.Queue, log_q: asyncio.Queue) -> None:
        assert isinstance(targets, dict), "Targets must be a dictionary"
        self.targets = targets
        self.max_proc = len(targets) if MAX_PROCESSES == -1 else MAX_PROCESSES
        self.log_q = log_q
        self.task_q = task_q
        self.running_proc = 0
        self.tasks = []

        # Clear log files when RunManager is instantiated
        clear_log_files()
        global_logger.info("Starting new sweep. Log files have been cleared.")

    async def start_run(self):
        # Create queue of available target nodes
        self.target_q = asyncio.PriorityQueue()
        for name, target in self.targets.items():
            await self.target_q.put(Target(name, target, retries=MAX_RETRIES))

        # Spawn workers
        [self.tasks.append(asyncio.create_task(self.spawn_worker())) for _ in range(self.max_proc)]
        await asyncio.gather(*self.tasks)

    # Function which connects to node and runs task
    async def spawn_worker(self) -> None:
        while not self.task_q.empty():
            connected = False
            target = None
            while not connected:
                target = await self.target_q.get()

                global_logger.info(f"Connecting to {target.name}")
                if target.details['type'] == 'ssh':
                    node = SSHNode(log_q=self.log_q, timeout=RUN_TIMEOUT, **target.details['params'])
                elif target.details['type'] == 'local_async':
                    node = LocalAsyncNode(log_q=self.log_q, available_gpus=target.details['params']['gpus'], timeout=RUN_TIMEOUT)
                else:
                    global_logger.warning(f"Unknown target type: {target.details['type']}")
                    self.target_q.task_done()
                    continue

                connected = await node.open_connection()

                if not connected:
                    if target.not_failed_us():
                        global_logger.info(f"Failed to connect to {target.name}, retrying")
                        target.retry()
                        await self.target_q.put(target)
                    self.target_q.task_done()
                    continue

                # Run task once a connection was opened
                if self.task_q.empty():  # In case someone else was faster NOTE should instead timeout here
                    global_logger.info(f"Task queue empty, closing connection to {target.name}")
                    self.target_q.task_done()
                    return

                # Get a task and run it
                cfg, label = await self.task_q.get()  # Get next task from queue

                # Remove the 'source ~/.bashrc &&' prefix
                cmd = cfg
                global_logger.info(f"Running task {cmd}, {label} on {target.name}")
                
                success = await node.run(cmd, label)
                if success:
                    global_logger.info(f"Task {label} completed successfully on {target.name}")
                    await self.target_q.put(target)
                else:
                    if target.not_failed_us():
                        global_logger.info(f"Task {label} failed on {target.name}, retrying")
                        target.retry()
                        await self.target_q.put(target)
                    await self.task_q.put((cfg, label)) # Need to do task again
                self.task_q.task_done()
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
        #To work with PQ => Return True if we have MORE retries left
        if self.retries > other.retries:
            return True

    def __str__(self):
        return f"Target({self.name})"

