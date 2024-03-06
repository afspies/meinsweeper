from abc import ABC, abstractmethod
import socket
import asyncio
import asyncssh
from pathlib import Path
import logging
import xml.etree.ElementTree as ET

from meinsweeper.modules.helpers.debug_logging import global_logger
from .nodes import *
import textwrap

MINIMUM_VRAM = 8  # in GigaBytes
USAGE_CRITERION = 0.8  # percentage (float) or -1 => no processes other than xorg
MAX_PROCESSES = -1  # -1 => no limit, Otherwise number of processes = min(#nodes, #tbc_runs)
RUN_TIMEOUT = 600 #240  # in seconds
MAX_RETRIES = 3  #! removed from PQ after this - should readd after some interval

class RunManager(object):
    """ The RunManager spawns processes on the target nodes from the pool of available ones
    """
    def __init__(self, targets: dict, task_q: asyncio.Queue, log_q: asyncio.Queue) -> None:
        self.targets = targets
        self.max_proc = len(targets) if MAX_PROCESSES == -1 else MAX_PROCESSES
        self.log_q = log_q
        self.task_q = task_q
        self.running_proc = 0
        self.tasks = []

    async def start_run(self):
        # Create queue of available target nodes
        self.target_q = asyncio.PriorityQueue(
        )  #maxsize=MAX_PROCESSES) #! This should be num tasks, also check behaviour
        [await self.target_q.put(Target(target, retries=MAX_RETRIES)) for target in self.targets]

        # Spawn workers
        # ? Should this be a while loop
        [self.tasks.append(asyncio.create_task(self.spawn_worker())) for _ in range(self.max_proc)]
        await asyncio.gather(*self.tasks)

    # Function which connects to node and runs task
    async def spawn_worker(self) -> None:
        while not self.task_q.empty():
            connected = False
            target = None
            retries = None
            while not connected:
                target = await self.target_q.get()

                # initialize logger
                # Create target
                global_logger.info(f"Connecting to {target}")
                if target.details['type'] == 'ssh':
                    node = SSHNode(log_q=self.log_q, timeout=RUN_TIMEOUT, **target.details['params'])
                    connected = await node.open_connection()

                if not connected:
                    global_logger.warning(f"Failed to connect to {target}")
                    if target.not_failed_us():
                        global_logger.info(f"Retrying {target}")
                        target.retry()
                        await self.target_q.put(target)
                    await asyncio.sleep(5)

            # Run task once a connection was opened
            if self.task_q.empty():  # In case someone else was faster NOTE should instead timeout here
                global_logger.info(f"Task queue empty, closing connection to {target}")
                self.target_q.task_done()
                return

            # Get a task and run it
            #! Replace with configurer objectj
            # NOTE should add timeout here in case of async funkiness
            cfg, label = await self.task_q.get()  # Get next task from queue

            # cfg will be of form {'param1': ..., 'param2': ..., 'param3': ...}
            cmd = 'source ~/.bashrc &&' + cfg
            global_logger.info(f"Running task {cmd}, {label} on {target}")
            
            #! Replace with runner class
            success = await node.run(cmd, label)
            if success:
                global_logger.info(f"Task {label} completed successfully on {target}")
                await self.target_q.put(target)
            else:
                if target.not_failed_us():
                    global_logger.info(f"Task {label} failed on {target}, retrying")
                    target.retry()
                    await self.target_q.put(target)
                await self.task_q.put((cfg, label)) # Need to do task again
            self.task_q.task_done()
            self.target_q.task_done()


class Target(object):
    """
    A wrapper object for a target node which allows ranking in the PQ
    based on retries 
    
    TODO Extend to include other attributes like vram, etc. for ranking 
    """
    def __init__(self, target, retries=3) -> None:
        self.details = target
        self.retries = retries

    def get_target(self):
        return self.target

    def retry(self):
        self.retries -= 1

    def not_failed_us(self):
        return self.retries > 0

    def __lt__(self, other):
        #To work with PQ => Return True if we have MORE retries left
        if self.retries > other.retries:
            return True
