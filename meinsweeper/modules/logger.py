from time import time as time_now
from .helpers.utils import get_time_diff

import logging
from logging import Logger

from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
import asyncio

TABLE_REFRESH_RATE = 3  # Hz

#-------------------------------------------------------
# This class will allow people to log in a fashion compliant with MeinSweeper
# RUNS ON REMOTE
class MSLogger():
    """Wrapper around a python logger which will take train/test 
    losses and print them in a format compatible with MeinSweeper"""
    def __init__(self):
        self.python_logger = logging.getLogger('meinsweeper_logger')
        self.python_logger.setLevel(logging.INFO)
        self.step = {'train': 0, 'val': 0, 'test': 0}

    def log_loss(self, loss: float, mode: str = 'train', step: int = None):
        assert mode in ['train', 'val', 'test'], f"Only modes 'train', 'val' and 'test' are supported"
        if step is None:
            step = self.step[mode]
            self.step[mode] += 1

        log_str = f'[[LOG_ACCURACY {mode.upper()}]] Step: {step}; Losses: {mode.capitalize()}: {loss}'
        self.python_logger.info(log_str)

    # def __getstate__(self):
    #     state = self.__dict__.copy()
    #     state["python_logger"] = None
    #     return state

    # def __setstate__(self, d):
    #     self.__dict__.update(d)  # I *think* this is a safe way to do it
    #     self.python_logger = logging.getLogger(
    #         'meinsweeper_logger'
    #     )


# ------------------------------------------------------
# This class will display logging info in a rich table
# RUNS ON HOST
class LogParser():
    def __init__(self, log_q: asyncio.PriorityQueue, steps: int = None) -> None:
        self.log_q = log_q
        self.table = DisplayTable('Training Progress', 'Info', steps=steps)
        self.display = self.table.table
        # can't get procs from table to count, as this includes failed runs
        self.start_time = time_now()

    async def start_logger(self):
        while True:
            i, addr, label = await self.log_q.get()
            self.table.update(i, addr, label)
            self.log_q.task_done()

    def complete(self, live_session):
        time_elapsed = get_time_diff(self.start_time)
        msg = f"🚀 Done! Completed {self.table.num_runs_completed} runs in {time_elapsed} 🚀"
        live_session.update(Panel(Text(msg, style="bold magenta", justify="center")))


# ------------------------------------------------------
class DisplayTable():
    def __init__(self, name: str, table_type: str, steps=None):
        self.name = name
        self.steps = steps
        self.table = Table(title=name)
        self.table.add_column("Progress", style="cyan", justify="right")
        self.num_runs_completed = 0
        self.progress_bars = Progress(
            TextColumn("[bold blue]{task.fields[name]}: {task.percentage:.0f}%"),
            BarColumn(),
            TimeRemainingColumn(),
            TextColumn(
                "[bold orange] Loss: {task.fields[loss_total]:.2e} Test_Acc {task.fields[test_acc]:.2e}"
            ),
        )
        self.table.add_row(Panel.fit(self.progress_bars, title=name, border_style="green", padding=(1, 1)))
        self.host_map = {}

    def update(self, progress: dict, host: str, label: str):
        progress, line = (
            progress  # progress is parsed line (i.e. losses or status) and line is raw
        )

        if progress is None:
            return
        progress['raw'] = str(progress)

        if host not in self.host_map:
            host_pid = self.add(host, label)
            progress['completed'] = 0
            progress['loss_total'] = 0
            progress['test_acc'] = 0
            self.progress_bars.update(host_pid, **progress)
        else:
            host_pid = self.host_map[host]
            failed = progress.get('status', '') == 'failed'
            if not failed:
                self.progress_bars.update(host_pid, **progress)
            else:
                self.progress_bars.update(host_pid, total=1, completed=1)

            if self.progress_bars.tasks[host_pid].finished:
                if not failed:
                    self.num_runs_completed += 1
                self.progress_bars.update(host_pid, visible=False)
                # self.progress_bars.tasks[host_pid].visible = False # self.progress_bars.remove_task(host_pid) #! doesn't work properly for some reason... (even if cleaning up pids)
                del self.host_map[host]

    def add(self, host: str, label: str):
        #! Representation should be generated elsewhere
        #! Also, need to handle stages
        # truncate label to 30 characters
        if len(label) > 30:
            label = label[:30] + (label[30:] and "...")
        pid = self.progress_bars.add_task(
            "", name=host.split('.')[0] + f' {label}', total=self.steps if self.steps else 100000
        )
        self.host_map[host] = pid
        return pid

    def get_num_procs(self):
        return len(self.progress_bars.tasks)

    def __str__(self):
        return str(self.table)