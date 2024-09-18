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
        self.step = {'train': 0, 'val': 0, 'test': 0}

    def log_loss(self, loss: float, mode: str = 'train', step: int = None):
        assert mode in ['train', 'val', 'test'], f"Only modes 'train', 'val' and 'test' are supported"
        if step is None:
            step = self.step[mode]
            self.step[mode] += 1

        log_str = f'[[LOG_ACCURACY {mode.upper()}]] Step: {step}; Losses: {mode.capitalize()}: {loss}'
        print(log_str, flush=True)

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
    def __init__(self, log_q: asyncio.Queue, steps: int = None) -> None:
        self.log_q = log_q
        self.table = DisplayTable('Training Progress', 'Info', steps=steps)
        self.display = self.table.table
        self.start_time = time_now()

    async def start_logger(self):
        while True:
            msg = await self.log_q.get()
            if isinstance(msg, tuple) and len(msg) == 3:
                content, addr, label = msg
                if isinstance(content, str):
                    # Instead of printing, update the table
                    self.table.update(content, addr, label)
                elif isinstance(content, dict):
                    self.table.update(content, addr, label)
            self.log_q.task_done()

    def complete(self, live_session):
        time_elapsed = get_time_diff(self.start_time)
        msg = f"🚀 Done! Completed {self.table.num_runs_completed} runs in {time_elapsed} 🚀"
        live_session.update(Panel(Text(msg, style="bold magenta", justify="center")))

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
                "[bold orange] Loss: {task.fields[loss_total]:.2e}"
            ),
        )
        self.table.add_row(Panel.fit(self.progress_bars, title=name, border_style="green", padding=(1, 1)))
        self.host_map = {}
        self.completed_tasks = set()
        self.job_pids = {}  # New dictionary to keep track of job PIDs

    def update(self, content, host: str, label: str):
        if isinstance(content, dict):
            progress = content
        elif isinstance(content, str):
            progress = self.parse_log_line(content)
        else:
            return

        if progress is None:
            return

        job_key = f"{host}_{label}"
        if job_key not in self.job_pids:
            host_pid = self.add(host, label)
            self.job_pids[job_key] = host_pid
        else:
            host_pid = self.job_pids[job_key]

        progress['completed'] = progress.get('completed', 0)
        progress['loss_total'] = progress.get('loss_total', 0)
        
        failed = progress.get('status', '') == 'failed'
        if not failed:
            self.progress_bars.update(host_pid, **progress)

        if progress.get('completed', 0) >= (self.steps or 100) or 'Job completed' in str(content):
            if host_pid not in self.completed_tasks:
                self.num_runs_completed += 1
                self.completed_tasks.add(host_pid)
            self.progress_bars.update(host_pid, completed=self.steps or 100, visible=False)

    @staticmethod
    def parse_log_line(line):
        if '[[LOG_ACCURACY' in line:
            mode = line.split('[[LOG_ACCURACY')[1].split(']]')[0].strip()
            parts = line.split(';')
            step = int(parts[0].split(':')[1].strip())
            loss = float(parts[1].split(':')[2].strip())
            return {'completed': step, 'loss_total': loss, 'mode': mode}
        return None

    def add(self, host: str, label: str):
        if len(label) > 30:
            label = label[:30] + (label[30:] and "...")
        pid = self.progress_bars.add_task(
            "", name=host.split('.')[0] + f' {label}', total=self.steps if self.steps else 100000,
            completed=0, loss_total=0
        )
        self.host_map[host] = pid
        return pid

    def get_num_procs(self):
        return len(self.progress_bars.tasks)

    def __str__(self):
        return str(self.table)

    def is_complete(self):
        return len(self.completed_tasks) == len(self.progress_bars.tasks)