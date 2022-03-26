
from time import sleep

from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

TABLE_REFRESH_RATE = 5 # Hz



class Logger():
    def __init__(self, log_q) -> None:
        self.log_q = log_q
        self.table = DisplayTable('Training Progress', 'Info')
        self.display = self.table.table

    async def start_logger(self):
        while True:
            i, t, cfg = await self.log_q.get()
            self.table.update(i, t, cfg)
            self.log_q.task_done()
        

class DisplayTable():
    def __init__(self, name: str, table_type: str):
        self.name = name
        # self.table_type = table_type
        self.table = Table(title=name)
        self.table.add_column("Progress", style="cyan", justify="right")
        # self.table.add_column("Info")
        self.progress_bars = Progress( TextColumn(
        "[bold blue]Progress for app {task.fields[name]}: {task.percentage:.0f}%"
             ),
            BarColumn(),TimeRemainingColumn(),)
        self.table.add_row(Panel.fit(self.progress_bars, title=name, border_style="green", padding=(1, 1)))
        self.host_map = {}

    def update(self, progress: str, host: str, cfg):
        if host not in self.host_map:
            self.add(host, cfg) 
        else:
            host_pid = self.host_map[host]
            self.progress_bars.advance(host_pid)
            if self.progress_bars.tasks[host_pid].finished:
                # self.progress_bars.remove_task(host_pid) #! doesn't work properly for some reason... (even if cleaning up pids)
                self.progress_bars.tasks[host_pid].visible = False
                del self.host_map[host]


    def add(self, host: str, cfg):
        #! Representation should be generated elsewhere
        self.host_map[host] = self.progress_bars.add_task("", name=host.split('.')[0]+f'_{cfg}', total=10)
    

    def __str__(self):
        return str(self.table)