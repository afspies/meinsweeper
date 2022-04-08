from time import time as time_now
from .utils import get_time_diff


from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table

TABLE_REFRESH_RATE = 3 # Hz
# class Logger():
#     def __init__(self, fuck) -> None:
#         pass

class Logger():
    def __init__(self, log_q) -> None:
        self.log_q = log_q
        self.table = DisplayTable('Training Progress', 'Info')
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

#! -------------------------------------------------------
#! This is a hack
log_file = open('log.txt', 'w', buffering=1)
log_file.write(f'start log at {time_now()}\n')
#! -------------------------------------------------------


class DisplayTable():
    def __init__(self, name: str, table_type: str):
        self.name = name
        # self.table_type = table_type
        self.table = Table(title=name)
        self.table.add_column("Progress", style="cyan", justify="right")        
        self.num_runs_completed = 0
        # self.table.add_column("Info")
        self.progress_bars = Progress( TextColumn(
        "[bold blue]{task.fields[name]}: {task.percentage:.0f}%"
             ),
            BarColumn(),TimeRemainingColumn(),TextColumn(
        "[bold orange] Loss: {task.fields[loss_total]} Test_Acc {task.fields[test_acc]}%"
            ))
        self.table.add_row(Panel.fit(self.progress_bars, title=name, border_style="green", padding=(1, 1)))
        self.host_map = {}

    def update(self, progress: dict, host: str, label: str):
        progress, line = progress
        log_file.write(f'{host} {label} {line}\n')
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
            self.progress_bars.update(host_pid, **progress)
            if progress.get('status', '')=='failed' or self.progress_bars.tasks[host_pid].finished:
                if self.progress_bars.tasks[host_pid].finished:
                    self.num_runs_completed += 1
                # self.progress_bars.remove_task(host_pid) #! doesn't work properly for some reason... (even if cleaning up pids)
                self.progress_bars.tasks[host_pid].visible = False
                del self.host_map[host]
    

    def add(self, host: str, label):
        #! Representation should be generated elsewhere
        #! Also, need to handle stages
        pid = self.progress_bars.add_task("", name=host.split('.')[0]+f' {label}', total=100000) 
        self.host_map[host] = pid
        return pid

    
    def get_num_procs(self):
        return len(self.progress_bars.tasks)

    def __str__(self):
        return str(self.table)