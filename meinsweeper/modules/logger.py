from time import time as time_now
from .helpers.utils import get_time_diff
import os
from datetime import datetime
import sys
import socket
import traceback
import threading

import logging
from logging import Logger

from rich.text import Text
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
import asyncio

TABLE_REFRESH_RATE = 3  # Hz

# Local logging configuration from environment variables
ENABLE_LOCAL_LOGGING = os.environ.get('MS_ENABLE_LOCAL_LOGGING', '').lower() == 'true'
LOG_DIR = os.environ.get('MS_LOG_DIR', 'logs')

# Get hostname for unique logging
HOSTNAME = socket.gethostname()

#-------------------------------------------------------
# This class will allow people to log in a fashion compliant with MeinSweeper
# RUNS ON REMOTE
# Add these imports at the top
import sys
from io import StringIO
import fcntl
import os

class LoggerWriter:
    def __init__(self, logger_func):
        self.logger_func = logger_func
        self._buffer = StringIO()
        
    def write(self, msg):
        if msg:  # Remove the isspace() check to capture all output
            self._buffer.write(msg)
            if '\n' in msg:  # Flush on newlines
                self.flush()
    
    def flush(self):
        msg = self._buffer.getvalue()
        if msg:
            self.logger_func(msg.rstrip())
            self._buffer.truncate(0)
            self._buffer.seek(0)

class MSLogger():
    def __init__(self):
        self.step = {'train': 0, 'val': 0, 'test': 0}
        self.log_lock = threading.Lock()  # Add thread safety
        
        if ENABLE_LOCAL_LOGGING:
            # Create logs directory if it doesn't exist
            os.makedirs(LOG_DIR, exist_ok=True)
            
            # Create timestamp and include hostname and process ID for unique log file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pid = os.getpid()
            log_file = os.path.join(LOG_DIR, f"meinsweeper_{HOSTNAME}_{timestamp}_pid{pid}.log")
            
            # Setup file logger with unique name including process ID
            self.file_logger = logging.getLogger(f'meinsweeper_logger_{HOSTNAME}_{timestamp}_pid{pid}')
            self.file_logger.setLevel(logging.DEBUG)
            
            # Create file handler with unbuffered write mode
            fh = logging.FileHandler(log_file, mode='x')
            fh.setLevel(logging.DEBUG)
            
            # Set file descriptor to unbuffered mode
            fd = fh.stream.fileno()
            flags = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_SYNC)
            
            # Create formatter with hostname and process ID
            formatter = logging.Formatter(
                f'%(asctime)s - {HOSTNAME} - PID:%(process)d - TID:%(thread)d - %(levelname)s - %(message)s'
            )
            fh.setFormatter(formatter)
            
            # Add handler to logger
            self.file_logger.addHandler(fh)
            
            # Store original stdout/stderr
            self._stdout = sys.stdout
            self._stderr = sys.stderr
            
            # Create and set up stdout/stderr redirectors
            self.stdout_logger = LoggerWriter(self.file_logger.info)
            self.stderr_logger = LoggerWriter(self.file_logger.error)
            
            # Redirect stdout and stderr
            sys.stdout = self.stdout_logger
            sys.stderr = self.stderr_logger
            
            # Set up exception hook
            self._original_excepthook = sys.excepthook
            sys.excepthook = self.handle_exception
            
            # Log initial system state
            self.file_logger.info("=== Logging Started ===")
            self.log_system_info()

    def log_system_info(self):
        """Log detailed system information"""
        with self.log_lock:
            self.file_logger.info("=== System Information ===")
            self.file_logger.info(f"Hostname: {HOSTNAME}")
            self.file_logger.info(f"Python version: {sys.version}")
            self.file_logger.info(f"Process ID: {os.getpid()}")
            self.file_logger.info(f"Parent Process ID: {os.getppid()}")
            self.file_logger.info(f"Working directory: {os.getcwd()}")
            self.file_logger.info(f"Command line: {' '.join(sys.argv)}")
            self.file_logger.info("Environment variables:")
            for key, value in sorted(os.environ.items()):
                self.file_logger.info(f"  {key}={value}")
            self.file_logger.info("======================")

    def handle_exception(self, exc_type, exc_value, exc_traceback):
        """Handle uncaught exceptions by logging them"""
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        with self.log_lock:
            self.file_logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

    def log_error(self, error_msg: str, include_trace: bool = True):
        """Log an error with optional stack trace"""
        with self.log_lock:
            if ENABLE_LOCAL_LOGGING:
                if include_trace:
                    self.file_logger.error(f"ERROR: {error_msg}\nTraceback:\n{traceback.format_exc()}")
                else:
                    self.file_logger.error(f"ERROR: {error_msg}")
            print(f"[[ERROR]] {error_msg}", flush=True)

    def log_warning(self, warning_msg: str):
        """Log a warning message"""
        with self.log_lock:
            if ENABLE_LOCAL_LOGGING:
                self.file_logger.warning(f"WARNING: {warning_msg}")
            print(f"[[WARNING]] {warning_msg}", flush=True)

    def log_debug(self, debug_msg: str):
        """Log a debug message"""
        with self.log_lock:
            if ENABLE_LOCAL_LOGGING:
                self.file_logger.debug(debug_msg)
            print(f"[[DEBUG]] {debug_msg}", flush=True)

    def log_loss(self, loss: float, mode: str = 'train', step: int = None):
        assert mode in ['train', 'val', 'test'], f"Only modes 'train', 'val' and 'test' are supported"
        if step is None:
            step = self.step[mode]
            self.step[mode] += 1

        try:
            log_str = f'[[LOG_ACCURACY {mode.upper()}]] Step: {step}; Losses: {mode.capitalize()}: {loss}'
            print(log_str, flush=True)
            
            if ENABLE_LOCAL_LOGGING:
                with self.log_lock:
                    self.file_logger.info(log_str)
        except Exception as e:
            self.log_error(f"Failed to log loss: {str(e)}")

    def __del__(self):
        """Cleanup when logger is destroyed"""
        if ENABLE_LOCAL_LOGGING:
            try:
                # Flush any remaining output
                sys.stdout.flush()
                sys.stderr.flush()
                
                # Restore original stdout/stderr
                sys.stdout = self._stdout
                sys.stderr = self._stderr
                
                # Restore original exception hook
                sys.excepthook = self._original_excepthook
                
                self.file_logger.info("Logger shutting down")
                
                # Close all handlers
                for handler in self.file_logger.handlers:
                    handler.flush()
                    handler.close()
                    self.file_logger.removeHandler(handler)
                    
            except Exception as e:
                # If we can't log normally, try printing directly
                print(f"Error during logger cleanup: {e}", file=self._stderr)
                
# class MSLogger():
#     """Wrapper around a python logger which will take train/test 
#     losses and print them in a format compatible with MeinSweeper"""
#     def __init__(self):
#         self.step = {'train': 0, 'val': 0, 'test': 0}
#         self.log_lock = threading.Lock()  # Add thread safety
        
#         # Setup local logging if enabled via environment variable
#         if ENABLE_LOCAL_LOGGING:
#             # Create logs directory if it doesn't exist
#             os.makedirs(LOG_DIR, exist_ok=True)
            
#             # Create timestamp and include hostname and process ID for unique log file
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             pid = os.getpid()
#             log_file = os.path.join(LOG_DIR, f"meinsweeper_{HOSTNAME}_{timestamp}_pid{pid}.log")
            
#             # Setup file logger with unique name including process ID
#             self.file_logger = logging.getLogger(f'meinsweeper_logger_{HOSTNAME}_{timestamp}_pid{pid}')
#             self.file_logger.setLevel(logging.DEBUG)
            
#             # Create file handler with exclusive write mode
#             fh = logging.FileHandler(log_file, mode='x')  # 'x' mode ensures file doesn't exist
#             fh.setLevel(logging.DEBUG)
            
#             # Create formatter with hostname and process ID
#             formatter = logging.Formatter(
#                 f'%(asctime)s - {HOSTNAME} - PID:%(process)d - TID:%(thread)d - %(levelname)s - %(message)s'
#             )
#             fh.setFormatter(formatter)
            
#             # Add handler to logger
#             self.file_logger.addHandler(fh)
            
#             # Log system information
#             self.file_logger.info(f"=== System Information ===")
#             self.file_logger.info(f"Hostname: {HOSTNAME}")
#             self.file_logger.info(f"Python version: {sys.version}")
#             self.file_logger.info(f"Process ID: {os.getpid()}")
#             self.file_logger.info(f"Working directory: {os.getcwd()}")
#             self.file_logger.info(f"Command line: {' '.join(sys.argv)}")
#             self.file_logger.info(f"Environment variables: {dict(os.environ)}")
#             self.file_logger.info(f"======================")
            
#             # Add exception hook to catch unhandled exceptions
#             sys.excepthook = self.handle_exception
            
#             # Redirect stdout and stderr to the log file
#             sys.stdout = LoggerWriter(self.file_logger.info)
#             sys.stderr = LoggerWriter(self.file_logger.error)
            
#             self.file_logger.info(f"Local logging initialized on {HOSTNAME} - Log file: {log_file}")

#     def handle_exception(self, exc_type, exc_value, exc_traceback):
#         """Handle uncaught exceptions by logging them"""
#         if issubclass(exc_type, KeyboardInterrupt):
#             # Call the default handler for KeyboardInterrupt
#             sys.__excepthook__(exc_type, exc_value, exc_traceback)
#             return

#         with self.log_lock:
#             self.file_logger.error("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

#     def log_error(self, error_msg: str, include_trace: bool = True):
#         """Log an error with optional stack trace"""
#         with self.log_lock:
#             if ENABLE_LOCAL_LOGGING:
#                 if include_trace:
#                     self.file_logger.error(f"ERROR: {error_msg}\nTraceback:\n{traceback.format_exc()}")
#                 else:
#                     self.file_logger.error(f"ERROR: {error_msg}")
#             # Also print to ensure it shows up in meinsweeper's logs
#             print(f"[[ERROR]] {error_msg}", flush=True)

#     def log_warning(self, warning_msg: str):
#         """Log a warning message"""
#         with self.log_lock:
#             if ENABLE_LOCAL_LOGGING:
#                 self.file_logger.warning(f"WARNING: {warning_msg}")
#             print(f"[[WARNING]] {warning_msg}", flush=True)

#     def log_debug(self, debug_msg: str):
#         """Log a debug message"""
#         with self.log_lock:
#             if ENABLE_LOCAL_LOGGING:
#                 self.file_logger.debug(debug_msg)
#             print(f"[[DEBUG]] {debug_msg}", flush=True)

#     def log_loss(self, loss: float, mode: str = 'train', step: int = None):
#         assert mode in ['train', 'val', 'test'], f"Only modes 'train', 'val' and 'test' are supported"
#         if step is None:
#             step = self.step[mode]
#             self.step[mode] += 1

#         try:
#             log_str = f'[[LOG_ACCURACY {mode.upper()}]] Step: {step}; Losses: {mode.capitalize()}: {loss}'
#             print(log_str, flush=True)
            
#             if ENABLE_LOCAL_LOGGING:
#                 with self.log_lock:
#                     self.file_logger.info(log_str)
#         except Exception as e:
#             self.log_error(f"Failed to log loss: {str(e)}")

#     def __del__(self):
#         """Cleanup when logger is destroyed"""
#         if ENABLE_LOCAL_LOGGING:
#             self.file_logger.info("Logger shutting down")
#             for handler in self.file_logger.handlers:
#                 handler.close()
#                 self.file_logger.removeHandler(handler)

# # Add this helper class to handle stdout/stderr redirection
# class LoggerWriter:
#     def __init__(self, logger_func):
#         self.logger_func = logger_func
#         self.buf = []

#     def write(self, msg):
#         if msg and not msg.isspace():
#             self.logger_func(msg.rstrip())
    
#     def flush(self):
#         pass

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
            try:
                msg = await asyncio.wait_for(self.log_q.get(), timeout=300)  # 5 minute timeout
                if isinstance(msg, tuple) and len(msg) == 3:
                    content, addr, label = msg
                    if isinstance(content, str):
                        self.table.update(content, addr, label)
                    elif isinstance(content, dict):
                        # Add timestamp to logged messages
                        content['timestamp'] = datetime.now().isoformat()
                        if content.get('status') == 'failed':
                            print(f"Failed job detected: {label} on {addr}")
                            print(f"Error details: {content}")
                        self.table.update(content, addr, label)
                self.log_q.task_done()
            except asyncio.TimeoutError:
                print("WARNING: No log messages received for 5 minutes")
                continue
            except Exception as e:
                print(f"Error in logger: {e}")
                traceback.print_exc()
                continue

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

        # Handle failed jobs
        if isinstance(progress, dict) and progress.get('status') == 'failed':
            # Hide the failed task and remove it from tracking
            self.progress_bars.update(host_pid, visible=False)
            if host_pid in self.completed_tasks:
                self.completed_tasks.remove(host_pid)
            if job_key in self.job_pids:
                del self.job_pids[job_key]
            return

        # Check for completion conditions
        completed = False
        if isinstance(progress, dict):
            completed = (
                progress.get('status') == 'completed' or
                progress.get('final', False) or
                progress.get('completed', 0) >= (self.steps or 100)
            )

        if completed and host_pid not in self.completed_tasks:
            self.num_runs_completed += 1
            self.completed_tasks.add(host_pid)
            # Hide the task and ensure it shows as 100% complete
            self.progress_bars.update(
                host_pid, 
                completed=self.steps or 100,
                visible=False
            )
            return

        # Only update non-completed tasks
        if not completed and not progress.get('status') == 'failed':
            progress['completed'] = progress.get('completed', 0)
            progress['loss_total'] = progress.get('loss_total', 0)
            self.progress_bars.update(host_pid, **progress)

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