import logging
import os
import shutil
from pathlib import Path
import queue
import threading
from meinsweeper.modules.helpers.utils import debug_print
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import uuid
from datetime import datetime
import sys
import socket
import psutil
import signal

# Add this near the top of the file
DEBUG = os.getenv('MEINSWEEPER_DEBUG', 'False').lower() == 'true'
LOGGING_DEBUG = os.getenv('MEINSWEEPER_LOGGING_DEBUG', 'False').lower() == 'true'

# Replace existing debug_print calls with logging-specific debug prints
def debug_print(*args, **kwargs):
    """Print debug messages when MEINSWEEPER_DEBUG environment variable is True"""
    if (LOGGING_DEBUG and any('log' in str(arg).lower() for arg in args)):
        print(*args, **kwargs)

LOG_DIR = Path(os.getenv("MEINSWEEPER_LOG_DIR", "./logs"))
LOGGING_ENABLED = os.getenv("MEINSWEEPER_LOGGING_ENABLED", "True").lower() == "true"

if LOGGING_ENABLED:
    # Only create the base log directory
    LOG_DIR.mkdir(exist_ok=True)
    if LOGGING_DEBUG:
        debug_print(f"Created base log directory: {LOG_DIR}")

# Add at the top of the file with other constants
SWEEP_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
SWEEP_LOG_DIR = LOG_DIR / SWEEP_ID
SWEEP_NODES_DIR = SWEEP_LOG_DIR / "nodes"  # Sweep-specific nodes directory
SWEEP_LOGGERS = {}  # Dictionary to track loggers for this sweep instance
LOGGER_LOCK = threading.Lock()  # Thread-safe logger management

if LOGGING_ENABLED:
    # Create complete sweep-specific directory structure
    SWEEP_LOG_DIR.mkdir(exist_ok=True)
    SWEEP_NODES_DIR.mkdir(exist_ok=True)
    if LOGGING_DEBUG:
        debug_print(f"Created sweep directory structure: {SWEEP_LOG_DIR}")

    # Create a marker file with process info to identify active sweeps
    marker_info = {
        'pid': os.getpid(),
        'hostname': socket.gethostname(),
        'start_time': datetime.now().isoformat(),
        'command': ' '.join(sys.argv)
    }
    
    marker_file = SWEEP_LOG_DIR / "active"
    with marker_file.open('w') as f:
        for key, value in marker_info.items():
            f.write(f"{key}: {value}\n")
    if LOGGING_DEBUG:
        debug_print(f"Created sweep marker file: {marker_file}")

def is_sweep_active(sweep_dir: Path) -> bool:
    """Check if a sweep is still active by verifying its process"""
    marker_file = sweep_dir / "active"
    if not marker_file.exists():
        return False
        
    try:
        # Read the marker file
        with marker_file.open('r') as f:
            lines = f.readlines()
        
        # Parse the PID
        pid_line = next(line for line in lines if line.startswith('pid:'))
        pid = int(pid_line.split(':')[1].strip())
        
        # Check if process is still running
        try:
            os.kill(pid, 0)  # Doesn't actually kill the process, just checks if it exists
            return True
        except ProcessLookupError:
            return False
            
    except (FileNotFoundError, ValueError, StopIteration):
        return False

def is_sweep_process_running(pid):
    """Check if a sweep process is still running"""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False

def cleanup_inactive_sweeps():
    """Remove inactive sweeps and their resources"""
    if not LOGGING_ENABLED:
        return
        
    for sweep_dir in LOG_DIR.glob("2*"):  # Match datetime-named directories
        if not sweep_dir.is_dir():
            continue
            
        marker_file = sweep_dir / "active"
        if not marker_file.exists():
            continue
            
        try:
            with marker_file.open('r') as f:
                lines = f.readlines()
                pid_line = next(line for line in lines if line.startswith('pid:'))
                pid = int(pid_line.split(':')[1].strip())
                
            if not is_sweep_process_running(pid):
                if LOGGING_DEBUG:
                    debug_print(f"Removing inactive sweep directory: {sweep_dir}")
                try:
                    shutil.rmtree(sweep_dir)
                except Exception as e:
                    if LOGGING_DEBUG:
                        debug_print(f"Error removing sweep directory: {e}")
                    
        except (FileNotFoundError, ValueError, StopIteration) as e:
            if LOGGING_DEBUG:
                debug_print(f"Error processing sweep directory {sweep_dir}: {e}")

# A lazy way to disable all loggers without needing to check each logging call
class SilentFilter(logging.Filter):
    def __init__(self, logging_enabled=True):
        self.logging_enabled = logging_enabled

    def filter(self, record):
        # If the global flag is set to False, suppress the log record
        return self.logging_enabled  # True => Will be logged

# Create a custom filter instance
log_filter = SilentFilter(logging_enabled=LOGGING_ENABLED)

# -- Configure General Logging --
# Configure the logger
global_logger = logging.getLogger("Run Manager")
global_logger.setLevel(logging.INFO)

if LOGGING_ENABLED:
    # Create a FileHandler to append logs to a shared log file
    file_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "run.log",  # Path to the log file
        maxBytes=50 * 1024 * 1024,  # Maximum log file size (50MB)
        backupCount=2,  # Keep at most 2 log files
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
        )
    )

    # Add the FileHandler to the central logger
    global_logger.addHandler(file_handler)
global_logger.addFilter(log_filter)

# -- Decorator to log function calls to general log --
def log_function_call(func):
    def wrapper(*args, **kwargs):
        global_logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

# -- Functions to initialize node-specific logs -- 
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

def log_worker(log_queue, file_handler):
    while True:
        record = log_queue.get()
        if record is None:
            break
        file_handler.emit(record)

def init_node_logger(name):
    if LOGGING_DEBUG:
        debug_print(f"Creating/getting logger for node: {name} in sweep {SWEEP_ID}")
    
    # Use simple names for loggers but store them in sweep-specific directory
    logger_name = f"node_{name}"
    
    with LOGGER_LOCK:
        if logger_name in SWEEP_LOGGERS:
            if LOGGING_DEBUG:
                debug_print(f"Returning existing logger for {name} from sweep {SWEEP_ID}")
            return SWEEP_LOGGERS[logger_name]

        if LOGGING_DEBUG:
            debug_print(f"Creating new logger for {name} in sweep {SWEEP_ID}")
        logger = logging.getLogger(f"{SWEEP_ID}.{logger_name}")  # Namespace the logger with sweep ID
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        
        if LOGGING_ENABLED:
            # Store logs in sweep-specific nodes directory
            log_path = SWEEP_NODES_DIR / f"{logger_name}.log"
            
            if LOGGING_DEBUG:
                debug_print(f"Creating log file at: {log_path}")
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=25 * 1024 * 1024,
                backupCount=2,
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    f"%(asctime)s - {SWEEP_ID} - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
                )
            )
            logger.addHandler(file_handler)

        logger.addFilter(log_filter)
        SWEEP_LOGGERS[logger_name] = logger
        if LOGGING_DEBUG:
            debug_print(f"Current sweep loggers: {list(SWEEP_LOGGERS.keys())}")
        return logger

def init_file_watcher():
    if LOGGING_ENABLED and DEBUG:
        event_handler = LogFileHandler()
        observer = Observer()
        observer.schedule(event_handler, str(LOG_DIR), recursive=True)
        observer.start()
        debug_print("File watcher initialized")
        return observer
    return None

def clear_log_files():
    if LOGGING_ENABLED:
        if LOGGING_DEBUG:
            debug_print(f"Clearing log files for sweep {SWEEP_ID}")
        observer = init_file_watcher()
        
        with LOGGER_LOCK:
            # Clean up all loggers in this sweep
            for logger_name, logger in SWEEP_LOGGERS.items():
                if LOGGING_DEBUG:
                    debug_print(f"Cleaning up logger: {logger_name}")
                for handler in logger.handlers[:]:
                    try:
                        handler.close()
                        logger.removeHandler(handler)
                    except Exception as e:
                        if LOGGING_DEBUG:
                            debug_print(f"Error cleaning up handler: {e}")

                # Remove logger from logging system
                del logging.root.manager.loggerDict[f"{SWEEP_ID}.{logger_name}"]

            # Clear our sweep's logger dictionary
            SWEEP_LOGGERS.clear()
            if LOGGING_DEBUG:
                debug_print(f"Cleared all loggers for sweep {SWEEP_ID}")

        if observer:
            time.sleep(1)
    else:
        if LOGGING_DEBUG:
            debug_print("Logging is disabled. No files to clear.")

def cleanup_sweep():
    """Clean up resources for current sweep"""
    if LOGGING_ENABLED:
        try:
            marker_file = SWEEP_LOG_DIR / "active"
            if marker_file.exists():
                marker_file.unlink()
            if LOGGING_DEBUG:
                debug_print(f"Removed active marker for sweep {SWEEP_ID}")
        except Exception as e:
            if LOGGING_DEBUG:
                debug_print(f"Error cleaning up sweep: {e}")

class LogFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.event_count = 0
        self.max_events = 10

    def on_created(self, event):
        if not event.is_directory:
            self.event_count += 1
            if LOGGING_DEBUG:
                debug_print(f"\nFile created: {event.src_path}")

    def on_modified(self, event):
        if not event.is_directory and self.event_count < self.max_events:
            self.event_count += 1
            if LOGGING_DEBUG:
                debug_print(f"\nFile modified: {event.src_path}")
            if self.event_count >= self.max_events:
                if LOGGING_DEBUG:
                    debug_print("\nReached maximum number of events to display...")