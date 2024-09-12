import logging
import os
import shutil
from pathlib import Path

# Add this near the top of the file
DEBUG = os.getenv('MEINSWEEPER_DEBUG', 'False').lower() == 'true'

LOG_DIR = Path(os.getenv("MEINSWEEPER_LOG_DIR", "./logs"))
LOGGING_ENABLED = os.getenv("MEINSWEEPER_LOGGING_ENABLED", "True").lower() == "true"

if LOGGING_ENABLED:
    # make sure the log, and log/nodes directories exist
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR/'nodes').mkdir(exist_ok=True)
else:
    print("FYI Meinsweeper Logging is disabled")


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
    # global_logger = logging.getLogger('Run Manager')
    global_logger.addHandler(file_handler)
global_logger.addFilter(log_filter)


# -- Decorator to log function calls to general log --
# Use function closure to capture logger
def log_function_call(func):
    def wrapper(*args, **kwargs):
        global_logger.info(f"Calling {func.__name__} with args: {args}, kwargs: {kwargs}")
        return func(*args, **kwargs)
    return wrapper

# -- Functions to initialize node-specific logs -- 
def init_node_logger(name):
    logger = logging.getLogger(f'node_{name}')
    logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)
    
    if LOGGING_ENABLED:
        # Create a unique log file for each thread
        log_filename = f"node_{name}.log"

        # Create a FileHandler for the thread's log file
        #! Add logic to keep log files corresponding to crashed runs and move them into
        #! A special folder
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "nodes" / log_filename,  # Path to the log file
            maxBytes=25 * 1024 * 1024,  # Maximum log file size (25MB)
            backupCount=2,  # Keep at most 2 log files
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s - %(message)s"
            )
        )

        # Add the FileHandler to the thread-specific logger
        logger.addHandler(file_handler)  # Changed from node_logger to logger
    logger.addFilter(log_filter)  # Changed from node_logger to logger
    return logger

def clear_log_files():
    if LOGGING_ENABLED:
        # Clear the main log file
        with open(LOG_DIR / "run.log", 'w') as f:
            f.write('')  # This will clear the file
        
        # Clear all node log files
        node_log_dir = LOG_DIR / "nodes"
        for file in node_log_dir.glob('*.log'):
            with open(file, 'w') as f:
                f.write('')  # This will clear each file

        print("All log files have been cleared.")
    else:
        print("Logging is disabled. No files to clear.")