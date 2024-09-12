import asyncio

from rich.live import Live

from .due import *

# global console # will be set in main
from .modules import LogParser, RunManager
from .modules.helpers.debug_logging import global_logger

# Change these according to your own configuration.

SSH_TIMEOUT = 10  #seconds

async def run_sweep(cmd_list, targets, steps=None):
    global_logger.info("Initializing new sweep")
    
    # cmd_list should be a list of strings or list of tuples (str, str)
    if isinstance(cmd_list[0], str): # no labels included
        cmd_list = [(cmd, f'job {i}') for i,cmd in enumerate(cmd_list)]
    
    sweep_q = asyncio.Queue()  # Workers take from - All sweep cfgs are enqueued here
    log_q = asyncio.Queue()  # Workers put into - All log cfgs are enqueued here

    [await sweep_q.put(sweep_run) for sweep_run in cmd_list]

    # Start the run
    run_mgr = RunManager(targets, sweep_q, log_q)
    logger = LogParser(log_q, steps=steps)

    with Live(logger.display, refresh_per_second=3) as live:
        runs = asyncio.create_task(run_mgr.start_run())
        logs = asyncio.create_task(logger.start_logger())

        await runs
        await sweep_q.join()  # Wait for all tasks to be processed
        await log_q.join()  # Wait for all logs to be processed

        logger.complete(live)
        logs.cancel()

    # Add this line to ensure all tasks are completed
    while not logger.table.is_complete():
        await asyncio.sleep(0.1)

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(
    Doi("10.1167/13.9.30"),
    description="light-weight framework for running experiments on arbitrary compute nodes",
    tags=["experiment-management"],
    path='meinsweeper'
)
