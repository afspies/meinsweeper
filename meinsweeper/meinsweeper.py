import asyncio

from rich.live import Live

from .due import *

# global console # will be set in main
from .modules import LogParser, RunManager
from .modules.helpers.debug_logging import init_node_logger

# Change these according to your own configuration.

SSH_TIMEOUT = 10  #seconds

async def run_sweep(cmd_list, targets, steps=None):
    logger = init_node_logger('run_sweep')
    logger.info(f"Initializing new sweep with {len(cmd_list)} tasks and {len(targets)} targets")
    
    # cmd_list should be a list of strings or list of tuples (str, str)
    if isinstance(cmd_list[0], str): # no labels included
        cmd_list = [(cmd, f'job {i}') for i,cmd in enumerate(cmd_list)]
    
    sweep_q = asyncio.Queue()  # Workers take from - All sweep cfgs are enqueued here
    log_q = asyncio.Queue()  # Workers put into - All log cfgs are enqueued here

    for sweep_run in cmd_list:
        await sweep_q.put(sweep_run)

    # Start the run
    run_mgr = RunManager(targets, sweep_q, log_q)
    log_parser = LogParser(log_q, steps=steps)

    with Live(log_parser.display, refresh_per_second=3) as live:
        runs = asyncio.create_task(run_mgr.start_run())
        logs = asyncio.create_task(log_parser.start_logger())

        await runs
        await sweep_q.join()  # Wait for all tasks to be processed
        await log_q.join()  # Wait for all logs to be processed

        log_parser.complete(live)
        logs.cancel()

    # Add this block to print out detailed error information
    failed_jobs = [job for job in log_parser.table.progress_bars.tasks if job.completed < (steps or 100)]
    if failed_jobs:
        logger.info("\nFailed jobs:")
        for job in failed_jobs:
            logger.info(f"Job {job.fields['name']}:")
            if 'error' in job.fields:
                logger.info(f"Error: {job.fields['error']}")
            if 'traceback' in job.fields:
                logger.info(f"Traceback:\n{job.fields['traceback']}")
            if 'return_code' in job.fields:
                logger.info(f"Return code: {job.fields['return_code']}")
            logger.info("")

    logger.info("Sweep completed")

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(
    Doi("10.1167/13.9.30"),
    description="light-weight framework for running experiments on arbitrary compute nodes",
    tags=["experiment-management"],
    path='meinsweeper'
)
