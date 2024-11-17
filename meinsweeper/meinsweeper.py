import asyncio
import traceback

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
    
    try:
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

            try:
                await runs
                await sweep_q.join()
                await log_q.join()
            except Exception as e:
                logger.error(f"Error during sweep execution: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
            finally:
                log_parser.complete(live)
                logs.cancel()

        # Improved error reporting
        failed_jobs = [job for job in log_parser.table.progress_bars.tasks if job.completed < (steps or 100)]
        if failed_jobs:
            logger.error("\n=== Failed Jobs Report ===")
            for job in failed_jobs:
                logger.error(f"\nJob {job.fields['name']}:")
                logger.error(f"Completed steps: {job.completed} / {steps or 100}")
                for field, value in job.fields.items():
                    if field in ['error', 'traceback', 'return_code', 'last_output']:
                        logger.error(f"{field}: {value}")
            logger.error("=== End Failed Jobs Report ===\n")

    except Exception as e:
        logger.error(f"Critical error in run_sweep: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
    finally:
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
