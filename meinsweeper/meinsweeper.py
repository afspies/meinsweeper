import asyncio

from rich.live import Live

from .due import *

# global console # will be set in main
from .modules import LogParser, RunManager

# Change these according to your own configuration.

SSH_TIMEOUT = 10  #seconds


def run_sweep(cmd_list, targets, steps=None):
    # cmd_list should be a list of strings or list of tuples (str, str)
    if isinstance(cmd_list[0], str): # no labels included
        cmd_list = [(cmd, f'job {i}') for i,cmd in enumerate(cmd_list)]
    asyncio.run(main(cmd_list, targets, steps=steps))

async def main(cmd_list, targets, steps=None):
    sweep_q = asyncio.Queue()  # Workers take from - All sweep cfgs are enqueued here
    log_q = asyncio.Queue()  # Workers put into - All log cfgs are enqueued here

    [await sweep_q.put(sweep_run) for sweep_run in cmd_list]

    # Start the run
    run_mgr = RunManager(targets, sweep_q, log_q)
    logger = LogParser(log_q, steps=steps)

    with Live(logger.display, refresh_per_second=3) as live:
        runs = asyncio.create_task(run_mgr.start_run())
        logs = asyncio.create_task(logger.start_logger())

        await asyncio.gather(runs)
        await log_q.join()  # Implicitly awaits logger too

        logger.complete(live)  #This could be moved into logger, but requires keeping track of pids
        logs.cancel()

# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(
    Doi("10.1167/13.9.30"),
    description="light-weight framework for running experiments on arbitrary compute nodes",
    tags=["experiment-management"],
    path='meinsweeper'
)
