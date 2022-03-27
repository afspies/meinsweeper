import os
from .due import *
import asyncio

from rich.live import Live
from .modules import RunManager, Logger, dict_product

# Change these according to your own configuration.

SSH_TIMEOUT = 5 #seconds

from itertools import product

def run_sweep(cfg):
    asyncio.run(main(cfg))

async def main(cfg):
    sweep_q = asyncio.Queue() # Workers take from - All sweep cfgs are enqueued here
    log_q = asyncio.Queue() # Workers put into - All log cfgs are enqueued here

    # Add the sweep configurations to the sweep q
    sweep_parameters = dict_product(cfg['sweep_params'])
    [await sweep_q.put(sweep_run) for sweep_run in sweep_parameters]

    # Start the run
    run_mgr = RunManager(cfg['targets'], sweep_q, log_q)
    logger = Logger(log_q)

    with Live(logger.display, refresh_per_second=3) as live:
        runs = asyncio.create_task(run_mgr.start_run())
        logs = asyncio.create_task(logger.start_logger())

        await asyncio.gather(runs)
        await log_q.join()  # Implicitly awaits logger too

        logger.complete(live) #This could be moved into logger, but requires keeping track of pids
        logs.cancel()


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description= "light-weight framework for running experiments on arbitrary compute nodes",
         tags=["experiment-management"],
         path='meinsweeper')