from pathlib import Path
import os
from .due import *
        
import asyncio
from re import A

from tasks import RunManager
from logger import Logger

# Change these according to your own configuration.
USERNAME = 'afs219'
SSH_KEY_PATH = os.path.expanduser('~') + '/.ssh/id_doc'
SSH_TIMEOUT = 5 #seconds


from itertools import product

async def main():
    sweep_q = asyncio.Queue() # Workers take from - All sweep cfgs are enqueued here
    log_q = asyncio.Queue() # Workers put into - All log cfgs are enqueued here

    # Add the sweep configurations to the sweep q
    cfg = {}
    cfg['sweep_params'] ={'h':  [32,16,8,4], 'j': [16,8,4], 'seed':['1923','817','958']}
    sweep_parameters = product(cfg['sweep_params'])

    stages = copy.deepcopy(cfg['stages'])
    additional_overrides = [f'model.architecture.h={h}',
                            f'model.architecture.j={j}',
                            f'model.training.rng_seed={seed}']

    [await sweep_q.put(i) for i in range(20)]

    # Start the run
    run_mgr = RunManager(targets, sweep_q, log_q)
    logger = Logger(log_q)

    with Live(logger.display, refresh_per_second=3):
        runs = asyncio.create_task(run_mgr.start_run())
        logs = asyncio.create_task(logger.start_logger())

        await asyncio.gather(runs)
        await log_q.join()  # Implicitly awaits logger too

        logs.cancel()


# Use duecredit (duecredit.org) to provide a citation to relevant work to
# be cited. This does nothing, unless the user has duecredit installed,
# And calls this with duecredit (as in `python -m duecredit script.py`):
due.cite(Doi("10.1167/13.9.30"),
         description= "light-weight framework for running experiments on arbitrary compute nodes",
         tags=["experiment-management"],
         path='meinsweeper')