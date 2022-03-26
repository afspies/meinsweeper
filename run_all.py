import os
import asyncio
from re import A

from tasks import RunManager
from logger import Logger

# Change these according to your own configuration.
USERNAME = 'afs219'
SSH_KEY_PATH = os.path.expanduser('~') + '/.ssh/id_doc'
SSH_TIMEOUT = 5 #seconds

# targets = [
    # {'type':'ssh', 'params':{'address':'gpu12.doc.ic.ac.uk','username':USERNAME, 'key_path':SSH_KEY_PATH}},
    # {'type':'ssh', 'params':{'address':'gpu14.doc.ic.ac.uk','username':USERNAME, 'key_path':SSH_KEY_PATH}}]
targets = []
for i in range(1, 10):
    targets.append({'type':'ssh', 'params':{'address':f'gpu{i:02}.doc.ic.ac.uk','username':USERNAME, 'key_path':SSH_KEY_PATH}})
# for i in range(1, 30):
    # targets.append({'type':'ssh', 'params':{'address':f'ray{i:02}.doc.ic.ac.uk','username':USERNAME, 'key_path':SSH_KEY_PATH}})

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
    
if __name__ == "__main__":
    asyncio.run(main())
