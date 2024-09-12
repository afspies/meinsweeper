import os
import pytest
from meinsweeper.meinsweeper import run_sweep
from meinsweeper.modules.logger import MSLogger
import asyncio

# Set the debug flag for the test
os.environ['MEINSWEEPER_DEBUG'] = 'True'

@pytest.mark.asyncio
async def test_run_sweep_local():
    # Create a dummy job command that uses MSLogger
    dummy_job_cmd = """
import time
import torch
from meinsweeper.modules.logger import MSLogger

print("Starting job")
logger = MSLogger()

# Log initial training step
logger.log_loss(0.5, mode='train', step=0)
print('Initial step logged')

# Create a large tensor on GPU to ensure GPU usage
device = torch.device('cuda')
tensor = torch.rand(10000, 10000, device=device)
print(f'Created tensor of shape {tensor.shape} on {device}')
0
# Wait for 5 seconds
print('Waiting for 5 seconds')
time.sleep(5)

# Log final training step
logger.log_loss(0.1, mode='train', step=100)
print('Final step logged')
print("Job completed")
    """

    # Create a list of commands for the sweep
    commands = [(dummy_job_cmd, f"job{i}") for i in range(3)]

    # Define targets (using local_async nodes)
    targets = {
        'gpu0': {
            'type': 'local_async',
            'params': {
                'gpus': ['0']
            }
        },
        'gpu1': {
            'type': 'local_async',
            'params': {
                'gpus': ['1']
            }
        }
    }

    # Create a log queue to capture all messages
    log_q = asyncio.Queue()

    # Run the sweep
    await run_sweep(commands, targets, steps=2)

    # Print all messages from the log queue
    print("\nLog messages:")
    while not log_q.empty():
        msg = await log_q.get()
        print(msg)

    print("Sweep completed")

if __name__ == "__main__":
    asyncio.run(test_run_sweep_local())