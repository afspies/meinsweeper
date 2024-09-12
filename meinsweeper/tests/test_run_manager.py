import asyncio
import pytest
import os
from meinsweeper.modules.run_manager import RunManager
from meinsweeper.modules.helpers.debug_logging import init_node_logger, DEBUG

# Set the debug flag for the test
os.environ['MEINSWEEPER_DEBUG'] = 'True'

@pytest.mark.asyncio
async def test_run_manager_with_local_async_node():
    # Create task and log queues
    task_q = asyncio.Queue()
    log_q = asyncio.Queue()

    # Create a dummy job command
    dummy_job_cmd = """
import torch
import time

device = torch.device('cuda')
tensor = torch.rand(10000, 10000, device=device)  # Create a large tensor
print(f'Created tensor of shape {tensor.shape} on {device}')
time.sleep(15)  # Wait for 15 seconds
print('Job completed')
    """

    # Add two tasks to the task queue
    await task_q.put((dummy_job_cmd, "job1"))
    await task_q.put((dummy_job_cmd, "job2"))

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

    # Create RunManager instance
    run_manager = RunManager(targets, task_q, log_q)

    # Start the run
    await run_manager.start_run()

    # Check the log queue for expected messages
    expected_messages = [
        "running",
        "Created tensor of shape torch.Size([10000, 10000]) on cuda",
        "Job completed"
    ]

    messages_found = {msg: 0 for msg in expected_messages}

    while not log_q.empty():
        msg = await log_q.get()
        print('got message', msg)
        
        if isinstance(msg, tuple) and len(msg) == 3:
            log_content = msg[0]
            print('log_content ', log_content)
            if isinstance(log_content, str):
                for expected in expected_messages:
                    if expected in log_content:
                        messages_found[expected] += 1
                        break
            elif isinstance(log_content, dict) and log_content.get('status') == 'running':
                messages_found['running'] += 1
            elif isinstance(log_content, tuple) and len(log_content) == 2 and log_content[1] == 'running':
                messages_found['running'] += 1

    # Check if we found all expected messages at least twice (once for each job)
    for msg, count in messages_found.items():
        assert count >= 2, f"Expected message not found enough times: {msg}. Found {count} times."

    # Ensure all tasks were completed
    assert task_q.empty(), "Not all tasks were completed"

if __name__ == "__main__":
    asyncio.run(test_run_manager_with_local_async_node())