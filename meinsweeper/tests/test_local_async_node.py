import asyncio
import pytest
import torch
import os
from meinsweeper.modules.nodes.local_async_node import LocalAsyncNode
from meinsweeper.modules.helpers.debug_logging import DEBUG

# Set the debug flag for the test
os.environ['MEINSWEEPER_DEBUG'] = 'True'

# Use pytest.mark.asyncio to mark the test as asynchronous
@pytest.mark.asyncio
async def test_local_async_node(caplog):
    caplog.set_level(logging.DEBUG if DEBUG else logging.INFO)
    
    # Create a mock log queue
    log_q = asyncio.Queue()

    # Create a LocalAsyncNode with GPUs 0 and 1 available (adjust as needed for your system)
    node = LocalAsyncNode(log_q=log_q, available_gpus=['0', '1'])

    # Open the connection (this will check for GPU availability)
    connected = await node.open_connection()
    assert connected, "Failed to open connection to LocalAsyncNode"

    # Create two dummy job commands
    dummy_job_cmd = """
import torch
import time

device = torch.device('cuda')
tensor = torch.rand(10000, 10000, device=device)  # Create a large tensor
print(f'Created tensor of shape {tensor.shape} on {device}')
time.sleep(15)  # Wait for 15 seconds
print('Job completed')
    """

    # Wrap the command in a Python interpreter call
    cmd1 = f"python -c \"{dummy_job_cmd}\""
    cmd2 = f"python -c \"{dummy_job_cmd}\""

    # Run the two jobs concurrently
    task1 = asyncio.create_task(node.run(cmd1, "job1"))
    task2 = asyncio.create_task(node.run(cmd2, "job2"))

    # Wait for both jobs to complete
    results = await asyncio.gather(task1, task2)

    # Check if both jobs completed successfully
    assert all(results), "Not all jobs completed successfully"

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

    # Print captured logs
    print("\nCaptured logs:")
    print(caplog.text)

    # Clean up
    node.cleanup()

# Remove the following lines as they're not needed when using pytest
# if __name__ == "__main__":
#     asyncio.run(test_local_async_node())